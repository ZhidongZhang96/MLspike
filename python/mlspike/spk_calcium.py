from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .params import calcium_default_params
from .utils import is_spike_count_vector, struct_merge, timevector

MAX_EXP_WINDOW = 5.0


def spk_calcium(*args, **kwargs):
    if len(args) == 0:
        raise ValueError("Missing arguments")
    if isinstance(args[0], str) and args[0] == "par":
        par = calcium_default_params()
        if len(args) > 1:
            par = _parse_par_args(par, args[1:])
        if kwargs:
            par = struct_merge(par, kwargs, recursive=True)
        return par

    spikes = args[0]
    if len(args) == 2 and isinstance(args[1], dict):
        par = struct_merge(calcium_default_params(), args[1], recursive=True, skip_none=True)
    else:
        par = calcium_default_params()
        par = _parse_par_args(par, args[1:])
        if kwargs:
            par = struct_merge(par, kwargs, recursive=True)

    if not isinstance(spikes, (list, tuple)):
        if np.ndim(spikes) == 1:
            spikes = [spikes]
        else:
            spikes = [np.asarray(spikes)[:, i] for i in range(np.asarray(spikes).shape[1])]
    elif len(spikes) == 0:
        spikes = [np.array([])]

    spikes = timevector(spikes, par["dt"], mode="times")
    if not isinstance(spikes, (list, tuple)):
        spikes = [spikes]

    ndata = len(spikes)
    nt = None
    if spikes and len(spikes[0]) > 0 and is_spike_count_vector(spikes[0]):
        nt = len(spikes[0])

    if par["T"] is None:
        if nt is None:
            par["T"] = max([np.max(s) if len(s) else 0 for s in spikes]) + 1
        else:
            par["T"] = nt * par["dt"]
    else:
        if nt is not None and not np.isclose(par["T"], nt * par["dt"]):
            raise ValueError("par.T is not consistent with length of spikes input(s)")

    if ndata == 1:
        return _forward(spikes[0], par)

    results = []
    for k, spk in enumerate(spikes):
        par_k = dict(par)
        if not np.isscalar(par["dt"]):
            par_k["dt"] = par["dt"][k]
        if not np.isscalar(par["T"]):
            par_k["T"] = par["T"][k]
        results.append(_forward(spk, par_k))

    F, F0, drift = zip(*results)
    return list(F), list(F0), list(drift)


def _parse_par_args(par: Dict[str, Any], args: Tuple[Any, ...]) -> Dict[str, Any]:
    parsed = {}
    i = 0
    while i < len(args):
        a = args[i]
        if np.isscalar(a):
            if parsed.get("dt") is None:
                parsed["dt"] = float(a)
            else:
                parsed["T"] = float(a)
        elif isinstance(a, str):
            if a in {"1exp", "3exps"}:
                parsed["type"] = a
            else:
                if i + 1 >= len(args):
                    raise ValueError("Missing value for parameter")
                i += 1
                b = args[i]
                if "." in a:
                    head, tail = a.split(".", 1)
                    parsed.setdefault(head, {})[tail] = b
                else:
                    parsed[a] = b
        i += 1
    return struct_merge(par, parsed, recursive=True)


def _forward(spikes: np.ndarray, par: Dict[str, Any]):
    dt = float(par["dt"])
    a = par["a"]
    tau = par["tau"]
    ton = par["ton"]
    hill = par["hill"]
    c0 = par["c0"]
    sat = par["saturation"]
    pnonlin = par["pnonlin"]
    nt = int(round(par["T"] / dt))

    spikes = np.asarray(spikes, dtype=float)
    spikesactive = spikes + par["delay"]

    if par["type"] == "1exp":
        increase = np.zeros(nt)
        for tk in spikesactive:
            ik = 1 + int(np.ceil(tk / dt))
            if 1 <= ik <= nt:
                increase[ik - 1] += np.exp(-(((ik - 1) * dt - tk) / tau))
        decay = np.exp(-dt / tau)
        ct = par.get("x0", 0.0)
        c = np.zeros(nt)
        for i in range(nt):
            ct = ct * decay + increase[i]
            c[i] = ct
        cn = (c0 + c) ** hill - c0**hill
        p = cn / (1 + sat * cn)
        if ton > 0:
            ptarget = p.copy()
            pspeed = (1 + sat * cn) / ton
            p = np.zeros(nt)
            pt = 0.0
            for t in range(nt):
                pt = ptarget[t] + (pt - ptarget[t]) * np.exp(-pspeed[t] * dt)
                p[t] = pt
        if pnonlin is not None:
            p = np.polyval(list(reversed(pnonlin)) + [1 - sum(pnonlin), 0], p)
        ypred = a * p
    elif par["type"] == "3exps":
        if ton > 0:
            raise ValueError("3exps with rise time not implemented")
        tt = np.arange(nt) * dt
        a1, a2 = a
        ton = tau[0]
        t1 = tau[1]
        t2 = tau[2]
        ypred = np.zeros(nt)
        for tk in spikesactive:
            tti = tt - tk
            idx = (tti > 0) & (tti < MAX_EXP_WINDOW)
            tti = tti[idx]
            ypred[idx] += (tti > 0) * (1 - np.exp(-tti / ton)) * (a1 * np.exp(-tti / t1) + a2 * np.exp(-tti / t2))
        if pnonlin is not None:
            raise ValueError("nonlinear output not supported with 3exps")
    else:
        raise ValueError(f"Unknown calcium type {par['type']}")

    F0 = (1 + ypred) * par["F0"]

    drift = np.zeros(nt)
    if par["drift"]["parameter"]:
        if par["drift"]["method"] == "state":
            innovation = par["drift"]["parameter"] * np.random.randn(nt)
            memory = np.exp(-np.arange(0, 40 + dt, dt) / 10)
            drift = np.convolve(innovation, memory, mode="full")[:nt]
        elif par["drift"]["method"] == "basis functions":
            ndrift = int(par["drift"]["parameter"][0] if isinstance(par["drift"]["parameter"], (list, tuple)) else par["drift"]["parameter"])
            drifts = np.linspace(-1, 1, nt)
            if ndrift > 1:
                phase = np.linspace(0, 2 * np.pi, nt)
                nsin = (ndrift - 1) // 2
                extra = []
                for k in range(1, nsin + 1):
                    extra.append(np.sin(k * phase))
                    extra.append(np.cos(k * phase))
                drifts = np.column_stack([drifts] + extra)
            if isinstance(par["drift"]["parameter"], (list, tuple)) and len(par["drift"]["parameter"]) > 1:
                amp = par["drift"]["parameter"][1]
            else:
                amp = 0.1
            beta = amp * np.random.randn(drifts.shape[1])
            drift = drifts @ beta
        else:
            raise ValueError("unknown drift method")

    if par["drift"]["effect"] == "additive":
        ypred = ypred + drift
    elif par["drift"]["effect"] == "multiplicative":
        ypred = (1 + ypred) * (1 + drift) - 1
    else:
        raise ValueError("unknown drift effect")

    if par["sigma"]:
        ypred = ypred + np.random.randn(nt) * par["sigma"]

    F = (1 + ypred) * par["F0"]
    return F.reshape(-1, 1), F0.reshape(-1, 1), drift.reshape(-1, 1)
