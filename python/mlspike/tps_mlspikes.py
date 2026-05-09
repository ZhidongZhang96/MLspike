from __future__ import annotations

from typing import Any, Dict, List, Tuple

import math

import numpy as np

from .params import tps_default_params
from .spk_autosigma import spk_autosigma
from .utils import (
    add,
    coerce,
    interp1_matrix,
    interp1_values,
    log2proba,
    logmultexp,
    logmultexp_column,
    logsample,
    logsumexp,
    mult,
    rms,
    struct_merge,
)


def tps_mlspikes(*args, **kwargs):
    if len(args) == 0:
        raise ValueError("Missing arguments")

    if isinstance(args[0], str) and args[0] == "par":
        dt = args[1] if len(args) > 1 else None
        return tps_default_params(dt=dt)

    x = args[0]
    par = args[1] if len(args) > 1 else None

    if isinstance(x, list):
        data = [np.asarray(c, dtype=float).reshape(-1) for c in x]
        xiscell = True
    else:
        xarr = np.asarray(x, dtype=float)
        if xarr.ndim > 1 and xarr.shape[1] > 1:
            data = [xarr[:, i] for i in range(xarr.shape[1])]
            xiscell = False
        else:
            data = [xarr.reshape(-1)]
            xiscell = False

    ndata = len(data)

    if isinstance(par, dict):
        base = tps_default_params()
        if np.isscalar(par.get("dt")) and ndata > 1:
            par_list = []
            for k in range(ndata):
                pk = struct_merge(base, par, recursive=True, skip_none=True)
                par_list.append(pk)
            pars = par_list
        else:
            pars = [struct_merge(base, par, recursive=True, skip_none=True)]
            if ndata > 1:
                pars = [struct_merge(base, par, recursive=True, skip_none=True) for _ in range(ndata)]
    else:
        if par is None:
            raise ValueError("dt must be provided")
        if np.isscalar(par):
            dt = float(par)
            pars = [tps_default_params(dt=dt) for _ in range(ndata)]
        else:
            raise ValueError("Invalid parameter format")

    for idx in range(ndata):
        if pars[idx]["dt"] is None:
            raise ValueError("sampling time not defined")

    finetunes = [p["finetune"] for p in pars]
    sigma_missing = all(p["sigma"] is None for p in finetunes)
    if sigma_missing:
        dt = pars[0]["dt"]
        sigma_est = spk_autosigma(data, dt, pars[0]["finetune"]["autosigmasettings"])
        for p in pars:
            p["finetune"]["sigma"] = sigma_est

    results = []
    for k, series in enumerate(data):
        results.append(_run_single(series, pars[k]))

    if len(results) == 1:
        return results[0]

    if not xiscell:
        merged = []
        for i in range(len(results[0])):
            if i in {2, 3}:
                merged.append(np.array([r[i] for r in results]))
            else:
                merged.append(np.stack([r[i] for r in results], axis=1))
        return tuple(merged)

    return tuple([list(output) for output in zip(*results)])


def _run_single(x: np.ndarray, par: Dict[str, Any]):
    if np.any(np.isnan(x)):
        return [], np.full_like(x, np.nan), [], par

    if np.sum(x == 0) > len(x) / 2:
        return _forward(np.asarray(x), par)

    return _backward(np.asarray(x), par)


def _forward(n: np.ndarray, par: Dict[str, Any]):
    n = np.asarray(n, dtype=float)
    dt = par["dt"]
    a = par["a"]
    tau = par["tau"]
    ton = par["ton"]
    sat = par["saturation"]
    pnonlin = par["pnonlin"]
    hill = par["hill"]
    c0 = par["c0"]

    T = len(n)

    decay = np.exp(-dt / tau)
    c = np.zeros(T)
    ct = 0.0
    for t in range(T):
        ct = ct * decay + n[t]
        c[t] = ct

    cn = c if hill == 1 else (c0 + c) ** hill - c0**hill
    p = cn / (1 + sat * cn)
    if ton > 0:
        ptarget = p.copy()
        pspeed = (1 + sat * cn) / ton
        p = np.zeros(T)
        pt = 0.0
        for t in range(T):
            pt = ptarget[t] + (pt - ptarget[t]) * np.exp(-pspeed[t] * dt)
            p[t] = pt
    if pnonlin is not None:
        p = np.polyval(list(reversed(pnonlin)) + [1 - sum(pnonlin), 0], p)

    ypred = 1 + a * p
    if par["drift"].get("estimate") is not None:
        if par["drift"]["effect"] == "additive":
            ypred = ypred + par["drift"]["estimate"]
        else:
            ypred = ypred * (1 + par["drift"]["estimate"])

    F0 = par["F0"] if par["F0"] is not None else 1.0
    return ypred * F0


def _backward(F: np.ndarray, par: Dict[str, Any]):
    F = np.asarray(F, dtype=float).reshape(-1)
    if par["F0"] is None:
        par = dict(par)
        F0min = max(min(F.min(), F.mean() / 1.5), F.min() - (F.max() - F.min()))
        if par["drift"]["parameter"]:
            F0max = np.percentile(F, 90)
        else:
            F0max = max(F.mean(), np.median(F))
        if F0min <= 0:
            F0min = F0max / 100
        par["F0"] = [F0min, F0max]

    do_drift = par["drift"]["parameter"] > 0
    do_baseline = not np.isscalar(par["F0"])

    if do_drift and not do_baseline:
        raise ValueError("Drift estimation requires baseline interval")

    if not do_baseline:
        par["algo"]["nb"] = 1
        par["F0"] = [par["F0"], par["F0"]]
        return _backward_fixbaseline(F, par)

    if not do_drift:
        return _backward_fixbaseline(F, par)

    return _backward_driftstate(F, par)


def _backward_fixbaseline(F: np.ndarray, par: Dict[str, Any]):
    if F.size == 0:
        return [], [], par

    if len(par["F0"]) != 2:
        raise ValueError("F0 should be an interval")

    F0 = float(np.mean(par["F0"]))
    baselineinterval = np.array(par["F0"], dtype=float) / F0
    a = par["a"]
    decay = np.exp(-par["dt"] / par["tau"])
    sat = par["saturation"]
    spikerate = par["finetune"]["spikerate"]
    sigmay = par["finetune"]["sigma"] / F0

    estimate = par["algo"]["estimate"].lower()
    do_map = estimate == "map"
    do_proba = estimate == "proba"
    do_sample = estimate in {"sample", "samples"}
    interpmode = "linear" if do_map else "linear"
    nsample = par["algo"]["nsample"] or 200

    nc = par["algo"]["nc"] or par["algo"]["nc_norise"]
    cmax = par["algo"]["cmax"]
    dc = cmax / (nc - 1)
    cc = np.arange(nc) * dc
    nb = par["algo"]["nb"] or par["algo"]["nb_nodrift"]
    bb = np.linspace(baselineinterval[0], baselineinterval[1], nb)

    nspikemax = par["algo"]["nspikemax"]
    MM = []
    cci = None
    for i in range(nspikemax + 1):
        if i == 0:
            cci = cc * decay
        else:
            cci = np.minimum(cci + 1, cmax)
        MM.append(interp1_matrix(cc, cci, mode=interpmode))

    if spikerate:
        nspikcost = np.ones(nspikemax + 1) if par["special"]["burstcostsone"] else np.arange(nspikemax + 1)
        lspike = spikerate * par["dt"] + np.log(np.array([math_factorial(n) for n in nspikcost])) - nspikcost * np.log(spikerate * par["dt"])
        pspike = np.exp(-lspike)
        pspike = pspike / np.sum(pspike)
        lspike = -np.log(pspike)
    else:
        lspike = np.zeros(nspikemax + 1)
        pspike = np.ones(nspikemax + 1) / (nspikemax + 1)

    if do_map:
        MM_cat = np.vstack(MM)
    else:
        MS = np.zeros_like(MM[0])
        for i in range(1, nspikemax + 1):
            MS += pspike[i] * MM[i]

    dye = 1 + a * cc / (1 + sat * cc)
    xxmeasure = dye[:, None] * bb[None, :]
    lmeasure = -np.log(1 / (np.sqrt(2 * np.pi) * sigmay))
    lcalcium = np.zeros((nc, nb))

    T = len(F)
    y = (F / F0).reshape(-1)

    if not do_map:
        L = np.zeros((nc, nb, T))
    N = np.zeros((nc, nb, T), dtype=int)

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            lt = np.zeros((nc, nb))
        else:
            if do_map:
                lt1 = (MM_cat @ lt).reshape(nspikemax + 1, nc, nb).transpose(1, 0, 2)
                lt1 = lspike[None, :, None] + lt1
                n1 = np.argmin(lt1, axis=1)
                lt = np.min(lt1, axis=1)
                N[:, :, t + 1] = n1
            else:
                lt = logmultexp(MS, lt)
        lt = lt + (lmeasure + (y[t] - xxmeasure) ** 2 / (2 * sigmay**2))
        if not do_map:
            L[:, :, t] = lt
        if t == 0:
            lt = lt + lcalcium

    if do_map:
        flat = np.argmin(lt)
        bidx = flat // nc
        cidx = flat % nc
        baseline = bb[bidx]
    elif do_proba:
        pt = np.exp(np.min(lt) - lt)
        pbaseline = np.sum(pt, axis=0)
        pbaseline = pbaseline / np.sum(pbaseline)
        baseline = np.sum(bb * pbaseline)
    else:
        baseline = float(np.mean(bb))

    if do_sample:
        nsample = int(nsample)
    else:
        nsample = 1

    n = np.zeros((T, nsample))
    xest = np.zeros((T, nsample))

    for t in range(T):
        if t == 0:
            if do_map:
                LL = lt[cidx, bidx]
                xest[t, 0] = cc[cidx]
            elif do_sample:
                cidx_s, bidx_s = logsample(lt, nsample)
                xest[t] = cc[cidx_s]
                baseline = bb[bidx_s]
                LL = None
            else:
                LL = logsumexp(lt)
                pt = log2proba(lt)
                xest[t] = np.sum(cc[:, None] * pt)
        else:
            if do_map:
                n[t, 0] = N[cidx, bidx, t]
                xest[t, 0] = np.minimum(xest[t - 1, 0] * decay + n[t, 0], cmax)
                cidx = int(np.clip(np.round(xest[t, 0] / dc), 0, nc - 1))
            elif do_sample:
                nspike = np.arange(nspikemax + 1)
                ct = xest[t - 1][:, None] * decay + nspike[None, :]
                Lt = L[:, :, t]
                if nb == 1:
                    ltk = np.zeros_like(ct)
                    for i in range(nsample):
                        ltk[i] = interp1_values(Lt[:, 0], ct[i] / dc, fill=np.inf)
                else:
                    ltk = np.zeros_like(ct)
                    for i in range(nsample):
                        col = Lt[:, bidx_s[i]]
                        ltk[i] = interp1_values(col, ct[i] / dc, fill=np.inf)
                lt2 = lspike[None, :] + ltk
                csel = logsample(lt2, mode="rows")
                n[t] = csel
                xest[t] = ct[np.arange(nsample), csel]
            else:
                lt1 = lt
                lmin = np.min(lt1)
                pt1 = np.exp(lmin - lt1)
                pt = MS @ pt1
                nt = np.zeros_like(pt)
                lt = lmin - np.log(np.maximum(pt, 1e-300))
                lty = lt + L[:, :, t]
                pty = log2proba(lty)
                n[t] = np.sum(nt * pty)
                xest[t] = np.sum(cc[:, None] * pty)
                lt = lt + (lmeasure + (y[t] - xxmeasure) ** 2 / (2 * sigmay**2))

    Ffit = (1 + a * xest) * baseline * F0
    drift = np.ones(T) * baseline * F0
    n_out = n if do_proba else n.astype(int)
    if nsample == 1:
        n_out = n_out.reshape(-1)
        Ffit = Ffit.reshape(-1)
        drift = drift.reshape(-1)
    return n_out, Ffit, drift, par


def _backward_driftstate(F: np.ndarray, par: Dict[str, Any]):
    if F.size == 0:
        return [], [], par

    if len(par["F0"]) != 2:
        raise ValueError("F0 should be an interval when estimating drift")

    if par["drift"]["effect"] == "additive":
        F0 = 1.0
        if not (par["F0"][0] < F0 < par["F0"][1]):
            raise ValueError("additive drifts require normalized signals")
    else:
        F0 = float(np.mean(par["F0"]))

    baselineinterval = np.array(par["F0"], dtype=float) / F0
    a = par["a"]
    decay = np.exp(-par["dt"] / par["tau"])
    sat = par["saturation"]
    hill = par["hill"]
    pnonlin = par["pnonlin"]
    spikerate = par["finetune"]["spikerate"]
    sigmay = par["finetune"]["sigma"] / F0
    sigmab = par["drift"]["parameter"] / F0 * np.sqrt(par["dt"])

    estimate = par["algo"]["estimate"].lower()
    do_map = estimate == "map"
    do_proba = estimate == "proba"
    do_sample = estimate in {"sample", "samples"}

    nc = par["algo"]["nc"] or par["algo"]["nc_norise"]
    cmax = par["algo"]["cmax"]
    dc = cmax / (nc - 1)
    cc = np.arange(nc) * dc
    nb = par["algo"]["nb"] or par["algo"]["nb_driftstate"]
    bb = np.linspace(baselineinterval[0], baselineinterval[1], nb)
    db = bb[1] - bb[0]

    nspikemax = par["algo"]["nspikemax"]
    MM = []
    cci = None
    for i in range(nspikemax + 1):
        if i == 0:
            cci = cc * decay
        else:
            cci = np.minimum(cci + 1, cmax)
        MM.append(interp1_matrix(cc, cci, mode="linear"))

    if spikerate:
        nspikcost = np.ones(nspikemax + 1) if par["special"]["burstcostsone"] else np.arange(nspikemax + 1)
        lspike = spikerate * par["dt"] + np.log(np.array([math_factorial(n) for n in nspikcost])) - nspikcost * np.log(spikerate * par["dt"])
        pspike = np.exp(-lspike)
        pspike = pspike / np.sum(pspike)
        lspike = -np.log(pspike)
    else:
        lspike = np.zeros(nspikemax + 1)
        pspike = np.ones(nspikemax + 1) / (nspikemax + 1)

    maxdrift = max(2, int(np.ceil(3 * sigmab / db)))
    drift_steps = np.arange(-maxdrift, maxdrift + 1)
    drift_kernel = np.exp(-0.5 * (drift_steps / (sigmab / db + 1e-9)) ** 2)
    drift_kernel = drift_kernel / drift_kernel.sum()
    ldrift = -np.log(np.maximum(drift_kernel, 1e-12))

    if do_map:
        MM_cat = np.vstack(MM)
    else:
        MS = np.zeros_like(MM[0])
        for i in range(nspikemax + 1):
            MS += pspike[i] * MM[i]

    ccn = (par["c0"] + cc) ** hill - par["c0"] ** hill
    if pnonlin is None:
        dye = 1 + a * ccn / (1 + sat * ccn)
    else:
        dye = 1 + a * np.polyval(list(reversed(pnonlin)) + [1 - sum(pnonlin), 0], ccn)

    if par["drift"]["effect"] == "additive":
        xxmeasure = (dye - 1)[:, None] + bb[None, :]
    else:
        xxmeasure = dye[:, None] * bb[None, :]
    lmeasure = -np.log(1 / (np.sqrt(2 * np.pi) * sigmay))
    lcalcium = np.zeros((nc, nb))

    T = len(F)
    y = (F / F0).reshape(-1)

    if not do_map:
        L = np.zeros((nc, nb, T))
    N = np.zeros((nc, nb, T), dtype=int)
    D = np.zeros((nc, nb, T), dtype=float)

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            lt = np.zeros((nc, nb))
        else:
            if do_map:
                lt1 = (MM_cat @ lt).reshape(nspikemax + 1, nc, nb).transpose(1, 0, 2)
                lt1 = lspike[None, :, None] + lt1
                n1 = np.argmin(lt1, axis=1)
                lt = np.min(lt1, axis=1)
                N[:, :, t + 1] = n1
            else:
                lt = logmultexp(MS, lt)

            if do_map:
                lt1 = np.zeros((nc, nb, drift_steps.size))
                for i, step in enumerate(drift_steps):
                    idx = np.clip(np.arange(nb) + step, 0, nb - 1)
                    lt1[:, :, i] = lt[:, idx] + ldrift[i]
                idrift = np.argmin(lt1, axis=2)
                lt = np.min(lt1, axis=2)
                D[:, :, t + 1] = drift_steps[idrift] * db
            else:
                lt = logmultexp_column(_drift_matrix(nb, drift_kernel), lt)

        lt = lt + (lmeasure + (y[t] - xxmeasure) ** 2 / (2 * sigmay**2))
        if not do_map:
            L[:, :, t] = lt
        if t == 0:
            lt = lt + lcalcium

    nsample = par["algo"]["nsample"] or 200
    if not do_sample:
        nsample = 1

    n = np.zeros((T, nsample))
    xest = np.zeros((T, 2, nsample))

    for t in range(T):
        if t == 0:
            if do_map:
                flat = np.argmin(lt)
                bidx = flat // nc
                cidx = flat % nc
                xest[t, 0, :] = cc[cidx]
                xest[t, 1, :] = bb[bidx]
            elif do_sample:
                cidx, bidx = logsample(lt, nsample)
                xest[t, 0, :] = cc[cidx]
                xest[t, 1, :] = bb[bidx]
            else:
                pt = log2proba(lt)
                xest[t, 0, 0] = np.sum(cc[:, None] * pt)
                xest[t, 1, 0] = np.sum(bb[None, :] * pt)
        else:
            if do_map:
                xest[t, 1, 0] = coerce(xest[t - 1, 1, 0] + D[cidx, bidx, t], (baselineinterval[0], baselineinterval[1]))
                bidx = int(np.clip(np.round((xest[t, 1, 0] - bb[0]) / db), 0, nb - 1))
                n[t, 0] = N[cidx, bidx, t]
                xest[t, 0, 0] = min(xest[t - 1, 0, 0] * decay + n[t, 0], cmax)
                cidx = int(np.clip(np.round(xest[t, 0, 0] / dc), 0, nc - 1))
            elif do_sample:
                for k in range(nsample):
                    ct = xest[t - 1, 0, k] * decay + np.arange(nspikemax + 1)
                    bt = xest[t - 1, 1, k] + drift_steps * db
                    ltk = np.zeros((nspikemax + 1, drift_steps.size))
                    Lt = L[:, :, t]
                    for i in range(nspikemax + 1):
                        cidx0 = np.clip(np.round(ct[i] / dc).astype(int), 0, nc - 1)
                        for j, bval in enumerate(bt):
                            bidx0 = np.clip(np.round((bval - bb[0]) / db).astype(int), 0, nb - 1)
                            ltk[i, j] = Lt[cidx0, bidx0]
                    ltk = ltk + lspike[:, None] + ldrift[None, :]
                    csel, bsel = logsample(ltk, mode="2D")
                    n[t, k] = csel
                    xest[t, 0, k] = ct[csel]
                    xest[t, 1, k] = bt[bsel]
            else:
                lt1 = lt
                lmin = np.min(lt1)
                pt1 = np.exp(lmin - lt1)
                pt1b = pt1 @ _drift_matrix(nb, drift_kernel)
                pt = MS @ pt1b
                lt = lmin - np.log(np.maximum(pt, 1e-300))
                lty = lt + L[:, :, t]
                pty = log2proba(lty)
                n[t, 0] = np.sum(pty)
                xest[t, 0, 0] = np.sum(cc[:, None] * pty)
                xest[t, 1, 0] = np.sum(bb[None, :] * pty)
                lt = lt + (lmeasure + (y[t] - xxmeasure) ** 2 / (2 * sigmay**2))

    if do_sample:
        n_out = n
        drift = xest[:, 1, :].T * F0
        Ffit = (1 + a * xest[:, 0, :].T) * drift
    else:
        n_out = n[:, 0]
        drift = xest[:, 1, 0] * F0
        Ffit = (1 + a * xest[:, 0, 0]) * drift

    return n_out, Ffit, drift, par


def _drift_matrix(nb: int, kernel: np.ndarray) -> np.ndarray:
    half = kernel.size // 2
    mat = np.zeros((nb, nb))
    for i in range(nb):
        for k, weight in enumerate(kernel):
            j = i + k - half
            if 0 <= j < nb:
                mat[i, j] += weight
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def math_factorial(x: int) -> int:
    return 1 if x <= 1 else int(math.factorial(int(x)))
