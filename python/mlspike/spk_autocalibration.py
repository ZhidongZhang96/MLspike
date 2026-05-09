from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

PERCENT_SCALE = 100.0

from .params import autocalibration_default_params
from .spk_autosigma import spk_autosigma
from .spk_calcium import spk_calcium
from .tps_mlspikes import tps_mlspikes
from .utils import filt, rms, struct_merge, timevector


def spk_autocalibration(calcium, dt_or_pax=None, sigmaonly: bool = False, **kwargs):
    if isinstance(calcium, str) and calcium == "par":
        dt = dt_or_pax if not isinstance(dt_or_pax, dict) else dt_or_pax.get("dt")
        pax = autocalibration_default_params(dt)
        if isinstance(dt_or_pax, dict):
            pax = struct_merge(pax, dt_or_pax, recursive=True)
        if kwargs:
            pax = struct_merge(pax, kwargs, recursive=True)
        return pax

    if isinstance(dt_or_pax, dict):
        pax = struct_merge(autocalibration_default_params(), dt_or_pax, recursive=True)
    else:
        pax = autocalibration_default_params(dt_or_pax)
    if kwargs:
        pax = struct_merge(pax, kwargs, recursive=True)

    calcium_list = _to_list(calcium)
    calcium_list = [np.asarray(c, dtype=float).reshape(-1) for c in calcium_list]
    calcium_list = [c / np.mean(c) for c in calcium_list]

    dt = pax["dt"]
    if np.isscalar(dt):
        dt_list = [float(dt) for _ in calcium_list]
    else:
        dt_list = list(dt)

    sigma_est = spk_autosigma(calcium_list, dt_list[0], pax["autosigmasettings"])
    if sigmaonly:
        return sigma_est

    par = tps_mlspikes("par")
    par["a"] = pax["eventa"]
    par["tau"] = pax["eventtau"]
    par["drift"]["parameter"] = pax["driftparam"]
    par["drift"]["baselinestart"] = pax["baselinestart"]
    par["finetune"]["spikerate"] = pax["eventrate"]
    par["finetune"]["sigma"] = sigma_est
    par["special"]["nonintegerspike_minamp"] = pax["nonintegerspike_minamp"]
    par["display"] = "none"
    par = struct_merge(par, pax.get("mlspikepar", {}), recursive=True)

    events = []
    kept_events = []
    kept_nn = []
    mod_calcium = []
    n_list = []
    fit_list = []
    drift_list = []

    for i, series in enumerate(calcium_list):
        par_i = struct_merge(par, {"dt": dt_list[i]}, recursive=True)
        n, fit, drift, _ = tps_mlspikes(series, par_i)
        n = np.asarray(n, dtype=float).reshape(-1)
        fit = np.asarray(fit, dtype=float).reshape(-1)
        drift = np.asarray(drift, dtype=float).reshape(-1)
        n_list.append(n)
        fit_list.append(fit)
        drift_list.append(drift)
        ev, kept, nn, modc = _detect_events(series, n, fit, drift, pax, par_i)
        events.append(ev)
        kept_events.append(kept)
        kept_nn.append(nn)
        mod_calcium.append(modc)

    if not any(len(k) for k in kept_events):
        return [], [], sigma_est, []

    idx = [i for i, k in enumerate(kept_events) if len(k)]
    kept_events1 = [kept_events[i] for i in idx]
    mod_calcium1 = [mod_calcium[i] for i in idx]
    dt1 = [dt_list[i] for i in idx]

    tau_candidates = np.linspace(pax["taumin"], pax["taumax"], 25)
    errors = []
    for tau in tau_candidates:
        e, _, _ = _energy(tau, kept_events1, mod_calcium1, dt1, pax)
        errors.append(e)
    tau = float(tau_candidates[int(np.argmin(errors))])

    _, fit1, amps = _energy(tau, kept_events1, mod_calcium1, dt1, pax)
    amps_flat = np.concatenate([a for a in amps if len(a)]) if amps else np.array([])
    if amps_flat.size == 0:
        return [], [], sigma_est, []
    amp0 = float(np.median(amps_flat))
    amp0 = float(np.clip(amp0, pax["amin"], pax["amax"]))

    eventdesc = []
    spikes1 = []
    for i, ev in enumerate(kept_events1):
        amps_i = amps[i]
        nums = np.clip(np.round(amps_i / amp0), 1, 20).astype(int)
        spikes = []
        for t, nsp in zip(ev, nums):
            spikes.extend([t] * int(nsp))
        spikes1.append(np.asarray(spikes, dtype=float))
        eventdesc.append({"time": ev, "amp": amps_i, "num": nums})

    amp_candidates = np.linspace(pax["amin"], pax["amax"], 15)
    tau_candidates = np.linspace(pax["taumin"], pax["taumax"], 15)
    best = None
    best_params = (amp0, tau)
    for amp in amp_candidates:
        for tau_c in tau_candidates:
            e, _, _ = _energycalib(amp, tau_c, spikes1, mod_calcium1, dt1, pax)
            if best is None or e < best:
                best = e
                best_params = (amp, tau_c)

    amp, tau = best_params
    return tau, amp, sigma_est, eventdesc


def _detect_events(calcium, n, fit, drift, pax, par):
    dt = par["dt"]
    T = dt * len(calcium)
    kevent = np.where(n > 0)[0]
    events = kevent * dt
    kept_events = []
    kept_nn = np.zeros_like(n)

    lastevent = -np.inf
    i = 0
    while i < len(kevent):
        idx = kevent[i]
        tj = events[i]
        if tj - lastevent <= pax["tbef"]:
            lastevent = tj
            i += 1
            continue
        last_idx = idx
        group = [i]
        if pax["eventtspan"] > 0:
            while i + 1 < len(kevent) and events[i + 1] - tj <= pax["eventtspan"]:
                i += 1
                group.append(i)
                last_idx = kevent[i]
        idxs = kevent[group]
        ni = np.sum(n[idxs])
        idx1 = int(round(np.sum(idxs * n[idxs]) / max(ni, 1)))
        if T - tj <= pax["taft"]:
            i += 1
            continue
        if i + 1 < len(kevent) and events[i + 1] - tj <= pax["taft"]:
            i += 1
            continue
        idxbef = max(idx1 - 1, 0)
        if fit[idxbef] - drift[idxbef] > pax["cmax"]:
            i += 1
            continue
        if n[last_idx] > pax["maxamp"]:
            i += 1
            continue
        kept_events.append(idx1 * dt)
        kept_nn[idx1] = round(ni * par["a"] * PERCENT_SCALE)
        lastevent = tj
        i += 1

    if len(kept_events) == 0:
        return events, kept_events, kept_nn, calcium

    othern = n.copy()
    othern[kevent] = 0
    pfwd = tps_mlspikes("par")
    pfwd["F0"] = 1.0
    pfwd["dt"] = dt
    pfwd["a"] = par["a"]
    pfwd["tau"] = par["tau"]
    othercalcium = tps_mlspikes(othern, pfwd)
    modcalcium = calcium - drift * (othercalcium - 1)

    return events, kept_events, kept_nn, modcalcium


def _energy(tau, events, F, dt, pax):
    pfwd0 = spk_calcium("par")
    pfwd0["a"] = pax["eventa"]
    pfwd0["tau"] = tau

    dif = []
    Fpred = []
    amps = []
    for i, ev in enumerate(events):
        Fi = np.asarray(F[i], dtype=float)
        nevent = len(ev)
        if nevent == 0:
            amps.append(np.array([]))
            Fpred.append(np.zeros_like(Fi))
            dif.append(Fi)
            continue
        A = np.zeros((len(Fi), nevent))
        pfwd = dict(pfwd0)
        pfwd["dt"] = dt[i]
        pfwd["T"] = len(Fi) * dt[i]
        for j, t in enumerate(ev):
            A[:, j] = spk_calcium(t, pfwd)[0].reshape(-1)
        A = filt(A, pax["tdrift"] / dt[i], mode="hmd")
        amps_i, *_ = np.linalg.lstsq(A, Fi, rcond=None)
        pred = A @ amps_i
        Fpred.append(pred)
        dif.append(Fi - pred)
        amps.append(amps_i)
    error = rms(np.concatenate(dif)) * PERCENT_SCALE
    return error, Fpred, amps


def _energycalib(amp, tau, spikes, F, dt, pax):
    pfwd = spk_calcium("par")
    pfwd["a"] = amp
    pfwd["tau"] = tau
    pfwd["saturation"] = pax["saturation"]
    pfwd["pnonlin"] = pax["pnonlin"]
    pfwd["hill"] = pax["hill"]
    pfwd["c0"] = pax["c0"]

    dif = []
    fit = []
    drift = []
    for i, spk in enumerate(spikes):
        Fi = np.asarray(F[i], dtype=float)
        pfwd_i = dict(pfwd)
        pfwd_i["dt"] = dt[i]
        pfwd_i["T"] = len(Fi) * dt[i]
        Fpred = spk_calcium(spk, pfwd_i)[0].reshape(-1)
        base = Fi / np.maximum(Fpred, 1e-12)
        if pax["tdrift"] == 0:
            drift_i = np.full_like(base, np.mean(base))
        else:
            drift_i = filt(base, pax["tdrift"] / dt[i], mode="lmd")
        fit_i = drift_i * Fpred
        dif.append(Fi - fit_i)
        fit.append(fit_i)
        drift.append(drift_i)
    error = rms(np.concatenate(dif)) * PERCENT_SCALE
    return error, fit, drift


def _to_list(values):
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]
