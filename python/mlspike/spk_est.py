from __future__ import annotations

from typing import Any, Dict

from .tps_mlspikes import tps_mlspikes
from .utils import timevector, struct_merge
from .params import tps_default_params


def spk_est(calcium, par: Dict[str, Any] | float | None = None):
    if isinstance(calcium, str) and calcium == "par":
        return tps_mlspikes("par")

    defaultpar = tps_mlspikes("par")
    if isinstance(par, (int, float)):
        dt = float(par)
        par = tps_default_params(dt=dt)
    elif par is None:
        raise ValueError("Parameter dt or par must be provided")
    else:
        par = struct_merge(defaultpar, par, recursive=True)

    n, fit, drift, parest = tps_mlspikes(calcium, par)
    estimate = par["algo"]["estimate"].lower()
    if estimate == "map":
        spk = timevector(n, par["dt"], mode="times")
    elif estimate == "proba":
        spk = n
    elif estimate in {"sample", "samples"}:
        spk = timevector(n, par["dt"], mode="times")
    else:
        spk = n
    return spk, fit, drift, parest
