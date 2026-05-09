from __future__ import annotations

from typing import Any, Dict

from .utils import struct_merge


def tps_default_params(dt: float | None = None) -> Dict[str, Any]:
    par = {
        "dt": dt,
        "F0": None,
        "a": 0.1,
        "tau": 1.0,
        "ton": 0.0,
        "saturation": 0.0,
        "pnonlin": None,
        "hill": 1.0,
        "c0": 0.0,
        "drift": {
            "effect": "multiplicative",
            "parameter": 0.0,
            "baselinestart": False,
        },
        "finetune": {
            "spikerate": 0.1,
            "sigma": None,
            "autosigmasettings": "correlated",
        },
        "algo": {
            "estimate": "MAP",
            "nspikemax": 3,
            "cmax": 10.0,
            "nc": None,
            "nc_norise": 100,
            "nc_rise": 50,
            "nb": None,
            "nb_nodrift": 40,
            "nb_driftstate": 100,
            "nb_driftrise": 50,
            "nsample": None,
            "interpmode": "spline",
            "dogpu": False,
        },
        "special": {
            "nonintegerspike_minamp": 0.0,
            "burstcostsone": False,
        },
        "display": "none",
        "dographsummary": False,
    }
    return par


def calcium_default_params(**overrides: Any) -> Dict[str, Any]:
    par = {
        "dt": None,
        "T": None,
        "type": "1exp",
        "F0": 1.0,
        "delay": 0.0,
        "a": 0.1,
        "tau": 1.0,
        "ton": 0.0,
        "saturation": 0.0,
        "pnonlin": None,
        "sigma": 0.0,
        "hill": 1.0,
        "c0": 0.0,
        "x0": 0.0,
        "drift": {
            "method": "basis functions",
            "effect": "multiplicative",
            "parameter": 0.0,
        },
    }
    if overrides:
        par = struct_merge(par, overrides, recursive=True)
    return par


def autosigma_default_params(preset: str | None = None) -> Dict[str, Any]:
    par = {
        "freqs": [3.0, 20.0],
        "bias": 1.0,
        "donormalize": False,
    }
    if preset == "white":
        par["freqs"] = 3.0
    elif preset == "correlated":
        par["freqs"] = [3.0, 20.0]
    elif preset == "correlatedbias":
        par["freqs"] = [3.0, 20.0]
        par["bias"] = 0.7
    return par


def autocalibration_default_params(dt: float | None = None) -> Dict[str, Any]:
    par = {
        "dt": dt,
        "amin": 0.04,
        "amax": 0.1,
        "taumin": 0.4,
        "taumax": 1.6,
        "autosigmasettings": "correlated",
        "eventa": 0.1,
        "eventtau": 0.8,
        "saturation": 0.0,
        "pnonlin": None,
        "hill": 1.0,
        "c0": 1.0,
        "tdrift": 5.0,
        "driftparam": 0.005,
        "baselinestart": False,
        "eventrate": 0.00001,
        "nonintegerspike_minamp": 0.4,
        "mlspikepar": {},
        "maxamp": 2.5,
        "eventtspan": 0.0,
        "tbef": 1.0,
        "cmax": 0.01,
        "taft": 1.0,
        "histosmooth": None,
        "costfactor": [1.0, 0.5, 0.0],
        "display": "none",
        "realspikes": [],
        "reala": 0.0,
        "realtau": 0.0,
    }
    return par
