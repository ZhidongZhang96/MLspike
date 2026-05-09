from __future__ import annotations

from typing import Iterable

import numpy as np

from .params import autosigma_default_params
from .utils import fft_frequencies, rms, struct_merge


def spk_autosigma(calcium, dt: float, psig: dict | str | None = None) -> float:
    if isinstance(calcium, (list, tuple)):
        data = [np.asarray(c, dtype=float).reshape(-1) for c in calcium]
    else:
        data = [np.asarray(calcium, dtype=float).reshape(-1)]

    if psig is None:
        params = autosigma_default_params()
    elif isinstance(psig, str):
        params = autosigma_default_params(psig)
    else:
        params = autosigma_default_params()
        params = struct_merge(params, psig, recursive=True)

    freqs = params["freqs"]
    sigma_est = []
    for x in data:
        if params.get("donormalize", False):
            x = x / np.mean(x)
        nt = len(x)
        if nt == 0:
            sigma_est.append(0.0)
            continue
        if freqs == "diff":
            sigma_est.append(rms(np.diff(x)) / np.sqrt(2))
            continue
        fnyquist = 1.0 / (2 * dt)
        if fnyquist < 1.5 * (freqs[0] if isinstance(freqs, (list, tuple)) else freqs):
            raise ValueError("sampling rate too low for proper estimation of the noise")
        if np.isscalar(freqs):
            freqs = [float(freqs), fnyquist]
        xf = np.fft.fft(x) / np.sqrt(nt)
        fftfreqs = np.abs(fft_frequencies(nt, 1 / dt))
        okfreq = (fftfreqs >= freqs[0]) & (fftfreqs <= freqs[1])
        sigma_est.append(rms(xf[okfreq]))

    sigma_est = params["bias"] * float(np.mean(sigma_est))
    return sigma_est
