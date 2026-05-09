from __future__ import annotations

from typing import Iterable, List

import numpy as np


def spk_gentrain(rate: float, T: float, mode: str = "fix-rate", parameters: Iterable[float] | None = None, repeat: int | None = None):
    if parameters is None:
        parameters = []
    parameters = list(parameters)
    nrepeat = 1 if repeat is None else int(repeat)

    def _fix_rate() -> List[np.ndarray]:
        nspike = np.random.poisson(rate * T, size=nrepeat)
        trains = []
        for k in range(nrepeat):
            trains.append(np.sort(np.random.rand(nspike[k]) * T))
        return trains

    def _periodic() -> List[np.ndarray]:
        precision = parameters[0] if parameters else 0.2
        period = 1 / rate
        trains = []
        for _ in range(nrepeat):
            isi = np.random.exponential(period, size=int(rate * T) + 1)
            while isi.sum() < T:
                isi = np.concatenate([isi, np.random.exponential(period, size=int(rate * T) + 1)])
            isi0 = np.concatenate([[period * np.random.rand()], np.full(len(isi) - 1, period)])
            isi = (1 - precision) * isi0 + precision * isi
            spk = np.cumsum(isi)
            trains.append(spk[spk < T])
        return trains

    def _bursty() -> List[np.ndarray]:
        nperburst = parameters[0] if len(parameters) >= 1 else 1
        isi = parameters[1] if len(parameters) >= 2 else 0.01
        burstrate = rate / nperburst
        trains = []
        for _ in range(nrepeat):
            nburst = np.random.poisson(burstrate * T)
            bursts = []
            for _ in range(nburst):
                nspk = np.random.poisson(nperburst)
                if nspk > 0:
                    bursts.append(np.random.rand() * T + np.cumsum(isi * (np.random.poisson(10, size=nspk) / 10)))
            if bursts:
                trains.append(np.sort(np.concatenate(bursts)))
            else:
                trains.append(np.array([]))
        return trains

    def _vary_rate() -> List[np.ndarray]:
        smoothtime = parameters[0] if len(parameters) >= 1 else 5
        rnonzero = parameters[1] if len(parameters) >= 2 else 0.7
        if rnonzero <= 0 or rnonzero >= 1:
            raise ValueError("rnonzero must be in (0,1)")
        nsub = 20
        dt = smoothtime / nsub
        nt = int(np.ceil(T / dt))
        x = np.random.randn(nt + 2 * nsub, nrepeat)
        kernel = np.ones(nsub) / nsub
        vrate = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, x)
        vrate = vrate[nsub : nsub + nt]
        hwhh = np.sqrt(2 * np.log(2))
        s = nsub * hwhh / (2 * np.pi)
        sr = np.sqrt(1 / (2 * np.sqrt(np.pi) * s))
        vrate = vrate / sr
        thr = np.quantile(vrate, 1 - rnonzero)
        vrate = np.maximum(0, vrate - thr)
        xvals = np.linspace(thr, 5, 4000)
        y = (xvals - thr) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xvals**2)
        avgr = np.trapz(y, xvals)
        vrate = vrate * (rate / avgr)

        dt1 = 1 / np.max(vrate) / 100
        nt1 = int(np.floor(T / dt1))
        vrate1 = np.interp(np.arange(nt1) * dt1, np.arange(nt) * dt, vrate[:, 0])
        trains = []
        for k in range(nrepeat):
            nspike = (np.random.rand(nt1) < vrate1 * dt1)
            spk = np.where(nspike)[0] * dt1
            trains.append(spk + dt1 * np.random.rand(len(spk)))
        return trains

    if mode == "fix-rate":
        trains = _fix_rate()
    elif mode == "vary-rate":
        trains = _vary_rate()
    elif mode == "bursty":
        trains = _bursty()
    elif mode == "periodic":
        trains = _periodic()
    else:
        raise ValueError(f"Unknown mode {mode}")

    return trains[0] if repeat is None else trains
