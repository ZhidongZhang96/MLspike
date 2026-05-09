from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


ArrayLike = Any


def as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def is_scalar(value: Any) -> bool:
    return np.isscalar(value)


def struct_merge(base: Mapping[str, Any], override: Mapping[str, Any], *, recursive: bool = True, skip_none: bool = False) -> dict:
    result: dict = {}
    for key, val in base.items():
        result[key] = val
    for key, val in override.items():
        if skip_none and val is None:
            continue
        if recursive and isinstance(val, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = struct_merge(result[key], val, recursive=recursive, skip_none=skip_none)
        else:
            result[key] = val
    return result


def row(vec: ArrayLike) -> np.ndarray:
    arr = np.asarray(vec)
    return arr.reshape(1, -1)


def column(vec: ArrayLike) -> np.ndarray:
    arr = np.asarray(vec)
    return arr.reshape(-1, 1)


def add(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    return np.asarray(a) + np.asarray(b)


def subtract(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    return np.asarray(a) - np.asarray(b)


def mult(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    return np.asarray(a) * np.asarray(b)


def div(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    return np.asarray(a) / np.asarray(b)


def coerce(value: ArrayLike, bounds: Tuple[float, float]) -> np.ndarray:
    return np.clip(value, bounds[0], bounds[1])


def timevector(values: ArrayLike, dt: ArrayLike, mode: str | None = None) -> Any:
    if isinstance(values, (list, tuple)):
        if is_scalar(dt):
            return [timevector(v, dt, mode=mode) for v in values]
        return [timevector(v, dt[i], mode=mode) for i, v in enumerate(values)]

    arr = np.asarray(values)
    if arr.size == 0:
        return arr.copy()

    if mode is None:
        if arr.ndim == 1 and np.all(np.mod(arr, 1) == 0) and arr.size >= 5 and np.any(arr == 0):
            mode = "times"
        else:
            mode = "counts"

    if mode == "times":
        return counts_to_times(arr, dt)
    if mode == "counts":
        return times_to_counts(arr, dt)
    raise ValueError(f"Unknown mode {mode}")


def counts_to_times(counts: np.ndarray, dt: ArrayLike) -> np.ndarray:
    counts = np.asarray(counts).astype(int)
    if not is_scalar(dt):
        dt = float(np.mean(np.diff(np.asarray(dt))))
    times = np.repeat(np.arange(len(counts)), counts) * float(dt)
    return times.astype(float)


def times_to_counts(times: np.ndarray, dt: ArrayLike) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return np.zeros(0, dtype=int)
    if is_scalar(dt):
        dt = float(dt)
        nt = int(np.ceil(times.max() / dt)) + 1
        counts = np.zeros(nt, dtype=int)
        idx = np.clip(np.round(times / dt).astype(int), 0, nt - 1)
        np.add.at(counts, idx, 1)
        return counts

    grid = np.asarray(dt, dtype=float)
    if grid.size < 2:
        return np.zeros_like(grid, dtype=int)
    step = float(np.median(np.diff(grid)))
    edges = np.concatenate(([grid[0] - step / 2], grid + step / 2))
    counts, _ = np.histogram(times, bins=edges)
    return counts


def interp1_matrix(x: np.ndarray, x_new: np.ndarray, mode: str = "linear", fill: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    n = x.shape[0]
    m = x_new.shape[0]
    mat = np.zeros((m, n), dtype=float)
    if n == 1:
        mat[:, 0] = 1.0
        return mat

    if mode not in {"linear", "spline"}:
        raise ValueError(f"Unsupported interpolation mode {mode}")

    idx = np.searchsorted(x, x_new, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    denom = np.where(x1 == x0, 1.0, x1 - x0)
    w = (x_new - x0) / denom
    rows = np.arange(m)
    mat[rows, idx] = 1.0 - w
    mat[rows, idx + 1] += w

    valid = (x_new >= x[0]) & (x_new <= x[-1])
    if not np.all(valid):
        mat[~valid, :] = fill
    return mat


def interp1_values(values: np.ndarray, idx: np.ndarray, fill: float = np.inf) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    idx = np.asarray(idx, dtype=float)
    n = values.shape[0]
    if n == 1:
        return np.full_like(idx, values[0], dtype=float)
    i0 = np.floor(idx).astype(int)
    i1 = i0 + 1
    w = idx - i0
    i0 = np.clip(i0, 0, n - 1)
    i1 = np.clip(i1, 0, n - 1)
    out = (1 - w) * values[i0] + w * values[i1]
    out[idx < 0] = fill
    out[idx > n - 1] = fill
    return out


def rms(values: np.ndarray) -> float:
    values = np.asarray(values)
    if np.iscomplexobj(values):
        values = np.abs(values)
    values = values.astype(float)
    return float(np.sqrt(np.mean(values ** 2)))


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    vmin = np.min(v, axis=axis, keepdims=True)
    out = vmin - np.log(np.sum(np.exp(vmin - v), axis=axis, keepdims=True))
    if axis is None:
        return out.squeeze()
    return np.squeeze(out, axis=axis)


def log2proba(logp: np.ndarray, axis: int | None = None) -> np.ndarray:
    logp = np.asarray(logp, dtype=float)
    lmin = np.min(logp, axis=axis, keepdims=True)
    prob = np.exp(lmin - logp)
    prob_sum = np.sum(prob, axis=axis, keepdims=True)
    prob = np.divide(prob, prob_sum, out=np.zeros_like(prob), where=prob_sum != 0)
    if axis is None:
        return prob
    return prob


def logmultexp(matrix: np.ndarray, logp: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=float)
    lmin = np.min(logp)
    prob = np.exp(lmin - logp)
    prob1 = matrix @ prob
    prob1 = np.maximum(prob1, 1e-300)
    return lmin - np.log(prob1)


def logmultexp_column(matrix: np.ndarray, logp: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=float)
    lmin = np.min(logp, axis=1, keepdims=True)
    prob = np.exp(lmin - logp)
    prob1 = prob @ matrix
    prob1 = np.maximum(prob1, 1e-300)
    return lmin - np.log(prob1)


def logsample(logp: np.ndarray, nsample: int = 1, mode: str | None = None):
    logp = np.asarray(logp, dtype=float)
    if mode == "rows":
        # Sample one index per row
        lmin = np.min(logp, axis=1, keepdims=True)
        prob = np.exp(lmin - logp)
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob = np.divide(prob, prob_sum, out=np.zeros_like(prob), where=prob_sum != 0)
        cdf = np.cumsum(prob, axis=1)
        r = np.random.rand(prob.shape[0], 1)
        idx = (cdf < r).sum(axis=1)
        return idx

    if mode == "2D":
        # logp shape: (n1, n2, nsample)
        n1, n2, ns = logp.shape
        cidx = np.zeros(ns, dtype=int)
        bidx = np.zeros(ns, dtype=int)
        for k in range(ns):
            flat = logp[:, :, k].reshape(-1)
            lmin = np.min(flat)
            prob = np.exp(lmin - flat)
            prob_sum = prob.sum()
            if prob_sum == 0:
                cidx[k] = 0
                bidx[k] = 0
                continue
            prob = prob / prob_sum
            cdf = np.cumsum(prob)
            r = np.random.rand()
            idx = np.searchsorted(cdf, r)
            cidx[k] = idx % n1
            bidx[k] = idx // n1
        return cidx, bidx

    # default: sample from full distribution
    flat = logp.reshape(-1)
    lmin = np.min(flat)
    prob = np.exp(lmin - flat)
    prob_sum = prob.sum()
    if prob_sum == 0:
        idx = np.zeros(nsample, dtype=int)
    else:
        prob = prob / prob_sum
        cdf = np.cumsum(prob)
        r = np.random.rand(nsample)
        idx = np.searchsorted(cdf, r)
    nrows = logp.shape[0]
    cidx = idx % nrows
    bidx = idx // nrows
    return cidx, bidx


def fft_frequencies(n: int, fs: float) -> np.ndarray:
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    return np.fft.fftshift(freqs)


def gaussian_kernel(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0])
    radius = int(math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def filt(values: np.ndarray, sigma: float, mode: str = "lmd", mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    kernel = gaussian_kernel(sigma)

    def _convolve(vec: np.ndarray) -> np.ndarray:
        return np.convolve(vec, kernel, mode="same")

    if values.ndim == 1:
        if mode == "mask":
            if mask is None:
                return values.copy()
            mask = np.asarray(mask, dtype=float)
            num = _convolve(values * mask)
            den = _convolve(mask)
            return np.divide(num, den, out=np.zeros_like(num), where=den != 0)
        low = _convolve(values)
        if mode == "lmd":
            return low
        if mode == "hmd":
            return values - low
        raise ValueError(f"Unknown filter mode {mode}")

    out = np.zeros_like(values)
    for i in range(values.shape[1]):
        out[:, i] = filt(values[:, i], sigma, mode=mode, mask=mask)
    return out


def normcdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / math.sqrt(2.0)))


def norminv(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must be in (0,1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    q = np.zeros_like(p)
    r = np.zeros_like(p)
    x = np.zeros_like(p)

    lower = p < plow
    upper = p > phigh
    middle = (~lower) & (~upper)

    if np.any(lower):
        ql = np.sqrt(-2 * np.log(p[lower]))
        x[lower] = (
            (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5])
            / ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
        )

    if np.any(upper):
        qu = np.sqrt(-2 * np.log(1 - p[upper]))
        x[upper] = -(
            (((((c[0] * qu + c[1]) * qu + c[2]) * qu + c[3]) * qu + c[4]) * qu + c[5])
            / ((((d[0] * qu + d[1]) * qu + d[2]) * qu + d[3]) * qu + 1)
        )

    if np.any(middle):
        q[middle] = p[middle] - 0.5
        r[middle] = q[middle] ** 2
        x[middle] = (
            (((((a[0] * r[middle] + a[1]) * r[middle] + a[2]) * r[middle] + a[3]) * r[middle] + a[4]) * r[middle] + a[5])
            * q[middle]
            / (((((b[0] * r[middle] + b[1]) * r[middle] + b[2]) * r[middle] + b[3]) * r[middle] + b[4]) * r[middle] + 1)
        )

    return x
