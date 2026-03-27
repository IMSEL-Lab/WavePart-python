from __future__ import annotations

from typing import Iterable

import numpy as np

from .core import _as_float_array, _atan2d, _cosd, _sind
from .types import FloatArray, WindLimitResult, WindLimitSeries


def _movmean(values: FloatArray, window: int) -> FloatArray:
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(values).astype(float)
    total = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="same")
    count = np.convolve(valid, kernel, mode="same")
    return total / count


def wildpoint(values: Iterable[float] | FloatArray, nn: float) -> FloatArray:
    x = _as_float_array(values)
    n = x.size
    y = x.copy()
    for i in range(n):
        i1 = max(0, i - 2)
        i2 = min(n, i + 3)
        y[i] = np.median(x[i1:i2])
    xdiff = np.abs(x - y)
    xstd = np.std(x, ddof=1) if n > 1 else 0.0
    y[xdiff > xstd * nn] = np.nan
    return y


def estimate_wind_limit(freq: Iterable[float] | FloatArray, spectrum_1d: Iterable[float] | FloatArray, *, exponent: int = 4) -> WindLimitResult:
    f = _as_float_array(freq)
    sf = _as_float_array(spectrum_1d)

    f1 = 0.35
    f2 = 0.70
    df = float(f[1] - f[0])
    s = sf / (np.sum(sf) * df)
    iw = np.where((f >= f1) & (f <= f2))[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        yni = np.log10(s[iw] * f[iw] ** exponent)
        ain = np.polyfit(f[iw], yni, 1)
        sn = np.polyval(ain, f)
        s10 = np.log10(s * f**exponent)
    dev = s10[1:] - sn[1:]
    varw = np.std(dev[iw], ddof=1)
    limw = np.mean(dev[iw])
    lim = limw - 3 * varw
    x = np.cumsum(dev - lim)

    upper = np.where(f > f2)[0]
    if1 = int(upper[0]) if upper.size else x.size - 1
    prefix = x[: if1 + 1]
    ik = int(np.argmin(prefix)) if prefix.size else 0
    fw = float(f[ik])

    mask = f > fw
    s1 = s * mask
    fwpk = float(f[int(np.argmax(s1))])
    return WindLimitResult(fw=fw, fwpk=fwpk)


def estimate_wind_limits(
    time: Iterable[float] | FloatArray,
    freq: Iterable[float] | FloatArray,
    direction: Iterable[float] | FloatArray,
    spectra: FloatArray,
    *,
    plot: bool = False,
) -> WindLimitSeries:
    del plot

    t = _as_float_array(time)
    f = _as_float_array(freq)
    d = _as_float_array(direction)
    spectra = np.asarray(spectra, dtype=float)

    n = t.size
    raw_fw = np.zeros(n, dtype=float)
    dpk = np.zeros(n, dtype=float)
    sf = np.zeros((f.size, n), dtype=float)
    sth = np.zeros((d.size, n), dtype=float)

    kernel = np.ones(5, dtype=float) / 5.0
    for i in range(n):
        energy = spectra[:, :, i]
        sf[:, i] = np.convolve(np.sum(energy, axis=1), kernel, mode="same")
        sth[:, i] = np.convolve(np.sum(energy, axis=0), kernel, mode="same")
        result = estimate_wind_limit(f, sf[:, i], exponent=4)
        raw_fw[i] = result.fw
        j = np.where(f == result.fwpk)[0]
        row = energy[int(j[0]), :] if j.size else energy[int(np.argmin(np.abs(f - result.fwpk))), :]
        dpk[i] = d[int(np.argmax(row))]

    const = 1
    mvpt = 15
    fw = wildpoint(raw_fw, const)
    fw = wildpoint(fw, const)
    good = ~np.isnan(fw)
    fw = np.interp(t, t[good], fw[good])
    fw = _movmean(fw, mvpt)
    fw = _movmean(fw, mvpt)

    dvals = dpk.copy()
    dvals[~good] = np.nan
    dx = _cosd(dvals)
    dy = _sind(dvals)
    dx = np.interp(t, t[good], dx[good])
    dy = np.interp(t, t[good], dy[good])
    dx = _movmean(dx, mvpt)
    dx = _movmean(dx, mvpt)
    dy = _movmean(dy, mvpt)
    dy = _movmean(dy, mvpt)
    dw = _atan2d(dy, dx)

    return WindLimitSeries(
        time=t,
        fw=fw,
        dw=dw,
        raw_fw=raw_fw,
        raw_peak_direction=dpk,
        frequency_spectra=sf,
        direction_spectra=sth,
    )
