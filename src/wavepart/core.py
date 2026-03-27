from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d

from .types import FloatArray, PartitionResult


def _as_float_array(values: Iterable[float] | FloatArray) -> FloatArray:
    return np.asarray(values, dtype=float).reshape(-1)


def _cosd(values: FloatArray) -> FloatArray:
    return np.cos(np.deg2rad(values))


def _sind(values: FloatArray) -> FloatArray:
    return np.sin(np.deg2rad(values))


def _atan2d(y: FloatArray, x: FloatArray) -> FloatArray:
    return np.rad2deg(np.arctan2(y, x))


def _matlab_argmax(values: FloatArray) -> int:
    return int(np.argmax(np.ravel(values, order="F")))


def _matlab_nanargmax(values: FloatArray) -> int:
    return int(np.nanargmax(np.ravel(values, order="F")))


def _matlab_ind2sub(shape: tuple[int, int], index: int) -> tuple[int, int]:
    return tuple(int(i) for i in np.unravel_index(index, shape, order="F"))


def _wrap_delta_theta(delta_theta: FloatArray) -> FloatArray:
    wrapped = delta_theta.copy()
    wrapped[wrapped > 180] -= 360
    wrapped[wrapped < -180] += 360
    return wrapped


def _peak_location(freq: FloatArray, direction: FloatArray, energy: FloatArray, mask: FloatArray) -> tuple[float, float, float]:
    idx = _matlab_argmax(energy * mask)
    i1, j1 = _matlab_ind2sub(energy.shape, idx)
    return float(freq[i1]), float(direction[j1]), float(energy[i1, j1])


def _renumber_with_wind(aa: np.ndarray, wind_label: int) -> np.ndarray:
    bb = aa.copy()
    current = 1
    for label in range(1, int(np.max(aa)) + 1):
        count = int(np.count_nonzero(aa == label))
        if count == 0:
            continue
        if label == wind_label:
            bb[aa == label] = 1
        else:
            current += 1
            bb[aa == label] = current
    return bb


def _renumber_swell(aa: np.ndarray) -> np.ndarray:
    bb = aa.copy()
    current = 1
    for label in range(2, int(np.max(aa)) + 1):
        count = int(np.count_nonzero(aa == label))
        if count == 0:
            continue
        current += 1
        bb[aa == label] = current
    return bb


def _find_first_index(values: FloatArray, target: float) -> int:
    matches = np.where(values == target)[0]
    if matches.size:
        return int(matches[0])
    return int(np.argmin(np.abs(values - target)))


def filter_dir_wavespec(energy: FloatArray, nc: int = 1, n: int = 3) -> FloatArray:
    if n % 2 == 0:
        n += 1
    kernel = np.ones((n, n), dtype=float) / float(n * n)
    mn = n // 2

    e1 = np.vstack(
        [
            np.repeat(energy[[0], :], mn, axis=0),
            energy,
            np.repeat(energy[[-1], :], mn, axis=0),
        ]
    )
    if mn > 0:
        e2 = np.hstack([e1[:, -mn:], e1, e1[:, :mn]])
    else:
        e2 = e1.copy()

    e3 = e2
    for _ in range(nc):
        e3 = convolve2d(e3, kernel, mode="same")

    if mn == 0:
        return e3
    return e3[mn:-mn, mn:-mn]


@dataclass(slots=True)
class _WatershedTrace:
    peak_i: int
    peak_j: int
    flat: bool


def _nextmax(energy: FloatArray, i: int, j: int, rng: np.random.Generator) -> _WatershedTrace:
    n_rows, n_cols = energy.shape
    pt = np.full((3, 3), np.nan, dtype=float)
    pt[1, 1] = energy[i, j]

    pt[1, 2] = energy[i, 0] if j == n_cols - 1 else energy[i, j + 1]
    pt[1, 0] = energy[i, -1] if j == 0 else energy[i, j - 1]

    if i < n_rows - 1:
        pt[2, 1] = energy[i + 1, j]
        pt[2, 2] = energy[i + 1, 0] if j == n_cols - 1 else energy[i + 1, j + 1]
        pt[2, 0] = energy[i + 1, -1] if j == 0 else energy[i + 1, j - 1]
    if i > 0:
        pt[0, 1] = energy[i - 1, j]
        pt[0, 2] = energy[i - 1, 0] if j == n_cols - 1 else energy[i - 1, j + 1]
        pt[0, 0] = energy[i - 1, -1] if j == 0 else energy[i - 1, j - 1]

    ptflat = np.unique(pt[~np.isnan(pt)])
    if ptflat.size == 1:
        if i == n_rows - 1:
            i1 = int(rng.integers(0, 2))
        elif i == 0:
            i1 = int(rng.integers(1, 3))
        else:
            i1 = int(rng.integers(0, 3))
        next_i = i + i1 - 1

        j1 = int(rng.integers(0, 3))
        if j == n_cols - 1 and j1 == 2:
            next_j = 0
        elif j == 0 and j1 == 0:
            next_j = n_cols - 1
        else:
            next_j = j + j1 - 1
        return _WatershedTrace(next_i, next_j, True)

    index = _matlab_nanargmax(pt)
    i1, j1 = _matlab_ind2sub(pt.shape, index)
    next_i = i + i1 - 1
    if j == n_cols - 1 and j1 == 2:
        next_j = 0
    elif j == 0 and j1 == 0:
        next_j = n_cols - 1
    else:
        next_j = j + j1 - 1
    return _WatershedTrace(next_i, next_j, False)


def watershed_ww3(energy: FloatArray, *, seed: int = 0) -> np.ndarray:
    aa = np.zeros(energy.shape, dtype=int)
    n_rows, n_cols = energy.shape
    peak_map = np.full((n_rows, n_cols, 2), -1, dtype=int)
    peak_lookup: dict[tuple[int, int], int] = {}
    rng = np.random.default_rng(seed)
    next_label = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if aa[i, j] != 0:
                continue

            path: list[tuple[int, int]] = []
            i0, j0 = i, j
            while True:
                if aa[i0, j0] != 0:
                    label = int(aa[i0, j0])
                    break
                if peak_map[i0, j0, 0] >= 0:
                    peak = (int(peak_map[i0, j0, 0]), int(peak_map[i0, j0, 1]))
                    label = peak_lookup[peak]
                    break

                path.append((i0, j0))
                trace = _nextmax(energy, i0, j0, rng)
                if i0 == trace.peak_i and j0 == trace.peak_j and not trace.flat:
                    peak = (i0, j0)
                    label = peak_lookup.get(peak)
                    if label is None:
                        next_label += 1
                        label = next_label
                        peak_lookup[peak] = label
                    break
                i0, j0 = trace.peak_i, trace.peak_j

            for pi, pj in path:
                aa[pi, pj] = label
                peak_map[pi, pj, :] = np.array((i0, j0))
            aa[i, j] = label

    return aa


def peakspread(freq: FloatArray, theta: FloatArray, energy: FloatArray, mask: FloatArray) -> float:
    ep = energy * mask
    nf, nd = energy.shape
    f = np.repeat(freq[:, None], nd, axis=1)
    th = np.repeat(theta[None, :], nf, axis=0)
    e = float(np.sum(ep))
    fx1 = float(np.sum(ep * f * _cosd(th)) / e)
    fy1 = float(np.sum(ep * f * _sind(th)) / e)
    fx2 = float(np.sum(ep * (f**2) * (_cosd(th) ** 2)) / e)
    fy2 = float(np.sum(ep * (f**2) * (_sind(th) ** 2)) / e)
    return float(fx2 - fx1**2 + fy2 - fy1**2)


def valley_min(
    energy: FloatArray,
    freq: FloatArray,
    direction: FloatArray,
    d1: float,
    d2: float,
    f1: float,
    f2: float,
    labels: np.ndarray | None = None,
) -> float:
    i1 = _find_first_index(freq, f1)
    j1 = _find_first_index(direction, d1)
    i2 = _find_first_index(freq, f2)
    j2 = _find_first_index(direction, d2)

    wrapped_energy = energy
    wrapped_direction = direction
    wrapped_labels = labels
    wrapped_j2 = j2
    wrapped_d2 = d2
    if abs(d1 - d2) > 180:
        if d2 > d1:
            if j2 > 0:
                wrapped_d2 = d2 - 360
                wrapped_energy = np.hstack([energy[:, j2:], energy[:, :j2]])
                wrapped_direction = np.concatenate([direction[j2:] - 360, direction[:j2]])
                if labels is not None:
                    wrapped_labels = np.hstack([labels[:, j2:], labels[:, :j2]])
                wrapped_j2 = 0
            else:
                wrapped_d2 = d2 - 360
        else:
            if j2 > 0:
                wrapped_d2 = d2 + 360
                wrapped_energy = np.hstack([energy[:, j2:], energy[:, :j2]])
                wrapped_direction = np.concatenate([direction[j2:], direction[:j2] + 360])
                if labels is not None:
                    wrapped_labels = np.hstack([labels[:, j2:], labels[:, :j2]])
                wrapped_j2 = 0
            else:
                wrapped_d2 = d2 + 360

    ln_dir = np.linspace(d1, wrapped_d2, 100)
    m_val = (f2 - f1) / (wrapped_d2 - d1) if wrapped_d2 != d1 else np.nan
    ln_f = f1 + m_val * (ln_dir - d1)

    interpolator = RegularGridInterpolator(
        (wrapped_direction, freq),
        wrapped_energy.T,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack([ln_dir, ln_f])
    ln_val = interpolator(points)
    if np.all(np.isnan(ln_val)):
        vmin = float("nan")
    else:
        vmin = float(np.nanmin(ln_val))

    if wrapped_labels is not None and not np.isnan(m_val):
        p1 = int(wrapped_labels[i1, j1])
        p2 = int(wrapped_labels[i2, wrapped_j2])
        for d, f in zip(ln_dir, ln_f, strict=False):
            di = int(np.argmin(np.abs(wrapped_direction - d)))
            dcheck = float(np.min(np.abs(wrapped_direction - d)))
            if dcheck > 20:
                raise ValueError("check valley_min")
            fi = int(np.argmin(np.abs(freq - f)))
            atest = int(wrapped_labels[fi, di])
            if atest == 0 or (atest != p1 and atest != p2):
                return float("nan")
    return vmin


def partition_spectrum(
    freq: Iterable[float] | FloatArray,
    direction: Iterable[float] | FloatArray,
    spectrum: FloatArray,
    *,
    wind_cutoff: float | None = None,
    keep_wind_region: bool = True,
    backend: str = "reference",
    plot_switch: int = 10,
) -> PartitionResult:
    if backend != "reference":
        raise ValueError("Only the 'reference' backend is implemented.")

    freq = _as_float_array(freq)
    direction = _as_float_array(direction)
    energy = np.asarray(spectrum, dtype=float)

    navg = 3
    wind_width = 90.0
    windminf = 0.12
    swell_hs_lim = 0.2
    d_lim = 90.0
    df = float(freq[1] - freq[0])
    dth = float(direction[1] - direction[0])
    min_sq_dist = (6 * df) ** 2
    a = 4e-5
    b = 0.04
    kappa = 0.4
    z = 0.4

    if wind_cutoff is not None:
        windminf = float(wind_cutoff)

    ef = filter_dir_wavespec(energy, 2, navg)
    min_e_allowed = a / (np.max(freq) ** 4 + b)
    if np.sum(ef - np.min(ef)) < min_e_allowed:
        return PartitionResult(labels=np.zeros_like(ef, dtype=int), smoothed_spectrum=ef, backend=backend)

    aa = watershed_ww3(ef)
    m, n = aa.shape
    n_partitions = int(np.max(aa))
    fp = np.ones(n_partitions, dtype=float)
    dp = np.ones(n_partitions, dtype=float)
    ep = np.ones(n_partitions, dtype=float)
    for label in range(1, n_partitions + 1):
        mask = (aa == label).astype(float)
        fp[label - 1], dp[label - 1], ep[label - 1] = _peak_location(freq, direction, ef, mask)

    ep[fp < windminf] = 0
    if np.sum(ep) < 0:
        raise ValueError("no wind partitions, try reducing input fw (ex. 0.1 Hz)")
    wn = int(np.argmax(ep)) + 1
    fpw = fp[wn - 1]
    dpw = dp[wn - 1]
    delta_th = _wrap_delta_theta(direction - dpw)
    inside = np.abs(delta_th) < wind_width
    fc = windminf / _cosd(delta_th * 90 / wind_width)
    fc[~inside] = np.nan

    for label in range(1, n_partitions + 1):
        if fp[label - 1] == fpw:
            continue
        dj = _find_first_index(direction, dp[label - 1])
        fcd = fc[dj]
        if not np.isnan(fcd) and fp[label - 1] > fcd:
            aa[aa == label] = wn

    if keep_wind_region:
        bb = np.zeros((m, n), dtype=int)
        for col, fcd in enumerate(fc):
            if not np.isnan(fcd):
                keep = freq > fcd
                aa[keep, col] = wn
                noise = freq < fcd
            else:
                noise = freq > -1
            bb[noise, col] = 255
        aa[(aa == wn) & (bb == 255)] = 0

    aa = _renumber_with_wind(aa, wn)

    n_partitions = int(np.max(aa))
    fp = np.zeros(n_partitions, dtype=float)
    dp = np.zeros(n_partitions, dtype=float)
    df2 = np.zeros(n_partitions, dtype=float)
    for label in range(2, n_partitions + 1):
        mask = (aa == label).astype(float)
        fp[label - 1], dp[label - 1], _ = _peak_location(freq, direction, ef, mask)
        df2[label - 1] = peakspread(freq, direction, ef, mask)

    fx = fp * _cosd(dp)
    fy = fp * _sind(dp)
    sq_dist = np.zeros((n_partitions, n_partitions), dtype=float)
    sq_sprd = np.zeros((n_partitions, n_partitions), dtype=float)
    for label in range(2, n_partitions + 1):
        df2_values = (fx[label - 1] - fx) ** 2 + (fy[label - 1] - fy) ** 2
        sq_dist[:, label - 1] = df2_values
        sq_sprd[:, label - 1] = np.ones_like(df2_values) * df2[label - 1]

    for label in range(2, n_partitions + 1):
        merge = np.where((sq_dist[:, label - 1] < min_sq_dist) | (sq_dist[:, label - 1] < kappa * sq_sprd[:, label - 1]))[0]
        for other in merge + 1:
            if other == label or other <= 1:
                continue
            ddth = abs(dp[label - 1] - dp[other - 1])
            if ddth > 180:
                ddth -= 360
            if abs(ddth) < d_lim:
                aa[aa == other] = label

    aa = _renumber_swell(aa)

    n_partitions = int(np.max(aa))
    p = np.zeros(n_partitions, dtype=float)
    fp = np.zeros(n_partitions, dtype=float)
    for label in range(2, n_partitions + 1):
        mask = (aa == label).astype(float)
        p[label - 1] = float(np.sum(ef * mask) * df * dth)
        fp[label - 1], _, _ = _peak_location(freq, direction, ef, mask)

    jo = np.arange(1, n_partitions + 1, dtype=int)
    mv = a / (fp**4 + b)
    jn = jo[(p >= mv) & (fp < 0.6)]
    op = jo[~np.isin(jo, jn)]
    for label in op:
        if label > 1:
            aa[aa == label] = 0

    bb = aa.copy()
    for i, label in enumerate(jn, start=1):
        bb[aa == label] = i + 1
    aa = bb

    n_partitions = int(np.max(aa))
    fp = np.zeros(n_partitions, dtype=float)
    dp = np.zeros(n_partitions, dtype=float)
    ep = np.zeros(n_partitions, dtype=float)
    for label in range(2, n_partitions + 1):
        mask = (aa == label).astype(float)
        fp[label - 1], dp[label - 1], ep[label - 1] = _peak_location(freq, direction, ef, mask)

    vmin = np.zeros((n_partitions, n_partitions), dtype=float)
    dd = np.zeros((n_partitions, n_partitions), dtype=float)
    ee = np.zeros((n_partitions, n_partitions), dtype=float)
    for label in range(2, n_partitions + 1):
        for other in range(2, n_partitions + 1):
            if label == other:
                continue
            f1 = fp[label - 1]
            f2 = fp[other - 1]
            d1 = dp[label - 1]
            d2 = dp[other - 1]
            ee[other - 1, label - 1] = min(ep[label - 1], ep[other - 1])
            dd1 = abs(d2 - d1)
            if dd1 > 180:
                dd1 -= 360
            dd[other - 1, label - 1] = dd1
            vmin[other - 1, label - 1] = valley_min(ef, freq, direction, d1, d2, f1, f2, aa)

    for label in range(2, n_partitions + 1):
        merge = np.where((vmin[:, label - 1] > z * ee[:, label - 1]) & (np.abs(dd[:, label - 1]) < d_lim))[0]
        for other in merge + 1:
            if other == label or other <= 1:
                continue
            aa[aa == other] = label

    aa = _renumber_swell(aa)

    n_partitions = int(np.max(aa))
    sum_sw = np.zeros(n_partitions, dtype=float)
    hsig = np.zeros(n_partitions, dtype=float)
    for label in range(2, n_partitions + 1):
        mask = aa == label
        sw = energy * mask
        sum_sw[label - 1] = float(np.sum(sw) * df * dth)
        hsig[label - 1] = 4 * np.sqrt(sum_sw[label - 1])

    for label in range(2, n_partitions + 1):
        if hsig[label - 1] < swell_hs_lim:
            aa[aa == label] = 0

    aa = _renumber_swell(aa)

    n_partitions = int(np.max(aa))
    if n_partitions > 1:
        sum_sw = np.zeros(n_partitions - 1, dtype=float)
        for label in range(2, n_partitions + 1):
            mask = aa == label
            sw = energy * mask
            sum_sw[label - 2] = float(np.sum(sw) * df * dth)
        order = np.argsort(sum_sw)[::-1]
        bb = aa.copy()
        for i, label_idx in enumerate(order, start=1):
            bb[aa == (label_idx + 2)] = i + 1
        aa = bb

    return PartitionResult(labels=aa.astype(int), smoothed_spectrum=ef, backend=backend)
