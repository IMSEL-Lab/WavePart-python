from __future__ import annotations

from typing import Iterable

import numpy as np

from .core import _as_float_array, _atan2d, _cosd, _sind, _fortran_argmax, _fortran_ind2sub
from .types import FloatArray, PartitionParameters


def dispersion(const: Iterable[float] | FloatArray) -> FloatArray:
    const = np.asarray(const, dtype=float)
    kh = np.full(const.shape, np.nan, dtype=float)
    zeros = const == 0
    kh[zeros] = 0.0
    positive = const > 0
    kh[positive] = np.sqrt(const[positive])
    gt_one = const > 1
    kh[gt_one] = const[gt_one]

    if np.any(positive):
        for _ in range(1000):
            residual = np.abs(const[positive] - kh[positive] * np.tanh(kh[positive]))
            if np.max(residual) <= 1e-6:
                break
            f = kh[positive] * np.tanh(kh[positive]) - const[positive]
            fprime = kh[positive] / np.cosh(kh[positive]) ** 2 + np.tanh(kh[positive])
            kh[positive] = kh[positive] - f / fprime
    return kh


def compute_partition_params(
    spectrum: FloatArray,
    freq: Iterable[float] | FloatArray,
    direction: Iterable[float] | FloatArray,
    labels: np.ndarray,
    *,
    depth: float = 1000.0,
) -> PartitionParameters:
    energy = np.asarray(spectrum, dtype=float)
    freq = _as_float_array(freq)
    direction = _as_float_array(direction)
    labels = np.asarray(labels, dtype=int)

    g = 9.81
    n_partitions = int(np.max(labels)) if labels.size else 0
    if n_partitions == 0:
        empty = np.zeros(0, dtype=float)
        return PartitionParameters(
            mean_frequency=empty,
            peak_frequency=empty,
            mean_direction=empty,
            peak_direction=empty,
            directional_spread=empty,
            total_energy=empty,
            peak_energy=empty,
            hrms=empty,
            hsig=empty,
            significant_slope=empty,
        )

    mean_frequency = np.ones(n_partitions, dtype=float)
    peak_frequency = np.ones(n_partitions, dtype=float)
    mean_direction = np.ones(n_partitions, dtype=float)
    peak_direction = np.ones(n_partitions, dtype=float)
    directional_spread = np.ones(n_partitions, dtype=float)
    total_energy = np.ones(n_partitions, dtype=float)
    peak_energy = np.ones(n_partitions, dtype=float)
    hrms = np.ones(n_partitions, dtype=float)
    hsig = np.ones(n_partitions, dtype=float)
    significant_slope = np.ones(n_partitions, dtype=float)

    df = float(freq[1] - freq[0])
    dth = float(direction[1] - direction[0])
    nf = energy.shape[0]
    the = np.repeat(direction[None, :], nf, axis=0)

    for label in range(1, n_partitions + 1):
        mask = labels == label
        epart = energy * mask
        idx = _fortran_argmax(epart)
        i1, j1 = _fortran_ind2sub(labels.shape, idx)
        ep = float(energy[i1, j1])
        et = float(np.sum(epart) * df * dth)
        hrms_val = 2 * np.sqrt(2 * et)
        hsig_val = 4 * np.sqrt(et)
        row_sum = np.sum(epart, axis=1)
        row_sum_5 = np.sum(epart**5, axis=1)
        fm = float(np.sum(row_sum * freq) / np.sum(row_sum))
        fp = float(np.sum(row_sum_5 * freq) / np.sum(row_sum_5))
        total = float(np.sum(epart))
        total_5 = float(np.sum(epart**5))
        sino = float(np.sum(epart * _sind(the)) / total)
        coso = float(np.sum(epart * _cosd(the)) / total)
        dm = float(_atan2d(sino, coso))
        sino5 = float(np.sum(epart**5 * _sind(the)) / total_5)
        coso5 = float(np.sum(epart**5 * _cosd(the)) / total_5)
        dp = float(_atan2d(sino5, coso5))
        epsilon = float(np.sqrt(1 - (sino**2 + coso**2)))
        sigma = float((1 + 0.1547 * epsilon**3) / np.sin(epsilon))
        kh = float(np.asarray(dispersion(np.array([(2 * np.pi * fp) ** 2 * depth / g])))[0])
        lp = float(2 * np.pi / (kh / depth))
        psi = float(hsig_val / lp)

        idx0 = label - 1
        mean_frequency[idx0] = fm
        peak_frequency[idx0] = fp
        mean_direction[idx0] = dm
        peak_direction[idx0] = dp
        directional_spread[idx0] = sigma
        total_energy[idx0] = et
        peak_energy[idx0] = ep
        hrms[idx0] = hrms_val
        hsig[idx0] = hsig_val
        significant_slope[idx0] = psi

    return PartitionParameters(
        mean_frequency=mean_frequency,
        peak_frequency=peak_frequency,
        mean_direction=mean_direction,
        peak_direction=peak_direction,
        directional_spread=directional_spread,
        total_energy=total_energy,
        peak_energy=peak_energy,
        hrms=hrms,
        hsig=hsig,
        significant_slope=significant_slope,
    )
