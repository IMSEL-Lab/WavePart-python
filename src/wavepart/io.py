from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from .core import partition_spectrum
from .params import compute_partition_params
from .types import AnalysisResult, AnalysisStep, LoadedSpectrumSet
from .wind import estimate_wind_limits


def load_spectrum_set(path: str | Path) -> LoadedSpectrumSet:
    path = Path(path)
    data = loadmat(path)
    freq = np.asarray(data["freq"], dtype=float).reshape(-1)
    direction = np.asarray(data["dir"], dtype=float).reshape(-1)
    spectra = np.asarray(data["S"], dtype=float)
    time = np.asarray(data["t"], dtype=float).reshape(-1) if "t" in data else None
    return LoadedSpectrumSet(path=path, freq=freq, direction=direction, spectra=spectra, time=time)


def analyze_mat_file(
    path: str | Path,
    *,
    indices: Iterable[int] | None = None,
    depth: float = 30.0,
    estimate_wind: bool = False,
    backend: str = "reference",
    keep_wind_region: bool = True,
) -> AnalysisResult:
    dataset = load_spectrum_set(path)
    wind_limits = None
    if estimate_wind:
        if dataset.time is None:
            raise ValueError("The MAT file does not contain 't', which is required for wind-limit estimation.")
        wind_limits = estimate_wind_limits(dataset.time, dataset.freq, dataset.direction, dataset.spectra)

    if indices is None:
        indices = range(dataset.spectra.shape[2])

    steps: list[AnalysisStep] = []
    for index in indices:
        zero_based = int(index)
        wind_cutoff = float(wind_limits.fw[zero_based]) if wind_limits is not None else None
        partition = partition_spectrum(
            dataset.freq,
            dataset.direction,
            dataset.spectra[:, :, zero_based],
            wind_cutoff=wind_cutoff,
            keep_wind_region=keep_wind_region,
            backend=backend,
        )
        params = compute_partition_params(
            partition.smoothed_spectrum,
            dataset.freq,
            dataset.direction,
            partition.labels,
            depth=depth,
        )
        steps.append(
            AnalysisStep(
                index=zero_based,
                partition=partition,
                parameters=params,
                wind_cutoff=wind_cutoff,
            )
        )

    return AnalysisResult(dataset=dataset, steps=steps, wind_limits=wind_limits)
