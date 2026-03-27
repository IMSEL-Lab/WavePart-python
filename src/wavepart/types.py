from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FloatArray = np.ndarray
IntArray = np.ndarray


@dataclass(slots=True)
class PartitionResult:
    labels: IntArray
    smoothed_spectrum: FloatArray
    backend: str = "reference"

    @property
    def partition_count(self) -> int:
        return int(np.max(self.labels)) if self.labels.size else 0


@dataclass(slots=True)
class WindLimitResult:
    fw: float
    fwpk: float


@dataclass(slots=True)
class WindLimitSeries:
    time: FloatArray
    fw: FloatArray
    dw: FloatArray
    raw_fw: FloatArray
    raw_peak_direction: FloatArray
    frequency_spectra: FloatArray
    direction_spectra: FloatArray


@dataclass(slots=True)
class PartitionParameters:
    mean_frequency: FloatArray
    peak_frequency: FloatArray
    mean_direction: FloatArray
    peak_direction: FloatArray
    directional_spread: FloatArray
    total_energy: FloatArray
    peak_energy: FloatArray
    hrms: FloatArray
    hsig: FloatArray
    significant_slope: FloatArray

    def to_raw_arrays(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        frequency_metrics = np.vstack([self.mean_frequency, self.peak_frequency])
        direction_metrics = np.vstack([self.mean_direction, self.peak_direction, self.directional_spread])
        energy_metrics = np.vstack([self.total_energy, self.peak_energy])
        height_metrics = np.vstack([self.hrms, self.hsig, self.significant_slope])
        return frequency_metrics, direction_metrics, energy_metrics, height_metrics

    def to_xarray(self) -> Any:
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError("xarray is not installed. Use the 'xarray' extra.") from exc

        partition = np.arange(1, self.mean_frequency.size + 1, dtype=int)
        return xr.Dataset(
            data_vars={
                "mean_frequency": ("partition", self.mean_frequency),
                "peak_frequency": ("partition", self.peak_frequency),
                "mean_direction": ("partition", self.mean_direction),
                "peak_direction": ("partition", self.peak_direction),
                "directional_spread": ("partition", self.directional_spread),
                "total_energy": ("partition", self.total_energy),
                "peak_energy": ("partition", self.peak_energy),
                "hrms": ("partition", self.hrms),
                "hsig": ("partition", self.hsig),
                "significant_slope": ("partition", self.significant_slope),
            },
            coords={"partition": partition},
        )


@dataclass(slots=True)
class LoadedSpectrumSet:
    path: Path
    freq: FloatArray
    direction: FloatArray
    spectra: FloatArray
    time: FloatArray | None = None


@dataclass(slots=True)
class AnalysisStep:
    index: int
    partition: PartitionResult
    parameters: PartitionParameters | None = None
    wind_cutoff: float | None = None


@dataclass(slots=True)
class AnalysisResult:
    dataset: LoadedSpectrumSet
    steps: list[AnalysisStep]
    wind_limits: WindLimitSeries | None = None
