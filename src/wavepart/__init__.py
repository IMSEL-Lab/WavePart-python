from .core import (
    filter_dir_wavespec,
    partition_spectrum,
    peakspread,
    valley_min,
    watershed_ww3,
)
from .io import analyze_mat_file, load_spectrum_set
from .params import compute_partition_params, dispersion
from .types import (
    AnalysisResult,
    AnalysisStep,
    LoadedSpectrumSet,
    PartitionParameters,
    PartitionResult,
    WindLimitResult,
    WindLimitSeries,
)
from .wind import estimate_wind_limit, estimate_wind_limits, wildpoint

try:
    from .plotting import plot_partition_polar, plot_partition_surface
except ImportError:  # pragma: no cover - optional plotting dependency
    plot_partition_polar = None
    plot_partition_surface = None

__all__ = [
    "AnalysisResult",
    "AnalysisStep",
    "LoadedSpectrumSet",
    "PartitionParameters",
    "PartitionResult",
    "WindLimitResult",
    "WindLimitSeries",
    "analyze_mat_file",
    "compute_partition_params",
    "dispersion",
    "estimate_wind_limit",
    "estimate_wind_limits",
    "filter_dir_wavespec",
    "load_spectrum_set",
    "partition_spectrum",
    "peakspread",
    "plot_partition_polar",
    "plot_partition_surface",
    "valley_min",
    "watershed_ww3",
    "wildpoint",
]
