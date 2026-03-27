from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .types import PartitionResult


_GARNET = "#73000A"
_BLACK = "#000000"
_WHITE = "#FFFFFF"
_NEUTRALS = ["#ECECEC", "#C7C7C7", "#A2A2A2", "#5C5C5C", "#363636"]
_ACCENTS = ["#CC2E40", "#466A9F", "#1F414D", "#65780B", "#CED318", "#A49137"]


def _partition_cmap(count: int) -> tuple[ListedColormap, BoundaryNorm]:
    colors = [_WHITE, _GARNET]
    while len(colors) < count + 1:
        colors.extend(_ACCENTS)
        colors.extend(_NEUTRALS)
    cmap = ListedColormap(colors[: count + 1])
    bounds = np.arange(-0.5, count + 1.5, 1.0)
    return cmap, BoundaryNorm(bounds, cmap.N)


def plot_partition_surface(freq: Iterable[float], direction: Iterable[float], spectrum: np.ndarray, partition: PartitionResult):
    freq = np.asarray(freq, dtype=float)
    direction = np.asarray(direction, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 7))
    mesh = ax.pcolormesh(freq, direction, partition.labels.T, shading="nearest", cmap=_partition_cmap(partition.partition_count)[0])
    contour = ax.contour(freq, direction, partition.smoothed_spectrum.T, colors=_BLACK, linewidths=0.75)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Direction (deg)")
    ax.set_title("WavePart Partition Surface")
    ax.set_facecolor(_WHITE)
    fig.patch.set_facecolor(_WHITE)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Partition Label")
    return fig, ax


def plot_partition_polar(freq: Iterable[float], direction: Iterable[float], partition: PartitionResult):
    freq = np.asarray(freq, dtype=float)
    direction = np.asarray(direction, dtype=float)
    dtheta = float(direction[1] - direction[0])
    dr = float(freq[1] - freq[0])
    theta_edges = np.deg2rad(np.concatenate([direction - dtheta / 2.0, [direction[-1] + dtheta / 2.0]]))
    radius_edges = np.concatenate([freq - dr / 2.0, [freq[-1] + dr / 2.0]])
    z = partition.labels

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    cmap, norm = _partition_cmap(partition.partition_count)
    mesh = ax.pcolormesh(theta_edges, radius_edges, z, shading="flat", cmap=cmap, norm=norm)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor(_WHITE)
    fig.patch.set_facecolor(_WHITE)
    ax.grid(color=_BLACK, linewidth=0.75, alpha=0.35)
    ax.set_title("WavePart Partition Polar Plot", pad=18)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.08)
    cbar.set_label("Partition Label")
    return fig, ax
