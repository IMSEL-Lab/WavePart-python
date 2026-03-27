from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from wavepart import partition_spectrum, plot_partition_polar, plot_partition_surface


def _partition_cmap(count: int) -> tuple[ListedColormap, BoundaryNorm]:
    colors = [
        "#FFFFFF",
        "#73000A",
        "#466A9F",
        "#1F414D",
        "#65780B",
        "#A49137",
        "#5C5C5C",
        "#363636",
    ]
    cmap = ListedColormap(colors[: count + 1])
    norm = BoundaryNorm(np.arange(-0.5, count + 1.5, 1.0), cmap.N)
    return cmap, norm


def _plot_3d_surface(freq, direction, result, idx: int):
    freq_grid, dir_grid = np.meshgrid(freq, direction, indexing="xy")
    z = result.smoothed_spectrum.T
    labels = result.labels.T
    cmap, norm = _partition_cmap(result.partition_count)
    facecolors = cmap(norm(labels))

    fig = plt.figure(figsize=(11, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        freq_grid,
        dir_grid,
        z,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        rcount=z.shape[0],
        ccount=z.shape[1],
    )
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(labels)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_ticks(np.arange(0, result.partition_count + 1))
    cbar.set_label("Partition Label")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Direction (deg)")
    ax.set_zlabel("Smoothed Energy")
    ax.set_title(f"Python Partition 3D View. Index {idx}")
    ax.view_init(elev=32, azim=-38)
    fig.patch.set_facecolor("white")
    return fig, ax


def main() -> int:
    repo_dir = Path(__file__).resolve().parents[1]
    assets_dir = repo_dir / "reports" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    data = loadmat(repo_dir / "data" / "wavespec2d_ex.mat")
    freq = data["freq"].reshape(-1)
    direction = data["dir"].reshape(-1)
    spectra = data["S"]

    for idx in (1, 287):
        spectrum = spectra[:, :, idx - 1]
        result = partition_spectrum(freq, direction, spectrum)

        fig, ax = plot_partition_surface(freq, direction, spectrum, result)
        ax.set_title(f"Python Partition Surface. Index {idx}")
        fig.savefig(assets_dir / f"python_case_{idx:03d}_surface.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _plot_3d_surface(freq, direction, result, idx)
        fig.savefig(assets_dir / f"python_case_{idx:03d}_3d.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plot_partition_polar(freq, direction, result)
        ax.set_title(f"Python Partition Polar. Index {idx}", pad=22)
        fig.savefig(assets_dir / f"python_case_{idx:03d}_polar.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
