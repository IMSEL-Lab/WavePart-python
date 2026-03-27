from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from wavepart import partition_spectrum


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


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    with np.load(repo / "data" / "sample_spectra.npz") as data:
        freq = data["freq"]
        direction = data["direction"]
        spectrum = data["spectra"][:, :, 286]

    result = partition_spectrum(freq, direction, spectrum)
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
    ax.set_title("WavePart 3D View")
    ax.view_init(elev=32, azim=-38)
    fig.patch.set_facecolor("white")
    output = repo / "docs" / "assets" / "wavepart-3d.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
