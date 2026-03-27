from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from scipy.io import loadmat

from wavepart import partition_spectrum, plot_partition_polar, plot_partition_surface


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
        fig.clf()

        fig, ax = plot_partition_polar(freq, direction, result)
        ax.set_title(f"Python Partition Polar. Index {idx}", pad=22)
        fig.savefig(assets_dir / f"python_case_{idx:03d}_polar.png", dpi=180, bbox_inches="tight")
        fig.clf()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
