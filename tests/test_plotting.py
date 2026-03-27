from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from scipy.io import loadmat

from wavepart import partition_spectrum
from wavepart.plotting import plot_partition_polar, plot_partition_surface


def test_plotting_helpers_render(tmp_path) -> None:
    data = loadmat("data/wavespec2d_ex.mat")
    freq = data["freq"].reshape(-1)
    direction = data["dir"].reshape(-1)
    spectrum = data["S"][:, :, 0]
    partition = partition_spectrum(freq, direction, spectrum)

    fig, _ = plot_partition_surface(freq, direction, spectrum, partition)
    surface_path = tmp_path / "surface.png"
    fig.savefig(surface_path, dpi=100)
    fig.clf()

    fig, _ = plot_partition_polar(freq, direction, partition)
    polar_path = tmp_path / "polar.png"
    fig.savefig(polar_path, dpi=100)
    fig.clf()

    assert surface_path.exists()
    assert polar_path.exists()
