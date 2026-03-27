from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from wavepart import compute_partition_params, partition_spectrum


def test_flat_spectrum_returns_noise() -> None:
    data = loadmat("data/wavespec2d_ex.mat")
    freq = data["freq"].reshape(-1)
    direction = data["dir"].reshape(-1)
    spectrum = np.zeros((freq.size, direction.size), dtype=float)
    result = partition_spectrum(freq, direction, spectrum)
    assert result.partition_count == 0
    assert np.array_equal(result.labels, np.zeros_like(result.labels))


def test_partition_labels_are_consecutive() -> None:
    data = loadmat("data/wavespec2d_ex.mat")
    freq = data["freq"].reshape(-1)
    direction = data["dir"].reshape(-1)
    spectrum = data["S"][:, :, 0]
    result = partition_spectrum(freq, direction, spectrum)
    labels = np.unique(result.labels)
    assert np.array_equal(labels, np.arange(labels[-1] + 1))


def test_wave_parameter_shapes_match_partition_count() -> None:
    data = loadmat("data/wavespec2d_ex.mat")
    freq = data["freq"].reshape(-1)
    direction = data["dir"].reshape(-1)
    spectrum = data["S"][:, :, 0]
    result = partition_spectrum(freq, direction, spectrum)
    params = compute_partition_params(result.smoothed_spectrum, freq, direction, result.labels, depth=30)
    assert params.mean_frequency.shape == (result.partition_count,)
    assert params.peak_frequency.shape == (result.partition_count,)
