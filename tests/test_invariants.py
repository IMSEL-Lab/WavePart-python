from __future__ import annotations

import numpy as np

from wavepart import compute_partition_params, partition_spectrum


def _load_sample_data():
    with np.load("data/sample_spectra.npz") as data:
        freq = data["freq"]
        direction = data["direction"]
        spectra = data["spectra"]
    return freq, direction, spectra


def test_flat_spectrum_returns_noise() -> None:
    freq, direction, _ = _load_sample_data()
    spectrum = np.zeros((freq.size, direction.size), dtype=float)
    result = partition_spectrum(freq, direction, spectrum)
    assert result.partition_count == 0
    assert np.array_equal(result.labels, np.zeros_like(result.labels))


def test_partition_labels_are_consecutive() -> None:
    freq, direction, spectra = _load_sample_data()
    spectrum = spectra[:, :, 0]
    result = partition_spectrum(freq, direction, spectrum)
    labels = np.unique(result.labels)
    assert np.array_equal(labels, np.arange(labels[-1] + 1))


def test_wave_parameter_shapes_match_partition_count() -> None:
    freq, direction, spectra = _load_sample_data()
    spectrum = spectra[:, :, 0]
    result = partition_spectrum(freq, direction, spectrum)
    params = compute_partition_params(result.smoothed_spectrum, freq, direction, result.labels, depth=30)
    assert params.mean_frequency.shape == (result.partition_count,)
    assert params.peak_frequency.shape == (result.partition_count,)
