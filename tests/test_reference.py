from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wavepart import compute_partition_params, estimate_wind_limits, partition_spectrum


def _reference_fixture():
    return np.load("tests/fixtures/reference_outputs.npz")


def _sample_data():
    with np.load("data/sample_spectra.npz") as data:
        return data["freq"], data["direction"], data["spectra"], data["time"]


def test_partition_matches_reference_cases() -> None:
    with _reference_fixture() as ref:
        freq, direction, spectra, _ = _sample_data()

        for index in np.asarray(ref["indices"]).reshape(-1):
            key = f"case_{int(index):03d}"
            result = partition_spectrum(freq, direction, spectra[:, :, int(index) - 1])
            params = compute_partition_params(result.smoothed_spectrum, freq, direction, result.labels, depth=30)
            frequency_metrics, direction_metrics, energy_metrics, height_metrics = params.to_raw_arrays()

            assert_array_equal(result.labels, np.asarray(ref[f"{key}_labels"], dtype=int))
            assert_allclose(result.smoothed_spectrum, np.asarray(ref[f"{key}_smoothed"], dtype=float), rtol=1e-6, atol=1e-8)
            assert_allclose(frequency_metrics, np.asarray(ref[f"{key}_frequency_metrics"], dtype=float), rtol=1e-6, atol=1e-8)
            assert_allclose(direction_metrics, np.asarray(ref[f"{key}_direction_metrics"], dtype=float), rtol=1e-6, atol=1e-8)
            assert_allclose(energy_metrics, np.asarray(ref[f"{key}_energy_metrics"], dtype=float), rtol=5e-6, atol=1e-7)
            assert_allclose(height_metrics, np.asarray(ref[f"{key}_height_metrics"], dtype=float), rtol=5e-6, atol=1e-7)


def test_wind_series_matches_reference() -> None:
    with _reference_fixture() as ref:
        freq, direction, spectra, time = _sample_data()
        result = estimate_wind_limits(time, freq, direction, spectra)
        assert_allclose(result.fw, np.asarray(ref["wind_fw"], dtype=float).reshape(-1), rtol=1e-6, atol=1e-8)
        assert_allclose(result.dw, np.asarray(ref["wind_dw"], dtype=float).reshape(-1), rtol=1e-6, atol=1e-8)
