from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from wavepart import compute_partition_params, estimate_wind_limits, partition_spectrum


def _reference_fixture():
    return loadmat("tests/fixtures/reference_outputs.mat", simplify_cells=True)


def _sample_data():
    data = loadmat("data/wavespec2d_ex.mat")
    return data["freq"].reshape(-1), data["dir"].reshape(-1), data["S"], data["t"].reshape(-1)


def test_partition_matches_reference_cases() -> None:
    ref = _reference_fixture()
    freq, direction, spectra, _ = _sample_data()

    for ii, index in enumerate(np.asarray(ref["idx"]).reshape(-1)):
        result = partition_spectrum(freq, direction, spectra[:, :, int(index) - 1])
        params = compute_partition_params(result.smoothed_spectrum, freq, direction, result.labels, depth=30)
        f_out, d_out, ee_out, h_out = params.to_matlab_arrays()

        assert_array_equal(result.labels, np.asarray(ref["AA_ref"][ii], dtype=int))
        assert_allclose(result.smoothed_spectrum, np.asarray(ref["Ef_ref"][ii], dtype=float), rtol=1e-6, atol=1e-8)
        assert_allclose(f_out, np.asarray(ref["f_ref"][ii], dtype=float), rtol=1e-6, atol=1e-8)
        assert_allclose(d_out, np.asarray(ref["D_ref"][ii], dtype=float), rtol=1e-6, atol=1e-8)
        assert_allclose(ee_out, np.asarray(ref["Ee_ref"][ii], dtype=float), rtol=1e-6, atol=1e-8)
        assert_allclose(h_out, np.asarray(ref["H_ref"][ii], dtype=float), rtol=1e-6, atol=1e-8)


def test_wind_series_matches_reference() -> None:
    ref = _reference_fixture()
    freq, direction, spectra, time = _sample_data()
    result = estimate_wind_limits(time, freq, direction, spectra)
    assert_allclose(result.fw, np.asarray(ref["fw_ref"], dtype=float).reshape(-1), rtol=1e-6, atol=1e-8)
    assert_allclose(result.dw, np.asarray(ref["dw_ref"], dtype=float).reshape(-1), rtol=1e-6, atol=1e-8)
