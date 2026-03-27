from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_partition_cli(tmp_path) -> None:
    output = tmp_path / "partition_case.npz"
    cmd = [
        sys.executable,
        "-m",
        "wavepart.cli",
        "partition",
        "data/sample_spectra.npz",
        "--index",
        "1",
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    payload = np.load(output)
    assert "labels" in payload
    assert int(payload["index"]) == 1


def test_wind_limits_cli(tmp_path) -> None:
    output = tmp_path / "wind_limits.npz"
    cmd = [
        sys.executable,
        "-m",
        "wavepart.cli",
        "wind-limits",
        "data/sample_spectra.npz",
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    payload = np.load(output)
    assert "fw" in payload
    assert "dw" in payload
