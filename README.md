# WavePart Python

![WavePart 3D comparison](reports/assets/python_case_287_3d.png)

Public Python port of WavePart for partitioning directional wave spectra into wind sea and multiple swell components. This repository preserves the original MATLAB implementation, adds a tested Python package, and includes side-by-side MATLAB versus Python comparison outputs.

## Overview

The Python package lives under `src/wavepart` and ports the core WavePart workflow:

- directional spectrum partitioning
- wind-limit estimation across a time series
- partition parameter extraction
- plotting helpers for surface, polar, and 3D views
- MATLAB-oracle parity fixtures and tests

The original MATLAB project this work is based on is:

- https://github.com/gvoulgaris0/WavePart

This repository keeps that original work visible and extends it with a public Python implementation.

## Installation

Install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[plot,dev]
```

Install without plotting extras:

```bash
python -m pip install -e .
```

## Quick Start

Run the CLI on the bundled sample data:

```bash
wavepart partition-mat data/wavespec2d_ex.mat --index 1 --output out/partition_case.npz
wavepart wind-limits data/wavespec2d_ex.mat --output out/wind_limits.npz
wavepart demo data/wavespec2d_ex.mat --index 287 --output-dir out/demo
```

Use the package directly:

```python
from scipy.io import loadmat
from wavepart import partition_spectrum, compute_partition_params

data = loadmat("data/wavespec2d_ex.mat")
freq = data["freq"].reshape(-1)
direction = data["dir"].reshape(-1)
spectrum = data["S"][:, :, 0]

partition = partition_spectrum(freq, direction, spectrum)
params = compute_partition_params(
    partition.smoothed_spectrum,
    freq,
    direction,
    partition.labels,
    depth=30.0,
)
```

## Verification

The Python implementation is verified against MATLAB-generated oracle fixtures from the bundled sample dataset.

Refresh the oracle fixture locally:

```bash
python scripts/generate_oracles.py
```

Run the test suite:

```bash
pytest
```

Current coverage includes:

- end-to-end MATLAB parity checks for representative time indices
- wind-limit parity checks
- invariant tests for label semantics and flat-spectrum behavior
- CLI smoke tests
- plotting smoke tests

## Repository Contents

- `src/wavepart`: Python package
- `tests`: parity and smoke tests
- `scripts`: export and oracle-generation helpers
- `reports`: MATLAB/Python comparison report and generated figures
- `data`: bundled sample directional wave spectra
- `*.m`: original MATLAB source files

## Attribution

Original MATLAB authors:

- Douglas Cahl
- George Voulgaris

Original institutional affiliation:

- School of the Earth, Ocean and Environment
- University of South Carolina

Relevant citation from the original project:

- Douglas, C. and Voulgaris, G., 2019, WavePART: MATLAB software for the partition of directional ocean wave spectra. Zenodo. doi:10.5281/zenodo.2638500

## License

This repository is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
