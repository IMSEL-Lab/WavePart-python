# WavePart ![image](https://user-images.githubusercontent.com/48567321/126871118-68ae6e82-15fe-48e5-ac1e-2d1d0b673a9b.png)


Set of Matlab(r) functions for the partition of directional wave spectra to its wind and different swell components. The partitions are initially identified using a watershed defining algorithm  and then are modified following mostly the method described in Hanson and Phillips (2001).

## Python Package Shell

A Python package scaffold now lives alongside the MATLAB sources. The package is installable from the repo root and exposes a thin CLI for the planned Python port.

Install the base package.

```bash
pip install -e .
```

Install optional plotting or labeled-array support.

```bash
pip install -e ".[plot,xarray]"
```

Inspect the CLI.

```bash
wavepart --help
wavepart partition-mat --mat data/wavespec2d_ex.mat --index 0 --depth 30
wavepart wind-limits --mat data/wavespec2d_ex.mat
wavepart demo --mat data/wavespec2d_ex.mat
```

The CLI shell is in place now. The numerical Python modules are being ported separately, so the commands will become fully functional as those modules land.

Authors:  
  Douglas Cahl and George Voulgaris  
  School of the Earth, Ocean and Environment  
  University of South Carolina  
  Columbia, SC, 29205, USA.  
  
Code Updates:
  -  1/22/2020 - waveparams.m - the mean freq. (fm) estimate was incorrect; it has been corrected.
  -  1/22/2020 - partition.m was updated to catch cases when a flat spectrum is given as input.
  -  1/22/2020 - master_partition.m was updated so that it calls waveparams.m with the correct number of outputs.
  -  1/21/2020 - waveparams.m - the Hrms estimate was incorrect; also the mean direction now is given in -180 to +180 deg range.
  
Citation:  
   -  Douglas, C. and Voulgaris, G., 2019, WavePART: MATLAB(r) software for the partition of directional ocean wave spectra.: Zenodo, doi:10.5281/zenodo.2638500. 

Relevant References:  
   -  J.L. Hanson and O.M. Philips, 2001. Automated Analysis of Ocean Surface Directional  Wave Spectra. Journal of Oceanic and Atmospheric Technology, 18, 278-293.   
   -  J. Portilla, F.J. Ocampo-Torres, and J. Monbaliu, 2009. Spectral Partitioning and Identification of Wind Sea and Swell.  Journal of Oceanic and Atmospheric Technology, 26, 107-121. DOI: 10.1175/2008JTECHO609.1   
   -  E. Cheynet, 2019. Pcolor in Polar Coordinates (https://www.mathworks.com/matlabcentral/fileexchange/49040-pcolor-in-polar-coordinates), MATLAB Central File Exchange. Retrieved March 16, 2019.  

## Python Port

An installable Python port lives under `src/wavepart`.

Install it in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[plot,dev]
```

Run the CLI:

```bash
wavepart partition-mat data/wavespec2d_ex.mat --index 1 --output out/partition_case.npz
wavepart wind-limits data/wavespec2d_ex.mat --output out/wind_limits.npz
wavepart demo data/wavespec2d_ex.mat --index 1 --output-dir out/demo
```

Refresh MATLAB parity fixtures for tests:

```bash
python scripts/generate_oracles.py
pytest
```
