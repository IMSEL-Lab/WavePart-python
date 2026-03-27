from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


MATLAB_TEMPLATE = """
addpath('{repo}');
load('{dataset}');
idx = [1 100 267 287];
AA_ref = cell(1, numel(idx));
Ef_ref = cell(1, numel(idx));
f_ref = cell(1, numel(idx));
D_ref = cell(1, numel(idx));
Ee_ref = cell(1, numel(idx));
H_ref = cell(1, numel(idx));
for ii = 1:numel(idx)
    i = idx(ii);
    E0 = S(:,:,i);
    [AA,Ef] = partition(freq,dir,E0,1);
    [f_out,D_out,Ee_out,H_out] = waveparamspart(Ef,freq,dir,AA,30);
    AA_ref{{ii}} = AA;
    Ef_ref{{ii}} = Ef;
    f_ref{{ii}} = f_out;
    D_ref{{ii}} = D_out;
    Ee_ref{{ii}} = Ee_out;
    H_ref{{ii}} = H_out;
end
[fw_ref,dw_ref] = readspectra(t,freq,dir,S,0);
save('{output}', 'idx', 'AA_ref', 'Ef_ref', 'f_ref', 'D_ref', 'Ee_ref', 'H_ref', 'fw_ref', 'dw_ref');
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MATLAB reference fixtures for WavePart.")
    parser.add_argument(
        "--matlab-bin",
        default=shutil.which("matlab") or "matlab",
        help="MATLAB binary to invoke.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/reference_outputs.mat"),
        help="Output MAT fixture path.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "data" / "wavespec2d_ex.mat"
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    matlab_code = MATLAB_TEMPLATE.format(
        repo=str(repo).replace("'", "''"),
        dataset=str(dataset).replace("'", "''"),
        output=str(output).replace("'", "''"),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".m", delete=False) as handle:
        handle.write(matlab_code)
        temp_path = Path(handle.name)

    try:
        temp_run = str(temp_path).replace("'", "''")
        subprocess.run(
            [args.matlab_bin, "-batch", f"run('{temp_run}')"],
            check=True,
            cwd=repo,
        )
    finally:
        temp_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
