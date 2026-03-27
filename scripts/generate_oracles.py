"""Placeholder utility for generating MATLAB reference fixtures."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print(f"Reference fixtures are expected under {repo_root / 'tests' / 'fixtures'}")
    print("Add a MATLAB-backed exporter when the Python core is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
