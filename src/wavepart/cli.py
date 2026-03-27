"""Command-line entry points for the WavePart Python package shell."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _import_symbol(module_name: str, symbol: str) -> Any:
    try:
        module = __import__(module_name, fromlist=[symbol])
    except ImportError as exc:  # pragma: no cover - exercised in packaging smoke tests
        raise SystemExit(
            f"{module_name}.{symbol} is not available yet. "
            "Install the core Python port before using this command."
        ) from exc
    try:
        return getattr(module, symbol)
    except AttributeError as exc:  # pragma: no cover - exercised in packaging smoke tests
        raise SystemExit(
            f"{module_name}.{symbol} is not available yet. "
            "Install the core Python port before using this command."
        ) from exc


def _load_mat(path: Path) -> dict[str, Any]:
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def _default_mat_path() -> Path:
    repo_candidate = Path(__file__).resolve().parents[2] / "data" / "wavespec2d_ex.mat"
    return repo_candidate


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _extract_wind_cutoff(result: Any, index: int) -> float | None:
    fw = _field(result, "fw")
    if fw is None:
        if isinstance(result, (list, tuple)) and result:
            fw = result[0]
        else:
            return None
    array = np.asarray(fw).squeeze()
    if array.ndim == 0:
        return float(array)
    return float(array[index])


def _write_output(path: Path | None, payload: Any) -> None:
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    if path is None:
        print(text)
        return
    path.write_text(text + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="wavepart")
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition-mat", help="Partition one spectrum from a MATLAB .mat file.")
    partition.add_argument("--mat", type=Path, default=_default_mat_path())
    partition.add_argument("--index", type=int, default=0)
    partition.add_argument("--depth", type=float, default=30.0)
    partition.add_argument("--wind-cutoff", type=float, default=None)
    partition.add_argument("--keep-wind-region", action=argparse.BooleanOptionalAction, default=True)
    partition.add_argument("--backend", default="reference")
    partition.add_argument("--output", type=Path, default=None)
    partition.set_defaults(func=_cmd_partition_mat)

    wind = subparsers.add_parser("wind-limits", help="Estimate wind limits from a MATLAB .mat file.")
    wind.add_argument("--mat", type=Path, default=_default_mat_path())
    wind.add_argument("--plot", action="store_true")
    wind.add_argument("--output", type=Path, default=None)
    wind.set_defaults(func=_cmd_wind_limits)

    demo = subparsers.add_parser("demo", help="Run the sample WavePart workflow.")
    demo.add_argument("--mat", type=Path, default=_default_mat_path())
    demo.add_argument("--index", type=int, default=0)
    demo.add_argument("--depth", type=float, default=30.0)
    demo.add_argument("--backend", default="reference")
    demo.add_argument("--output", type=Path, default=None)
    demo.set_defaults(func=_cmd_demo)

    return parser.parse_args(argv)


def _cmd_partition_mat(args: argparse.Namespace) -> Any:
    partition_spectrum = _import_symbol("wavepart.partition", "partition_spectrum")
    params_fn = _import_symbol("wavepart.waveparamspart", "compute_partition_params")
    data = _load_mat(args.mat)
    result = partition_spectrum(
        np.asarray(data["freq"]).squeeze(),
        np.asarray(data["dir"]).squeeze(),
        np.asarray(data["S"])[:, :, args.index],
        wind_cutoff=args.wind_cutoff,
        keep_wind_region=args.keep_wind_region,
        backend=args.backend,
    )
    params = params_fn(
        np.asarray(data["S"])[:, :, args.index],
        np.asarray(data["freq"]).squeeze(),
        np.asarray(data["dir"]).squeeze(),
        result.labels,
        depth=args.depth,
    )
    payload = {
        "partition": result,
        "parameters": params,
    }
    _write_output(args.output, payload)
    return payload


def _cmd_wind_limits(args: argparse.Namespace) -> Any:
    estimate_wind_limits = _import_symbol("wavepart.readspectra", "estimate_wind_limits")
    data = _load_mat(args.mat)
    result = estimate_wind_limits(
        np.asarray(data["t"]).squeeze(),
        np.asarray(data["freq"]).squeeze(),
        np.asarray(data["dir"]).squeeze(),
        np.asarray(data["S"]),
    )
    _write_output(args.output, result)
    return result


def _cmd_demo(args: argparse.Namespace) -> Any:
    estimate_wind_limits = _import_symbol("wavepart.readspectra", "estimate_wind_limits")
    partition_spectrum = _import_symbol("wavepart.partition", "partition_spectrum")
    params_fn = _import_symbol("wavepart.waveparamspart", "compute_partition_params")
    data = _load_mat(args.mat)
    fw = estimate_wind_limits(
        np.asarray(data["t"]).squeeze(),
        np.asarray(data["freq"]).squeeze(),
        np.asarray(data["dir"]).squeeze(),
        np.asarray(data["S"]),
    )
    wind_cutoff = _extract_wind_cutoff(fw, args.index)
    partition = partition_spectrum(
        np.asarray(data["freq"]).squeeze(),
        np.asarray(data["dir"]).squeeze(),
        np.asarray(data["S"])[:, :, args.index],
        wind_cutoff=wind_cutoff,
        backend=args.backend,
    )
    params = params_fn(
        np.asarray(data["S"])[:, :, args.index],
        np.asarray(data["freq"]).squeeze(),
        np.asarray(data["dir"]).squeeze(),
        partition.labels,
        depth=args.depth,
    )
    payload = {
        "wind_limits": fw,
        "partition": partition,
        "parameters": params,
    }
    _write_output(args.output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution path
    raise SystemExit(main())
