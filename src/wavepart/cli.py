from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analyze_spectrum_file, estimate_wind_limits, load_spectrum_set


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Path to an NPZ file containing freq, direction, spectra, and optional time.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wavepart", description="WavePart Python CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    part = subparsers.add_parser("partition", help="Partition a single spectrum from an NPZ dataset.")
    _add_common_args(part)
    part.add_argument("--index", type=int, default=1, help="1-based spectrum index to partition.")
    part.add_argument("--depth", type=float, default=30.0, help="Water depth in meters.")
    part.add_argument("--output", type=Path, required=True, help="Output NPZ file.")
    part.add_argument("--plot-dir", type=Path, help="Optional directory for surface and polar PNG outputs.")
    part.add_argument("--wind-cutoff", type=float, help="Optional wind cutoff frequency in Hz.")
    part.add_argument("--no-wind-region", action="store_true", help="Disable wind-region reassignment.")

    wind = subparsers.add_parser("wind-limits", help="Estimate wind limits for all spectra in an NPZ dataset.")
    _add_common_args(wind)
    wind.add_argument("--output", type=Path, required=True, help="Output NPZ file.")

    demo = subparsers.add_parser("demo", help="Render example figures for one spectrum from an NPZ dataset.")
    _add_common_args(demo)
    demo.add_argument("--index", type=int, default=1, help="1-based spectrum index to analyze.")
    demo.add_argument("--depth", type=float, default=30.0, help="Water depth in meters.")
    demo.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs.")
    demo.add_argument("--estimate-wind", action="store_true", help="Estimate wind cutoff from the time series first.")
    return parser


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _save_partition_outputs(output: Path, step) -> None:
    params = step.parameters
    frequency_metrics, direction_metrics, energy_metrics, height_metrics = params.to_raw_arrays() if params is not None else (None, None, None, None)
    np.savez(
        output,
        labels=step.partition.labels,
        smoothed_spectrum=step.partition.smoothed_spectrum,
        frequency_metrics=frequency_metrics,
        direction_metrics=direction_metrics,
        energy_metrics=energy_metrics,
        height_metrics=height_metrics,
        wind_cutoff=step.wind_cutoff,
        index=step.index + 1,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "partition":
        analysis = analyze_spectrum_file(
            args.input,
            indices=[args.index - 1],
            depth=args.depth,
            backend="reference",
            keep_wind_region=not args.no_wind_region,
        )
        step = analysis.steps[0]
        if args.wind_cutoff is not None:
            from .core import partition_spectrum
            from .params import compute_partition_params

            dataset = analysis.dataset
            partition = partition_spectrum(
                dataset.freq,
                dataset.direction,
                dataset.spectra[:, :, step.index],
                wind_cutoff=args.wind_cutoff,
                keep_wind_region=not args.no_wind_region,
            )
            step.partition = partition
            step.parameters = compute_partition_params(
                partition.smoothed_spectrum,
                dataset.freq,
                dataset.direction,
                partition.labels,
                depth=args.depth,
            )
            step.wind_cutoff = args.wind_cutoff

        args.output.parent.mkdir(parents=True, exist_ok=True)
        _save_partition_outputs(args.output, step)
        if args.plot_dir is not None:
            from .plotting import plot_partition_polar, plot_partition_surface

            args.plot_dir.mkdir(parents=True, exist_ok=True)
            fig, _ = plot_partition_surface(
                analysis.dataset.freq,
                analysis.dataset.direction,
                analysis.dataset.spectra[:, :, step.index],
                step.partition,
            )
            fig.savefig(args.plot_dir / "partition_surface.png", dpi=150, bbox_inches="tight")
            fig.clf()
            fig, _ = plot_partition_polar(analysis.dataset.freq, analysis.dataset.direction, step.partition)
            fig.savefig(args.plot_dir / "partition_polar.png", dpi=150, bbox_inches="tight")
            fig.clf()

        _write_summary(
            args.output.with_suffix(".json"),
            {
                "index": step.index + 1,
                "partition_count": step.partition.partition_count,
                "wind_cutoff": step.wind_cutoff,
            },
        )
        return 0

    if args.command == "wind-limits":
        dataset = load_spectrum_set(args.input)
        if dataset.time is None:
            raise ValueError("The input file must contain 'time' to estimate wind limits.")
        wind = estimate_wind_limits(dataset.time, dataset.freq, dataset.direction, dataset.spectra)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            time=wind.time,
            fw=wind.fw,
            dw=wind.dw,
            raw_fw=wind.raw_fw,
            raw_peak_direction=wind.raw_peak_direction,
        )
        _write_summary(args.output.with_suffix(".json"), {"count": int(wind.time.size)})
        return 0

    if args.command == "demo":
        from .plotting import plot_partition_polar, plot_partition_surface

        args.output_dir.mkdir(parents=True, exist_ok=True)
        analysis = analyze_spectrum_file(
            args.input,
            indices=[args.index - 1],
            depth=args.depth,
            estimate_wind=args.estimate_wind,
        )
        step = analysis.steps[0]
        fig, _ = plot_partition_surface(
            analysis.dataset.freq,
            analysis.dataset.direction,
            analysis.dataset.spectra[:, :, step.index],
            step.partition,
        )
        fig.savefig(args.output_dir / "partition_surface.png", dpi=150, bbox_inches="tight")
        fig.clf()
        fig, _ = plot_partition_polar(analysis.dataset.freq, analysis.dataset.direction, step.partition)
        fig.savefig(args.output_dir / "partition_polar.png", dpi=150, bbox_inches="tight")
        fig.clf()
        _write_summary(
            args.output_dir / "summary.json",
            {
                "index": step.index + 1,
                "partition_count": step.partition.partition_count,
                "wind_cutoff": step.wind_cutoff,
            },
        )
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
