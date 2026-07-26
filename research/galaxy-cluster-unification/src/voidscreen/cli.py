from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import load_provenance, pack_dataset
from .experiment import ExperimentSettings, device_report, run_experiment


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _common_parser(description: str) -> argparse.ArgumentParser:
    root = _project_root()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=root / "configs" / "baseline.json")
    parser.add_argument("--data", type=Path, default=root / "data" / "raw" / "sparc")
    parser.add_argument("--environment-csv", type=Path)
    parser.add_argument("--environment-column", default="void_score")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _dataset_and_settings(args: argparse.Namespace):
    config = _load_config(args.config)
    packed = pack_dataset(
        args.data,
        quality_max=int(config["quality_max"]),
        minimum_inclination_deg=float(config["minimum_inclination_deg"]),
        minimum_points=int(config["minimum_points"]),
        train_fraction=float(config["train_fraction"]),
        minimum_train_points=int(config["minimum_train_points"]),
        minimum_holdout_points=int(config["minimum_holdout_points"]),
        environment_csv=args.environment_csv,
        environment_score_column=args.environment_column,
    )
    settings = ExperimentSettings(
        seed=int(args.seed if args.seed is not None else config["seed"]),
        disk_mass_to_light_prior=float(config["disk_mass_to_light_prior"]),
        bulge_mass_to_light_prior=float(config["bulge_mass_to_light_prior"]),
        log_mass_to_light_prior_sigma=float(config["log_mass_to_light_prior_sigma"]),
        velocity_error_floor_kms=float(config["velocity_error_floor_kms"]),
        rar_acceleration_m_s2=float(config["rar_acceleration_m_s2"]),
        hubble_km_s_mpc=float(config["hubble_km_s_mpc"]),
        learning_rate=float(
            args.learning_rate if args.learning_rate is not None else config["learning_rate"]
        ),
        steps=int(args.steps if args.steps is not None else config["steps"]),
    )
    return packed, settings


def main_fit(argv: list[str] | None = None) -> int:
    root = _project_root()
    parser = _common_parser("Fit one SPARC rotation-curve model with radial holdout")
    parser.add_argument(
        "--model", choices=["newtonian", "rar", "nfw", "void", "potential"], default="void"
    )
    parser.add_argument("--fixed-flat-power", action="store_true")
    parser.add_argument("--boundary-layer", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "results" / "single_fit")
    args = parser.parse_args(argv)
    if args.fixed_flat_power and args.model != "void":
        parser.error("--fixed-flat-power is only valid with --model void")
    if args.boundary_layer and args.model != "potential":
        parser.error("--boundary-layer is only valid with --model potential")
    packed, settings = _dataset_and_settings(args)
    print(
        f"Loaded {packed.n_galaxies} galaxies / {packed.n_points} points "
        f"({packed.n_train} train, {packed.n_holdout} outer holdout)"
    )
    summary = run_experiment(
        packed,
        model_name=args.model,
        output_dir=args.output,
        device_name=args.device,
        dtype_name=args.dtype,
        settings=settings,
        fixed_flat_power=args.fixed_flat_power,
        environment_enabled=args.environment_csv is not None,
        boundary_layer_enabled=args.boundary_layer,
        progress=not args.quiet,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _save_comparison_plot(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ordered = frame.sort_values("outer_chi2_per_point")
    axes[0].bar(ordered["run_label"], ordered["outer_chi2_per_point"], color="#4C78A8")
    axes[0].set(ylabel="Outer holdout chi² / point")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(ordered["run_label"], ordered["outer_rmse_kms"], color="#F58518")
    axes[1].set(ylabel="Outer holdout RMSE (km/s)")
    axes[1].tick_params(axis="x", rotation=35)
    figure.suptitle("SPARC radial-extrapolation comparison")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main_compare(argv: list[str] | None = None) -> int:
    root = _project_root()
    parser = _common_parser("Compare preregistered SPARC models on one radial split")
    parser.add_argument("--output", type=Path, default=root / "results" / "comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "newtonian",
            "rar",
            "nfw",
            "void",
            "void_p05",
            "void_env",
            "potential",
            "potential_env",
            "potential_boundary",
        ],
    )
    args = parser.parse_args(argv)
    packed, settings = _dataset_and_settings(args)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    selected_models = args.models or ["newtonian", "rar", "nfw", "void", "void_p05"]
    if args.environment_csv is not None and "void_env" not in selected_models:
        selected_models.append("void_env")
    environment_models = {"void_env", "potential_env"}
    if args.environment_csv is None and environment_models.intersection(selected_models):
        parser.error("void_env and potential_env require --environment-csv")
    for label in selected_models:
        model_name = (
            "void"
            if label in {"void_p05", "void_env"}
            else "potential"
            if label in {"potential_env", "potential_boundary"}
            else label
        )
        fixed = label == "void_p05"
        print(f"\n=== {label} ===")
        summary = run_experiment(
            packed,
            model_name=model_name,
            output_dir=args.output / label,
            device_name=args.device,
            dtype_name=args.dtype,
            settings=settings,
            fixed_flat_power=fixed,
            environment_enabled=label in environment_models,
            boundary_layer_enabled=label == "potential_boundary",
            run_label=label,
            progress=not args.quiet,
        )
        summaries.append(summary)
    rows = [
        {
            "run_label": summary["run_label"],
            "parameter_count": summary["parameter_count"],
            "aic_train": summary["aic_train"],
            "bic_train": summary["bic_train"],
            "train_chi2_per_point": summary["train"]["chi2_per_point"],
            "outer_chi2_per_point": summary["outer_holdout"]["chi2_per_point"],
            "outer_rmse_kms": summary["outer_holdout"]["rmse_kms"],
            "outer_mae_kms": summary["outer_holdout"]["mae_kms"],
        }
        for summary in summaries
    ]
    frame = pd.DataFrame(rows).sort_values("outer_chi2_per_point")
    frame.to_csv(args.output / "comparison.csv", index=False)
    (args.output / "comparison.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_comparison_plot(frame, args.output / "comparison.png")
    print("\n" + frame.to_string(index=False))
    return 0


def main_device() -> int:
    print(json.dumps(device_report(), indent=2))
    return 0


def main_inspect(argv: list[str] | None = None) -> int:
    parser = _common_parser("Inspect SPARC inputs and the deterministic radial split")
    args = parser.parse_args(argv)
    packed, settings = _dataset_and_settings(args)
    provenance = load_provenance(args.data)
    report = {
        "galaxies": packed.n_galaxies,
        "points": packed.n_points,
        "train_points": packed.n_train,
        "outer_holdout_points": packed.n_holdout,
        "environment_score_column": packed.environment_score_column,
        "environment_fingerprint_sha256": packed.environment_fingerprint,
        "data_fingerprint_sha256": packed.data_fingerprint,
        "environment_enabled": args.environment_csv is not None,
        "settings": settings.__dict__,
        "provenance": provenance,
    }
    print(json.dumps(report, indent=2))
    return 0
