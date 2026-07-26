from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from cross_validate_cf4 import (
    _balanced_folds,
    _bootstrap_comparison,
    _fold_dataset,
    _run_or_load,
    _settings,
)

from voidscreen.data import pack_dataset
from voidscreen.experiment import resolve_device

ROOT = Path(__file__).resolve().parents[1]


def _variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "rar_reference",
            "model": "rar",
            "environment_enabled": False,
            "boundary_layer_enabled": False,
            "score_column": "void_score",
            "fit": False,
        },
        {
            "name": "potential_P0",
            "model": "potential",
            "environment_enabled": False,
            "boundary_layer_enabled": False,
            "score_column": "void_score",
            "fit": True,
        },
        {
            "name": "potential_W1_void_wall",
            "model": "potential",
            "environment_enabled": True,
            "boundary_layer_enabled": False,
            "score_column": "void_score",
            "fit": True,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict wall-balanced galaxy validation of frozen W1 void geometry."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=ROOT / "data" / "derived" / "void_wall_scores_local.csv",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "void_wall_galaxy_cv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    settings = _settings(args.config, steps=args.steps)
    packed = pack_dataset(
        args.data,
        environment_csv=args.environment_csv,
        environment_score_column="void_score",
    )
    raw_score = packed.environment_raw.copy()
    fold_assignment = _balanced_folds(raw_score, args.folds)
    wall_table = pd.read_csv(args.environment_csv).set_index("galaxy")
    assignments = pd.DataFrame(
        {
            "galaxy": packed.galaxy_names,
            "fold": fold_assignment,
            "void_wall_score": raw_score,
            "inside_catalog_void": wall_table.loc[
                list(packed.galaxy_names), "inside_catalog_void"
            ].to_numpy(dtype=bool),
        }
    ).sort_values("galaxy")
    args.output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output / "fold_assignments.csv", index=False)

    device = resolve_device(args.device)
    summaries: list[dict[str, Any]] = []
    galaxy_frames: list[pd.DataFrame] = []
    for heldout_fold in range(args.folds):
        fold_data = _fold_dataset(
            packed,
            raw_score,
            fold_assignment,
            heldout_fold,
            "void_score",
        )
        for variant in _variants():
            print(f"fold={heldout_fold} variant={variant['name']}", flush=True)
            summary, _, per_galaxy = _run_or_load(
                variant=variant,
                heldout_fold=heldout_fold,
                fold_data=fold_data,
                settings=settings,
                device=device,
                output_dir=args.output,
            )
            summaries.append(summary)
            per_galaxy.insert(0, "variant", variant["name"])
            per_galaxy.insert(0, "fold", heldout_fold)
            galaxy_frames.append(per_galaxy)

    galaxy_metrics = pd.concat(galaxy_frames, ignore_index=True)
    galaxy_metrics.to_csv(args.output / "all_heldout_galaxy_metrics.csv", index=False)
    fold_summary = pd.DataFrame(
        [
            {
                "fold": summary["fold"],
                "variant": summary["variant"],
                "power_p": summary["global_parameters"].get("power_p"),
                "transition_potential_chi": summary["global_parameters"].get(
                    "transition_potential_chi"
                ),
                "void_wall_zeta": summary["global_parameters"].get(
                    "environment_shift_zeta_dex_per_sigma"
                ),
                "heldout_chi2_per_point": summary["heldout_galaxies"]["chi2_per_point"],
                "heldout_rmse_kms": summary["heldout_galaxies"]["rmse_kms"],
            }
            for summary in summaries
        ]
    )
    fold_summary.to_csv(args.output / "fold_summary.csv", index=False)

    metrics_by_variant: dict[str, Any] = {}
    for variant, selected in galaxy_metrics.groupby("variant"):
        metrics_by_variant[variant] = {
            "galaxies": len(selected),
            "points": int(selected["n"].sum()),
            "chi2_per_point": float(selected["chi2"].sum() / selected["n"].sum()),
            "rmse_kms": float(math.sqrt(selected["squared_error"].sum() / selected["n"].sum())),
        }

    p0 = galaxy_metrics.loc[galaxy_metrics["variant"] == "potential_P0"]
    w1 = galaxy_metrics.loc[galaxy_metrics["variant"] == "potential_W1_void_wall"]
    rar = galaxy_metrics.loc[galaxy_metrics["variant"] == "rar_reference"]
    w1_comparison = _bootstrap_comparison(p0, w1, draws=args.bootstrap_draws, seed=settings.seed)
    zeta_values = [
        float(summary["global_parameters"]["environment_shift_zeta_dex_per_sigma"])
        for summary in summaries
        if summary["variant"] == "potential_W1_void_wall"
    ]
    w1_comparison["fold_zeta_values"] = zeta_values
    w1_comparison["all_fold_zetas_positive"] = all(value > 0.0 for value in zeta_values)
    w1_comparison["one_sided_bootstrap_p_improvement"] = 1.0 - float(
        w1_comparison["bootstrap_probability_candidate_improves_chi2"]
    )
    p0_vs_rar = _bootstrap_comparison(rar, p0, draws=args.bootstrap_draws, seed=settings.seed + 1)

    report = {
        "status": "completed strict wall-balanced galaxy-level cross-validation",
        "folds": args.folds,
        "steps_per_fitted_model": args.steps,
        "bootstrap_draws": args.bootstrap_draws,
        "device": str(device),
        "design": (
            "Folds are balanced on the frozen external void-wall score. Global parameters train "
            "on all radii of non-held-out galaxies; no held-out velocity enters optimization."
        ),
        "wall_geometry_report": str(
            ROOT / "data" / "derived" / "void_wall_scores_local_report.json"
        ),
        "inside_counts_by_fold": assignments.groupby("fold")["inside_catalog_void"]
        .sum()
        .astype(int)
        .to_dict(),
        "variant_metrics": metrics_by_variant,
        "potential_W1_vs_P0": w1_comparison,
        "potential_P0_vs_RAR": p0_vs_rar,
    }
    (args.output / "cross_validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
