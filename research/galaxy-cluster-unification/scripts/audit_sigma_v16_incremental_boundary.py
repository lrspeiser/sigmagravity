#!/usr/bin/env python3
"""Fair nested boundary control preserving the spent internal solution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from infer_sigma_v16_spent_boundary import (
    boundary_prediction,
    feature_names,
    sample_cluster,
)

from voidscreen.sigma_boundary_inference import shear_alignment_and_power_closed
from voidscreen.sigma_covariant_feature_inference import (
    EquivariantDataset,
    fit_equivariant_ridge_features,
    predict_residual,
    score_prediction,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def with_added_base(
    dataset: EquivariantDataset,
    addition: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> EquivariantDataset:
    return EquivariantDataset(
        name=dataset.name,
        mask=dataset.mask,
        base=tuple(base + extra for base, extra in zip(dataset.base, addition, strict=True)),
        target=dataset.target,
        features=dataset.features,
    )


def symmetric_error(scores: list[dict[str, float]]) -> float:
    return float(np.sqrt(np.mean([row["full_field_NRMSE"] ** 2 for row in scores])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit incremental outer-baryon transfer with the internal fit preserved."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v16d_incremental_boundary_control.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v16d_incremental_boundary_control",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "frozen before fitting the nested boundary increments":
        raise RuntimeError("the v16D nested-control protocol is not frozen")
    base_path = ROOT / config["base_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    sampled = [sample_cluster(cluster, base) for cluster in base["sample"]["clusters"]]
    datasets = [row[0] for row in sampled]
    primary_padding = int(base["boundary_decomposition"]["primary_fourier_padding_factor"])
    decompositions = [row[2][primary_padding] for row in sampled]
    names = feature_names(base)
    internal_names = names["internal_only"]
    internal_alpha = float(config["internal_ridge_alpha"])
    boundary_alpha_grid = [float(value) for value in config["boundary_ridge_alpha_grid"]]

    internal_direction_rows = []
    internal_fits = []
    adjusted_by_direction = []
    for train_index, test_index in ((0, 1), (1, 0)):
        internal_fit = fit_equivariant_ridge_features(
            [datasets[train_index]],
            feature_names=internal_names,
            alpha=internal_alpha,
        )
        adjusted = [
            with_added_base(dataset, predict_residual(dataset, internal_fit.coefficients))
            for dataset in datasets
        ]
        internal_direction_rows.append(
            {
                "train_cluster": datasets[train_index].name,
                "test_cluster": datasets[test_index].name,
                **score_prediction(adjusted[test_index], {}),
            }
        )
        internal_fits.append(internal_fit)
        adjusted_by_direction.append(adjusted)
    internal_symmetric = symmetric_error(internal_direction_rows)

    family_results = {}
    for family in config["boundary_families"]:
        boundary_names = [name for name in names[family] if name.startswith("boundary_")]
        alpha_rows = []
        for alpha in boundary_alpha_grid:
            directions = []
            for direction_index, (train_index, test_index) in enumerate(((0, 1), (1, 0))):
                adjusted = adjusted_by_direction[direction_index]
                boundary_fit = fit_equivariant_ridge_features(
                    [adjusted[train_index]],
                    feature_names=boundary_names,
                    alpha=alpha,
                )
                full_score = score_prediction(
                    adjusted[test_index],
                    boundary_fit.coefficients,
                )
                boundary_1, boundary_2 = boundary_prediction(
                    adjusted[test_index],
                    boundary_fit.coefficients,
                )
                target_boundary = decompositions[test_index]
                boundary_score = shear_alignment_and_power_closed(
                    boundary_1,
                    boundary_2,
                    target_boundary.boundary_shear_1,
                    target_boundary.boundary_shear_2,
                    adjusted[test_index].mask,
                )
                directions.append(
                    {
                        "train_cluster": datasets[train_index].name,
                        "test_cluster": datasets[test_index].name,
                        **full_score,
                        **boundary_score,
                    }
                )
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "symmetric_cross_cluster_full_field_NRMSE": symmetric_error(directions),
                    "directions": directions,
                }
            )
        selected = min(
            alpha_rows,
            key=lambda row: (
                row["symmetric_cross_cluster_full_field_NRMSE"],
                row["alpha"],
            ),
        )
        zero_limit = next(row for row in alpha_rows if row["alpha"] == max(boundary_alpha_grid))
        family_results[family] = {
            "boundary_feature_names": boundary_names,
            "selected_alpha": selected["alpha"],
            "symmetric_cross_cluster_full_field_NRMSE": selected[
                "symmetric_cross_cluster_full_field_NRMSE"
            ],
            "cross_cluster_scores": selected["directions"],
            "alpha_sweep": alpha_rows,
            "zero_boundary_limit_NRMSE": zero_limit["symmetric_cross_cluster_full_field_NRMSE"],
            "zero_boundary_limit_difference_from_internal": float(
                abs(zero_limit["symmetric_cross_cluster_full_field_NRMSE"] - internal_symmetric)
            ),
        }

    best_family = min(
        family_results,
        key=lambda family: family_results[family]["symmetric_cross_cluster_full_field_NRMSE"],
    )
    best = family_results[best_family]
    relative_improvement = float(
        (internal_symmetric - best["symmetric_cross_cluster_full_field_NRMSE"]) / internal_symmetric
    )
    gates = config["diagnostic_gates"]
    zero_limit_gate = bool(
        all(
            result["zero_boundary_limit_difference_from_internal"]
            <= gates["maximum_zero_boundary_limit_difference"]
            for result in family_results.values()
        )
    )
    improvement_gate = bool(
        relative_improvement >= gates["minimum_relative_improvement_over_preserved_internal"]
    )
    absolute_gate = bool(
        best["symmetric_cross_cluster_full_field_NRMSE"]
        <= gates["maximum_symmetric_cross_cluster_full_field_NRMSE"]
    )
    alignment_gate = bool(
        all(
            row["boundary_shear_alignment_cosine"]
            >= gates["minimum_boundary_shear_alignment_cosine"]
            for row in best["cross_cluster_scores"]
        )
    )
    power_gate = bool(
        all(
            row["boundary_shear_power_closed"] >= gates["minimum_boundary_shear_power_closed"]
            for row in best["cross_cluster_scores"]
        )
    )
    advance = bool(
        zero_limit_gate and improvement_gate and absolute_gate and alignment_gate and power_gate
    )
    decision = (
        "measured outer baryons add a transferable boundary field; derive a covariant propagator"
        if advance
        else "the fair zero-limit control rejects the measured static outer-baryon boundary increment"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for family, result in family_results.items():
        axes[0].semilogx(
            [max(row["alpha"], 1.0e-8) for row in result["alpha_sweep"]],
            [row["symmetric_cross_cluster_full_field_NRMSE"] for row in result["alpha_sweep"]],
            marker="o",
            label=family,
        )
    axes[0].axhline(internal_symmetric, color="black", linestyle="--", label="preserved internal")
    axes[0].set(
        xlabel="boundary ridge alpha",
        ylabel="cross-cluster NRMSE",
        title="Nested zero-boundary limit",
    )
    axes[0].legend(fontsize=8)
    directions = best["cross_cluster_scores"]
    positions = np.arange(len(directions))
    axes[1].bar(
        positions - 0.18,
        [row["boundary_shear_alignment_cosine"] for row in directions],
        width=0.36,
        label="alignment",
    )
    axes[1].bar(
        positions + 0.18,
        [row["boundary_shear_power_closed"] for row in directions],
        width=0.36,
        label="power closed",
    )
    axes[1].set_xticks(
        positions,
        [f"{row['train_cluster']} to {row['test_cluster']}" for row in directions],
        rotation=15,
    )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(title=f"{best_family} boundary increment", ylabel="dimensionless")
    axes[1].legend(fontsize=8)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "incremental_boundary_control.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed Sigma v16D incremental boundary control",
        "protocol_version": config["protocol_version"],
        "sample_is_spent": True,
        "observational_validation_claim": False,
        "per_cluster_gravity_parameters": 0,
        "per_cluster_shear_or_orientation_parameters": 0,
        "input_hashes": {
            "config": sha256(args.config),
            "base_config": sha256(base_path),
        },
        "internal_ridge_alpha": internal_alpha,
        "preserved_internal_cross_cluster_scores": internal_direction_rows,
        "preserved_internal_symmetric_NRMSE": internal_symmetric,
        "family_results": family_results,
        "best_boundary_family": best_family,
        "relative_boundary_improvement": relative_improvement,
        "gate_results": {
            "zero_boundary_limit_reproduces_internal": zero_limit_gate,
            "material_improvement": improvement_gate,
            "absolute_source_sufficiency": absolute_gate,
            "boundary_alignment_both_directions": alignment_gate,
            "boundary_power_both_directions": power_gate,
            "advance": advance,
        },
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(f"internal={internal_symmetric:.6f}")
    for family, result in family_results.items():
        print(
            f"{family}: alpha={result['selected_alpha']:g}, "
            f"cross={result['symmetric_cross_cluster_full_field_NRMSE']:.6f}, "
            f"zero_limit_delta={result['zero_boundary_limit_difference_from_internal']:.3e}"
        )
    print(decision)


if __name__ == "__main__":
    main()
