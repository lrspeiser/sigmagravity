#!/usr/bin/env python3
"""Galaxy-level formula holdout for P0593 RAR-completed redistribution."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0580_conservative_return_sparc import galaxy_force_profile, score  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity, characteristic_acceleration  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


def object_remainder(name: str) -> int:
    return int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16) % 4


def partition(name: str) -> str:
    return "formula_holdout" if object_remainder(name) == 0 else "discovery"


def per_galaxy_rmse(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    work = frame[["galaxy", "velocity_observed_adjusted_km_s", prediction_column]].copy()
    work["squared_residual"] = np.square(
        work[prediction_column] - work.velocity_observed_adjusted_km_s
    )
    return (
        work.groupby("galaxy", as_index=False)
        .squared_residual.mean()
        .assign(RMSE_km_s=lambda x: np.sqrt(x.squared_residual))
        .drop(columns="squared_residual")
    )


def partition_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    metrics = score(frame, frame[prediction_column].to_numpy(float))
    return {
        "pooled_RMSE_km_s": metrics["outer_RMSE_km_s"],
        "equal_galaxy_RMSE_km_s": metrics["outer_equal_galaxy_RMSE_km_s"],
        "mean_residual_km_s": metrics["outer_mean_residual_km_s"],
    }


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0593b_diffusion_formula_holdout_protocol.json").read_text(encoding="utf-8")
    )
    parent = json.loads((ROOT / protocol["parent_protocol"]).read_text(encoding="utf-8"))
    galaxy_cfg = parent["galaxy_test"]
    all_points = pd.read_csv(ROOT / galaxy_cfg["points"])
    points = all_points[
        (all_points.model == galaxy_cfg["model"]) & (all_points.scenario == galaxy_cfg["scenario"])
    ].copy()
    points["formula_partition"] = points.galaxy.map(partition)
    outer = points[points.split == galaxy_cfg["split"]].copy()
    counts = outer.groupby("formula_partition").galaxy.nunique().to_dict()
    expected = protocol["partition"]
    if counts != {
        "discovery": expected["expected_discovery_galaxies"],
        "formula_holdout": expected["expected_holdout_galaxies"],
    }:
        raise RuntimeError(f"P0593B whole-galaxy partition changed: {counts}")

    profiles = {galaxy: galaxy_force_profile(block) for galaxy, block in points.groupby("galaxy", sort=False)}
    strengths = {
        "diffuse": parent["factorial"]["diffuse_strength"],
        "contract": parent["factorial"]["contract_strength"],
    }
    route_cache: dict[tuple[str, str, float], np.ndarray] = {}
    for galaxy, profile in profiles.items():
        for geometry, values in strengths.items():
            for strength in values:
                routed, _ = redistributed_cumulative_mass(
                    profile["radius_kpc"],
                    profile["mass_solar"],
                    r80=profile["R80_kpc"],
                    position_scale=1.0 if geometry == "diffuse" else 1.0 - float(strength),
                    width_over_r80=(
                        float(strength)
                        if geometry == "diffuse"
                        else parent["constants"]["contract_width_over_R80"]
                    ),
                    bins=parent["constants"]["radial_bins"],
                )
                route_cache[(galaxy, geometry, float(strength))] = routed

    candidates = []
    for geometry in parent["factorial"]["route_geometry"]:
        for strength, fraction, gate_power in itertools.product(
            strengths[geometry],
            parent["factorial"]["route_fraction"],
            parent["factorial"]["gate_power"],
        ):
            candidates.append((geometry, float(strength), float(fraction), float(gate_power)))
    if len(candidates) != protocol["candidate_filter"]["candidate_count"]:
        raise RuntimeError("P0593B candidate count changed")

    score_rows: list[dict] = []
    predictions: dict[str, pd.DataFrame] = {}
    for geometry, strength, fraction, gate_power in candidates:
        parts = []
        for galaxy, profile in profiles.items():
            activation = 1.0 if gate_power == 0.0 else low_acceleration_activation(
                characteristic_acceleration(profile),
                a0_m_s2=parent["constants"]["a0_m_s2"],
                power=gate_power,
            )
            routed_fraction = fraction * activation
            routed = route_cache[(galaxy, geometry, strength)]
            effective_mass = (1.0 - routed_fraction) * profile["mass_solar"] + routed_fraction * routed
            radius = profile["radius_kpc"]
            g_eff = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
            velocity = acceleration_velocity(
                radius, rar_acceleration(g_eff, parent["constants"]["a0_m_s2"])
            )
            frame = profile["frame"].copy()
            frame["prediction_km_s"] = velocity
            parts.append(frame[frame.split == galaxy_cfg["split"]].copy())
        prediction = pd.concat(parts, ignore_index=True)
        prediction["formula_partition"] = prediction.galaxy.map(partition)
        candidate_id = f"{geometry}__s{strength:g}__f{fraction:g}__n{gate_power:g}__RAR"
        row = {
            "candidate_id": candidate_id,
            "route_geometry": geometry,
            "strength": strength,
            "route_fraction": fraction,
            "gate_power": gate_power,
        }
        for split in ("discovery", "formula_holdout"):
            metrics = partition_metrics(
                prediction[prediction.formula_partition == split], "prediction_km_s"
            )
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
        score_rows.append(row)
        predictions[candidate_id] = prediction

    scores = pd.DataFrame(score_rows)
    selection = protocol["selection"]
    scores = scores.sort_values(
        [selection["metric"], *selection["tie_break"]], kind="stable"
    ).reset_index(drop=True)
    selected = scores.iloc[0]
    selected_id = str(selected.candidate_id)
    selected_predictions = predictions[selected_id].copy()
    selected_predictions["fixed_RAR_km_s"] = selected_predictions.velocity_RAR_same_nuisance_km_s

    references = {}
    for split in ("discovery", "formula_holdout"):
        block = selected_predictions[selected_predictions.formula_partition == split]
        references[split] = partition_metrics(block, "fixed_RAR_km_s")
    holdout = selected_predictions[selected_predictions.formula_partition == "formula_holdout"].copy()
    selected_galaxy = per_galaxy_rmse(holdout, "prediction_km_s").rename(
        columns={"RMSE_km_s": "selected_RMSE_km_s"}
    )
    reference_galaxy = per_galaxy_rmse(holdout, "fixed_RAR_km_s").rename(
        columns={"RMSE_km_s": "fixed_RAR_RMSE_km_s"}
    )
    galaxy_scores = selected_galaxy.merge(reference_galaxy, on="galaxy", validate="one_to_one")
    galaxy_scores["improvement_fraction"] = 1.0 - (
        galaxy_scores.selected_RMSE_km_s / galaxy_scores.fixed_RAR_RMSE_km_s
    )
    selected_holdout = {
        key.removeprefix("formula_holdout_"): float(value)
        for key, value in selected.items()
        if key.startswith("formula_holdout_")
    }
    selected_discovery = {
        key.removeprefix("discovery_"): float(value)
        for key, value in selected.items()
        if key.startswith("discovery_")
    }
    equal_improvement = 1.0 - (
        selected_holdout["equal_galaxy_RMSE_km_s"]
        / references["formula_holdout"]["equal_galaxy_RMSE_km_s"]
    )
    pooled_improvement = 1.0 - (
        selected_holdout["pooled_RMSE_km_s"] / references["formula_holdout"]["pooled_RMSE_km_s"]
    )

    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    delta_mse = np.square(galaxy_scores.fixed_RAR_RMSE_km_s.to_numpy(float)) - np.square(
        galaxy_scores.selected_RMSE_km_s.to_numpy(float)
    )
    draws = int(protocol["bootstrap"]["draws"])
    samples = rng.integers(0, len(delta_mse), size=(draws, len(delta_mse)))
    boot_improvement = np.mean(delta_mse[samples], axis=1) > 0.0
    bootstrap_probability = float(np.mean(boot_improvement))
    galaxies_improved = int(np.sum(galaxy_scores.improvement_fraction > 0.0))
    galaxies_improved_fraction = galaxies_improved / len(galaxy_scores)
    gates_cfg = protocol["advance_gates"]
    gates = {
        "holdout_equal_galaxy_improvement_pass": bool(
            equal_improvement >= gates_cfg["holdout_equal_galaxy_improvement_fraction_min"]
        ),
        "holdout_pooled_RMSE_not_worse_pass": bool(pooled_improvement >= 0.0),
        "holdout_galaxies_improved_fraction_pass": bool(
            galaxies_improved_fraction >= gates_cfg["holdout_galaxies_improved_fraction_min"]
        ),
        "bootstrap_probability_improvement_pass": bool(
            bootstrap_probability >= gates_cfg["bootstrap_probability_improvement_min"]
        ),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    selected_predictions.to_csv(output / protocol["outputs"]["selected_predictions"], index=False)
    galaxy_scores.sort_values("improvement_fraction", ascending=False).to_csv(
        output / protocol["outputs"]["galaxy_scores"], index=False
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].bar(
        ["fixed RAR", "selected"],
        [references["formula_holdout"]["equal_galaxy_RMSE_km_s"], selected_holdout["equal_galaxy_RMSE_km_s"]],
    )
    axes[0].set(ylabel="holdout equal-galaxy RMSE (km/s)", title="formula holdout")
    axes[1].hist(100.0 * galaxy_scores.improvement_fraction, bins=14, color="tab:blue")
    axes[1].axvline(0.0, color="black", ls="--")
    axes[1].set(xlabel="per-galaxy RMSE improvement (%)", ylabel="galaxies", title="40 unseen galaxies")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    all_data_best = pd.read_csv(
        ROOT / parent["outputs"]["directory"] / parent["outputs"]["candidate_scores"]
    ).query("scalar_completion == 'RAR'").sort_values("outer_RMSE_km_s").iloc[0]
    report = {
        "report_version": "P0593B-DIFFUSION-FORMULA-HOLDOUT-RESULTS-0.1.0",
        "status": "complete_formula_holdout",
        "coverage": {
            "candidates": len(scores),
            "discovery_galaxies": counts["discovery"],
            "holdout_galaxies": counts["formula_holdout"],
            "holdout_points": len(holdout),
        },
        "selection_metric": selection["metric"],
        "selected_candidate": {
            "candidate_id": selected_id,
            "route_geometry": selected.route_geometry,
            "strength": float(selected.strength),
            "route_fraction": float(selected.route_fraction),
            "gate_power": float(selected.gate_power),
        },
        "all_data_screen_best_candidate": str(all_data_best.candidate_id),
        "selected_discovery": selected_discovery,
        "selected_holdout": selected_holdout,
        "fixed_RAR_reference": references,
        "holdout_equal_galaxy_improvement_fraction": equal_improvement,
        "holdout_pooled_improvement_fraction": pooled_improvement,
        "holdout_galaxies_improved": galaxies_improved,
        "holdout_galaxies_improved_fraction": galaxies_improved_fraction,
        "bootstrap_probability_equal_galaxy_MSE_improvement": bootstrap_probability,
        "gates": gates,
        "all_advance_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0593B diffusion formula holdout\n\n"
        f"The candidate selected only on {counts['discovery']} discovery galaxies was `{selected_id}`. "
        f"On {counts['formula_holdout']} unseen galaxies it changed equal-galaxy RMSE by "
        f"{100.0 * equal_improvement:+.2f}% and pooled RMSE by {100.0 * pooled_improvement:+.2f}% "
        f"versus fixed RAR; {galaxies_improved}/{len(galaxy_scores)} galaxies improved. "
        f"The galaxy bootstrap probability of lower equal-galaxy MSE was {100.0 * bootstrap_probability:.1f}%. "
        f"Advance gates: {'PASS' if all(gates.values()) else 'FAIL'}.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
