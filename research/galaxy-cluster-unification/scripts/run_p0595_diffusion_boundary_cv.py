#!/usr/bin/env python3
"""Extend the diffuse/RAR boundaries with whole-galaxy cross-validation."""

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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0580_conservative_return_sparc import galaxy_force_profile  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity, characteristic_acceleration  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


def galaxy_fold(name: str, folds: int) -> int:
    return int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8], 16) % int(folds)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0595_diffusion_boundary_cv_protocol.json").read_text(encoding="utf-8")
    )
    parent = json.loads((ROOT / protocol["parent_protocol"]).read_text(encoding="utf-8"))
    galaxy_cfg = parent["galaxy_test"]
    all_points = pd.read_csv(ROOT / galaxy_cfg["points"])
    points = all_points[
        (all_points.model == galaxy_cfg["model"]) & (all_points.scenario == galaxy_cfg["scenario"])
    ].copy()
    points["source_point_index"] = points.index
    outer = points[points.split == galaxy_cfg["split"]].copy().reset_index(drop=True)
    folds = int(protocol["cross_validation"]["folds"])
    outer["cv_fold"] = outer.galaxy.map(lambda value: galaxy_fold(value, folds))
    profiles = {galaxy: galaxy_force_profile(block) for galaxy, block in points.groupby("galaxy", sort=False)}
    route_cache = {}
    for galaxy, profile in profiles.items():
        for q_value in protocol["grid"]["q_R80"]:
            routed, _ = redistributed_cumulative_mass(
                profile["radius_kpc"],
                profile["mass_solar"],
                r80=profile["R80_kpc"],
                position_scale=1.0,
                width_over_r80=float(q_value),
                bins=parent["constants"]["radial_bins"],
            )
            route_cache[(galaxy, float(q_value))] = routed

    candidates = list(
        itertools.product(
            protocol["grid"]["q_R80"],
            protocol["grid"]["route_fraction"],
            protocol["grid"]["gate_power"],
        )
    )
    if len(candidates) != protocol["grid"]["candidate_count"]:
        raise RuntimeError("P0595 candidate grid changed")
    observed = outer.velocity_observed_adjusted_km_s.to_numpy(float)
    galaxies = outer.galaxy.to_numpy(str)
    unique_galaxies = np.asarray(sorted(set(galaxies)))
    galaxy_to_fold = {galaxy: galaxy_fold(galaxy, folds) for galaxy in unique_galaxies}
    predictions: dict[str, np.ndarray] = {}
    candidate_galaxy_mse: dict[str, pd.Series] = {}

    for q_value, fraction, gate_power in candidates:
        prediction = np.empty(len(outer), dtype=float)
        for galaxy, indices in outer.groupby("galaxy", sort=False).indices.items():
            profile = profiles[galaxy]
            activation = 1.0 if float(gate_power) == 0.0 else low_acceleration_activation(
                characteristic_acceleration(profile),
                a0_m_s2=parent["constants"]["a0_m_s2"],
                power=float(gate_power),
            )
            routed_fraction = float(fraction) * activation
            routed = route_cache[(galaxy, float(q_value))]
            effective_mass = (1.0 - routed_fraction) * profile["mass_solar"] + routed_fraction * routed
            radius = profile["radius_kpc"]
            g_eff = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
            full_velocity = acceleration_velocity(
                radius, rar_acceleration(g_eff, parent["constants"]["a0_m_s2"])
            )
            frame = profile["frame"]
            mask = frame.split.to_numpy(str) == galaxy_cfg["split"]
            velocity_by_source = dict(zip(frame.loc[mask, "source_point_index"], full_velocity[mask]))
            target_indices = np.asarray(indices, dtype=int)
            prediction[target_indices] = [
                velocity_by_source[source]
                for source in outer.loc[target_indices, "source_point_index"].to_numpy(int)
            ]
        candidate_id = f"q{float(q_value):g}__f{float(fraction):g}__n{float(gate_power):g}__RAR"
        predictions[candidate_id] = prediction
        candidate_galaxy_mse[candidate_id] = (
            pd.Series(np.square(prediction - observed), index=galaxies).groupby(level=0).mean()
        )

    fold_rows = []
    for candidate_id, galaxy_mse in candidate_galaxy_mse.items():
        q_value, fraction, gate_power = next(
            values
            for values in candidates
            if candidate_id == f"q{float(values[0]):g}__f{float(values[1]):g}__n{float(values[2]):g}__RAR"
        )
        for fold in range(folds):
            train_names = [name for name in unique_galaxies if galaxy_to_fold[name] != fold]
            test_names = [name for name in unique_galaxies if galaxy_to_fold[name] == fold]
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "q_R80": float(q_value),
                    "route_fraction": float(fraction),
                    "gate_power": float(gate_power),
                    "fold": fold,
                    "training_galaxies": len(train_names),
                    "test_galaxies": len(test_names),
                    "training_equal_galaxy_RMSE_km_s": float(np.sqrt(galaxy_mse.loc[train_names].mean())),
                    "test_equal_galaxy_RMSE_km_s": float(np.sqrt(galaxy_mse.loc[test_names].mean())),
                }
            )
    fold_scores = pd.DataFrame(fold_rows)
    selection_rows = []
    oof_prediction = np.empty(len(outer), dtype=float)
    for fold in range(folds):
        selected = fold_scores[fold_scores.fold == fold].sort_values(
            [
                "training_equal_galaxy_RMSE_km_s",
                "route_fraction",
                "q_R80",
                "gate_power",
                "candidate_id",
            ],
            kind="stable",
        ).iloc[0]
        mask = outer.cv_fold.to_numpy(int) == fold
        oof_prediction[mask] = predictions[str(selected.candidate_id)][mask]
        reference_test = outer.loc[mask, "velocity_RAR_same_nuisance_km_s"].to_numpy(float)
        reference_rmse = float(np.sqrt(np.mean(np.square(reference_test - observed[mask]))))
        reference_galaxy_mse = pd.Series(
            np.square(reference_test - observed[mask]), index=outer.loc[mask, "galaxy"].to_numpy(str)
        ).groupby(level=0).mean()
        reference_equal_rmse = float(np.sqrt(reference_galaxy_mse.mean()))
        selected_pooled_rmse = float(np.sqrt(np.mean(np.square(oof_prediction[mask] - observed[mask]))))
        selection_rows.append(
            {
                **selected.to_dict(),
                "test_pooled_RMSE_km_s": selected_pooled_rmse,
                "fixed_RAR_test_pooled_RMSE_km_s": reference_rmse,
                "test_pooled_improvement_fraction": 1.0 - selected_pooled_rmse / reference_rmse,
                "fixed_RAR_test_equal_galaxy_RMSE_km_s": reference_equal_rmse,
                "test_equal_galaxy_improvement_fraction": 1.0
                - float(selected.test_equal_galaxy_RMSE_km_s) / reference_equal_rmse,
            }
        )
    selections = pd.DataFrame(selection_rows)
    outer["oof_prediction_km_s"] = oof_prediction
    outer["fixed_RAR_km_s"] = outer.velocity_RAR_same_nuisance_km_s
    outer["selected_candidate_id"] = outer.cv_fold.map(
        selections.set_index("fold").candidate_id.to_dict()
    )

    galaxy_rows = []
    for galaxy, block in outer.groupby("galaxy"):
        obs = block.velocity_observed_adjusted_km_s.to_numpy(float)
        oof_mse = float(np.mean(np.square(block.oof_prediction_km_s.to_numpy(float) - obs)))
        rar_mse = float(np.mean(np.square(block.fixed_RAR_km_s.to_numpy(float) - obs)))
        profile = profiles[galaxy]
        galaxy_rows.append(
            {
                "galaxy": galaxy,
                "fold": galaxy_to_fold[galaxy],
                "oof_RMSE_km_s": np.sqrt(oof_mse),
                "fixed_RAR_RMSE_km_s": np.sqrt(rar_mse),
                "delta_MSE_km_s2": rar_mse - oof_mse,
                "improvement_fraction": 1.0 - np.sqrt(oof_mse) / np.sqrt(rar_mse),
                "concentration_R50_over_R80": profile["concentration_R50_over_R80"],
                "R80_kpc": profile["R80_kpc"],
                "g_R80_m_s2": characteristic_acceleration(profile),
            }
        )
    galaxy_scores = pd.DataFrame(galaxy_rows)
    morphology = pd.read_csv(ROOT / protocol["morphology"]["path"])
    morphology = morphology.drop(columns=["fold"], errors="ignore").drop_duplicates("galaxy")
    galaxy_scores = galaxy_scores.merge(morphology, on="galaxy", how="left", validate="one_to_one")
    galaxy_scores["log10_baryonic_mass_solar"] = np.log10(galaxy_scores.baryonic_mass_solar)
    galaxy_scores["log10_effective_surface_brightness"] = np.log10(
        galaxy_scores.effective_surface_brightness
    )
    galaxy_scores["log10_R80_kpc"] = np.log10(galaxy_scores.R80_kpc)
    galaxy_scores["log10_g_R80_over_a0"] = np.log10(
        galaxy_scores.g_R80_m_s2 / parent["constants"]["a0_m_s2"]
    )
    association_rows = []
    for feature in protocol["morphology"]["features"]:
        block = galaxy_scores[[feature, "delta_MSE_km_s2"]].dropna()
        correlation, p_value = spearmanr(block[feature], block.delta_MSE_km_s2)
        association_rows.append(
            {
                "feature": feature,
                "galaxies": len(block),
                "spearman_rho": float(correlation),
                "p_value": float(p_value),
            }
        )
    associations = pd.DataFrame(association_rows)
    associations["BH_FDR_q"] = bh_qvalues(associations.p_value.to_numpy(float))
    associations = associations.sort_values("p_value").reset_index(drop=True)

    oof_equal_rmse = float(np.sqrt(np.mean(np.square(galaxy_scores.oof_RMSE_km_s))))
    rar_equal_rmse = float(np.sqrt(np.mean(np.square(galaxy_scores.fixed_RAR_RMSE_km_s))))
    oof_pooled_rmse = float(np.sqrt(np.mean(np.square(oof_prediction - observed))))
    rar_pooled_rmse = float(
        np.sqrt(np.mean(np.square(outer.fixed_RAR_km_s.to_numpy(float) - observed)))
    )
    equal_improvement = 1.0 - oof_equal_rmse / rar_equal_rmse
    pooled_improvement = 1.0 - oof_pooled_rmse / rar_pooled_rmse
    galaxies_improved = int(np.sum(galaxy_scores.delta_MSE_km_s2 > 0.0))
    galaxies_improved_fraction = galaxies_improved / len(galaxy_scores)
    folds_improved = int(np.sum(selections.test_equal_galaxy_improvement_fraction > 0.0))
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    delta = galaxy_scores.delta_MSE_km_s2.to_numpy(float)
    sample_indices = rng.integers(0, len(delta), size=(protocol["bootstrap"]["draws"], len(delta)))
    bootstrap_probability = float(np.mean(np.mean(delta[sample_indices], axis=1) > 0.0))
    all_data = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "equal_galaxy_RMSE_km_s": float(np.sqrt(values.mean())),
            }
            for candidate_id, values in candidate_galaxy_mse.items()
        ]
    )
    global_best = all_data.sort_values(["equal_galaxy_RMSE_km_s", "candidate_id"]).iloc[0]
    selected_specs = selections[["q_R80", "route_fraction", "gate_power"]]
    boundary_flags = {
        "q_R80_at_upper_boundary": bool(selected_specs.q_R80.max() == max(protocol["grid"]["q_R80"])),
        "route_fraction_at_lower_boundary": bool(selected_specs.route_fraction.min() == min(protocol["grid"]["route_fraction"])),
        "gate_power_at_upper_boundary": bool(selected_specs.gate_power.max() == max(protocol["grid"]["gate_power"])),
    }
    gate_cfg = protocol["advance_gates"]
    gates = {
        "oof_equal_galaxy_improvement_pass": bool(
            equal_improvement >= gate_cfg["oof_equal_galaxy_improvement_fraction_min"]
        ),
        "oof_galaxies_improved_fraction_pass": bool(
            galaxies_improved_fraction >= gate_cfg["oof_galaxies_improved_fraction_min"]
        ),
        "fold_count_pass": bool(folds_improved >= gate_cfg["folds_improved_min"]),
        "bootstrap_probability_pass": bool(
            bootstrap_probability >= gate_cfg["bootstrap_probability_improvement_min"]
        ),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output / protocol["outputs"]["candidate_fold_scores"], index=False)
    selections.to_csv(output / protocol["outputs"]["fold_selections"], index=False)
    outer.to_csv(output / protocol["outputs"]["oof_predictions"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    associations.to_csv(output / protocol["outputs"]["morphology_associations"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].scatter(galaxy_scores.fixed_RAR_RMSE_km_s, galaxy_scores.oof_RMSE_km_s, alpha=0.7)
    limit = max(galaxy_scores.fixed_RAR_RMSE_km_s.max(), galaxy_scores.oof_RMSE_km_s.max()) * 1.03
    axes[0].plot([0, limit], [0, limit], ls="--", color="black")
    axes[0].set(xlabel="fixed RAR galaxy RMSE", ylabel="OOF spatial+RAR galaxy RMSE", title="whole-galaxy CV")
    display = associations.sort_values("spearman_rho")
    axes[1].barh(display.feature, display.spearman_rho)
    axes[1].axvline(0.0, color="black")
    axes[1].set(xlabel="Spearman rho with improvement in MSE", title="exploratory morphology links")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0595-DIFFUSION-BOUNDARY-CV-RESULTS-0.1.0",
        "status": "complete_whole_galaxy_cross_validation",
        "coverage": {
            "galaxies": len(galaxy_scores),
            "outer_points": len(outer),
            "candidates": len(candidates),
            "folds": folds,
        },
        "oof": {
            "equal_galaxy_RMSE_km_s": oof_equal_rmse,
            "fixed_RAR_equal_galaxy_RMSE_km_s": rar_equal_rmse,
            "equal_galaxy_improvement_fraction": equal_improvement,
            "pooled_RMSE_km_s": oof_pooled_rmse,
            "fixed_RAR_pooled_RMSE_km_s": rar_pooled_rmse,
            "pooled_improvement_fraction": pooled_improvement,
            "galaxies_improved": galaxies_improved,
            "galaxies_improved_fraction": galaxies_improved_fraction,
            "folds_improved": folds_improved,
            "bootstrap_probability_equal_galaxy_MSE_improvement": bootstrap_probability,
        },
        "fold_selections": selections.to_dict("records"),
        "global_all_data_best_candidate": global_best.to_dict(),
        "unique_fold_selected_candidates": int(selections.candidate_id.nunique()),
        "boundary_flags": boundary_flags,
        "strongest_morphology_associations": associations.head(5).to_dict("records"),
        "gates": gates,
        "all_advance_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0595 diffusion boundary cross-validation\n\n"
        f"Across five whole-galaxy folds, the diffuse/RAR family changed equal-galaxy RMSE from "
        f"{rar_equal_rmse:.3f} to {oof_equal_rmse:.3f} km/s ({100.0 * equal_improvement:+.2f}%) and "
        f"pooled RMSE by {100.0 * pooled_improvement:+.2f}%. It improved {galaxies_improved}/"
        f"{len(galaxy_scores)} galaxies and {folds_improved}/5 held folds; bootstrap improvement "
        f"probability was {100.0 * bootstrap_probability:.1f}%. The folds selected "
        f"{selections.candidate_id.nunique()} distinct candidates. Strongest exploratory morphology "
        f"association: {associations.iloc[0].feature}, rho={associations.iloc[0].spearman_rho:+.3f}, "
        f"FDR q={associations.iloc[0].BH_FDR_q:.3g}.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
