#!/usr/bin/env python3
"""Cross-validate a dimensionless R50/R80 gate on diffuse redistribution."""

from __future__ import annotations

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

from run_p0580_conservative_return_sparc import galaxy_force_profile  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity, characteristic_acceleration  # noqa: E402
from run_p0595_diffusion_boundary_cv import galaxy_fold  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    radial_shape_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


def equal_galaxy_mse(residual: np.ndarray, galaxies: np.ndarray) -> pd.Series:
    return pd.Series(np.square(residual), index=galaxies).groupby(level=0).mean()


def json_records(frame: pd.DataFrame) -> list[dict]:
    """Serialize nullable table values as JSON null instead of nonstandard NaN."""
    return json.loads(frame.to_json(orient="records"))


def select_oof(
    fold_scores: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    outer: pd.DataFrame,
    observed: np.ndarray,
    *,
    family: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    eligible = fold_scores if family == "shape_gate" else fold_scores[fold_scores.shape_gate == "none"]
    oof = np.empty(len(outer), dtype=float)
    selections = []
    for fold in sorted(outer.cv_fold.unique()):
        selected = eligible[eligible.fold == fold].sort_values(
            [
                "training_equal_galaxy_RMSE_km_s",
                "route_fraction_max",
                "q_R80",
                "acceleration_gate_power",
                "shape_gate",
                "candidate_id",
            ],
            kind="stable",
        ).iloc[0]
        mask = outer.cv_fold.to_numpy(int) == fold
        oof[mask] = predictions[str(selected.candidate_id)][mask]
        ref = outer.loc[mask, "velocity_RAR_same_nuisance_km_s"].to_numpy(float)
        names = outer.loc[mask, "galaxy"].to_numpy(str)
        ref_equal = float(np.sqrt(equal_galaxy_mse(ref - observed[mask], names).mean()))
        selections.append(
            {
                **selected.to_dict(),
                "family": family,
                "fixed_RAR_test_equal_galaxy_RMSE_km_s": ref_equal,
                "test_equal_galaxy_improvement_fraction": 1.0
                - float(selected.test_equal_galaxy_RMSE_km_s) / ref_equal,
            }
        )
    return pd.DataFrame(selections), oof


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0596_radial_shape_gate_cv_protocol.json").read_text(encoding="utf-8")
    )
    parent = json.loads((ROOT / protocol["parent_protocol"]).read_text(encoding="utf-8"))
    galaxy_cfg = parent["galaxy_test"]
    raw = pd.read_csv(ROOT / galaxy_cfg["points"])
    points = raw[(raw.model == galaxy_cfg["model"]) & (raw.scenario == galaxy_cfg["scenario"])].copy()
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

    specs = []
    for q_value in protocol["grid"]["q_R80"]:
        for fraction in protocol["grid"]["route_fraction_max"]:
            for accel_power in protocol["grid"]["acceleration_gate_power"]:
                for shape in protocol["grid"]["shape_gate"]:
                    candidate_id = (
                        f"q{float(q_value):g}__f{float(fraction):g}__n{float(accel_power):g}"
                        f"__h{shape['id']}__RAR"
                    )
                    specs.append(
                        {
                            "candidate_id": candidate_id,
                            "q_R80": float(q_value),
                            "route_fraction_max": float(fraction),
                            "acceleration_gate_power": float(accel_power),
                            "shape_gate": shape["id"],
                            "shape_midpoint": shape["midpoint"],
                            "shape_width": shape["width"],
                        }
                    )
    if len(specs) != protocol["grid"]["candidate_count"]:
        raise RuntimeError("P0596 candidate grid changed")
    observed = outer.velocity_observed_adjusted_km_s.to_numpy(float)
    point_galaxies = outer.galaxy.to_numpy(str)
    galaxy_names = np.asarray(sorted(profiles))
    galaxy_to_fold = {name: galaxy_fold(name, folds) for name in galaxy_names}
    predictions = {}
    galaxy_mse_by_candidate = {}
    fold_rows = []
    overall_rows = []

    for spec in specs:
        prediction = np.empty(len(outer), dtype=float)
        for galaxy, indices in outer.groupby("galaxy", sort=False).indices.items():
            profile = profiles[galaxy]
            accel_activation = 1.0 if spec["acceleration_gate_power"] == 0.0 else low_acceleration_activation(
                characteristic_acceleration(profile),
                a0_m_s2=parent["constants"]["a0_m_s2"],
                power=spec["acceleration_gate_power"],
            )
            shape_activation = 1.0 if spec["shape_gate"] == "none" else radial_shape_activation(
                profile["concentration_R50_over_R80"],
                midpoint=spec["shape_midpoint"],
                width=spec["shape_width"],
            )
            fraction = spec["route_fraction_max"] * accel_activation * shape_activation
            routed = route_cache[(galaxy, spec["q_R80"])]
            effective_mass = (1.0 - fraction) * profile["mass_solar"] + fraction * routed
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
        candidate_id = spec["candidate_id"]
        predictions[candidate_id] = prediction
        galaxy_mse = equal_galaxy_mse(prediction - observed, point_galaxies)
        galaxy_mse_by_candidate[candidate_id] = galaxy_mse
        overall_rows.append({**spec, "all_equal_galaxy_RMSE_km_s": float(np.sqrt(galaxy_mse.mean()))})
        for fold in range(folds):
            train = [name for name in galaxy_names if galaxy_to_fold[name] != fold]
            test = [name for name in galaxy_names if galaxy_to_fold[name] == fold]
            fold_rows.append(
                {
                    **spec,
                    "fold": fold,
                    "training_galaxies": len(train),
                    "test_galaxies": len(test),
                    "training_equal_galaxy_RMSE_km_s": float(np.sqrt(galaxy_mse.loc[train].mean())),
                    "test_equal_galaxy_RMSE_km_s": float(np.sqrt(galaxy_mse.loc[test].mean())),
                }
            )
    fold_scores = pd.DataFrame(fold_rows)
    overall = pd.DataFrame(overall_rows)
    shape_selections, shape_oof = select_oof(
        fold_scores, predictions, outer, observed, family="shape_gate"
    )
    control_selections, control_oof = select_oof(
        fold_scores, predictions, outer, observed, family="no_shape"
    )
    selections = pd.concat([shape_selections, control_selections], ignore_index=True)
    outer["shape_gate_oof_km_s"] = shape_oof
    outer["no_shape_oof_km_s"] = control_oof
    outer["fixed_RAR_km_s"] = outer.velocity_RAR_same_nuisance_km_s
    outer["shape_selected_candidate"] = outer.cv_fold.map(
        shape_selections.set_index("fold").candidate_id.to_dict()
    )

    galaxy_rows = []
    for galaxy, block in outer.groupby("galaxy"):
        obs = block.velocity_observed_adjusted_km_s.to_numpy(float)
        row = {"galaxy": galaxy, "cv_fold": galaxy_to_fold[galaxy]}
        for label, column in (
            ("shape_gate", "shape_gate_oof_km_s"),
            ("no_shape", "no_shape_oof_km_s"),
            ("fixed_RAR", "fixed_RAR_km_s"),
        ):
            row[f"{label}_MSE_km_s2"] = float(np.mean(np.square(block[column].to_numpy(float) - obs)))
            row[f"{label}_RMSE_km_s"] = np.sqrt(row[f"{label}_MSE_km_s2"])
        row["shape_vs_RAR_delta_MSE_km_s2"] = row["fixed_RAR_MSE_km_s2"] - row["shape_gate_MSE_km_s2"]
        row["shape_vs_no_shape_delta_MSE_km_s2"] = row["no_shape_MSE_km_s2"] - row["shape_gate_MSE_km_s2"]
        galaxy_rows.append(row)
    galaxy_scores = pd.DataFrame(galaxy_rows)

    def aggregate(label: str, prediction: np.ndarray) -> dict[str, float]:
        mse = galaxy_scores[f"{label}_MSE_km_s2"].to_numpy(float)
        return {
            "equal_galaxy_RMSE_km_s": float(np.sqrt(np.mean(mse))),
            "pooled_RMSE_km_s": float(np.sqrt(np.mean(np.square(prediction - observed)))),
        }

    metrics = {
        "shape_gate": aggregate("shape_gate", shape_oof),
        "no_shape": aggregate("no_shape", control_oof),
        "fixed_RAR": aggregate("fixed_RAR", outer.fixed_RAR_km_s.to_numpy(float)),
    }
    improvement_vs_rar = 1.0 - (
        metrics["shape_gate"]["equal_galaxy_RMSE_km_s"]
        / metrics["fixed_RAR"]["equal_galaxy_RMSE_km_s"]
    )
    improvement_vs_control = 1.0 - (
        metrics["shape_gate"]["equal_galaxy_RMSE_km_s"]
        / metrics["no_shape"]["equal_galaxy_RMSE_km_s"]
    )
    galaxies_improved = int(np.sum(galaxy_scores.shape_vs_RAR_delta_MSE_km_s2 > 0.0))
    galaxies_improved_fraction = galaxies_improved / len(galaxy_scores)
    folds_improved = int(np.sum(shape_selections.test_equal_galaxy_improvement_fraction > 0.0))
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    delta = galaxy_scores.shape_vs_RAR_delta_MSE_km_s2.to_numpy(float)
    samples = rng.integers(0, len(delta), size=(protocol["bootstrap"]["draws"], len(delta)))
    bootstrap_probability = float(np.mean(np.mean(delta[samples], axis=1) > 0.0))

    impact_rows = []
    for parameter in ("q_R80", "route_fraction_max", "acceleration_gate_power", "shape_gate"):
        grouped = overall.groupby(parameter).all_equal_galaxy_RMSE_km_s.median().sort_values()
        impact_rows.append(
            {
                "parameter": parameter,
                "best_level": str(grouped.index[0]),
                "worst_level": str(grouped.index[-1]),
                "median_RMSE_span_km_s": float(grouped.iloc[-1] - grouped.iloc[0]),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values("median_RMSE_span_km_s", ascending=False)
    global_best = overall.sort_values(["all_equal_galaxy_RMSE_km_s", "candidate_id"]).iloc[0]
    cfg = protocol["advance_gates"]
    gates = {
        "improvement_vs_fixed_RAR_pass": bool(
            improvement_vs_rar >= cfg["oof_equal_galaxy_improvement_vs_fixed_RAR_fraction_min"]
        ),
        "improvement_vs_no_shape_pass": bool(
            improvement_vs_control >= cfg["oof_equal_galaxy_improvement_vs_no_shape_family_fraction_min"]
        ),
        "fold_count_pass": bool(folds_improved >= cfg["folds_improved_vs_fixed_RAR_min"]),
        "galaxy_fraction_pass": bool(
            galaxies_improved_fraction >= cfg["galaxies_improved_vs_fixed_RAR_fraction_min"]
        ),
        "bootstrap_probability_pass": bool(
            bootstrap_probability >= cfg["bootstrap_probability_vs_fixed_RAR_min"]
        ),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output / protocol["outputs"]["candidate_fold_scores"], index=False)
    selections.to_csv(output / protocol["outputs"]["fold_selections"], index=False)
    outer.to_csv(output / protocol["outputs"]["oof_predictions"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].bar(
        ["fixed RAR", "no shape", "shape gate"],
        [metrics[key]["equal_galaxy_RMSE_km_s"] for key in ("fixed_RAR", "no_shape", "shape_gate")],
    )
    axes[0].set(ylabel="OOF equal-galaxy RMSE (km/s)", title="radial-shape gate")
    display = impacts.sort_values("median_RMSE_span_km_s")
    axes[1].barh(display.parameter, display.median_RMSE_span_km_s)
    axes[1].set(xlabel="median RMSE span (km/s)", title="parameter impact")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0596-RADIAL-SHAPE-GATE-CV-RESULTS-0.1.0",
        "status": "complete_whole_galaxy_cross_validation",
        "coverage": {"galaxies": len(galaxy_scores), "outer_points": len(outer), "candidates": len(specs), "folds": folds},
        "oof_metrics": metrics,
        "equal_galaxy_improvement_fraction_vs_fixed_RAR": improvement_vs_rar,
        "equal_galaxy_improvement_fraction_vs_no_shape_family": improvement_vs_control,
        "galaxies_improved_vs_fixed_RAR": galaxies_improved,
        "galaxies_improved_vs_fixed_RAR_fraction": galaxies_improved_fraction,
        "folds_improved_vs_fixed_RAR": folds_improved,
        "bootstrap_probability_equal_galaxy_MSE_improvement_vs_fixed_RAR": bootstrap_probability,
        "shape_fold_selections": json_records(shape_selections),
        "no_shape_fold_selections": json_records(control_selections),
        "unique_shape_selected_candidates": int(shape_selections.candidate_id.nunique()),
        "global_all_data_best_candidate": global_best.to_dict(),
        "parameter_impacts": impacts.to_dict("records"),
        "gates": gates,
        "all_advance_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0596 radial-shape gate cross-validation\n\n"
        f"The shape-gated family changed out-of-fold equal-galaxy RMSE by "
        f"{100.0 * improvement_vs_rar:+.2f}% versus fixed RAR and {100.0 * improvement_vs_control:+.2f}% "
        f"versus the same diffuse family without a shape gate. It improved {galaxies_improved}/"
        f"{len(galaxy_scores)} galaxies and {folds_improved}/5 held folds; bootstrap probability versus "
        f"fixed RAR was {100.0 * bootstrap_probability:.1f}%. The folds selected "
        f"{shape_selections.candidate_id.nunique()} distinct shape candidates.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
