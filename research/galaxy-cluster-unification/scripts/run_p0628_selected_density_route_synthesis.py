#!/usr/bin/env python3
"""P0628: out-of-fold and cross-domain synthesis of the selected candidate."""

from __future__ import annotations

import json
import math
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

from run_p0623_density_path_survival import predict_velocity, score_arrays  # noqa: E402
from run_p0625_bounded_porosity_survival import apply_record, prepare_frame  # noqa: E402
from run_p0626_compact_scalar_angular_route import fit_atomic  # noqa: E402


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def strict_json(value):
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def score_prediction(frame, name: str, prediction) -> dict:
    return {"model": name, **score_arrays(frame, np.asarray(prediction, dtype=float))}


def build_oof(frame, lookup, beta: float):
    blocks = []
    for fold in range(5):
        train = frame[frame.galaxy_fold.ne(fold)]
        test = frame[frame.galaxy_fold.eq(fold)].copy()
        records = fit_atomic(train, lookup)
        q_constant = apply_record(records["constant"], np.ones(len(test)))
        q_potential = apply_record(
            records["inverse_hill0_m1__potential_depth"], test.potential_depth
        )
        surface = test.force_equivalent_mass_solar.to_numpy(float) / (
            np.pi * np.square(test.force_equivalent_r80_kpc.to_numpy(float))
        )
        q_surface = apply_record(
            records["inverse_hillfloor_m2__mean_surface_R80"], surface
        )
        q_or = np.maximum(q_potential, q_surface)
        q_selected = q_constant + beta * (q_or - q_constant)
        test["q_constant"] = q_constant
        test["q_potential"] = q_potential
        test["q_surface"] = q_surface
        test["q_OR"] = q_or
        test["q_selected"] = q_selected
        test["OR_component"] = np.where(
            q_potential >= q_surface, "potential", "surface_density"
        )
        test["velocity_constant_km_s"] = predict_velocity(test, q_constant)
        test["velocity_selected_km_s"] = predict_velocity(test, q_selected)
        test["velocity_RAR_km_s"] = test.velocity_RAR_same_nuisance_km_s
        test["prediction_fold"] = fold
        blocks.append(test)
    return pd.concat(blocks, ignore_index=True)


def overall_scores(oof, p0623):
    rows = [
        score_prediction(oof, "P0628_selected_OOF", oof.velocity_selected_km_s),
        score_prediction(oof, "fold_refit_constant_OOF", oof.velocity_constant_km_s),
        score_prediction(oof, "fixed_RAR_same_nuisance", oof.velocity_RAR_km_s),
    ]
    raw = pd.read_csv(ROOT / p0623["inputs"]["SPARC_points"])
    nfw = raw[
        raw.model.str.lower().eq("nfw")
        & raw.scenario.eq("invariant")
        & raw.split.eq("outer_holdout")
    ].copy()
    if len(nfw) and nfw.galaxy.nunique() == oof.galaxy.nunique():
        rows.append(score_prediction(nfw, "weak_prior_NFW_inner_refit", nfw.velocity_predicted_km_s))
    mond = raw[
        raw.model.str.lower().eq("simple_mond")
        & raw.scenario.eq("invariant")
        & raw.split.eq("outer_holdout")
    ].copy()
    if len(mond) and mond.galaxy.nunique() == oof.galaxy.nunique():
        rows.append(score_prediction(mond, "simple_MOND_inner_refit", mond.velocity_predicted_km_s))
    result = pd.DataFrame(rows)
    constant = float(
        result.loc[result.model.eq("fold_refit_constant_OOF"), "equal_galaxy_RMSE_km_s"].iloc[0]
    )
    result["improvement_vs_constant_fraction"] = 1.0 - result.equal_galaxy_RMSE_km_s / constant
    return result


def per_galaxy_scores(oof):
    local = oof.copy()
    for model, column in {
        "selected": "velocity_selected_km_s",
        "constant": "velocity_constant_km_s",
        "RAR": "velocity_RAR_km_s",
    }.items():
        local[f"{model}_residual"] = local[column] - local.velocity_observed_adjusted_km_s
        local[f"{model}_squared"] = np.square(local[f"{model}_residual"])
    result = local.groupby("galaxy", sort=True).agg(
        fold=("prediction_fold", "first"),
        points=("galaxy", "size"),
        baryonic_mass_solar=("force_equivalent_mass_solar", "first"),
        gas_fraction=("gas_fraction", "first"),
        stellar_bulge_fraction=("stellar_bulge_fraction", "first"),
        hubble_type=("hubble_type", "first"),
        mean_surface_R80=("mean_surface_R80", "first"),
        potential_depth_median=("potential_depth", "median"),
        selected_MSE=("selected_squared", "mean"),
        constant_MSE=("constant_squared", "mean"),
        RAR_MSE=("RAR_squared", "mean"),
        selected_mean_residual=("selected_residual", "mean"),
        constant_mean_residual=("constant_residual", "mean"),
        RAR_mean_residual=("RAR_residual", "mean"),
        selected_q_mean=("q_selected", "mean"),
        potential_activation_fraction=("OR_component", lambda x: float(np.mean(x == "potential"))),
    ).reset_index()
    for model in ("selected", "constant", "RAR"):
        result[f"{model}_RMSE_km_s"] = np.sqrt(result[f"{model}_MSE"])
    result["mass_regime"] = np.select(
        [result.baryonic_mass_solar < 1.0e9, result.baryonic_mass_solar > 1.0e10],
        ["dwarf_below_1e9", "giant_above_1e10"],
        default="intermediate_1e9_to_1e10",
    )
    result["gas_regime"] = np.select(
        [result.gas_fraction < 0.2, result.gas_fraction > 0.5],
        ["gas_poor_below_0p2", "gas_rich_above_0p5"],
        default="gas_intermediate",
    )
    result["bulge_regime"] = np.select(
        [result.stellar_bulge_fraction < 0.05, result.stellar_bulge_fraction > 0.2],
        ["bulgeless_below_0p05", "bulged_above_0p2"],
        default="modest_bulge",
    )
    result["hubble_regime"] = np.select(
        [result.hubble_type <= 4, result.hubble_type >= 8],
        ["early_T_le_4", "late_T_ge_8"],
        default="middle_T_5_to_7",
    )
    result["surface_regime"] = pd.qcut(
        result.mean_surface_R80,
        3,
        labels=["low_surface_tertile", "middle_surface_tertile", "high_surface_tertile"],
    ).astype(str)
    return result


def regime_scores(per_galaxy):
    rows = []
    for dimension in (
        "mass_regime",
        "gas_regime",
        "bulge_regime",
        "hubble_regime",
        "surface_regime",
    ):
        for regime, block in per_galaxy.groupby(dimension, sort=True):
            row = {"dimension": dimension, "regime": regime, "galaxies": len(block)}
            for model in ("selected", "constant", "RAR"):
                row[f"{model}_equal_galaxy_RMSE_km_s"] = float(
                    np.sqrt(block[f"{model}_MSE"].mean())
                )
                row[f"{model}_mean_galaxy_residual_km_s"] = float(
                    block[f"{model}_mean_residual"].mean()
                )
            row["selected_improvement_vs_constant_fraction"] = 1.0 - (
                row["selected_equal_galaxy_RMSE_km_s"]
                / row["constant_equal_galaxy_RMSE_km_s"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(per_galaxy, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    count = len(per_galaxy)
    indices = rng.integers(0, count, size=(draws, count))
    selected = per_galaxy.selected_MSE.to_numpy(float)
    constant = per_galaxy.constant_MSE.to_numpy(float)
    rar = per_galaxy.RAR_MSE.to_numpy(float)
    selected_rmse = np.sqrt(np.mean(selected[indices], axis=1))
    constant_rmse = np.sqrt(np.mean(constant[indices], axis=1))
    rar_rmse = np.sqrt(np.mean(rar[indices], axis=1))
    return pd.DataFrame(
        {
            "draw": np.arange(draws),
            "selected_RMSE_km_s": selected_rmse,
            "constant_RMSE_km_s": constant_rmse,
            "RAR_RMSE_km_s": rar_rmse,
            "selected_improvement_vs_constant_fraction": 1.0
            - selected_rmse / constant_rmse,
            "selected_improvement_vs_RAR_fraction": 1.0 - selected_rmse / rar_rmse,
        }
    )


def activation_scores(oof, per_galaxy):
    rows = []
    for regime, galaxies in per_galaxy.groupby("mass_regime"):
        block = oof[oof.galaxy.isin(galaxies.galaxy)]
        rows.append(
            {
                "mass_regime": regime,
                "galaxies": block.galaxy.nunique(),
                "points": len(block),
                "potential_branch_fraction": float(np.mean(block.OR_component.eq("potential"))),
                "surface_branch_fraction": float(np.mean(block.OR_component.eq("surface_density"))),
                "q_constant_mean": float(block.q_constant.mean()),
                "q_potential_mean": float(block.q_potential.mean()),
                "q_surface_mean": float(block.q_surface.mean()),
                "q_selected_mean": float(block.q_selected.mean()),
            }
        )
    return pd.DataFrame(rows)


def cross_domain_scorecard(galaxy_scores, p0628):
    p0627_atlas = pd.read_csv(ROOT / "results/p0627_or_strength_phase_atlas/atlas_summary.csv")
    selected = p0627_atlas[
        np.isclose(p0627_atlas.beta, 0.5) & np.isclose(p0627_atlas.phase_degrees, -67.5)
    ].iloc[0]
    raw = pd.read_csv(ROOT / "results/p0627_or_strength_phase_atlas/raw_scores.csv")
    raw_selected = raw[
        np.isclose(raw.beta, 0.5) & np.isclose(raw.phase_degrees, -67.5)
    ]
    raw_scalar = raw[np.isclose(raw.beta, 0.5) & raw.phase_degrees.isna()].set_index(
        "system_label"
    )
    compact = float(p0628["cross_domain_comparators"]["compact_halo_RMS_arcsec"])
    selected_galaxy = galaxy_scores[galaxy_scores.model.eq("P0628_selected_OOF")].iloc[0]
    rar_galaxy = galaxy_scores[galaxy_scores.model.eq("fixed_RAR_same_nuisance")].iloc[0]
    rows = [
        {
            "domain": "galaxy_rotation",
            "metric": "five-fold OOF equal-galaxy RMSE km/s",
            "candidate_value": selected_galaxy.equal_galaxy_RMSE_km_s,
            "comparator": "fixed RAR same nuisance",
            "comparator_value": rar_galaxy.equal_galaxy_RMSE_km_s,
            "ratio": selected_galaxy.equal_galaxy_RMSE_km_s / rar_galaxy.equal_galaxy_RMSE_km_s,
            "evidence": "raw rotation speeds; project-spent OOF",
        },
        {
            "domain": "derived_cluster_lensing",
            "metric": "equal-system RMSE dex",
            "candidate_value": 0.193724,
            "comparator": "constant scalar",
            "comparator_value": 0.194199,
            "ratio": 0.193724 / 0.194199,
            "evidence": "NFW-derived acceleration target",
        },
        {
            "domain": "raw_cluster_lensing",
            "metric": "five-system fixed-geometry equal-system RMS arcsec",
            "candidate_value": selected.equal_system_RMS_arcsec,
            "comparator": "original P0618 scalar",
            "comparator_value": selected.original_P0618_scalar_RMS_arcsec,
            "ratio": selected.equal_system_RMS_arcsec / selected.original_P0618_scalar_RMS_arcsec,
            "evidence": "raw image positions; opened-data fixed geometry",
        },
        {
            "domain": "raw_cluster_lensing",
            "metric": "five-system fixed-geometry equal-system RMS arcsec",
            "candidate_value": selected.equal_system_RMS_arcsec,
            "comparator": "limited compact halo historical validation",
            "comparator_value": compact,
            "ratio": selected.equal_system_RMS_arcsec / compact,
            "evidence": "scope-limited non-matched dark-matter comparator",
        },
        {
            "domain": "Solar",
            "metric": "Mercury mas/century",
            "candidate_value": (-1.7176636455056473 - 2.129530870646183) / 2.0,
            "comparator": "absolute proxy margin",
            "comparator_value": 3.1,
            "ratio": abs((-1.7176636455056473 - 2.129530870646183) / 2.0) / 3.1,
            "evidence": "analytic proxy",
        },
    ]
    for row in raw_selected.itertuples():
        scalar_value = float(raw_scalar.loc[row.system_label, "heldout_RMS_arcsec"])
        rows.append(
            {
                "domain": f"raw_{row.system_label}",
                "metric": "heldout RMS arcsec",
                "candidate_value": row.heldout_RMS_arcsec,
                "comparator": "same-beta scalar",
                "comparator_value": scalar_value,
                "ratio": float(row.heldout_RMS_arcsec) / scalar_value,
                "evidence": "raw image positions; per-system value reported in raw artifact",
            }
        )
    return pd.DataFrame(rows)


def make_figure(output, scores, regimes, bootstrap_frame, cross):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(scores.model, scores.equal_galaxy_RMSE_km_s)
    axes[0, 0].set_ylabel("equal-galaxy RMSE (km/s)")
    axes[0, 0].set_title("Five-fold galaxy OOF")
    axes[0, 0].tick_params(axis="x", rotation=35, labelsize=7)
    mass = regimes[regimes.dimension.eq("mass_regime")]
    x = np.arange(len(mass))
    axes[0, 1].bar(x - 0.2, mass.constant_mean_galaxy_residual_km_s, 0.4, label="constant")
    axes[0, 1].bar(x + 0.2, mass.selected_mean_galaxy_residual_km_s, 0.4, label="selected")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x, mass.regime, rotation=25, fontsize=7)
    axes[0, 1].set_ylabel("mean galaxy residual (km/s)")
    axes[0, 1].set_title("Dwarf/giant bias")
    axes[0, 1].legend()
    axes[1, 0].hist(
        100 * bootstrap_frame.selected_improvement_vs_constant_fraction,
        bins=50,
        color="#3182bd",
    )
    axes[1, 0].axvline(0.0, color="black")
    axes[1, 0].set_xlabel("paired bootstrap improvement vs constant (%)")
    axes[1, 0].set_title("Spent-sample repeatability")
    shown = cross[cross.domain.isin(["galaxy_rotation", "derived_cluster_lensing", "raw_cluster_lensing"])]
    axes[1, 1].barh(shown.comparator, shown.ratio)
    axes[1, 1].axvline(1.0, color="black", linestyle="--")
    axes[1, 1].set_xlabel("candidate / comparator error")
    axes[1, 1].set_title("Comparator context (not matched evidence)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    protocol = load_json("configs/p0628_selected_density_route_synthesis_protocol.json")
    p0627 = load_json("configs/p0627_or_strength_phase_atlas_protocol.json")
    p0626 = load_json(p0627["parent_protocols"][0])
    p0625 = load_json(p0626["parent_protocols"][0])
    p0623 = load_json(p0625["parent_protocols"][0])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    frame, lookup, _, _ = prepare_frame(p0623)
    beta = float(protocol["selected_candidate"]["beta"])
    oof = build_oof(frame, lookup, beta)
    keep = [
        "galaxy",
        "galaxy_fold",
        "prediction_fold",
        "radius_adjusted_kpc",
        "velocity_observed_adjusted_km_s",
        "velocity_constant_km_s",
        "velocity_selected_km_s",
        "velocity_RAR_km_s",
        "q_constant",
        "q_potential",
        "q_surface",
        "q_OR",
        "q_selected",
        "OR_component",
        "force_equivalent_mass_solar",
        "force_equivalent_r80_kpc",
        "mean_surface_R80",
        "potential_depth",
    ]
    oof[keep].to_csv(output / protocol["outputs"]["galaxy_points"], index=False)
    scores = overall_scores(oof, p0623)
    scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    per_galaxy = per_galaxy_scores(oof)
    per_galaxy.to_csv(output / protocol["outputs"]["galaxy_per_object"], index=False)
    regimes = regime_scores(per_galaxy)
    regimes.to_csv(output / protocol["outputs"]["regimes"], index=False)
    bootstrap_frame = bootstrap(
        per_galaxy,
        int(protocol["galaxy_evaluation"]["bootstrap_draws"]),
        int(protocol["galaxy_evaluation"]["bootstrap_seed"]),
    )
    bootstrap_frame.to_csv(output / protocol["outputs"]["bootstrap"], index=False)
    activation = activation_scores(oof, per_galaxy)
    activation.to_csv(output / protocol["outputs"]["activation"], index=False)
    cross = cross_domain_scorecard(scores, protocol)
    cross.to_csv(output / protocol["outputs"]["cross_domain"], index=False)

    selected = scores[scores.model.eq("P0628_selected_OOF")].iloc[0]
    constant = scores[scores.model.eq("fold_refit_constant_OOF")].iloc[0]
    rar = scores[scores.model.eq("fixed_RAR_same_nuisance")].iloc[0]
    mass = regimes[regimes.dimension.eq("mass_regime")].set_index("regime")
    bootstrap_ci = np.quantile(
        bootstrap_frame.selected_improvement_vs_constant_fraction, [0.025, 0.5, 0.975]
    )
    fold4 = score_arrays(
        oof[oof.prediction_fold.eq(4)],
        oof.loc[oof.prediction_fold.eq(4), "velocity_selected_km_s"],
    )
    fold4_constant = score_arrays(
        oof[oof.prediction_fold.eq(4)],
        oof.loc[oof.prediction_fold.eq(4), "velocity_constant_km_s"],
    )
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete_selected_candidate_synthesis",
        "selected_formula": protocol["selected_candidate"],
        "galaxy": {
            "selected_equal_galaxy_RMSE_km_s": selected.equal_galaxy_RMSE_km_s,
            "constant_equal_galaxy_RMSE_km_s": constant.equal_galaxy_RMSE_km_s,
            "RAR_equal_galaxy_RMSE_km_s": rar.equal_galaxy_RMSE_km_s,
            "improvement_vs_constant_fraction": selected.improvement_vs_constant_fraction,
            "ratio_to_RAR": selected.equal_galaxy_RMSE_km_s / rar.equal_galaxy_RMSE_km_s,
            "bootstrap_improvement_95_interval": bootstrap_ci.tolist(),
            "P0623_fold4_selected_RMSE_km_s": fold4["equal_galaxy_RMSE_km_s"],
            "P0623_fold4_constant_RMSE_km_s": fold4_constant["equal_galaxy_RMSE_km_s"],
        },
        "mass_regimes": strict_json(mass.reset_index().to_dict(orient="records")),
        "activation": strict_json(activation.to_dict(orient="records")),
        "cross_domain": strict_json(cross.to_dict(orient="records")),
        "conclusion": {
            "dwarf_and_giant_bias_both_improve": bool(
                abs(mass.loc["dwarf_below_1e9", "selected_mean_galaxy_residual_km_s"])
                < abs(mass.loc["dwarf_below_1e9", "constant_mean_galaxy_residual_km_s"])
                and abs(mass.loc["giant_above_1e10", "selected_mean_galaxy_residual_km_s"])
                < abs(mass.loc["giant_above_1e10", "constant_mean_galaxy_residual_km_s"])
            ),
            "beats_fixed_RAR": bool(selected.equal_galaxy_RMSE_km_s < rar.equal_galaxy_RMSE_km_s),
            "beats_limited_compact_halo": bool(
                cross.loc[cross.comparator.eq("limited compact halo historical validation"), "ratio"].iloc[0]
                < 1.0
            ),
            "validated_field_theory": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(strict_json(report), indent=2), encoding="utf-8"
    )
    make_figure(output / protocol["outputs"]["figure"], scores, regimes, bootstrap_frame, cross)
    dwarf = mass.loc["dwarf_below_1e9"]
    giant = mass.loc["giant_above_1e10"]
    lines = [
        "# P0628 selected density-route synthesis",
        "",
        "## Bottom line",
        "",
        f"Five-fold OOF equal-galaxy RMSE is **{selected.equal_galaxy_RMSE_km_s:.3f} km/s**, "
        f"versus **{constant.equal_galaxy_RMSE_km_s:.3f}** for the refit constant and "
        f"**{rar.equal_galaxy_RMSE_km_s:.3f}** for fixed RAR.",
        f"The paired spent-sample bootstrap improvement versus constant is "
        f"**{100*bootstrap_ci[1]:.2f}%** (95% interval {100*bootstrap_ci[0]:.2f}% to {100*bootstrap_ci[2]:.2f}%).",
        "",
        f"Dwarf mean residual changes from **{dwarf.constant_mean_galaxy_residual_km_s:+.2f}** to "
        f"**{dwarf.selected_mean_galaxy_residual_km_s:+.2f} km/s**; giant residual changes from "
        f"**{giant.constant_mean_galaxy_residual_km_s:+.2f}** to "
        f"**{giant.selected_mean_galaxy_residual_km_s:+.2f} km/s**.",
        "",
        "The candidate remains worse than fixed RAR on galaxies and about twice the limited compact-halo raw-lensing RMS. Its raw aggregate gain over the original scalar is only 0.123% and is not uniform across clusters.",
        "",
        "This is a hypothesis-generating result on project-spent data, not a validated field theory.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(strict_json(report["galaxy"]), indent=2), flush=True)
    print(json.dumps(strict_json(report["conclusion"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
