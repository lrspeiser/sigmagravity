#!/usr/bin/env python3
"""P0629: continuous compact-potential to extended-porosity hierarchy."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_apogee_cross_domain import radius_at_mass_fraction  # noqa: E402
from run_arc_invariant_absolute_lensing import (  # noqa: E402
    cluster_score,
    prepare_clusters,
    response_for_frame,
    response_parameters,
)
from run_p0554_local_cross_domain_sensitivity import raw_contexts  # noqa: E402
from run_p0615_self_coupled_quadrupole_route import derived_state  # noqa: E402
from run_p0617_self_coupled_support_phase_atlas import contexts_and_frozen_geometry, lens_score  # noqa: E402
from run_p0618_universal_route_phase import phase_field  # noqa: E402
from run_p0623_density_path_survival import (  # noqa: E402
    A0,
    G_SI,
    KPC_M,
    M_SUN_KG,
    predict_velocity,
    score_arrays,
)
from run_p0625_bounded_porosity_survival import apply_record, prepare_frame  # noqa: E402
from run_p0626_compact_scalar_angular_route import fit_atomic  # noqa: E402
from voidscreen.arc_invariants import generalized_arc_response, spherical_profile_invariants  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


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


def identifier(scale: float, beta: float) -> str:
    return f"L{scale:g}_beta{beta:g}".replace(".", "p")


def weight(r80, scale: float):
    return 1.0 / (1.0 + np.square(np.asarray(r80, dtype=float) / float(scale)))


def hierarchical_q(q_constant, q_potential, q_surface, r80, scale: float, beta: float):
    q_or = np.maximum(q_potential, q_surface)
    q_porous = q_constant + float(beta) * (q_or - q_constant)
    compact = weight(r80, scale)
    return compact * q_potential + (1.0 - compact) * q_porous


def atomic_values(records, frame):
    q_constant = apply_record(records["constant"], np.ones(len(frame)))
    q_potential = apply_record(
        records["inverse_hill0_m1__potential_depth"], frame.potential_depth
    )
    surface = frame.force_equivalent_mass_solar.to_numpy(float) / (
        np.pi * np.square(frame.force_equivalent_r80_kpc.to_numpy(float))
    )
    q_surface = apply_record(records["inverse_hillfloor_m2__mean_surface_R80"], surface)
    return q_constant, q_potential, q_surface


def galaxy_evaluation(frame, lookup, scales, betas):
    prediction_blocks = []
    for fold in range(5):
        train = frame[frame.galaxy_fold.ne(fold)]
        test = frame[frame.galaxy_fold.eq(fold)].copy()
        records = fit_atomic(train, lookup)
        q_constant, q_potential, q_surface = atomic_values(records, test)
        baseline = test[["galaxy", "force_equivalent_mass_solar"]].copy()
        baseline["candidate_id"] = "constant"
        baseline["prediction"] = predict_velocity(test, q_constant)
        baseline["observed"] = test.velocity_observed_adjusted_km_s.to_numpy(float)
        baseline["fold"] = fold
        prediction_blocks.append(baseline)
        for scale in scales:
            for beta in betas:
                q_eff = hierarchical_q(
                    q_constant,
                    q_potential,
                    q_surface,
                    test.force_equivalent_r80_kpc,
                    scale,
                    beta,
                )
                local = test[["galaxy", "force_equivalent_mass_solar"]].copy()
                local["candidate_id"] = identifier(scale, beta)
                local["prediction"] = predict_velocity(test, q_eff)
                local["observed"] = test.velocity_observed_adjusted_km_s.to_numpy(float)
                local["fold"] = fold
                prediction_blocks.append(local)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    predictions["residual"] = predictions.prediction - predictions.observed
    predictions["squared"] = np.square(predictions.residual)
    per_galaxy = predictions.groupby(["candidate_id", "galaxy"], sort=False).agg(
        MSE=("squared", "mean"),
        mean_residual=("residual", "mean"),
        baryonic_mass_solar=("force_equivalent_mass_solar", "first"),
        fold=("fold", "first"),
    ).reset_index()
    per_galaxy["mass_regime"] = np.select(
        [per_galaxy.baryonic_mass_solar < 1.0e9, per_galaxy.baryonic_mass_solar > 1.0e10],
        ["dwarf_below_1e9", "giant_above_1e10"],
        default="intermediate_1e9_to_1e10",
    )
    score_rows = []
    baseline = per_galaxy[per_galaxy.candidate_id.eq("constant")]
    baseline_rmse = float(np.sqrt(baseline.MSE.mean()))
    baseline_fold = baseline.groupby("fold").MSE.mean()
    for candidate_id, block in per_galaxy.groupby("candidate_id"):
        fold_mse = block.groupby("fold").MSE.mean()
        score_rows.append(
            {
                "candidate_id": candidate_id,
                "equal_galaxy_RMSE_km_s": float(np.sqrt(block.MSE.mean())),
                "improvement_vs_constant_fraction": 1.0
                - float(np.sqrt(block.MSE.mean())) / baseline_rmse,
                "fold_wins": int(np.sum(fold_mse < baseline_fold)),
            }
        )
    regime_rows = []
    constant_regimes = baseline.groupby("mass_regime").agg(
        constant_MSE=("MSE", "mean"),
        constant_mean_residual=("mean_residual", "mean"),
    )
    for (candidate_id, regime), block in per_galaxy.groupby(["candidate_id", "mass_regime"]):
        regime_rows.append(
            {
                "candidate_id": candidate_id,
                "mass_regime": regime,
                "galaxies": len(block),
                "equal_galaxy_RMSE_km_s": float(np.sqrt(block.MSE.mean())),
                "mean_residual_km_s": float(block.mean_residual.mean()),
                "constant_equal_galaxy_RMSE_km_s": float(
                    np.sqrt(constant_regimes.loc[regime, "constant_MSE"])
                ),
                "constant_mean_residual_km_s": float(
                    constant_regimes.loc[regime, "constant_mean_residual"]
                ),
            }
        )
    return pd.DataFrame(score_rows), pd.DataFrame(regime_rows), per_galaxy


def derived_evaluation(parent, spec, records, scales, betas):
    clusters, _ = prepare_clusters(parent)
    response = response_for_frame(
        clusters,
        spec,
        q=1.0,
        a0=A0,
        radius_column="radius_kpc",
        gbar_column="gbar_m_s2",
    )
    q_constant, q_potential, q_surface = atomic_values(records, clusters)
    photon = float(spec["photon_extra_multiplier"])
    rows = []
    baseline_prediction = clusters.gbar_m_s2.to_numpy(float) * (
        1.0 + photon * q_constant * response["unit_fractional_response"]
    )
    baseline = cluster_score(clusters, baseline_prediction)
    for scale in scales:
        for beta in betas:
            q_eff = hierarchical_q(
                q_constant,
                q_potential,
                q_surface,
                clusters.force_equivalent_r80_kpc,
                scale,
                beta,
            )
            prediction = clusters.gbar_m_s2.to_numpy(float) * (
                1.0 + photon * q_eff * response["unit_fractional_response"]
            )
            metrics = cluster_score(clusters, prediction)
            rows.append(
                {
                    "candidate_id": identifier(scale, beta),
                    "coherence_scale_kpc": scale,
                    "porous_beta": beta,
                    "q_min": np.min(q_eff),
                    "q_median": np.median(q_eff),
                    "q_max": np.max(q_eff),
                    **metrics,
                    "improvement_vs_constant_fraction": 1.0
                    - metrics["cluster_equal_system_RMSE_dex"]
                    / baseline["cluster_equal_system_RMSE_dex"],
                }
            )
    return pd.DataFrame(rows)


def raw_parent_fields(spec, records, anchors, local, scales, betas):
    radius_grid = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(radius_grid, anchor_radius, anchor_gbar, outer_slope=-2.0)
    invariants = spherical_profile_invariants(radius_grid, gbar)
    anchor_mass = anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    total = float(np.maximum.accumulate(anchor_mass)[-1])
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    q_constant = apply_record(records["constant"], np.ones_like(radius_grid))
    q_potential = apply_record(
        records["inverse_hill0_m1__potential_depth"], invariants["potential_depth"]
    )
    q_surface = apply_record(
        records["inverse_hillfloor_m2__mean_surface_R80"],
        np.full_like(radius_grid, total / (np.pi * r80**2)),
    )
    response = generalized_arc_response(
        gbar,
        radius_grid,
        np.full_like(radius_grid, total),
        np.full_like(radius_grid, r50 / r80),
        potential_depth=invariants["potential_depth"],
        potential_length_kpc=invariants["potential_length_kpc"],
        potential_path_ratio=invariants["potential_path_ratio"],
        enclosed_mass_log_slope=invariants["enclosed_mass_log_slope"],
        **response_parameters(spec, q=1.0, a0=A0),
    )
    photon = float(spec["photon_extra_multiplier"])
    impact = np.geomspace(0.05, 500.0, 700)
    angular_scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    fields = {}
    for scale in scales:
        for beta in betas:
            q_eff = hierarchical_q(
                q_constant, q_potential, q_surface, np.full_like(radius_grid, r80), scale, beta
            )
            acceleration = gbar * (
                1.0 + photon * q_eff * response["unit_fractional_response"]
            )

            def lookup(radius, local_acceleration=acceleration):
                return np.exp(
                    np.interp(np.log(radius), np.log(radius_grid), np.log(local_acceleration))
                )

            alpha = spherical_deflection_radians(
                impact * angular_scale,
                lookup,
                maximum_radius_kpc=1.0e6,
                integration_points=800,
            )
            fields[(scale, beta)] = RadialDeflectionField(impact, alpha)
    return fields


def raw_evaluation(spec, records, scales, betas, phases):
    p0615 = load_json("configs/p0615_self_coupled_quadrupole_route_protocol.json")
    p0581 = load_json(p0615["inputs"]["P0581_protocol"])
    prepared = contexts_and_frozen_geometry(p0615)
    p0554 = load_json("configs/p0554_local_cross_domain_sensitivity_protocol.json")
    anchor_map = {context.label: context.anchors for context in raw_contexts(p0554)}
    rows = []
    for context, cohort, parameters, sources in prepared:
        label = context.system["label"]
        anchors = context.anchors if hasattr(context, "anchors") else anchor_map[label]
        fields = raw_parent_fields(spec, records, anchors, context.local, scales, betas)
        for (scale, beta), parent_field in fields.items():
            local_context = SimpleNamespace(**context.__dict__)
            local_context.parent = parent_field
            state = derived_state(local_context)
            epsilon = float(state["amplitudes"]["quadratic_Q2_over_total"])
            for phase in phases:
                route, _ = phase_field(p0581, local_context, state, phase)
                metrics = lens_score(local_context, parameters, sources, route, epsilon)
                rows.append(
                    {
                        "cohort": cohort,
                        "system_label": label,
                        "candidate_id": identifier(scale, beta),
                        "coherence_scale_kpc": scale,
                        "porous_beta": beta,
                        "phase_degrees": phase,
                        "epsilon": epsilon,
                        **metrics,
                    }
                )
        print(f"P0629 raw {label}: {len(scales)*len(betas)*len(phases)}", flush=True)
    return pd.DataFrame(rows)


def candidate_matrix(galaxy, regimes, derived, raw, scales, betas, phases):
    original = pd.read_csv(ROOT / "results/p0618_universal_route_phase/scores.csv")
    original = original[original.variant_id.eq("scalar_control")]
    original_rms = float(np.sqrt(np.mean(np.square(original.heldout_RMS_arcsec))))
    rows = []
    regime_lookup = regimes.set_index(["candidate_id", "mass_regime"])
    derived_lookup = derived.set_index("candidate_id")
    galaxy_lookup = galaxy.set_index("candidate_id")
    for scale in scales:
        for beta in betas:
            candidate_id = identifier(scale, beta)
            dwarf = regime_lookup.loc[(candidate_id, "dwarf_below_1e9")]
            giant = regime_lookup.loc[(candidate_id, "giant_above_1e10")]
            for phase in phases:
                block = raw[
                    raw.candidate_id.eq(candidate_id) & np.isclose(raw.phase_degrees, phase)
                ]
                all_roots = bool(block.heldout_all_roots.astype(bool).all())
                raw_rms = (
                    float(np.sqrt(np.mean(np.square(block.heldout_RMS_arcsec))))
                    if all_roots
                    else np.inf
                )
                grow = galaxy_lookup.loc[candidate_id]
                drow = derived_lookup.loc[candidate_id]
                dwarf_bias_improves = abs(dwarf.mean_residual_km_s) < abs(
                    dwarf.constant_mean_residual_km_s
                )
                giant_bias_improves = abs(giant.mean_residual_km_s) < abs(
                    giant.constant_mean_residual_km_s
                )
                passes = bool(
                    dwarf_bias_improves
                    and giant_bias_improves
                    and grow.fold_wins >= 3
                    and grow.improvement_vs_constant_fraction >= 0.05
                    and drow.improvement_vs_constant_fraction >= 0.0
                    and all_roots
                    and raw_rms <= original_rms
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "coherence_scale_kpc": scale,
                        "porous_beta": beta,
                        "phase_degrees": phase,
                        "galaxy_RMSE_km_s": grow.equal_galaxy_RMSE_km_s,
                        "galaxy_improvement_fraction": grow.improvement_vs_constant_fraction,
                        "galaxy_fold_wins": int(grow.fold_wins),
                        "dwarf_mean_residual_km_s": dwarf.mean_residual_km_s,
                        "dwarf_constant_mean_residual_km_s": dwarf.constant_mean_residual_km_s,
                        "dwarf_bias_improves": dwarf_bias_improves,
                        "giant_mean_residual_km_s": giant.mean_residual_km_s,
                        "giant_constant_mean_residual_km_s": giant.constant_mean_residual_km_s,
                        "giant_bias_improves": giant_bias_improves,
                        "derived_cluster_improvement_fraction": drow.improvement_vs_constant_fraction,
                        "Solar_potential_endpoint_pass": True,
                        "raw_roots": int(block.heldout_converged_roots.sum()),
                        "raw_RMS_arcsec": raw_rms,
                        "raw_original_scalar_RMS_arcsec": original_rms,
                        "raw_improvement_vs_original_fraction": 1.0 - raw_rms / original_rms,
                        "all_diagnostic_rules_pass": passes,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["all_diagnostic_rules_pass", "raw_RMS_arcsec", "galaxy_RMSE_km_s"],
        ascending=[False, True, True],
    )


def make_figure(output, matrix):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    grouped = matrix.drop_duplicates(["coherence_scale_kpc", "porous_beta"])
    for beta, block in grouped.groupby("porous_beta"):
        axes[0].plot(
            block.coherence_scale_kpc,
            100 * block.galaxy_improvement_fraction,
            marker="o",
            label=f"beta={beta:g}",
        )
    axes[0].set(xlabel="coherence scale (kpc)", ylabel="galaxy gain (%)", title="Compact potential branch")
    axes[0].legend()
    axes[1].scatter(
        matrix.giant_mean_residual_km_s,
        matrix.dwarf_mean_residual_km_s,
        c=matrix.coherence_scale_kpc,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="giant mean residual", ylabel="dwarf mean residual", title="Bias tradeoff")
    axes[2].scatter(
        100 * matrix.galaxy_improvement_fraction,
        100 * matrix.raw_improvement_vs_original_fraction,
        c=np.where(matrix.all_diagnostic_rules_pass, "green", "gray"),
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(xlabel="galaxy gain (%)", ylabel="raw gain (%)", title="Cross-domain frontier")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    protocol = load_json("configs/p0629_hierarchical_density_survival_protocol.json")
    p0628 = load_json("configs/p0628_selected_density_route_synthesis_protocol.json")
    p0627 = load_json("configs/p0627_or_strength_phase_atlas_protocol.json")
    p0626 = load_json(p0627["parent_protocols"][0])
    p0625 = load_json(p0626["parent_protocols"][0])
    p0623 = load_json(p0625["parent_protocols"][0])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scales = [float(value) for value in protocol["formula"]["coherence_scales_kpc"]]
    betas = [float(value) for value in protocol["formula"]["porous_beta_values"]]
    phases = [float(value) for value in protocol["formula"]["route_phases_degrees"]]
    frame, lookup, spec, parent = prepare_frame(p0623)
    galaxy, regimes, _ = galaxy_evaluation(frame, lookup, scales, betas)
    galaxy.to_csv(output / protocol["outputs"]["galaxy"], index=False)
    regimes.to_csv(output / protocol["outputs"]["mass_regimes"], index=False)
    records = fit_atomic(frame[frame.galaxy_fold.isin([0, 1, 2, 3])], lookup)
    derived = derived_evaluation(parent, spec, records, scales, betas)
    derived.to_csv(output / protocol["outputs"]["derived"], index=False)
    raw = raw_evaluation(spec, records, scales, betas, phases)
    raw.to_csv(output / protocol["outputs"]["raw"], index=False)
    matrix = candidate_matrix(galaxy, regimes, derived, raw, scales, betas, phases)
    matrix.to_csv(output / protocol["outputs"]["candidate_matrix"], index=False)
    passers = matrix[matrix.all_diagnostic_rules_pass]
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete_opened_data_hierarchical_screen",
        "best_row": strict_json(matrix.iloc[0].to_dict()),
        "diagnostic_passers": strict_json(passers.to_dict(orient="records")),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    make_figure(output / protocol["outputs"]["figure"], matrix)
    best = matrix.iloc[0]
    lines = [
        "# P0629 hierarchical density survival",
        "",
        f"Best opened-data row: **L={best.coherence_scale_kpc:g} kpc, beta={best.porous_beta:g}, phase={best.phase_degrees:+g} deg**.",
        f"Galaxy gain: **{100*best.galaxy_improvement_fraction:+.2f}%**; dwarf residual "
        f"**{best.dwarf_mean_residual_km_s:+.2f} km/s**; giant residual "
        f"**{best.giant_mean_residual_km_s:+.2f} km/s**.",
        f"Derived cluster gain: **{100*best.derived_cluster_improvement_fraction:+.3f}%**; "
        f"raw gain: **{100*best.raw_improvement_vs_original_fraction:+.3f}%** with "
        f"**{int(best.raw_roots)}/18** roots.",
        f"Rows clearing all spent-data diagnostic rules: **{len(passers)}**.",
        "",
        "This is a multiplicity-exposed hypothesis screen, not external validation.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
