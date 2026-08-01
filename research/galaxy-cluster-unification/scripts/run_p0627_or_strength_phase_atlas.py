#!/usr/bin/env python3
"""P0627: atlas of one universal OR strength and one universal route phase."""

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


def beta_label(beta: float) -> str:
    return f"beta_{beta:.2f}".replace(".", "p")


def phase_label(phase: float) -> str:
    return f"phase_{phase:+05.1f}".replace(".", "p")


def atomic_q(records, potential, surface):
    shape = np.asarray(potential).shape
    constant = apply_record(records["constant"], np.ones(shape))
    q_potential = apply_record(records["inverse_hill0_m1__potential_depth"], potential)
    q_surface = apply_record(records["inverse_hillfloor_m2__mean_surface_R80"], surface)
    return constant, np.maximum(q_potential, q_surface)


def q_beta(constant, q_or, beta: float):
    return constant + float(beta) * (q_or - constant)


def galaxy_scores(frame, lookup, folds, betas):
    rows = []
    for fold in folds:
        train = frame[frame.galaxy_fold.isin(folds) & frame.galaxy_fold.ne(fold)]
        test = frame[frame.galaxy_fold.eq(fold)]
        records = fit_atomic(train, lookup)
        surface = test.force_equivalent_mass_solar.to_numpy(float) / (
            np.pi * np.square(test.force_equivalent_r80_kpc.to_numpy(float))
        )
        constant, q_or = atomic_q(records, test.potential_depth.to_numpy(float), surface)
        for beta in betas:
            q_eff = q_beta(constant, q_or, beta)
            metrics = score_arrays(test, predict_velocity(test, q_eff))
            rows.append(
                {
                    "beta": beta,
                    "heldout_fold": fold,
                    "equal_galaxy_RMSE_km_s": metrics["equal_galaxy_RMSE_km_s"],
                    "pooled_RMSE_km_s": metrics["pooled_RMSE_km_s"],
                }
            )
        baseline_metrics = score_arrays(test, predict_velocity(test, constant))
        rows.append(
            {
                "beta": 0.0,
                "heldout_fold": fold,
                "equal_galaxy_RMSE_km_s": baseline_metrics["equal_galaxy_RMSE_km_s"],
                "pooled_RMSE_km_s": baseline_metrics["pooled_RMSE_km_s"],
            }
        )
    folds_frame = pd.DataFrame(rows)
    baseline = folds_frame[folds_frame.beta.eq(0.0)][
        ["heldout_fold", "equal_galaxy_RMSE_km_s"]
    ].rename(columns={"equal_galaxy_RMSE_km_s": "baseline_RMSE"})
    folds_frame = folds_frame.merge(baseline, on="heldout_fold", validate="many_to_one")
    folds_frame["fold_improvement_fraction"] = (
        1.0 - folds_frame.equal_galaxy_RMSE_km_s / folds_frame.baseline_RMSE
    )
    scores = folds_frame.groupby("beta").agg(
        mean_MSE=("equal_galaxy_RMSE_km_s", lambda x: np.mean(np.square(x))),
        pooled_MSE=("pooled_RMSE_km_s", lambda x: np.mean(np.square(x))),
        fold_wins=("fold_improvement_fraction", lambda x: int(np.sum(np.asarray(x) > 0.0))),
    ).reset_index()
    scores["cv_equal_galaxy_RMSE_km_s"] = np.sqrt(scores.pop("mean_MSE"))
    scores["cv_pooled_RMSE_km_s"] = np.sqrt(scores.pop("pooled_MSE"))
    baseline_rmse = float(scores.loc[scores.beta.eq(0.0), "cv_equal_galaxy_RMSE_km_s"].iloc[0])
    scores["improvement_vs_constant_fraction"] = (
        1.0 - scores.cv_equal_galaxy_RMSE_km_s / baseline_rmse
    )
    return scores.sort_values("beta"), folds_frame


def derived_scores(parent, spec, records, betas):
    clusters, _ = prepare_clusters(parent)
    response = response_for_frame(
        clusters,
        spec,
        q=1.0,
        a0=A0,
        radius_column="radius_kpc",
        gbar_column="gbar_m_s2",
    )
    surface = clusters.force_equivalent_mass_solar.to_numpy(float) / (
        np.pi * np.square(clusters.force_equivalent_r80_kpc.to_numpy(float))
    )
    constant, q_or = atomic_q(records, clusters.potential_depth.to_numpy(float), surface)
    photon = float(spec["photon_extra_multiplier"])
    rows = []
    for beta in [0.0, *betas]:
        q_eff = q_beta(constant, q_or, beta)
        prediction = clusters.gbar_m_s2.to_numpy(float) * (
            1.0 + photon * q_eff * response["unit_fractional_response"]
        )
        rows.append(
            {
                "beta": beta,
                "q_min": np.min(q_eff),
                "q_median": np.median(q_eff),
                "q_max": np.max(q_eff),
                **cluster_score(clusters, prediction),
            }
        )
    scores = pd.DataFrame(rows)
    baseline = float(scores.loc[scores.beta.eq(0.0), "cluster_equal_system_RMSE_dex"].iloc[0])
    scores["improvement_vs_constant_fraction"] = (
        1.0 - scores.cluster_equal_system_RMSE_dex / baseline
    )
    return scores


def beta_fields(spec, records, anchors, raw_protocol, betas):
    radius_grid = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(radius_grid, anchor_radius, anchor_gbar, outer_slope=-2.0)
    invariants = spherical_profile_invariants(radius_grid, gbar)
    anchor_mass = anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    total = float(np.maximum.accumulate(anchor_mass)[-1])
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    surface = np.full_like(radius_grid, total / (np.pi * r80**2))
    constant, q_or = atomic_q(records, invariants["potential_depth"], surface)
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
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    fields = {}
    for beta in betas:
        q_eff = q_beta(constant, q_or, beta)
        acceleration = gbar * (
            1.0 + photon * q_eff * response["unit_fractional_response"]
        )

        def lookup(radius, local_acceleration=acceleration):
            return np.exp(
                np.interp(np.log(radius), np.log(radius_grid), np.log(local_acceleration))
            )

        alpha = spherical_deflection_radians(
            impact * scale,
            lookup,
            maximum_radius_kpc=1.0e6,
            integration_points=800,
        )
        fields[beta] = RadialDeflectionField(impact, alpha)
    return fields


def raw_atlas(spec, records, betas, phases):
    p0615 = load_json("configs/p0615_self_coupled_quadrupole_route_protocol.json")
    p0581 = load_json(p0615["inputs"]["P0581_protocol"])
    prepared = contexts_and_frozen_geometry(p0615)
    p0554 = load_json("configs/p0554_local_cross_domain_sensitivity_protocol.json")
    anchor_map = {context.label: context.anchors for context in raw_contexts(p0554)}
    rows = []
    for context, cohort, parameters, sources in prepared:
        label = context.system["label"]
        anchors = context.anchors if hasattr(context, "anchors") else anchor_map[label]
        fields = beta_fields(spec, records, anchors, context.local, betas)
        for beta, parent_field in fields.items():
            local_context = SimpleNamespace(**context.__dict__)
            local_context.parent = parent_field
            scalar = lens_score(local_context, parameters, sources, None, 0.0)
            rows.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "beta": beta,
                    "phase_degrees": np.nan,
                    "variant_id": f"{beta_label(beta)}_scalar",
                    "epsilon": 0.0,
                    **scalar,
                }
            )
            state = derived_state(local_context)
            epsilon = float(state["amplitudes"]["quadratic_Q2_over_total"])
            for phase in phases:
                route, _ = phase_field(p0581, local_context, state, phase)
                metrics = lens_score(local_context, parameters, sources, route, epsilon)
                rows.append(
                    {
                        "cohort": cohort,
                        "system_label": label,
                        "beta": beta,
                        "phase_degrees": phase,
                        "variant_id": f"{beta_label(beta)}_{phase_label(phase)}",
                        "epsilon": epsilon,
                        **metrics,
                    }
                )
        print(f"P0627 raw {label}: {len(betas)} betas x {len(phases)} phases", flush=True)
    return pd.DataFrame(rows)


def summarize_atlas(raw, galaxy, derived):
    original = pd.read_csv(ROOT / "results/p0618_universal_route_phase/scores.csv")
    original = original[original.variant_id.eq("scalar_control")].set_index("system_label")
    original_rms = float(np.sqrt(np.mean(np.square(original.heldout_RMS_arcsec))))
    scalar = raw[raw.phase_degrees.isna()].set_index(["beta", "system_label"])
    rows = []
    for (beta, phase), block in raw[raw.phase_degrees.notna()].groupby(
        ["beta", "phase_degrees"]
    ):
        block = block.set_index("system_label")
        scalar_block = scalar.loc[beta]
        all_roots = bool(block.heldout_all_roots.astype(bool).all())
        improvements = []
        for label in block.index:
            if bool(block.loc[label].heldout_all_roots) and bool(
                scalar_block.loc[label].heldout_all_roots
            ):
                improvements.append(
                    1.0
                    - float(block.loc[label].heldout_RMS_arcsec)
                    / float(scalar_block.loc[label].heldout_RMS_arcsec)
                )
            else:
                improvements.append(-np.inf)
        routed_rms = (
            float(np.sqrt(np.mean(np.square(block.heldout_RMS_arcsec))))
            if all_roots
            else np.inf
        )
        scalar_all = bool(scalar_block.heldout_all_roots.astype(bool).all())
        scalar_rms = (
            float(np.sqrt(np.mean(np.square(scalar_block.heldout_RMS_arcsec))))
            if scalar_all
            else np.inf
        )
        grow = galaxy[galaxy.beta.eq(beta)].iloc[0]
        drow = derived[derived.beta.eq(beta)].iloc[0]
        rows.append(
            {
                "beta": beta,
                "phase_degrees": phase,
                "all_18_roots": all_roots,
                "roots": int(block.heldout_converged_roots.sum()),
                "systems_not_worse_than_own_scalar": int(np.sum(np.asarray(improvements) >= 0.0)),
                "worst_system_improvement_fraction": float(np.min(improvements)),
                "mean_system_improvement_fraction": float(np.mean(improvements)),
                "equal_system_RMS_arcsec": routed_rms,
                "own_scalar_equal_system_RMS_arcsec": scalar_rms,
                "improvement_vs_own_scalar_fraction": 1.0 - routed_rms / scalar_rms,
                "original_P0618_scalar_RMS_arcsec": original_rms,
                "improvement_vs_original_P0618_scalar_fraction": 1.0 - routed_rms / original_rms,
                "galaxy_improvement_fraction": grow.improvement_vs_constant_fraction,
                "galaxy_fold_wins": int(grow.fold_wins),
                "derived_cluster_improvement_fraction": drow.improvement_vs_constant_fraction,
                "Solar_convex_endpoint_pass": True,
            }
        )
    atlas = pd.DataFrame(rows)
    atlas["all_cross_domain_diagnostic_rules"] = (
        (atlas.galaxy_improvement_fraction >= 0.05)
        & (atlas.derived_cluster_improvement_fraction >= 0.0)
        & atlas.Solar_convex_endpoint_pass
        & atlas.all_18_roots
        & (atlas.improvement_vs_own_scalar_fraction >= 0.0)
        & (atlas.improvement_vs_original_P0618_scalar_fraction >= 0.0)
    )
    return atlas.sort_values(
        [
            "all_18_roots",
            "systems_not_worse_than_own_scalar",
            "worst_system_improvement_fraction",
            "equal_system_RMS_arcsec",
            "galaxy_improvement_fraction",
        ],
        ascending=[False, False, False, True, False],
    )


def make_figure(output, galaxy, derived, atlas):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(galaxy.beta, 100 * galaxy.improvement_vs_constant_fraction, marker="o", label="galaxy")
    axes[0].plot(derived.beta, 100 * derived.improvement_vs_constant_fraction, marker="s", label="derived cluster")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="universal beta", ylabel="improvement (%)", title="Scalar tradeoff")
    axes[0].legend()
    pivot = atlas.pivot(index="beta", columns="phase_degrees", values="improvement_vs_own_scalar_fraction")
    image = axes[1].imshow(100 * pivot.to_numpy(), aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
    axes[1].set(
        xticks=np.arange(len(pivot.columns)),
        xticklabels=[f"{x:g}" for x in pivot.columns],
        yticks=np.arange(len(pivot.index)),
        yticklabels=[f"{x:g}" for x in pivot.index],
        xlabel="phase (degrees)",
        ylabel="beta",
        title="Route change vs own scalar (%)",
    )
    fig.colorbar(image, ax=axes[1])
    best_by_beta = atlas.sort_values("equal_system_RMS_arcsec").groupby("beta").first().reset_index()
    axes[2].plot(best_by_beta.beta, best_by_beta.equal_system_RMS_arcsec, marker="o")
    axes[2].axhline(float(atlas.original_P0618_scalar_RMS_arcsec.iloc[0]), color="black", linestyle="--")
    axes[2].set(xlabel="beta", ylabel="raw equal-system RMS (arcsec)", title="Best phase per beta (diagnostic)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    protocol = load_json("configs/p0627_or_strength_phase_atlas_protocol.json")
    p0626 = load_json(protocol["parent_protocols"][0])
    p0625 = load_json(p0626["parent_protocols"][0])
    p0623 = load_json(p0625["parent_protocols"][0])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    betas = [float(value) for value in protocol["formula"]["beta_values"]]
    phases = [float(value) for value in protocol["formula"]["phase_degrees"]]
    frame, lookup, spec, parent = prepare_frame(p0623)
    folds = p0623["sample"]["development_galaxy_folds"]
    galaxy, _ = galaxy_scores(frame, lookup, folds, betas)
    galaxy.to_csv(output / protocol["outputs"]["galaxy"], index=False)
    records = fit_atomic(frame[frame.galaxy_fold.isin(folds)], lookup)
    derived = derived_scores(parent, spec, records, betas)
    derived.to_csv(output / protocol["outputs"]["derived"], index=False)
    raw = raw_atlas(spec, records, betas, phases)
    raw.to_csv(output / protocol["outputs"]["raw"], index=False)
    atlas = summarize_atlas(raw, galaxy, derived)
    atlas.to_csv(output / protocol["outputs"]["atlas"], index=False)
    selected = atlas.iloc[0]
    passers = atlas[atlas.all_cross_domain_diagnostic_rules]
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete_opened_data_atlas",
        "selected_lexicographic": strict_json(selected.to_dict()),
        "cross_domain_rule_passers": strict_json(
            passers[["beta", "phase_degrees"]].to_dict(orient="records")
        ),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    make_figure(output / protocol["outputs"]["figure"], galaxy, derived, atlas)
    lines = [
        "# P0627 universal OR-strength and route-phase atlas",
        "",
        f"Lexicographic diagnostic selection: **beta={selected.beta:g}, phase={selected.phase_degrees:+g} deg**.",
        "",
        f"Galaxy CV gain: **{100*selected.galaxy_improvement_fraction:+.2f}%**; derived-cluster gain: **{100*selected.derived_cluster_improvement_fraction:+.2f}%**.",
        f"Raw roots: **{int(selected.roots)}/18**; route versus own scalar: **{100*selected.improvement_vs_own_scalar_fraction:+.3f}%**; versus original P0618 scalar: **{100*selected.improvement_vs_original_P0618_scalar_fraction:+.3f}%**.",
        f"Full spent-data diagnostic rule passers: **{len(passers)}**.",
        "",
        "This is an opened-data atlas with two-dimensional multiplicity. A selected pair must be frozen before any future system.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
