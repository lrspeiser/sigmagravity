#!/usr/bin/env python3
"""P0626: compact scalar survival plus the frozen universal angular route."""

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
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import raw_contexts  # noqa: E402
from run_p0615_self_coupled_quadrupole_route import derived_state  # noqa: E402
from run_p0617_self_coupled_support_phase_atlas import (  # noqa: E402
    contexts_and_frozen_geometry,
    lens_score,
)
from run_p0618_universal_route_phase import phase_field  # noqa: E402
from run_p0623_density_path_survival import (  # noqa: E402
    A0,
    G_SI,
    KPC_M,
    M_SUN_KG,
    predict_velocity,
    safe_positive,
    score_arrays,
)
from run_p0625_bounded_porosity_survival import (  # noqa: E402
    apply_record,
    fit_record,
    prepare_frame,
    solar_transfer,
)
from voidscreen.arc_invariants import generalized_arc_response, spherical_profile_invariants  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


ATOMIC_IDS = [
    "constant",
    "inverse_hill0_m1__potential_depth",
    "inverse_hillfloor_m2__mean_surface_R80",
]
SCALAR_VARIANTS = ["constant", "OR_direct", "OR_compact30", "OR_compact100"]


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


def compact_weight(r80_kpc, coherence_kpc: float) -> np.ndarray:
    return 1.0 / (1.0 + np.square(np.asarray(r80_kpc, dtype=float) / coherence_kpc))


def variant_q(q_constant, q_potential, q_surface, r80_kpc):
    q_or = np.maximum(q_potential, q_surface)
    q_constant = np.broadcast_to(np.asarray(q_constant, dtype=float), q_or.shape)
    return {
        "constant": q_constant,
        "OR_direct": q_or,
        "OR_compact30": q_constant + compact_weight(r80_kpc, 30.0) * (q_or - q_constant),
        "OR_compact100": q_constant + compact_weight(r80_kpc, 100.0) * (q_or - q_constant),
    }


def fit_atomic(frame, lookup):
    return {candidate_id: fit_record(frame, lookup[candidate_id]) for candidate_id in ATOMIC_IDS}


def frame_q(records, frame):
    q_constant = apply_record(records["constant"], np.ones(len(frame)))
    q_potential = apply_record(
        records["inverse_hill0_m1__potential_depth"], frame.potential_depth.to_numpy(float)
    )
    surface = frame.force_equivalent_mass_solar.to_numpy(float) / (
        np.pi * np.square(frame.force_equivalent_r80_kpc.to_numpy(float))
    )
    q_surface = apply_record(records["inverse_hillfloor_m2__mean_surface_R80"], surface)
    return variant_q(q_constant, q_potential, q_surface, frame.force_equivalent_r80_kpc)


def galaxy_scores(frame, lookup, folds):
    rows = []
    for fold in folds:
        train = frame[frame.galaxy_fold.isin(folds) & frame.galaxy_fold.ne(fold)]
        test = frame[frame.galaxy_fold.eq(fold)]
        records = fit_atomic(train, lookup)
        for variant_id, q_eff in frame_q(records, test).items():
            metrics = score_arrays(test, predict_velocity(test, q_eff))
            rows.append(
                {
                    "variant_id": variant_id,
                    "heldout_fold": fold,
                    "equal_galaxy_RMSE_km_s": metrics["equal_galaxy_RMSE_km_s"],
                    "pooled_RMSE_km_s": metrics["pooled_RMSE_km_s"],
                    "q_min": np.min(q_eff),
                    "q_median": np.median(q_eff),
                    "q_max": np.max(q_eff),
                }
            )
    fold_frame = pd.DataFrame(rows)
    baseline = fold_frame[fold_frame.variant_id.eq("constant")][
        ["heldout_fold", "equal_galaxy_RMSE_km_s"]
    ].rename(columns={"equal_galaxy_RMSE_km_s": "baseline_RMSE_km_s"})
    fold_frame = fold_frame.merge(baseline, on="heldout_fold", validate="many_to_one")
    fold_frame["fold_improvement_fraction"] = (
        1.0 - fold_frame.equal_galaxy_RMSE_km_s / fold_frame.baseline_RMSE_km_s
    )
    scores = fold_frame.groupby("variant_id", sort=False).agg(
        mean_MSE=("equal_galaxy_RMSE_km_s", lambda x: np.mean(np.square(x))),
        pooled_MSE=("pooled_RMSE_km_s", lambda x: np.mean(np.square(x))),
        fold_wins=("fold_improvement_fraction", lambda x: int(np.sum(np.asarray(x) > 0.0))),
        q_min=("q_min", "min"),
        q_median=("q_median", "median"),
        q_max=("q_max", "max"),
    ).reset_index()
    scores["cv_equal_galaxy_RMSE_km_s"] = np.sqrt(scores.pop("mean_MSE"))
    scores["cv_pooled_RMSE_km_s"] = np.sqrt(scores.pop("pooled_MSE"))
    baseline_rmse = float(scores.loc[scores.variant_id.eq("constant"), "cv_equal_galaxy_RMSE_km_s"].iloc[0])
    scores["improvement_vs_constant_fraction"] = (
        1.0 - scores.cv_equal_galaxy_RMSE_km_s / baseline_rmse
    )
    return scores, fold_frame


def derived_scores(parent, spec, records):
    clusters, _ = prepare_clusters(parent)
    response = response_for_frame(
        clusters,
        spec,
        q=1.0,
        a0=A0,
        radius_column="radius_kpc",
        gbar_column="gbar_m_s2",
    )
    unit = response["unit_fractional_response"]
    q_values = frame_q(records, clusters)
    rows = []
    photon = float(spec["photon_extra_multiplier"])
    for variant_id, q_eff in q_values.items():
        prediction = clusters.gbar_m_s2.to_numpy(float) * (1.0 + photon * q_eff * unit)
        rows.append(
            {
                "variant_id": variant_id,
                "q_min": np.min(q_eff),
                "q_median": np.median(q_eff),
                "q_max": np.max(q_eff),
                **cluster_score(clusters, prediction),
            }
        )
    scores = pd.DataFrame(rows)
    baseline = float(scores.loc[scores.variant_id.eq("constant"), "cluster_equal_system_RMSE_dex"].iloc[0])
    scores["improvement_vs_constant_fraction"] = (
        1.0 - scores.cluster_equal_system_RMSE_dex / baseline
    )
    return scores


def solar_scores(spec, records):
    atomic_plus_or = solar_transfer(
        spec,
        records,
        [
            {
                "candidate_id": "OR_direct",
                "components": [
                    "inverse_hill0_m1__potential_depth",
                    "inverse_hillfloor_m2__mean_surface_R80",
                ],
                "operator": "maximum",
            }
        ],
    )
    selected = atomic_plus_or[atomic_plus_or.candidate_id.isin(["constant", "OR_direct"])].copy()
    direct = selected[selected.candidate_id.eq("OR_direct")].iloc[0].to_dict()
    for label in ("OR_compact30", "OR_compact100"):
        row = dict(direct)
        row["candidate_id"] = label
        selected = pd.concat([selected, pd.DataFrame([row])], ignore_index=True)
    return selected.rename(columns={"candidate_id": "variant_id"})


def build_scalar_fields(spec, records, anchors, raw_protocol):
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
    surface = total / (np.pi * r80**2)
    q_surface = apply_record(
        records["inverse_hillfloor_m2__mean_surface_R80"], np.full_like(radius_grid, surface)
    )
    q_values = variant_q(q_constant, q_potential, q_surface, np.full_like(radius_grid, r80))
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
    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    fields = {}
    for variant_id, q_eff in q_values.items():
        acceleration = gbar * (1.0 + photon * q_eff * response["unit_fractional_response"])

        def lookup(radius, local_acceleration=acceleration):
            return np.exp(
                np.interp(np.log(radius), np.log(radius_grid), np.log(local_acceleration))
            )

        alpha = spherical_deflection_radians(
            impact_arcsec * scale,
            lookup,
            maximum_radius_kpc=1.0e6,
            integration_points=800,
        )
        fields[variant_id] = RadialDeflectionField(impact_arcsec, alpha)
    return fields, r80, q_values


def raw_scores(spec, records):
    p0615 = load_json("configs/p0615_self_coupled_quadrupole_route_protocol.json")
    p0581 = load_json(p0615["inputs"]["P0581_protocol"])
    prepared = contexts_and_frozen_geometry(p0615)
    p0554 = load_json("configs/p0554_local_cross_domain_sensitivity_protocol.json")
    anchor_map = {context.label: context.anchors for context in raw_contexts(p0554)}
    rows = []
    for context, cohort, parameters, sources in prepared:
        label = context.system["label"]
        anchors = context.anchors if hasattr(context, "anchors") else anchor_map[label]
        fields, r80, q_values = build_scalar_fields(spec, records, anchors, context.local)
        for scalar_id, parent_field in fields.items():
            local_context = SimpleNamespace(**context.__dict__)
            local_context.parent = parent_field
            scalar = lens_score(local_context, parameters, sources, None, 0.0)
            state = derived_state(local_context)
            epsilon = float(state["amplitudes"]["quadratic_Q2_over_total"])
            route_field, _ = phase_field(p0581, local_context, state, 90.0)
            routed = lens_score(local_context, parameters, sources, route_field, epsilon)
            rows.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "scalar_variant": scalar_id,
                    "variant_id": f"{scalar_id}_scalar",
                    "route_applied": False,
                    "R80_kpc": r80,
                    "compact_weight_30": float(compact_weight(r80, 30.0)),
                    "compact_weight_100": float(compact_weight(r80, 100.0)),
                    "epsilon": 0.0,
                    **scalar,
                }
            )
            rows.append(
                {
                    "cohort": cohort,
                    "system_label": label,
                    "scalar_variant": scalar_id,
                    "variant_id": f"{scalar_id}_plus_route",
                    "route_applied": True,
                    "R80_kpc": r80,
                    "compact_weight_30": float(compact_weight(r80, 30.0)),
                    "compact_weight_100": float(compact_weight(r80, 100.0)),
                    "epsilon": epsilon,
                    **routed,
                }
            )
        print(f"P0626 raw {label}: {2 * len(fields)} variants", flush=True)
    return pd.DataFrame(rows)


def raw_summary(raw):
    rows = []
    baseline = raw[raw.variant_id.eq("constant_scalar")].set_index("system_label")
    for variant_id, block in raw.groupby("variant_id"):
        complete = block[block.heldout_all_roots.astype(bool)]
        common = block[
            block.heldout_all_roots.astype(bool)
            & block.system_label.map(baseline.heldout_all_roots.astype(bool))
        ]
        base_common = baseline.loc[common.system_label]
        rms = float(np.sqrt(np.mean(np.square(common.heldout_RMS_arcsec)))) if len(common) else np.nan
        base_rms = (
            float(np.sqrt(np.mean(np.square(base_common.heldout_RMS_arcsec))))
            if len(common)
            else np.nan
        )
        rows.append(
            {
                "variant_id": variant_id,
                "roots": int(block.heldout_converged_roots.sum()),
                "images": int(block.heldout_images.sum()),
                "complete_systems": len(complete),
                "common_systems_vs_constant": len(common),
                "common_RMS_arcsec": rms,
                "constant_common_RMS_arcsec": base_rms,
                "improvement_vs_constant_fraction": 1.0 - rms / base_rms if len(common) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    scalar_lookup = result.set_index("variant_id")
    changes = []
    for row in result.itertuples():
        if row.variant_id.endswith("_plus_route"):
            scalar_id = row.variant_id.replace("_plus_route", "_scalar")
            scalar = scalar_lookup.loc[scalar_id]
            if row.complete_systems == 5 and scalar.complete_systems == 5:
                changes.append(1.0 - row.common_RMS_arcsec / scalar.common_RMS_arcsec)
            else:
                changes.append(np.nan)
        else:
            changes.append(np.nan)
    result["route_improvement_vs_own_scalar_fraction"] = changes
    return result


def make_figure(output, galaxy, derived, raw_summary_frame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(galaxy.variant_id, 100 * galaxy.improvement_vs_constant_fraction)
    axes[0].set_title("Galaxy development CV")
    axes[0].set_ylabel("improvement (%)")
    axes[0].tick_params(axis="x", rotation=60)
    axes[1].bar(derived.variant_id, 100 * derived.improvement_vs_constant_fraction)
    axes[1].set_title("20 derived clusters")
    axes[1].tick_params(axis="x", rotation=60)
    axes[2].bar(raw_summary_frame.variant_id, raw_summary_frame.roots)
    axes[2].axhline(18, color="black", linestyle="--")
    axes[2].set_title("Raw roots")
    axes[2].tick_params(axis="x", rotation=75, labelsize=6)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    protocol = load_json("configs/p0626_compact_scalar_angular_route_protocol.json")
    p0625 = load_json(protocol["parent_protocols"][0])
    p0623 = load_json(p0625["parent_protocols"][0])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    frame, lookup, spec, parent = prepare_frame(p0623)
    folds = p0623["sample"]["development_galaxy_folds"]
    galaxy, _ = galaxy_scores(frame, lookup, folds)
    galaxy.to_csv(output / protocol["outputs"]["galaxy"], index=False)
    development = frame[frame.galaxy_fold.isin(folds)]
    records = fit_atomic(development, lookup)
    derived = derived_scores(parent, spec, records)
    derived.to_csv(output / protocol["outputs"]["derived_cluster"], index=False)
    solar = solar_scores(spec, records)
    solar.to_csv(output / protocol["outputs"]["solar"], index=False)
    raw = raw_scores(spec, records)
    raw.to_csv(output / protocol["outputs"]["raw"], index=False)
    raw_aggregate = raw_summary(raw)

    candidate_rows = []
    for scalar_id in SCALAR_VARIANTS:
        grow = galaxy[galaxy.variant_id.eq(scalar_id)].iloc[0]
        drow = derived[derived.variant_id.eq(scalar_id)].iloc[0]
        srow = solar[solar.variant_id.eq(scalar_id)].iloc[0]
        for route in (False, True):
            raw_id = f"{scalar_id}_{'plus_route' if route else 'scalar'}"
            rrow = raw_aggregate[raw_aggregate.variant_id.eq(raw_id)].iloc[0]
            scalar_viable = bool(
                grow.fold_wins >= 3
                and srow.all_solar_proxies_pass
                and drow.improvement_vs_constant_fraction >= 0.0
            )
            route_viable = bool(
                route
                and rrow.roots == 18
                and np.isfinite(rrow.route_improvement_vs_own_scalar_fraction)
                and rrow.route_improvement_vs_own_scalar_fraction >= 0.0
            )
            candidate_rows.append(
                {
                    "variant_id": raw_id,
                    "scalar_variant": scalar_id,
                    "route_applied": route,
                    "galaxy_improvement_fraction": grow.improvement_vs_constant_fraction,
                    "galaxy_fold_wins": int(grow.fold_wins),
                    "derived_cluster_improvement_fraction": drow.improvement_vs_constant_fraction,
                    "solar_pass": bool(srow.all_solar_proxies_pass),
                    "raw_roots": int(rrow.roots),
                    "raw_complete_systems": int(rrow.complete_systems),
                    "raw_improvement_vs_constant_fraction": rrow.improvement_vs_constant_fraction,
                    "route_improvement_vs_own_scalar_fraction": rrow.route_improvement_vs_own_scalar_fraction,
                    "scalar_viable": scalar_viable,
                    "route_viable": route_viable,
                    "unified_diagnostic_pass": bool(
                        scalar_viable
                        and route_viable
                        and rrow.improvement_vs_constant_fraction >= 0.0
                    ),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete",
        "candidates": strict_json(candidates.to_dict(orient="records")),
        "unified_diagnostic_passers": candidates[
            candidates.unified_diagnostic_pass
        ].variant_id.tolist(),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    make_figure(output / protocol["outputs"]["figure"], galaxy, derived, raw_aggregate)
    lines = [
        "# P0626 compact scalar plus angular route",
        "",
        "| Variant | Galaxy gain | Derived gain | Solar | Raw roots | Raw gain | Route vs own scalar | Unified diagnostic |",
        "|---|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for row in candidates.itertuples():
        route_change = (
            f"{100*row.route_improvement_vs_own_scalar_fraction:+.2f}%"
            if np.isfinite(row.route_improvement_vs_own_scalar_fraction)
            else "n/a"
        )
        lines.append(
            f"| `{row.variant_id}` | {100*row.galaxy_improvement_fraction:+.2f}% | "
            f"{100*row.derived_cluster_improvement_fraction:+.2f}% | "
            f"{'pass' if row.solar_pass else 'fail'} | {row.raw_roots}/18 | "
            f"{100*row.raw_improvement_vs_constant_fraction:+.2f}% | {route_change} | "
            f"{'pass' if row.unified_diagnostic_pass else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "All systems are project-spent and the angular phase was selected earlier on these lenses.",
        ]
    )
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(candidates.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
