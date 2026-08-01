#!/usr/bin/env python3
"""P0625: bounded pair survival and deep-or-porous composite tests."""

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
    prepare_galaxies,
    response_for_frame,
    response_parameters,
)
from run_p0554_local_cross_domain_sensitivity import RawLens, raw_contexts, raw_score  # noqa: E402
from run_p0617_self_coupled_support_phase_atlas import contexts_and_frozen_geometry  # noqa: E402
from run_p0623_density_path_survival import (  # noqa: E402
    A0,
    G_SI,
    KPC_M,
    M_SUN_KG,
    build_candidates,
    build_feature_frame,
    fit_candidate,
    pair_proximity,
    predict_velocity,
    q_from_parameters,
    safe_positive,
    score_arrays,
)
from run_p0624_deep_porous_cross_domain import (  # noqa: E402
    SOLAR_RADIUS_M,
    member_pair_feature,
    smooth_cluster_feature,
    solar_feature,
)
from voidscreen.arc_apogee import AU_M, JULIAN_YEAR_DAYS, RAD_TO_MAS  # noqa: E402
from voidscreen.arc_invariants import C_M_S, generalized_arc_response, spherical_profile_invariants  # noqa: E402
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


def prepare_frame(p0623: dict):
    parent = load_json(p0623["inputs"]["SPARC_parent_protocol"])
    points, _ = prepare_galaxies(parent, A0)
    outer = points[points.split.eq("outer_holdout")].copy()
    p0554 = load_json(p0623["inputs"]["P0554_protocol"])
    spec = dict(p0554["baseline"])
    spec.pop("universal_q")
    spec["candidate_id"] = "P0625_unit_parent"
    response = response_for_frame(
        outer,
        spec,
        q=1.0,
        a0=A0,
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
    )
    outer["unit_P0554_response"] = response["unit_fractional_response"]
    morphology = pd.read_csv(ROOT / p0623["inputs"]["SPARC_morphology"])
    frame, _, _, feature_columns = build_feature_frame(outer, morphology, p0623)
    lookup = {candidate.candidate_id: candidate for candidate in build_candidates(feature_columns, p0623)}
    return frame, lookup, spec, parent


def fit_record(frame: pd.DataFrame, candidate):
    if candidate.family == "constant":
        center, scale = np.nan, np.nan
        z = np.zeros(len(frame))
    else:
        values = np.log10(safe_positive(frame[candidate.feature]))
        center = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float(q75 - q25)
        if scale < 1.0e-6:
            scale = float(np.std(values)) or 1.0
        z = (values - center) / scale
    parameters = fit_candidate(frame, candidate, z)
    return {
        "candidate": candidate,
        "center": center,
        "scale": scale,
        "parameters": parameters,
    }


def apply_record(record: dict, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    candidate = record["candidate"]
    if candidate.family == "constant":
        z = np.zeros(values.shape)
    else:
        z = (np.log10(safe_positive(values)) - record["center"]) / record["scale"]
    return q_from_parameters(candidate, record["parameters"], z)


def record_q(record: dict, frame: pd.DataFrame) -> np.ndarray:
    feature = record["candidate"].feature
    values = np.ones(len(frame)) if feature == "none" else frame[feature].to_numpy(float)
    return apply_record(record, values)


def combine(operator: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if operator == "maximum":
        return np.maximum(left, right)
    if operator == "root_mean_square":
        return np.sqrt((np.square(left) + np.square(right)) / 2.0)
    raise ValueError(operator)


def all_q(records: dict, composites: list[dict], value_provider) -> dict[str, np.ndarray]:
    q = {candidate_id: value_provider(candidate_id, record) for candidate_id, record in records.items()}
    for item in composites:
        left, right = item["components"]
        q[item["candidate_id"]] = combine(item["operator"], q[left], q[right])
    return q


def galaxy_cv(frame, atomic_ids, composites, lookup, development_folds):
    fold_rows = []
    for fold in development_folds:
        train = frame[frame.galaxy_fold.isin(development_folds) & frame.galaxy_fold.ne(fold)]
        test = frame[frame.galaxy_fold.eq(fold)]
        records = {candidate_id: fit_record(train, lookup[candidate_id]) for candidate_id in atomic_ids}
        q_values = all_q(records, composites, lambda _, record: record_q(record, test))
        for candidate_id, q_eff in q_values.items():
            metrics = score_arrays(test, predict_velocity(test, q_eff))
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "heldout_fold": fold,
                    "equal_galaxy_RMSE_km_s": metrics["equal_galaxy_RMSE_km_s"],
                    "pooled_RMSE_km_s": metrics["pooled_RMSE_km_s"],
                    "mean_residual_km_s": metrics["mean_residual_km_s"],
                    "q_min": float(np.min(q_eff)),
                    "q_median": float(np.median(q_eff)),
                    "q_max": float(np.max(q_eff)),
                }
            )
    folds = pd.DataFrame(fold_rows)
    baseline = folds[folds.candidate_id.eq("constant")][
        ["heldout_fold", "equal_galaxy_RMSE_km_s"]
    ].rename(columns={"equal_galaxy_RMSE_km_s": "baseline_RMSE_km_s"})
    folds = folds.merge(baseline, on="heldout_fold", validate="many_to_one")
    folds["fold_improvement_fraction"] = 1.0 - folds.equal_galaxy_RMSE_km_s / folds.baseline_RMSE_km_s
    scores = folds.groupby("candidate_id", sort=False).agg(
        mean_MSE=("equal_galaxy_RMSE_km_s", lambda x: np.mean(np.square(x))),
        pooled_MSE=("pooled_RMSE_km_s", lambda x: np.mean(np.square(x))),
        fold_wins=("fold_improvement_fraction", lambda x: int(np.sum(np.asarray(x) > 0.0))),
        mean_fold_improvement_fraction=("fold_improvement_fraction", "mean"),
        q_min=("q_min", "min"),
        q_median=("q_median", "median"),
        q_max=("q_max", "max"),
    ).reset_index()
    scores["cv_equal_galaxy_RMSE_km_s"] = np.sqrt(scores.pop("mean_MSE"))
    scores["cv_pooled_RMSE_km_s"] = np.sqrt(scores.pop("pooled_MSE"))
    baseline_rmse = float(scores.loc[scores.candidate_id.eq("constant"), "cv_equal_galaxy_RMSE_km_s"].iloc[0])
    scores["improvement_vs_constant_fraction"] = 1.0 - scores.cv_equal_galaxy_RMSE_km_s / baseline_rmse
    return scores.sort_values("cv_equal_galaxy_RMSE_km_s"), folds


def frozen_records(frame, atomic_ids, lookup, development_folds):
    development = frame[frame.galaxy_fold.isin(development_folds)]
    records = {candidate_id: fit_record(development, lookup[candidate_id]) for candidate_id in atomic_ids}
    rows = []
    for candidate_id, record in records.items():
        q_eff = record_q(record, development)
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": record["candidate"].family,
                "feature": record["candidate"].feature,
                "parameters_json": json.dumps(record["parameters"].tolist()),
                "feature_log10_center": record["center"],
                "feature_log10_IQR": record["scale"],
                "development_q_min": np.min(q_eff),
                "development_q_median": np.median(q_eff),
                "development_q_max": np.max(q_eff),
            }
        )
    return records, pd.DataFrame(rows)


def derived_transfer(parent, spec, records, composites):
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
    q_values = all_q(
        records,
        composites,
        lambda _, record: apply_record(
            record,
            np.ones(len(clusters))
            if record["candidate"].feature == "none"
            else smooth_cluster_feature(clusters, record["candidate"].feature),
        ),
    )
    rows = []
    photon = float(spec["photon_extra_multiplier"])
    for candidate_id, q_eff in q_values.items():
        prediction = clusters.gbar_m_s2.to_numpy(float) * (1.0 + photon * q_eff * unit)
        rows.append(
            {
                "candidate_id": candidate_id,
                "q_min": np.min(q_eff),
                "q_median": np.median(q_eff),
                "q_max": np.max(q_eff),
                **cluster_score(clusters, prediction),
            }
        )
    result = pd.DataFrame(rows)
    baseline = float(result.loc[result.candidate_id.eq("constant"), "cluster_equal_system_RMSE_dex"].iloc[0])
    result["improvement_vs_constant_fraction"] = 1.0 - result.cluster_equal_system_RMSE_dex / baseline
    return result.sort_values("cluster_equal_system_RMSE_dex")


def solar_q_arrays(records, composites, radius_m):
    radius = np.asarray(radius_m, dtype=float)
    potential = G_SI * M_SUN_KG / radius / C_M_S**2
    return all_q(
        records,
        composites,
        lambda _, record: apply_record(
            record,
            solar_feature(record["candidate"].feature, radius, potential),
        ),
    )


def solar_transfer(spec, records, composites):
    def unit_fraction(radius_m):
        radius = np.asarray(radius_m, dtype=float)
        gbar = G_SI * M_SUN_KG / np.square(radius)
        response = generalized_arc_response(
            gbar,
            radius / KPC_M,
            np.ones_like(radius),
            np.ones_like(radius),
            potential_depth=G_SI * M_SUN_KG / radius / C_M_S**2,
            potential_length_kpc=radius / KPC_M,
            potential_path_ratio=np.ones_like(radius),
            enclosed_mass_log_slope=np.zeros_like(radius),
            **response_parameters(spec, q=1.0, a0=A0),
        )
        return response["fractional_dynamical_response"]

    grid = np.geomspace(1.6 * SOLAR_RADIUS_M, 8.43 * AU_M, 1000)
    unit_grid = unit_fraction(grid)
    grid_q = solar_q_arrays(records, composites, grid)
    semimajor = 0.38709893 * AU_M
    eccentricity = 0.205630
    period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, 32768, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    mercury_radius = semimajor * one_minus_e2 / (1.0 + eccentricity * cosine)
    mercury_unit = unit_fraction(mercury_radius)
    mercury_q = solar_q_arrays(records, composites, mercury_radius)
    rows = []
    photon = float(spec["photon_extra_multiplier"])
    for candidate_id, q_eff in grid_q.items():
        dynamic = unit_grid * q_eff
        earth = float(np.interp(AU_M, grid, dynamic))
        saturn = float(np.interp(8.43 * AU_M, grid, dynamic))
        radial = -(G_SI * M_SUN_KG / np.square(mercury_radius)) * mercury_unit * mercury_q[candidate_id]
        time_weight = one_minus_e2**1.5 / np.square(1.0 + eccentricity * cosine)
        mean_r_cosine = float(np.mean(radial * cosine * time_weight))
        period_seconds = period_days * 86400.0
        mean_motion = 2.0 * np.pi / period_seconds
        mean_rate = -math.sqrt(one_minus_e2) / (mean_motion * semimajor * eccentricity) * mean_r_cosine
        mercury = mean_rate * period_seconds * (100.0 * JULIAN_YEAR_DAYS / period_days) * RAD_TO_MAS
        rows.append(
            {
                "candidate_id": candidate_id,
                "q_min": np.min(q_eff),
                "q_median": np.median(q_eff),
                "q_max": np.max(q_eff),
                "maximum_dynamic_fraction_limb_to_Saturn": np.max(dynamic),
                "maximum_lensing_fraction_limb_to_Saturn": np.max(photon * dynamic),
                "Earth_orbit_fractional_change": earth,
                "Saturn_orbit_fractional_change": saturn,
                "Mercury_precession_mas_per_century": mercury,
                "Cassini_proxy_pass": bool(np.max(photon * dynamic) <= 2.3e-5),
                "Earth_proxy_pass": bool(earth <= 1.0e-10),
                "Mercury_proxy_pass": bool(abs(mercury) <= 3.1),
            }
        )
    result = pd.DataFrame(rows)
    result["all_solar_proxies_pass"] = (
        result.Cassini_proxy_pass & result.Earth_proxy_pass & result.Mercury_proxy_pass
    )
    return result


def raw_feature(record, radius_grid, invariants, total, r80, member_context):
    feature = record["candidate"].feature
    if feature == "none":
        return np.ones_like(radius_grid)
    if feature == "potential_depth":
        return invariants["potential_depth"]
    if feature == "mean_surface_R80":
        return np.full_like(radius_grid, total / (np.pi * r80**2))
    if feature.startswith("pair_"):
        return np.full_like(radius_grid, member_pair_feature(member_context, total, feature))
    raise ValueError(feature)


def raw_fields(spec, records, composites, anchors, raw_protocol, member_context):
    radius_grid = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(radius_grid, anchor_radius, anchor_gbar, outer_slope=-2.0)
    invariants = spherical_profile_invariants(radius_grid, gbar)
    anchor_mass = anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    total = float(np.maximum.accumulate(anchor_mass)[-1])
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    q_values = all_q(
        records,
        composites,
        lambda _, record: apply_record(
            record, raw_feature(record, radius_grid, invariants, total, r80, member_context)
        ),
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
    fields = {}
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    photon = float(spec["photon_extra_multiplier"])
    for candidate_id, q_eff in q_values.items():
        acceleration = gbar * (1.0 + photon * q_eff * response["unit_fractional_response"])

        def lookup(radius, local_acceleration=acceleration):
            return np.exp(
                np.interp(np.log(radius), np.log(radius_grid), np.log(local_acceleration))
            )

        physical_alpha = spherical_deflection_radians(
            impact_arcsec * scale,
            lookup,
            maximum_radius_kpc=1.0e6,
            integration_points=800,
        )
        fields[candidate_id] = RadialDeflectionField(impact_arcsec, physical_alpha)
    return fields


def raw_transfer(spec, records, composites):
    p0554 = load_json("configs/p0554_local_cross_domain_sensitivity_protocol.json")
    contexts = raw_contexts(p0554)
    p0615 = load_json("configs/p0615_self_coupled_quadrupole_route_protocol.json")
    member_map = {
        context.system["label"]: context
        for context, _, _, _ in contexts_and_frozen_geometry(p0615)
    }
    rows = []
    for context in contexts:
        print(f"P0625 raw: {context.label}", flush=True)
        fields = raw_fields(
            spec,
            records,
            composites,
            context.anchors,
            context.local,
            member_map[context.label],
        )
        lens = RawLens(context.local, fields)
        for candidate_id in fields:
            _, sources = lens.profiled_residuals(candidate_id, context.geometry, context.training)
            train = lens.exact_predictions(
                candidate_id, context.geometry, sources, context.training, stage="training"
            )
            held = lens.exact_predictions(
                candidate_id, context.geometry, sources, context.heldout, stage="heldout"
            )
            train_metrics = raw_score(train, lens.sigma)
            held_metrics = raw_score(held, lens.sigma)
            rows.append(
                {
                    "system_label": context.label,
                    "candidate_id": candidate_id,
                    "training_RMS_arcsec": train_metrics["exact_radial_RMS_arcsec"],
                    "training_roots_converged": train_metrics["converged_roots"],
                    "heldout_images": len(context.heldout),
                    "heldout_RMS_arcsec": held_metrics["exact_radial_RMS_arcsec"],
                    "heldout_roots_converged": held_metrics["converged_roots"],
                    "heldout_all_roots": held_metrics["all_roots_converged"],
                }
            )
    result = pd.DataFrame(rows)
    baseline = result[result.candidate_id.eq("constant")].set_index("system_label")
    result["improvement_vs_constant_fraction"] = result.apply(
        lambda row: (
            1.0 - row.heldout_RMS_arcsec / baseline.loc[row.system_label].heldout_RMS_arcsec
            if row.heldout_all_roots
            and baseline.loc[row.system_label].heldout_all_roots
            and np.isfinite(row.heldout_RMS_arcsec)
            and np.isfinite(baseline.loc[row.system_label].heldout_RMS_arcsec)
            else np.nan
        ),
        axis=1,
    )
    return result


def figure(output, galaxy, derived, solar, raw):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    show = galaxy.sort_values("improvement_vs_constant_fraction").tail(12)
    axes[0, 0].barh(show.candidate_id, 100 * show.improvement_vs_constant_fraction)
    axes[0, 0].set_title("Galaxy development CV")
    axes[0, 0].set_xlabel("improvement vs constant (%)")
    show = derived.sort_values("improvement_vs_constant_fraction").tail(12)
    axes[0, 1].barh(show.candidate_id, 100 * show.improvement_vs_constant_fraction)
    axes[0, 1].set_title("20 derived clusters")
    axes[0, 1].set_xlabel("improvement vs constant (%)")
    axes[1, 0].bar(solar.candidate_id, np.abs(solar.Mercury_precession_mas_per_century))
    axes[1, 0].axhline(3.1, color="red", linestyle="--")
    axes[1, 0].tick_params(axis="x", rotation=75, labelsize=6)
    axes[1, 0].set_title("Solar proxy")
    roots = raw.groupby("candidate_id").agg(roots=("heldout_roots_converged", "sum"))
    axes[1, 1].bar(roots.index, roots.roots)
    axes[1, 1].axhline(18, color="black", linestyle="--")
    axes[1, 1].tick_params(axis="x", rotation=75, labelsize=6)
    axes[1, 1].set_title("Raw roots recovered (of 18)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    protocol = load_json("configs/p0625_bounded_porosity_survival_protocol.json")
    p0623 = load_json(protocol["parent_protocols"][0])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    frame, lookup, spec, parent = prepare_frame(p0623)
    atomic_ids = protocol["atomic_candidates"]
    composites = protocol["composite_candidates"]
    development_folds = p0623["sample"]["development_galaxy_folds"]
    galaxy, folds = galaxy_cv(frame, atomic_ids, composites, lookup, development_folds)
    galaxy.to_csv(output / protocol["outputs"]["galaxy_cv"], index=False)
    folds.to_csv(output / protocol["outputs"]["galaxy_folds"], index=False)
    records, parameters = frozen_records(frame, atomic_ids, lookup, development_folds)
    parameters.to_csv(output / protocol["outputs"]["frozen_parameters"], index=False)
    derived = derived_transfer(parent, spec, records, composites)
    derived.to_csv(output / protocol["outputs"]["derived_cluster"], index=False)
    solar = solar_transfer(spec, records, composites)
    solar.to_csv(output / protocol["outputs"]["solar"], index=False)
    raw = raw_transfer(spec, records, composites)
    raw.to_csv(output / protocol["outputs"]["raw"], index=False)

    constant_roots = int(raw[raw.candidate_id.eq("constant")].heldout_roots_converged.sum())
    summary_rows = []
    for candidate_id in galaxy.candidate_id:
        grow = galaxy[galaxy.candidate_id.eq(candidate_id)].iloc[0]
        drow = derived[derived.candidate_id.eq(candidate_id)].iloc[0]
        srow = solar[solar.candidate_id.eq(candidate_id)].iloc[0]
        rblock = raw[raw.candidate_id.eq(candidate_id)]
        roots = int(rblock.heldout_roots_converged.sum())
        viable = bool(
            grow.fold_wins >= 3
            and srow.all_solar_proxies_pass
            and roots >= constant_roots
            and drow.improvement_vs_constant_fraction >= 0.0
        )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "galaxy_cv_improvement_fraction": grow.improvement_vs_constant_fraction,
                "galaxy_fold_wins": int(grow.fold_wins),
                "derived_cluster_improvement_fraction": drow.improvement_vs_constant_fraction,
                "solar_pass": bool(srow.all_solar_proxies_pass),
                "raw_roots": roots,
                "raw_images": int(rblock.heldout_images.sum()),
                "raw_complete_systems": int(rblock.heldout_all_roots.sum()),
                "cross_domain_diagnostic_gate": viable,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["cross_domain_diagnostic_gate", "galaxy_cv_improvement_fraction"], ascending=False
    )
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete",
        "candidates": strict_json(summary.to_dict(orient="records")),
        "diagnostic_gate_passers": summary[summary.cross_domain_diagnostic_gate].candidate_id.tolist(),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    figure(output / protocol["outputs"]["figure"], galaxy, derived, solar, raw)
    lines = [
        "# P0625 bounded porosity survival",
        "",
        "This stage replaces the failed unbounded pair extrapolation with bounded laws and tests parameter-free OR combinations.",
        "",
        "| Candidate | Galaxy CV gain | Fold wins | Derived cluster gain | Solar | Raw roots | Diagnostic gate |",
        "|---|---:|---:|---:|:---:|---:|:---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| `{row.candidate_id}` | {100*row.galaxy_cv_improvement_fraction:+.2f}% | "
            f"{row.galaxy_fold_wins}/4 | {100*row.derived_cluster_improvement_fraction:+.2f}% | "
            f"{'pass' if row.solar_pass else 'fail'} | {row.raw_roots}/{row.raw_images} | "
            f"{'pass' if row.cross_domain_diagnostic_gate else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The gate is a project-spent diagnostic, not independent validation or a theory claim.",
        ]
    )
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
