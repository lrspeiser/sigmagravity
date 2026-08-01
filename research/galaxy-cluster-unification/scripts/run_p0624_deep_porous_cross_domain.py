#!/usr/bin/env python3
"""P0624: transfer frozen density/path-survival laws to clusters and the Sun."""

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

from run_arc_apogee_cross_domain import radius_at_mass_fraction  # noqa: E402
from run_arc_invariant_absolute_lensing import (  # noqa: E402
    cluster_score,
    prepare_clusters,
    prepare_galaxies,
    response_for_frame,
    response_parameters,
)
from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    RawLens,
    raw_contexts,
    raw_score,
)
from run_p0617_self_coupled_support_phase_atlas import (  # noqa: E402
    contexts_and_frozen_geometry,
)
from run_p0623_density_path_survival import (  # noqa: E402
    A0,
    G_SI,
    KPC_M,
    M_SUN_KG,
    build_candidates,
    build_feature_frame,
    fit_candidate,
    pair_proximity,
    q_from_parameters,
    safe_positive,
)
from voidscreen.arc_apogee import (  # noqa: E402
    AU_M,
    JULIAN_YEAR_DAYS,
    RAD_TO_MAS,
)
from voidscreen.arc_invariants import (  # noqa: E402
    C_M_S,
    generalized_arc_response,
    spherical_profile_invariants,
)
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


SOLAR_RADIUS_M = 6.957e8


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


def prepare_galaxy_development(p0623: dict):
    parent = load_json(p0623["inputs"]["SPARC_parent_protocol"])
    points, _ = prepare_galaxies(parent, A0)
    outer = points[points.split.eq("outer_holdout")].copy()
    p0554 = load_json(p0623["inputs"]["P0554_protocol"])
    spec = dict(p0554["baseline"])
    spec.pop("universal_q")
    spec["candidate_id"] = "P0624_unit_parent"
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
    development = frame[frame.galaxy_fold.isin(p0623["sample"]["development_galaxy_folds"])].copy()
    candidates = build_candidates(feature_columns, p0623)
    return development, {candidate.candidate_id: candidate for candidate in candidates}, spec


def fit_frozen_candidates(development, candidate_lookup, p0624):
    records = []
    frozen = {}
    for item in p0624["frozen_candidates"]:
        candidate = candidate_lookup[item["candidate_id"]]
        if candidate.family == "constant":
            z = np.zeros(len(development))
            center, scale = np.nan, np.nan
        else:
            log_feature = np.log10(safe_positive(development[candidate.feature]))
            center = float(np.median(log_feature))
            q25, q75 = np.quantile(log_feature, [0.25, 0.75])
            scale = float(q75 - q25)
            if scale < 1.0e-6:
                scale = float(np.std(log_feature)) or 1.0
            z = (log_feature - center) / scale
        parameters = fit_candidate(development, candidate, z)
        q_eff = q_from_parameters(candidate, parameters, z)
        record = {
            "candidate_id": candidate.candidate_id,
            "role": item["role"],
            "family": candidate.family,
            "feature": candidate.feature,
            "fixed_slope": candidate.slope,
            "parameter_count": candidate.parameter_count,
            "parameters_json": json.dumps([float(value) for value in parameters]),
            "feature_log10_center": center,
            "feature_log10_IQR": scale,
            "development_q_min": float(np.min(q_eff)),
            "development_q_median": float(np.median(q_eff)),
            "development_q_max": float(np.max(q_eff)),
        }
        records.append(record)
        frozen[candidate.candidate_id] = {
            "candidate": candidate,
            "parameters": parameters,
            "center": center,
            "scale": scale,
            "role": item["role"],
        }
    return pd.DataFrame(records), frozen


def apply_frozen(record: dict, feature_values) -> np.ndarray:
    feature_values = np.asarray(feature_values, dtype=float)
    if record["candidate"].family == "constant":
        z = np.zeros(feature_values.shape)
    else:
        z = (np.log10(safe_positive(feature_values)) - record["center"]) / record["scale"]
    return q_from_parameters(record["candidate"], record["parameters"], z)


def smooth_cluster_feature(frame: pd.DataFrame, feature: str) -> np.ndarray | None:
    mass = frame.force_equivalent_mass_solar.to_numpy(float)
    r80 = frame.force_equivalent_r80_kpc.to_numpy(float)
    if feature == "potential_depth":
        return frame.potential_depth.to_numpy(float)
    if feature == "mean_surface_R80":
        return mass / (np.pi * r80**2)
    if feature == "mean_volume_R80":
        return mass / ((4.0 / 3.0) * np.pi * r80**3)
    if feature == "acceleration_R80":
        return G_SI * mass * M_SUN_KG / np.square(r80 * KPC_M)
    if feature == "baryonic_mass":
        return mass
    if feature == "R80":
        return r80
    if feature in ("pair_surface_L30p0kpc", "pair_count_L30p0kpc"):
        kernel = 30.0
        scale = r80 / 3.0
        proximity = np.asarray(
            [pair_proximity(np.asarray([1.0]), np.asarray([value]), kernel) for value in scale]
        )
        if feature.startswith("pair_surface"):
            return mass * proximity / (2.0 * np.pi * kernel**2)
        return mass**2 * proximity
    return None


def derived_cluster_transfer(parent, spec, frozen):
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
    photon = float(spec["photon_extra_multiplier"])
    rows, predictions = [], []
    for candidate_id, record in frozen.items():
        feature = record["candidate"].feature
        values = np.ones(len(clusters)) if feature == "none" else smooth_cluster_feature(clusters, feature)
        if values is None:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "role": record["role"],
                    "available": False,
                    "unavailable_reason": "no domain-consistent derived-cluster feature",
                }
            )
            continue
        q_eff = apply_frozen(record, values)
        prediction = clusters.gbar_m_s2.to_numpy(float) * (1.0 + photon * q_eff * unit)
        metrics = cluster_score(clusters, prediction)
        rows.append(
            {
                "candidate_id": candidate_id,
                "role": record["role"],
                "available": True,
                "unavailable_reason": "",
                "q_min": float(np.min(q_eff)),
                "q_median": float(np.median(q_eff)),
                "q_max": float(np.max(q_eff)),
                **metrics,
            }
        )
        local = clusters[["system", "radius_kpc", "gbar_m_s2", "observed_g_m_s2"]].copy()
        local["candidate_id"] = candidate_id
        local["feature_value"] = values
        local["q_eff"] = q_eff
        local["predicted_g_m_s2"] = prediction
        predictions.append(local)
    scores = pd.DataFrame(rows)
    if predictions:
        prediction_frame = pd.concat(predictions, ignore_index=True)
    else:
        prediction_frame = pd.DataFrame()
    baseline = float(
        scores.loc[scores.candidate_id.eq("constant"), "cluster_equal_system_RMSE_dex"].iloc[0]
    )
    scores["improvement_vs_constant_fraction"] = np.where(
        scores.available.astype(bool),
        1.0 - scores.cluster_equal_system_RMSE_dex / baseline,
        np.nan,
    )
    return scores, prediction_frame


def solar_feature(candidate_feature: str, radius_m: np.ndarray, potential_depth: np.ndarray):
    if candidate_feature == "none":
        return np.ones_like(radius_m)
    if candidate_feature == "potential_depth":
        return potential_depth
    solar_r80_kpc = 0.8 * SOLAR_RADIUS_M / KPC_M
    if candidate_feature == "mean_surface_R80":
        return np.full_like(radius_m, 1.0 / (np.pi * solar_r80_kpc**2))
    if candidate_feature == "pair_surface_L30p0kpc":
        proximity = pair_proximity(
            np.asarray([1.0]), np.asarray([solar_r80_kpc / 3.0]), 30.0
        )
        return np.full_like(radius_m, proximity / (2.0 * np.pi * 30.0**2))
    if candidate_feature == "pair_count_L30p0kpc":
        proximity = pair_proximity(
            np.asarray([1.0]), np.asarray([solar_r80_kpc / 3.0]), 30.0
        )
        return np.full_like(radius_m, proximity)
    if candidate_feature == "outward_radial_column":
        # Outside a compact source, a literal outward matter column is nearly zero.
        return np.full_like(radius_m, 1.0e-30)
    return np.full_like(radius_m, np.nan)


def solar_diagnostics(spec: dict, record: dict):
    def fraction(radius_m):
        radius = np.asarray(radius_m, dtype=float)
        gbar = G_SI * M_SUN_KG / np.square(radius)
        potential = G_SI * M_SUN_KG / radius / C_M_S**2
        response = generalized_arc_response(
            gbar,
            radius / KPC_M,
            np.ones_like(radius),
            np.ones_like(radius),
            potential_depth=potential,
            potential_length_kpc=radius / KPC_M,
            potential_path_ratio=np.ones_like(radius),
            enclosed_mass_log_slope=np.zeros_like(radius),
            **response_parameters(spec, q=1.0, a0=A0),
        )
        feature = solar_feature(record["candidate"].feature, radius, potential)
        q_eff = apply_frozen(record, feature)
        return response["fractional_dynamical_response"] * q_eff, q_eff

    radius = np.geomspace(1.6 * SOLAR_RADIUS_M, 8.43 * AU_M, 1000)
    dynamic, q_grid = fraction(radius)
    photon = float(spec["photon_extra_multiplier"])
    earth = float(np.interp(AU_M, radius, dynamic))
    saturn = float(np.interp(8.43 * AU_M, radius, dynamic))

    semimajor = 0.38709893 * AU_M
    eccentricity = 0.205630
    period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, 32768, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    mercury_radius = semimajor * one_minus_e2 / (1.0 + eccentricity * cosine)
    mercury_fraction, mercury_q = fraction(mercury_radius)
    radial_perturbation = -(G_SI * M_SUN_KG / np.square(mercury_radius)) * mercury_fraction
    time_weight = one_minus_e2**1.5 / np.square(1.0 + eccentricity * cosine)
    mean_r_cosine = float(np.mean(radial_perturbation * cosine * time_weight))
    period_seconds = period_days * 86400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = (
        -math.sqrt(one_minus_e2)
        / (mean_motion * semimajor * eccentricity)
        * mean_r_cosine
    )
    mercury = (
        mean_rate
        * period_seconds
        * (100.0 * JULIAN_YEAR_DAYS / period_days)
        * RAD_TO_MAS
    )
    return {
        "q_limb_to_saturn_min": float(np.min(q_grid)),
        "q_limb_to_saturn_median": float(np.median(q_grid)),
        "q_limb_to_saturn_max": float(np.max(q_grid)),
        "q_mercury_min": float(np.min(mercury_q)),
        "q_mercury_max": float(np.max(mercury_q)),
        "maximum_dynamic_fraction_limb_to_Saturn": float(np.max(dynamic)),
        "maximum_lensing_fraction_limb_to_Saturn": float(np.max(photon * dynamic)),
        "Earth_orbit_fractional_change": earth,
        "Saturn_orbit_fractional_change": saturn,
        "Mercury_precession_mas_per_century": float(mercury),
        "Cassini_proxy_pass": bool(np.max(photon * dynamic) <= 2.3e-5),
        "Earth_proxy_pass": bool(earth <= 1.0e-10),
        "Mercury_proxy_pass": bool(abs(mercury) <= 3.1),
    }


def member_pair_feature(member_context, total_mass: float, feature: str) -> float:
    members = member_context.members
    xy = members[["x_arcsec", "y_arcsec"]].to_numpy(float)
    scale = float(
        member_context.local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    xy *= scale
    weights = members.base_weight.to_numpy(float)
    weights /= weights.sum()
    distance2 = np.sum(np.square(xy[:, None, :] - xy[None, :, :]), axis=2)
    kernel = 30.0
    internal_sigma = 2.0
    denominator = kernel**2 + 2.0 * internal_sigma**2
    overlap = kernel**2 / denominator * np.exp(-distance2 / (2.0 * denominator))
    proximity = float(np.sum(weights[:, None] * weights[None, :] * overlap))
    if feature.startswith("pair_surface"):
        return total_mass * proximity / (2.0 * np.pi * kernel**2)
    return total_mass**2 * proximity


def variable_raw_field(spec, record, anchors, raw_protocol, member_context):
    radius_grid = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(radius_grid, anchor_radius, anchor_gbar, outer_slope=-2.0)
    invariants = spherical_profile_invariants(radius_grid, gbar)
    anchor_mass = anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    total = float(np.maximum.accumulate(anchor_mass)[-1])
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    feature_name = record["candidate"].feature
    if feature_name == "none":
        feature = np.ones_like(radius_grid)
    elif feature_name == "potential_depth":
        feature = invariants["potential_depth"]
    elif feature_name == "mean_surface_R80":
        feature = np.full_like(radius_grid, total / (np.pi * r80**2))
    elif feature_name.startswith("pair_"):
        value = member_pair_feature(member_context, total, feature_name)
        feature = np.full_like(radius_grid, value)
    else:
        raise ValueError(f"raw transfer is unavailable for {feature_name}")
    q_eff = apply_frozen(record, feature)
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
    acceleration = gbar * (
        1.0 + float(spec["photon_extra_multiplier"]) * q_eff * response["unit_fractional_response"]
    )

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(acceleration)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    physical_alpha = spherical_deflection_radians(
        impact_arcsec * scale,
        lookup,
        maximum_radius_kpc=1.0e6,
        integration_points=800,
    )
    field = RadialDeflectionField(impact_arcsec, physical_alpha)
    sample = np.unique(np.geomspace(1, len(radius_grid), 300).astype(int) - 1)
    profile = pd.DataFrame(
        {
            "radius_kpc": radius_grid[sample],
            "gbar_m_s2": gbar[sample],
            "potential_depth": invariants["potential_depth"][sample],
            "feature_value": feature[sample],
            "q_eff": q_eff[sample],
            "lensing_acceleration_m_s2": acceleration[sample],
        }
    )
    return field, profile


def raw_transfer(spec, frozen, p0624):
    allowed_roles = set(p0624["raw_cluster"]["roles_to_run"])
    selected = {key: value for key, value in frozen.items() if value["role"] in allowed_roles}
    p0554 = load_json("configs/p0554_local_cross_domain_sensitivity_protocol.json")
    contexts = raw_contexts(p0554)
    p0615 = load_json("configs/p0615_self_coupled_quadrupole_route_protocol.json")
    member_prepared = contexts_and_frozen_geometry(p0615)
    member_map = {context.system["label"]: context for context, _, _, _ in member_prepared}
    score_rows, prediction_blocks = [], []
    for context in contexts:
        fields = {}
        profiles = []
        for candidate_id, record in selected.items():
            print(f"raw {context.label}: {candidate_id}", flush=True)
            field, profile = variable_raw_field(
                spec, record, context.anchors, context.local, member_map[context.label]
            )
            fields[candidate_id] = field
            profile.insert(0, "candidate_id", candidate_id)
            profile.insert(0, "system_label", context.label)
            profiles.append(profile)
        lens = RawLens(context.local, fields)
        for candidate_id, record in selected.items():
            _, sources = lens.profiled_residuals(candidate_id, context.geometry, context.training)
            train = lens.exact_predictions(
                candidate_id, context.geometry, sources, context.training, stage="training"
            )
            held = lens.exact_predictions(
                candidate_id, context.geometry, sources, context.heldout, stage="heldout"
            )
            train_metrics = raw_score(train, lens.sigma)
            held_metrics = raw_score(held, lens.sigma)
            score_rows.append(
                {
                    "system_label": context.label,
                    "raw_group": context.group,
                    "candidate_id": candidate_id,
                    "role": record["role"],
                    "training_images": len(context.training),
                    "training_RMS_arcsec": train_metrics["exact_radial_RMS_arcsec"],
                    "training_roots_converged": train_metrics["converged_roots"],
                    "heldout_images": len(context.heldout),
                    "heldout_RMS_arcsec": held_metrics["exact_radial_RMS_arcsec"],
                    "heldout_roots_converged": held_metrics["converged_roots"],
                    "heldout_all_roots": held_metrics["all_roots_converged"],
                }
            )
            for block in (train, held):
                local = block.copy()
                local.insert(0, "candidate_id", candidate_id)
                local.insert(0, "system_label", context.label)
                prediction_blocks.append(local)
    scores = pd.DataFrame(score_rows)
    baseline = scores[scores.candidate_id.eq("constant")].set_index("system_label")
    scores["improvement_vs_constant_fraction"] = scores.apply(
        lambda row: (
            1.0 - float(row.heldout_RMS_arcsec) / float(baseline.loc[row.system_label].heldout_RMS_arcsec)
            if bool(row.heldout_all_roots)
            and bool(baseline.loc[row.system_label].heldout_all_roots)
            and np.isfinite(float(row.heldout_RMS_arcsec))
            and np.isfinite(float(baseline.loc[row.system_label].heldout_RMS_arcsec))
            else np.nan
        ),
        axis=1,
    )
    return scores, pd.concat(prediction_blocks, ignore_index=True)


def make_figure(output, derived, solar, raw):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    available = derived[derived.available.astype(bool)]
    axes[0].bar(available.candidate_id, available.cluster_equal_system_RMSE_dex)
    axes[0].set_ylabel("equal-system RMSE (dex)")
    axes[0].set_title("20 derived CLASH systems")
    axes[0].tick_params(axis="x", rotation=75, labelsize=6)
    axes[1].bar(solar.candidate_id, np.abs(solar.Mercury_precession_mas_per_century))
    axes[1].axhline(3.1, color="red", linestyle="--")
    axes[1].set_ylabel("|Mercury proxy| (mas/century)")
    axes[1].set_title("Solar compact-source screen")
    axes[1].tick_params(axis="x", rotation=75, labelsize=6)
    complete = raw[raw.heldout_all_roots.astype(bool)].copy()
    aggregate = complete.groupby("candidate_id").heldout_RMS_arcsec.apply(
        lambda values: np.sqrt(np.mean(np.square(values)))
    )
    axes[2].bar(aggregate.index, aggregate.values)
    axes[2].set_ylabel("finite complete-system RMS (arcsec)")
    axes[2].set_title("Five fixed-geometry raw lenses")
    axes[2].tick_params(axis="x", rotation=75, labelsize=6)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    p0624 = load_json("configs/p0624_deep_porous_cross_domain_protocol.json")
    p0623 = load_json(p0624["parent"]["protocol"])
    output = ROOT / p0624["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    development, candidate_lookup, spec = prepare_galaxy_development(p0623)
    parameters, frozen = fit_frozen_candidates(development, candidate_lookup, p0624)
    parameters.to_csv(output / p0624["outputs"]["frozen_parameters"], index=False)
    parent = load_json(p0623["inputs"]["SPARC_parent_protocol"])

    derived, derived_predictions = derived_cluster_transfer(parent, spec, frozen)
    derived.to_csv(output / p0624["outputs"]["derived_cluster"], index=False)
    derived_predictions.to_csv(output / p0624["outputs"]["derived_predictions"], index=False)
    print("Derived cluster transfer complete", flush=True)

    solar_rows = []
    for candidate_id, record in frozen.items():
        solar_rows.append(
            {
                "candidate_id": candidate_id,
                "role": record["role"],
                **solar_diagnostics(spec, record),
            }
        )
    solar = pd.DataFrame(solar_rows)
    solar["all_solar_proxies_pass"] = (
        solar.Cassini_proxy_pass & solar.Earth_proxy_pass & solar.Mercury_proxy_pass
    )
    solar.to_csv(output / p0624["outputs"]["solar"], index=False)
    print("Solar transfer complete", flush=True)

    raw, raw_predictions = raw_transfer(spec, frozen, p0624)
    raw.to_csv(output / p0624["outputs"]["raw"], index=False)
    raw_predictions.to_csv(output / p0624["outputs"]["raw_predictions"], index=False)

    primary = "inverse_hill0_m1__potential_depth"
    derived_lookup = derived.set_index("candidate_id")
    solar_lookup = solar.set_index("candidate_id")
    raw_primary = raw[raw.candidate_id.eq(primary)]
    report = {
        "protocol_version": p0624["protocol_version"],
        "status": "complete",
        "coverage": {
            "frozen_candidates": len(frozen),
            "derived_cluster_systems": 20,
            "derived_cluster_points": len(derived_predictions[derived_predictions.candidate_id.eq("constant")]),
            "raw_cluster_systems": raw.system_label.nunique(),
            "raw_heldout_images": int(raw[raw.candidate_id.eq("constant")].heldout_images.sum()),
        },
        "primary_potential": {
            "derived_cluster_equal_system_RMSE_dex": float(
                derived_lookup.loc[primary, "cluster_equal_system_RMSE_dex"]
            ),
            "derived_improvement_vs_constant_fraction": float(
                derived_lookup.loc[primary, "improvement_vs_constant_fraction"]
            ),
            "all_solar_proxies_pass": bool(solar_lookup.loc[primary, "all_solar_proxies_pass"]),
            "raw_roots": int(raw_primary.heldout_roots_converged.sum()),
            "raw_images": int(raw_primary.heldout_images.sum()),
            "raw_complete_systems": int(raw_primary.heldout_all_roots.sum()),
        },
        "derived_cluster_scores": strict_json(derived.to_dict(orient="records")),
        "solar_scores": strict_json(solar.to_dict(orient="records")),
        "raw_scores": strict_json(raw.to_dict(orient="records")),
        "claim_limits": p0624["claim_limits"],
    }
    (output / p0624["outputs"]["report"]).write_text(
        json.dumps(strict_json(report), indent=2), encoding="utf-8"
    )
    make_figure(output / p0624["outputs"]["figure"], derived, solar, raw)
    lines = [
        "# P0624 deep-versus-porous cross-domain stress test",
        "",
        "All formula parameters below were fitted once on P0623 development galaxies and then frozen.",
        "",
        "## Derived clusters",
        "",
    ]
    for row in derived[derived.available.astype(bool)].sort_values("cluster_equal_system_RMSE_dex").itertuples():
        lines.append(
            f"- `{row.candidate_id}`: {row.cluster_equal_system_RMSE_dex:.4f} dex "
            f"({100.0 * row.improvement_vs_constant_fraction:+.2f}% vs frozen constant)."
        )
    lines.extend(["", "## Solar", ""])
    for row in solar.itertuples():
        lines.append(
            f"- `{row.candidate_id}`: Mercury {row.Mercury_precession_mas_per_century:+.3f} "
            f"mas/century; all proxies pass = **{row.all_solar_proxies_pass}**."
        )
    lines.extend(["", "## Raw cluster roots", ""])
    for candidate_id, block in raw.groupby("candidate_id"):
        lines.append(
            f"- `{candidate_id}`: {int(block.heldout_roots_converged.sum())}/"
            f"{int(block.heldout_images.sum())} roots across {int(block.heldout_all_roots.sum())}/"
            f"{len(block)} complete systems."
        )
    lines.extend(
        [
            "",
            "These systems are project-spent. The result is a mechanism stress test, not external validation.",
        ]
    )
    (output / p0624["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["primary_potential"], indent=2), flush=True)


if __name__ == "__main__":
    main()
