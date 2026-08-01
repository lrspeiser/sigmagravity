#!/usr/bin/env python3
"""Micro-sweep the arc law against galaxies, absolute lensing, and the Sun."""

from __future__ import annotations

import hashlib
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

from run_arc_apogee_cross_domain import (  # noqa: E402
    fit_q,
    galaxy_properties,
    radius_at_mass_fraction,
    score_predictions,
    velocity_prediction,
)
from run_arc_apogee_boundary_refinement import cross_galaxy_score  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    RawLens,
    load_baryonic_anchors,
    load_images,
    near_bound,
    score as raw_score,
    spec_for,
)
from voidscreen.arc_invariants import (  # noqa: E402
    generalized_arc_response,
    generalized_solar_diagnostics,
    spherical_profile_invariants,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.arc_apogee import G_SI, M_SUN_KG  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.unified import load_clash_acceleration_frame, rar_acceleration  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(float(value)) else None
    return value


def add_profile_invariants(
    frame: pd.DataFrame,
    *,
    system_column: str,
    radius_column: str,
    gbar_column: str,
) -> pd.DataFrame:
    pieces = []
    for _, block in frame.groupby(system_column, sort=False):
        local = block.copy()
        values = spherical_profile_invariants(
            local[radius_column].to_numpy(float), local[gbar_column].to_numpy(float)
        )
        for name, array in values.items():
            local[name] = array
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True)


def cluster_properties(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, block in frame.groupby("system", sort=True):
        radius = block.radius_kpc.to_numpy(float)
        gbar = block.gbar_m_s2.to_numpy(float)
        mass = gbar * np.square(radius * KPC_M) / (G_SI * M_SUN_KG)
        order = np.argsort(radius)
        total = float(np.maximum.accumulate(mass[order])[-1])
        r50 = radius_at_mass_fraction(radius, mass, 0.5)
        r80 = radius_at_mass_fraction(radius, mass, 0.8)
        rows.append(
            {
                "system": system,
                "force_equivalent_mass_solar": total,
                "force_equivalent_r50_kpc": r50,
                "force_equivalent_r80_kpc": r80,
                "force_equivalent_concentration_r50_over_r80": r50 / r80,
            }
        )
    return pd.DataFrame(rows)


def response_parameters(spec: dict, *, q: float, a0: float) -> dict:
    return {
        "residence_strength": float(q),
        "alpha": float(spec["alpha"]),
        "apogee_ratio": float(spec["apogee_ratio"]),
        "screen_a0_m_s2": float(a0),
        "screen_exponent": float(spec["screen_exponent"]),
        "screen_scale": float(spec["screen_scale"]),
        "mass_radius_delta": float(spec["mass_radius_delta"]),
        "extent_leak": float(spec["extent_leak"]),
        "invariant_mode": str(spec["invariant_mode"]),
        "invariant_power": float(spec["invariant_power"]),
        "invariant_scale": float(spec["invariant_scale"]),
        "secondary_path_ratio_power": float(spec.get("secondary_path_ratio_power", 0.0)),
        "photon_extra_multiplier": float(spec["photon_extra_multiplier"]),
        "residence_softness": float(spec.get("residence_softness", 1.0)),
        "screen_softness": float(spec.get("screen_softness", 1.0)),
        "potential_softness": float(spec.get("potential_softness", 1.0)),
        "potential_path_cross": float(spec.get("potential_path_cross", 1.0)),
        "response_addition_softness": float(
            spec.get("response_addition_softness", 1.0)
        ),
        "lensing_addition_softness": float(
            spec.get("lensing_addition_softness", 1.0)
        ),
        "extent_scale_coupling": float(spec.get("extent_scale_coupling", 0.0)),
        "potential_scale_coupling": float(spec.get("potential_scale_coupling", 0.0)),
        "mass_growth_power": float(spec.get("mass_growth_power", 0.0)),
    }


def response_for_frame(
    frame: pd.DataFrame,
    spec: dict,
    *,
    q: float,
    a0: float,
    radius_column: str,
    gbar_column: str,
) -> dict[str, np.ndarray]:
    return generalized_arc_response(
        frame[gbar_column].to_numpy(float),
        frame[radius_column].to_numpy(float),
        frame.force_equivalent_mass_solar.to_numpy(float),
        frame.force_equivalent_concentration_r50_over_r80.to_numpy(float),
        potential_depth=frame.potential_depth.to_numpy(float),
        potential_length_kpc=frame.potential_length_kpc.to_numpy(float),
        potential_path_ratio=frame.potential_path_ratio.to_numpy(float),
        enclosed_mass_log_slope=frame.enclosed_mass_log_slope.to_numpy(float),
        **response_parameters(spec, q=q, a0=a0),
    )


def build_specs(protocol: dict) -> list[dict]:
    baseline = dict(protocol["baseline"])
    specs = [{**baseline, "family": "baseline", "changed_parameter": "none", "changed_value": "baseline"}]
    for parameter, values in protocol["one_at_a_time"].items():
        for value in values:
            specs.append(
                {
                    **baseline,
                    parameter: value,
                    "family": parameter,
                    "changed_parameter": parameter,
                    "changed_value": str(value),
                }
            )
    for family in protocol["invariant_families"]:
        for power in family["powers"]:
            for scale in family["scales"]:
                specs.append(
                    {
                        **baseline,
                        "invariant_mode": family["mode"],
                        "invariant_power": power,
                        "invariant_scale": scale,
                        "family": family["mode"],
                        "changed_parameter": f"{family['mode']}_power_scale",
                        "changed_value": f"{power:g}@{scale:g}",
                    }
                )
    for index, spec in enumerate(specs):
        spec["candidate_id"] = f"I{index:03d}"
    return specs


def prepare_galaxies(protocol: dict, a0: float):
    raw = pd.read_csv(ROOT / protocol["inputs"]["SPARC_points"])
    points = raw[raw.model.eq("fixed_RAR") & raw.scenario.eq("invariant")].copy()
    morphology = pd.read_csv(ROOT / protocol["inputs"]["SPARC_morphology"])
    properties = galaxy_properties(points, morphology, a0)
    frame = points.merge(properties, on="galaxy", validate="many_to_one")
    frame = add_profile_invariants(
        frame,
        system_column="galaxy",
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
    )
    return frame, properties


def prepare_clusters(protocol: dict):
    frame = load_clash_acceleration_frame(ROOT / protocol["inputs"]["CLASH_acceleration"])
    properties = cluster_properties(frame)
    frame = frame.merge(properties, on="system", validate="many_to_one")
    # load_clash_acceleration_frame already supplies the same potential closure;
    # recompute all invariants together so their numerical definitions are locked.
    frame = frame.drop(
        columns=[name for name in ("potential_depth", "potential_length_kpc", "potential_path_ratio", "enclosed_mass_log_slope") if name in frame],
        errors="ignore",
    )
    frame = add_profile_invariants(
        frame,
        system_column="system",
        radius_column="radius_kpc",
        gbar_column="gbar_m_s2",
    )
    return frame, properties


def cluster_score(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    residual = np.log10(prediction) - frame.log_gtot.to_numpy(float)
    per_system = pd.Series(np.square(residual), index=frame.system).groupby(level=0).mean()
    return {
        "cluster_RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "cluster_equal_system_RMSE_dex": float(np.sqrt(per_system.mean())),
        "cluster_mean_residual_dex": float(np.mean(residual)),
        "cluster_median_observed_over_predicted": float(
            np.median(frame.observed_g_m_s2.to_numpy(float) / prediction)
        ),
    }


def pareto_ids(scores: pd.DataFrame) -> set[str]:
    eligible = scores[scores.all_solar_proxies_pass].copy()
    output = set()
    for row in eligible.itertuples(index=False):
        dominated = eligible[
            (eligible.cross_galaxy_outer_RMSE_km_s <= row.cross_galaxy_outer_RMSE_km_s)
            & (eligible.cluster_RMSE_dex <= row.cluster_RMSE_dex)
            & (
                (eligible.cross_galaxy_outer_RMSE_km_s < row.cross_galaxy_outer_RMSE_km_s)
                | (eligible.cluster_RMSE_dex < row.cluster_RMSE_dex)
            )
        ]
        if dominated.empty:
            output.add(row.candidate_id)
    return output


def raw_field(spec: dict, q: float, anchors: pd.DataFrame, raw_protocol: dict, a0: float):
    radius_grid = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius_grid, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    invariants = spherical_profile_invariants(radius_grid, gbar)
    anchor_mass = anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    total = float(np.maximum.accumulate(anchor_mass)[-1])
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    response = generalized_arc_response(
        gbar,
        radius_grid,
        np.full_like(radius_grid, total),
        np.full_like(radius_grid, r50 / r80),
        potential_depth=invariants["potential_depth"],
        potential_length_kpc=invariants["potential_length_kpc"],
        potential_path_ratio=invariants["potential_path_ratio"],
        enclosed_mass_log_slope=invariants["enclosed_mass_log_slope"],
        **response_parameters(spec, q=q, a0=a0),
    )
    acceleration = gbar * response["lensing_enhancement"]

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
    sample_index = np.unique(
        np.geomspace(1, len(radius_grid), 300).astype(int) - 1
    )
    profile = pd.DataFrame(
        {
            "candidate_id": spec["candidate_id"],
            "radius_kpc": radius_grid[sample_index],
            "gbar_m_s2": gbar[sample_index],
            "dynamic_acceleration_m_s2": (
                gbar * response["dynamical_enhancement"]
            )[sample_index],
            "lensing_acceleration_m_s2": acceleration[sample_index],
            "potential_path_ratio": invariants["potential_path_ratio"][sample_index],
            "enclosed_mass_log_slope": invariants["enclosed_mass_log_slope"][sample_index],
        }
    )
    return field, profile


def run_raw_shortlist(protocol: dict, shortlist: pd.DataFrame, specs_by_id: dict, a0: float):
    raw_protocol = json.loads((ROOT / protocol["inputs"]["RXJ2129_protocol"]).read_text())
    images = load_images(raw_protocol)
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    training = images[~images.image_id.isin(heldout_ids)].copy()
    heldout = images[images.image_id.isin(heldout_ids)].copy()
    anchors = load_baryonic_anchors(raw_protocol)
    fields = {}
    profiles = []
    for row in shortlist.itertuples(index=False):
        spec = specs_by_id[row.candidate_id]
        fields[row.candidate_id], profile = raw_field(
            spec, float(row.universal_q), anchors, raw_protocol, a0
        )
        profiles.append(profile)
    lens = RawLens(raw_protocol, fields)
    previous = pd.read_csv(ROOT / protocol["inputs"]["RXJ2129_previous_parameters"])
    block = previous[
        previous.stage.eq("training") & previous.model.eq("locked_universal_candidate")
    ]
    initial = block.set_index("parameter").loc[list(spec_for("fixed").labels), "value"].to_numpy(float)
    prediction_frames = []
    parameter_rows = []
    raw_rows = []
    starts = int(protocol["raw_lensing"]["optimization_starts"])
    seed = int(raw_protocol["optimization"]["random_seed"])
    for offset, row in enumerate(shortlist.itertuples(index=False)):
        model = row.candidate_id
        fit = lens.fit(
            model,
            training,
            starts=starts,
            seed=seed + 500 + offset,
            initial_override=initial,
        )
        train_pred = lens.exact_predictions(
            model, fit["result"].x, fit["sources"], training, stage="training"
        )
        held_pred = lens.exact_predictions(
            model, fit["result"].x, fit["sources"], heldout, stage="heldout"
        )
        prediction_frames.extend([train_pred, held_pred])
        for label, value in zip(spec_for(model).labels, fit["result"].x):
            parameter_rows.append(
                {
                    "candidate_id": model,
                    "selection_role": row.selection_role,
                    "parameter": label,
                    "value": float(value),
                    "near_bound": near_bound(model, fit["result"].x)[label],
                }
            )
        train_score = raw_score(train_pred, lens.sigma, free_parameters=20)
        held_score = raw_score(held_pred, lens.sigma)
        raw_rows.append(
            {
                "candidate_id": model,
                "selection_role": row.selection_role,
                "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                "heldout_RMS_arcsec": held_score["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": held_score["converged_roots"],
                "heldout_all_roots_converged": held_score["all_roots_converged"],
                "maximum_heldout_residual_arcsec": held_score["maximum_radial_residual_arcsec"],
                "any_geometry_near_bound": bool(any(near_bound(model, fit["result"].x).values())),
            }
        )
    return (
        pd.DataFrame(raw_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(parameter_rows),
        pd.concat(profiles, ignore_index=True),
    )


def make_figure(scores, impacts, cluster_predictions, raw_scores, references, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for family, block in scores.groupby("family", sort=False):
        ax.scatter(
            block.cross_galaxy_outer_RMSE_km_s,
            block.cluster_RMSE_dex,
            s=30 if family != "baseline" else 90,
            alpha=0.75,
            label=family.replace("_", " "),
        )
    ax.axvline(references["RAR_galaxy_RMSE_km_s"], color="black", ls="--")
    ax.axhline(references["fixed_RAR_cluster_RMSE_dex"], color="gray", ls=":")
    ax.set(xlabel="held-out galaxy RMSE (km/s)", ylabel="CLASH absolute RMSE (dex)", title="One universal q; no cluster amplitude fit")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 1]
    ordered = impacts.sort_values("cluster_impact_span_dex")
    ax.barh(ordered.family, ordered.cluster_impact_span_dex, color="tab:purple")
    ax.set(xlabel="CLASH RMSE span (dex)", title="Which tiny change moves lensing most?")

    ax = axes[1, 0]
    selected = cluster_predictions[cluster_predictions.selection_role.notna()]
    for role, block in selected.groupby("selection_role"):
        radial = block.groupby("radius_kpc").residual_dex.mean()
        ax.plot(radial.index, radial.values, "o-", label=role.replace("_", " "))
    ax.axhline(0.0, color="black", ls="--")
    ax.set_xscale("log")
    ax.set(xlabel="cluster radius (kpc)", ylabel="mean log prediction/target", title="Absolute cluster residual by radius")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    values = raw_scores.heldout_RMS_arcsec.fillna(0.0)
    ax.barh(raw_scores.selection_role.str.replace("_", " "), values, color="tab:blue")
    ax.axvline(0.5, color="crimson", ls="--", label="0.5 arcsec gate")
    ax.set(xlabel="held-out RX J2129 RMS (arcsec)", title="Raw-image shortlist selected before image scoring")
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    config_path = ROOT / "configs" / "arc_invariant_absolute_lensing_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    a0 = float(protocol["constants"]["a0_m_s2"])
    bounds = protocol["constants"]["universal_q_bounds"]
    galaxy, galaxy_props = prepare_galaxies(protocol, a0)
    clusters, _ = prepare_clusters(protocol)
    specs = build_specs(protocol)
    score_rows = []
    cluster_prediction_frames = []
    galaxy_cache = {}
    for index, spec in enumerate(specs):
        frame = galaxy.copy()
        unit = response_for_frame(
            frame, spec, q=1.0, a0=a0, radius_column="radius_adjusted_kpc", gbar_column="g_bar_m_s2"
        )
        frame["arc_coordinate"] = unit["unit_fractional_response"]
        inner = frame[frame.split.eq("inner_train")]
        outer = frame[frame.split.eq("outer_holdout")]
        q = fit_q(inner, bounds)
        galaxy_score = score_predictions(outer, velocity_prediction(outer, q))
        cross_score, fold_q = cross_galaxy_score(inner, outer, galaxy_props, bounds)
        solar = generalized_solar_diagnostics(**response_parameters(spec, q=q, a0=a0))
        cluster_response = response_for_frame(
            clusters, spec, q=q, a0=a0, radius_column="radius_kpc", gbar_column="gbar_m_s2"
        )
        predicted_dynamic = clusters.gbar_m_s2.to_numpy(float) * cluster_response["dynamical_enhancement"]
        predicted_lens = clusters.gbar_m_s2.to_numpy(float) * cluster_response["lensing_enhancement"]
        lens_score = cluster_score(clusters, predicted_lens)
        score_rows.append(
            {
                **spec,
                "universal_q": q,
                "fold_q_min": float(np.min(fold_q)),
                "fold_q_max": float(np.max(fold_q)),
                "outer_RMSE_km_s": galaxy_score["RMSE_km_s"],
                "cross_galaxy_outer_RMSE_km_s": cross_score["RMSE_km_s"],
                "cross_galaxy_equal_RMSE_km_s": cross_score["equal_galaxy_RMSE_km_s"],
                **lens_score,
                **solar,
                "all_solar_proxies_pass": bool(
                    solar["Cassini_proxy_pass"] and solar["Earth_proxy_pass"] and solar["Mercury_proxy_pass"]
                ),
            }
        )
        cluster_local = clusters.copy()
        cluster_local["candidate_id"] = spec["candidate_id"]
        cluster_local["predicted_dynamic_m_s2"] = predicted_dynamic
        cluster_local["predicted_lensing_m_s2"] = predicted_lens
        cluster_local["residual_dex"] = np.log10(predicted_lens) - cluster_local.log_gtot
        cluster_prediction_frames.append(cluster_local)
        galaxy_cache[spec["candidate_id"]] = (frame, q)
        if index % 10 == 0:
            print(f"micro-sweep {index + 1}/{len(specs)}", flush=True)
    scores = pd.DataFrame(score_rows)
    scores["pareto"] = scores.candidate_id.isin(pareto_ids(scores))

    outer = galaxy[galaxy.split.eq("outer_holdout")]
    rar_galaxy = score_predictions(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float))
    cluster_baryon = cluster_score(clusters, clusters.gbar_m_s2.to_numpy(float))
    cluster_rar = cluster_score(clusters, rar_acceleration(clusters.gbar_m_s2.to_numpy(float), a0))
    references = {
        "RAR_galaxy_RMSE_km_s": rar_galaxy["RMSE_km_s"],
        "baryons_cluster_RMSE_dex": cluster_baryon["cluster_RMSE_dex"],
        "fixed_RAR_cluster_RMSE_dex": cluster_rar["cluster_RMSE_dex"],
    }
    eligible = scores[scores.all_solar_proxies_pass]
    baseline = scores[scores.family.eq("baseline")].iloc[0]
    best_galaxy = eligible.sort_values("cross_galaxy_outer_RMSE_km_s").iloc[0]
    zero_slip = eligible[eligible.photon_extra_multiplier.eq(1.0)]
    best_zero_slip = zero_slip.sort_values("cluster_RMSE_dex").iloc[0]
    galaxy_limit = 1.5 * rar_galaxy["RMSE_km_s"]
    constrained = eligible[eligible.cross_galaxy_outer_RMSE_km_s <= galaxy_limit]
    best_constrained = constrained.sort_values("cluster_RMSE_dex").iloc[0]
    photon = constrained[constrained.photon_extra_multiplier.gt(1.0)]
    best_photon = photon.sort_values("cluster_RMSE_dex").iloc[0] if len(photon) else best_constrained
    selection_pairs = [
        ("baseline", baseline),
        ("best_galaxy", best_galaxy),
        ("best_zero_slip_cluster", best_zero_slip),
        ("best_cluster_within_1p5_RAR", best_constrained),
        ("best_photon_multiplier", best_photon),
    ]
    unique = {}
    for role, row in selection_pairs:
        if row.candidate_id not in unique:
            unique[row.candidate_id] = {**row.to_dict(), "selection_role": role}
        else:
            unique[row.candidate_id]["selection_role"] += f"+{role}"
    shortlist = pd.DataFrame(unique.values())
    specs_by_id = {spec["candidate_id"]: spec for spec in specs}

    raw_scores, raw_predictions, raw_parameters, raw_profiles = run_raw_shortlist(
        protocol, shortlist, specs_by_id, a0
    )
    selected_ids = set(shortlist.candidate_id)
    cluster_predictions = pd.concat(cluster_prediction_frames, ignore_index=True)
    role_map = shortlist.set_index("candidate_id").selection_role
    cluster_predictions["selection_role"] = cluster_predictions.candidate_id.map(role_map)
    selected_galaxy = []
    for candidate_id in selected_ids:
        frame, q = galaxy_cache[candidate_id]
        local = frame.copy()
        local["candidate_id"] = candidate_id
        local["selection_role"] = role_map[candidate_id]
        local["universal_q"] = q
        local["predicted_velocity_km_s"] = velocity_prediction(local, q)
        selected_galaxy.append(local)
    selected_galaxy = pd.concat(selected_galaxy, ignore_index=True)

    impacts = []
    for family, block in scores.groupby("family", sort=False):
        impacts.append(
            {
                "family": family,
                "variants": int(len(block)),
                "galaxy_impact_span_km_s": float(block.cross_galaxy_outer_RMSE_km_s.max() - block.cross_galaxy_outer_RMSE_km_s.min()),
                "cluster_impact_span_dex": float(block.cluster_RMSE_dex.max() - block.cluster_RMSE_dex.min()),
                "best_cluster_candidate": str(block.sort_values("cluster_RMSE_dex").iloc[0].candidate_id),
                "best_cluster_RMSE_dex": float(block.cluster_RMSE_dex.min()),
            }
        )
    impacts = pd.DataFrame(impacts).sort_values("cluster_impact_span_dex", ascending=False)

    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    selected_galaxy.to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    cluster_predictions.to_csv(output / protocol["outputs"]["cluster_predictions"], index=False)
    raw_predictions.to_csv(output / protocol["outputs"]["raw_predictions"], index=False)
    raw_parameters.to_csv(output / protocol["outputs"]["raw_parameters"], index=False)
    raw_profiles.to_csv(output / "raw_RXJ2129_profiles.csv", index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed absolute-lensing arc-invariant micro-sweep",
        "protocol_sha256": sha256(config_path),
        "coverage": {
            "variants": int(len(scores)),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "SPARC_outer_points": int(len(outer)),
            "CLASH_systems": int(clusters.system.nunique()),
            "CLASH_points": int(len(clusters)),
            "raw_RXJ2129_shortlist": int(len(shortlist)),
            "raw_RXJ2129_training_images": 15,
            "raw_RXJ2129_heldout_images": 7,
        },
        "references": references,
        "baseline": baseline.to_dict(),
        "best_galaxy": best_galaxy.to_dict(),
        "best_zero_slip_cluster": best_zero_slip.to_dict(),
        "best_cluster_within_1p5_RAR_galaxy_error": best_constrained.to_dict(),
        "best_photon_multiplier": best_photon.to_dict(),
        "pareto_candidates": scores[scores.pareto].sort_values("cross_galaxy_outer_RMSE_km_s").to_dict("records"),
        "raw_RXJ2129_scores": raw_scores.to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "# Arc-invariant absolute-lensing micro-sweep",
        "",
        f"Tested {len(scores)} small universal variations on {galaxy.galaxy.nunique()} galaxies, {clusters.system.nunique()} CLASH systems, Solar proxies, and {len(shortlist)} raw RX J2129 shortlist laws.",
        f"Baseline absolute CLASH RMSE: {baseline.cluster_RMSE_dex:.4f} dex; fixed RAR: {cluster_rar['cluster_RMSE_dex']:.4f} dex.",
        f"Best zero-slip CLASH RMSE: {best_zero_slip.cluster_RMSE_dex:.4f} dex with galaxy RMSE {best_zero_slip.cross_galaxy_outer_RMSE_km_s:.3f} km/s.",
        f"Best constrained CLASH RMSE: {best_constrained.cluster_RMSE_dex:.4f} dex with galaxy RMSE {best_constrained.cross_galaxy_outer_RMSE_km_s:.3f} km/s.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    make_figure(
        scores,
        impacts,
        cluster_predictions,
        raw_scores,
        references,
        output / protocol["outputs"]["figure"],
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
