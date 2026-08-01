#!/usr/bin/env python3
"""Screen a conservative radial remap of the P0554 extra response."""

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

from run_arc_invariant_absolute_lensing import (  # noqa: E402
    prepare_clusters,
    prepare_galaxies,
    response_for_frame,
    response_parameters,
)
from run_arc_apogee_cross_domain import radius_at_mass_fraction  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0554_baryonic_network_rxj1347_holdout import pair_split  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import A0, json_safe, sha256  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, score as raw_score  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    load_anchors,
    load_system_images,
    system_protocol,
)
from voidscreen.arc_invariants import (  # noqa: E402
    generalized_arc_response,
    spherical_profile_invariants,
)
from voidscreen.arc_apogee import G_SI, M_SUN_KG  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.radial_route import (  # noqa: E402
    potential_transition_scale,
    remap_total_acceleration,
    remapped_solar_diagnostics,
)
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


def object_remainder(name: str) -> int:
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 4


def partition(name: str) -> str:
    return "formula_holdout" if object_remainder(name) == 0 else "discovery"


def candidates(protocol: dict) -> list[dict]:
    rows = [
        {
            "candidate_id": "parent",
            "route_fraction": 0.0,
            "radial_scale": 1.0,
            "role": "exact P0554 parent",
        }
    ]
    for fraction in protocol["grid"]["route_fractions"]:
        for scale in protocol["grid"]["radial_scales"]:
            rows.append(
                {
                    "candidate_id": f"f{int(round(100 * fraction)):03d}_l{int(round(100 * scale)):03d}",
                    "route_fraction": float(fraction),
                    "radial_scale": float(scale),
                    "role": "radial remap",
                }
            )
    expected = int(protocol["grid"]["candidate_count"])
    if len(rows) != expected or len({row["candidate_id"] for row in rows}) != expected:
        raise RuntimeError("candidate grid count or identifiers do not match protocol")
    return rows


def remap_profiles(
    frame: pd.DataFrame,
    parent_response: dict[str, np.ndarray],
    *,
    system_column: str,
    radius_column: str,
    gbar_column: str,
    route_fraction: float,
    radial_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.reset_index(drop=True)
    gbar = frame[gbar_column].to_numpy(float)
    parent_dynamic = gbar * parent_response["dynamical_enhancement"]
    parent_lensing = gbar * parent_response["lensing_enhancement"]
    dynamic = np.empty(len(frame), dtype=float)
    lensing = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby(system_column, sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        order = np.argsort(frame.loc[indices, radius_column].to_numpy(float), kind="stable")
        positions = indices[order]
        radius = frame.loc[positions, radius_column].to_numpy(float)
        scale = radial_scale
        if np.ndim(radial_scale) != 0:
            scale = np.asarray(radial_scale, dtype=float)[positions]
        dynamic[positions] = remap_total_acceleration(
            radius,
            gbar[positions],
            parent_dynamic[positions],
            route_fraction=route_fraction,
            radial_scale=scale,
        )
        lensing[positions] = remap_total_acceleration(
            radius,
            gbar[positions],
            parent_lensing[positions],
            route_fraction=route_fraction,
            radial_scale=scale,
        )
    return dynamic, lensing


def galaxy_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - frame.velocity_observed_adjusted_km_s.to_numpy(float)
    equal = pd.Series(np.square(residual), index=frame.galaxy).groupby(level=0).mean()
    return {
        "point_RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_object_RMSE": float(np.sqrt(equal.mean())),
        "mean_residual": float(np.mean(residual)),
    }


def cluster_metrics(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    residual = np.log10(predicted) - frame.log_gtot.to_numpy(float)
    equal = pd.Series(np.square(residual), index=frame.system).groupby(level=0).mean()
    return {
        "point_RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_object_RMSE": float(np.sqrt(equal.mean())),
        "mean_residual": float(np.mean(residual)),
    }


def raw_radial_field(
    spec: dict,
    q: float,
    anchors: pd.DataFrame,
    local: dict,
    *,
    route_fraction: float,
    radial_scale: float,
    candidate_id: str,
    potential_log_scale_amplitude: float | None = None,
    potential_pivot: float = 2.0e-6,
    potential_sharpness: float = 1.0,
) -> tuple[RadialDeflectionField, pd.DataFrame]:
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
        **response_parameters(spec, q=q, a0=A0),
    )
    parent_dynamic = gbar * response["dynamical_enhancement"]
    parent_lensing = gbar * response["lensing_enhancement"]
    scale_profile = radial_scale
    if potential_log_scale_amplitude is not None:
        scale_profile = potential_transition_scale(
            invariants["potential_depth"],
            log_scale_amplitude=float(potential_log_scale_amplitude),
            pivot=float(potential_pivot),
            sharpness=float(potential_sharpness),
        )
    dynamic = remap_total_acceleration(
        radius_grid,
        gbar,
        parent_dynamic,
        route_fraction=route_fraction,
        radial_scale=scale_profile,
    )
    lensing = remap_total_acceleration(
        radius_grid,
        gbar,
        parent_lensing,
        route_fraction=route_fraction,
        radial_scale=scale_profile,
    )

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(lensing)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    physical_alpha = spherical_deflection_radians(
        impact_arcsec * scale,
        lookup,
        maximum_radius_kpc=1.0e6,
        integration_points=800,
    )
    sample = np.unique(np.geomspace(1, len(radius_grid), 300).astype(int) - 1)
    profile = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "radius_kpc": radius_grid[sample],
            "gbar_m_s2": gbar[sample],
            "parent_dynamic_m_s2": parent_dynamic[sample],
            "routed_dynamic_m_s2": dynamic[sample],
            "parent_lensing_m_s2": parent_lensing[sample],
            "routed_lensing_m_s2": lensing[sample],
            "radial_scale": np.broadcast_to(scale_profile, radius_grid.shape)[sample],
        }
    )
    return RadialDeflectionField(impact_arcsec, physical_alpha), profile


def run_rxj1347(protocol, spec, q, selected):
    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_protocol"]).read_text(encoding="utf-8"))
    system = next(item for item in raw_protocol["systems"] if item["label"] == "RXJ1347")
    local = system_protocol(raw_protocol, system)
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    images = load_system_images(catalog, system)
    training, heldout = pair_split(images)
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    anchors = load_anchors(tian, system["label"])
    geometry_frame = pd.read_csv(ROOT / protocol["inputs"]["RXJ1347_baseline_geometry"])
    geometry = geometry_frame.iloc[0][list(FIXED_LABELS)].to_numpy(float)
    # The transfer candidate was already selected from discovery scores before
    # this function is called.  Score the whole frozen grid afterward to learn
    # whether any RXJ1347 response is isolated or part of a stable trend.
    variants = candidates(protocol)
    rows, predictions, profiles = [], [], []
    for item in variants:
        field, profile = raw_radial_field(
            spec,
            q,
            anchors,
            local,
            route_fraction=float(item["route_fraction"]),
            radial_scale=float(item["radial_scale"]),
            candidate_id=str(item["candidate_id"]),
        )
        profiles.append(profile)
        model = str(item["candidate_id"])
        lens = MorphologyLens(
            local, {model: field}, parent=model, morphology=None, fraction=0.0
        )
        _, sources = lens.profiled_residuals(model, geometry, training)
        exact = lens.exact_predictions(model, geometry, sources, heldout, stage="heldout")
        metrics = raw_score(exact, lens.sigma)
        rows.append(
            {
                "candidate_id": model,
                "route_fraction": float(item["route_fraction"]),
                "radial_scale": float(item["radial_scale"]),
                "heldout_images": len(heldout),
                "heldout_exact_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": metrics["converged_roots"],
                "heldout_all_roots": metrics["all_roots_converged"],
            }
        )
        exact.insert(0, "candidate_id", model)
        predictions.append(exact)
    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True),
        pd.concat(profiles, ignore_index=True),
    )


def make_figure(scores, partitions, raw_scores, selected_id, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    nonparent = scores[scores.candidate_id.ne("parent")]
    selected = scores[scores.candidate_id.eq(selected_id)].iloc[0]
    ax = axes[0, 0]
    scatter = ax.scatter(
        100 * nonparent.discovery_galaxy_gain,
        100 * nonparent.discovery_cluster_gain,
        c=nonparent.radial_scale,
        s=30 + 70 * nonparent.route_fraction,
        cmap="coolwarm",
        alpha=0.8,
    )
    ax.scatter(100 * selected.discovery_galaxy_gain, 100 * selected.discovery_cluster_gain, marker="*", s=220, c="gold", edgecolor="black")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="discovery galaxy gain (%)", ylabel="discovery cluster gain (%)", title="Universal radial-remap screen")
    fig.colorbar(scatter, ax=ax, label="radial scale lambda")

    ax = axes[0, 1]
    hold = partitions[(partitions.candidate_id == selected_id) & (partitions.partition == "formula_holdout")]
    values = []
    for domain in ("galaxy", "cluster"):
        row = hold[hold.domain == domain].iloc[0]
        values.append(100 * row.gain_vs_parent)
    ax.bar(["galaxies", "clusters"], values, color=["tab:blue", "tab:orange"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="holdout gain vs parent (%)", title="Frozen formula transfer")

    ax = axes[1, 0]
    solar = scores.sort_values("joint_discovery_gain", ascending=False).head(12)
    ax.scatter(solar.joint_discovery_gain * 100, solar.Mercury_precession_mas_per_century, c=solar.all_solar_proxies_pass.map({True: "tab:green", False: "tab:red"}))
    ax.axhspan(-3.1, 3.1, color="0.9")
    ax.set(xlabel="worst discovery-domain gain (%)", ylabel="Mercury proxy (mas/century)", title="Solar safety of leading variants")

    ax = axes[1, 1]
    raw_x = raw_scores.route_fraction * (raw_scores.radial_scale - 1.0)
    raw_colors = np.where(raw_scores.candidate_id.eq(selected_id), "gold", np.where(raw_scores.candidate_id.eq("parent"), "black", "tab:purple"))
    ax.scatter(raw_x, raw_scores.heldout_exact_RMS_arcsec, c=raw_colors, alpha=0.8)
    ax.axhline(raw_scores[raw_scores.candidate_id.eq("parent")].heldout_exact_RMS_arcsec.iloc[0], color="0.4", ls="--")
    ax.set(xlabel="effective radial displacement f(lambda-1)", ylabel="RXJ1347 pair-heldout RMS (arcsec)", title="All post-selection raw-image scores")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0554_radial_flux_remap_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    parent_protocol = json.loads((ROOT / protocol["inputs"]["parent_protocol"]).read_text(encoding="utf-8"))
    spec = dict(protocol["parent"])
    q = float(spec.pop("universal_q"))
    spec.pop("candidate")
    spec["candidate_id"] = "P0554_radial_parent"
    galaxy, _ = prepare_galaxies(parent_protocol, A0)
    cluster, _ = prepare_clusters(parent_protocol)
    galaxy = galaxy.reset_index(drop=True)
    cluster = cluster.reset_index(drop=True)
    galaxy["partition"] = galaxy.galaxy.map(partition)
    cluster["partition"] = cluster.system.map(partition)
    parent_galaxy_response = response_for_frame(
        galaxy,
        spec,
        q=q,
        a0=A0,
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
    )
    parent_cluster_response = response_for_frame(
        cluster,
        spec,
        q=q,
        a0=A0,
        radius_column="radius_kpc",
        gbar_column="gbar_m_s2",
    )
    outer = galaxy.split.eq("outer_holdout").to_numpy()
    candidate_rows, partition_rows = [], []
    galaxy_predictions, cluster_predictions = [], []
    prediction_cache = {}
    for index, item in enumerate(candidates(protocol), start=1):
        print(f"scalar candidate {index}/{protocol['grid']['candidate_count']}: {item['candidate_id']}", flush=True)
        gdyn, _ = remap_profiles(
            galaxy,
            parent_galaxy_response,
            system_column="galaxy",
            radius_column="radius_adjusted_kpc",
            gbar_column="g_bar_m_s2",
            route_fraction=item["route_fraction"],
            radial_scale=item["radial_scale"],
        )
        _, clens = remap_profiles(
            cluster,
            parent_cluster_response,
            system_column="system",
            radius_column="radius_kpc",
            gbar_column="gbar_m_s2",
            route_fraction=item["route_fraction"],
            radial_scale=item["radial_scale"],
        )
        velocity = np.sqrt(np.maximum(gdyn * galaxy.radius_adjusted_kpc.to_numpy(float) * KPC_M / 1.0e6, 0.0))
        prediction_cache[item["candidate_id"]] = (velocity, clens)
        solar = remapped_solar_diagnostics(
            response_parameters=response_parameters(spec, q=q, a0=A0),
            route_fraction=item["route_fraction"],
            radial_scale=item["radial_scale"],
        )
        candidate_rows.append(
            {
                **item,
                **solar,
                "all_solar_proxies_pass": bool(solar["Cassini_proxy_pass"] and solar["Earth_proxy_pass"] and solar["Mercury_proxy_pass"]),
            }
        )
        for split in ("discovery", "formula_holdout"):
            gm = outer & galaxy.partition.eq(split).to_numpy()
            cm = cluster.partition.eq(split).to_numpy()
            for domain, metrics in (
                ("galaxy", galaxy_metrics(galaxy.loc[gm], velocity[gm])),
                ("cluster", cluster_metrics(cluster.loc[cm], clens[cm])),
            ):
                partition_rows.append({"candidate_id": item["candidate_id"], "domain": domain, "partition": split, **metrics})

    scores = pd.DataFrame(candidate_rows)
    partitions = pd.DataFrame(partition_rows)
    parents = partitions[partitions.candidate_id.eq("parent")].set_index(["domain", "partition"])
    partitions["parent_equal_object_RMSE"] = [parents.loc[(row.domain, row.partition), "equal_object_RMSE"] for row in partitions.itertuples()]
    partitions["gain_vs_parent"] = 1.0 - partitions.equal_object_RMSE / partitions.parent_equal_object_RMSE
    discovery = partitions[partitions.partition.eq("discovery")].pivot(index="candidate_id", columns="domain", values="gain_vs_parent")
    scores = scores.merge(
        discovery.rename(columns={"galaxy": "discovery_galaxy_gain", "cluster": "discovery_cluster_gain"}),
        left_on="candidate_id",
        right_index=True,
        validate="one_to_one",
    )
    scores["joint_discovery_gain"] = scores[["discovery_galaxy_gain", "discovery_cluster_gain"]].min(axis=1)
    scores["mean_discovery_gain"] = scores[["discovery_galaxy_gain", "discovery_cluster_gain"]].mean(axis=1)
    scores["effective_log_shift"] = scores.route_fraction * np.log(scores.radial_scale)
    eligible = scores[scores.candidate_id.ne("parent") & scores.all_solar_proxies_pass].copy()
    eligible["distance_from_parent_scale"] = np.abs(eligible.radial_scale - 1.0)
    eligible = eligible.sort_values(
        ["joint_discovery_gain", "mean_discovery_gain", "route_fraction", "distance_from_parent_scale"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    if eligible.empty:
        raise RuntimeError("no non-parent radial remap passed the Solar gate")
    selected = eligible.iloc[0].to_dict()
    selected_id = str(selected["candidate_id"])
    scores["selection_role"] = np.where(scores.candidate_id.eq(selected_id), "frozen_transfer_candidate", np.where(scores.candidate_id.eq("parent"), "parent", "screened"))

    for item in candidates(protocol):
        velocity, clens = prediction_cache[item["candidate_id"]]
        gm = outer
        local_g = galaxy.loc[gm, ["galaxy", "partition", "radius_adjusted_kpc", "velocity_observed_adjusted_km_s"]].copy()
        local_g.insert(0, "candidate_id", item["candidate_id"])
        local_g["predicted_km_s"] = velocity[gm]
        local_g["residual_km_s"] = local_g.predicted_km_s - local_g.velocity_observed_adjusted_km_s
        galaxy_predictions.append(local_g)
        local_c = cluster[["system", "partition", "radius_kpc", "observed_g_m_s2"]].copy()
        local_c.insert(0, "candidate_id", item["candidate_id"])
        local_c["predicted_g_m_s2"] = clens
        local_c["residual_dex"] = np.log10(clens) - cluster.log_gtot.to_numpy(float)
        cluster_predictions.append(local_c)

    raw_scores, raw_predictions, raw_profiles = run_rxj1347(protocol, spec, q, selected)
    parent_raw = raw_scores[raw_scores.candidate_id.eq("parent")].iloc[0]
    raw_scores["gain_vs_parent"] = 1.0 - raw_scores.heldout_exact_RMS_arcsec / float(parent_raw.heldout_exact_RMS_arcsec)
    scores = scores.merge(
        raw_scores[["candidate_id", "gain_vs_parent"]].rename(columns={"gain_vs_parent": "RXJ1347_raw_gain"}),
        on="candidate_id",
        validate="one_to_one",
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    partitions.to_csv(output / protocol["outputs"]["partition_scores"], index=False)
    pd.concat(galaxy_predictions, ignore_index=True).to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    pd.concat(cluster_predictions, ignore_index=True).to_csv(output / protocol["outputs"]["cluster_predictions"], index=False)
    raw_scores.to_csv(output / protocol["outputs"]["raw_scores"], index=False)
    raw_predictions.to_csv(output / protocol["outputs"]["raw_predictions"], index=False)
    raw_profiles.to_csv(output / protocol["outputs"]["radial_profiles"], index=False)

    selected_partitions = partitions[partitions.candidate_id.eq(selected_id)]
    selected_raw = raw_scores[raw_scores.candidate_id.eq(selected_id)].iloc[0]
    holdout_gains = {
        row.domain: float(row.gain_vs_parent)
        for row in selected_partitions[selected_partitions.partition.eq("formula_holdout")].itertuples()
    }
    discovery_gains = {
        row.domain: float(row.gain_vs_parent)
        for row in selected_partitions[selected_partitions.partition.eq("discovery")].itertuples()
    }
    raw_gain = float(selected_raw.gain_vs_parent)
    shifted = scores[scores.candidate_id.ne("parent")].copy()
    effective_shift_fits = {}
    for domain_column in ("discovery_galaxy_gain", "discovery_cluster_gain", "RXJ1347_raw_gain"):
        coefficients = np.polyfit(shifted.effective_log_shift, shifted[domain_column], 2)
        predicted = np.polyval(coefficients, shifted.effective_log_shift)
        residual_sum = float(np.sum(np.square(shifted[domain_column] - predicted)))
        total_sum = float(np.sum(np.square(shifted[domain_column] - shifted[domain_column].mean())))
        effective_shift_fits[domain_column] = {
            "quadratic_coefficients": coefficients.tolist(),
            "R_squared": float(1.0 - residual_sum / total_sum),
        }
    transfer_gate = bool(
        holdout_gains["galaxy"] > 0.0
        and holdout_gains["cluster"] > 0.0
        and selected_raw.heldout_all_roots
        and raw_gain > 0.0
    )
    top = scores.sort_values(["joint_discovery_gain", "mean_discovery_gain"], ascending=False).head(12)
    report = {
        "report_version": "P0554-RADIAL-FLUX-REMAP-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "candidates": len(scores),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "SPARC_discovery_galaxies": int(galaxy[galaxy.partition.eq("discovery")].galaxy.nunique()),
            "SPARC_formula_holdout_galaxies": int(galaxy[galaxy.partition.eq("formula_holdout")].galaxy.nunique()),
            "CLASH_systems": int(cluster.system.nunique()),
            "CLASH_discovery_systems": int(cluster[cluster.partition.eq("discovery")].system.nunique()),
            "CLASH_formula_holdout_systems": int(cluster[cluster.partition.eq("formula_holdout")].system.nunique()),
            "RXJ1347_pair_heldout_images": int(selected_raw.heldout_images),
        },
        "selected": {
            "candidate_id": selected_id,
            "route_fraction": float(selected["route_fraction"]),
            "radial_scale": float(selected["radial_scale"]),
            "discovery_gains": discovery_gains,
            "formula_holdout_gains": holdout_gains,
            "RXJ1347_raw_gain": raw_gain,
            "RXJ1347_all_roots": bool(selected_raw.heldout_all_roots),
            "solar": {key: json_safe(selected[key]) for key in ["maximum_dynamic_fraction_limb_to_Saturn", "maximum_lensing_fraction_limb_to_Saturn", "Earth_orbit_fractional_change", "Saturn_orbit_fractional_change", "Mercury_precession_mas_per_century", "all_solar_proxies_pass"]},
        },
        "top_discovery_candidates": top.to_dict("records"),
        "transfer_gate": {
            "requires_positive_galaxy_cluster_and_raw_RXJ1347_holdout_gains": True,
            "passed": transfer_gate,
        },
        "universal_findings": {
            "all_nonparent_discovery_joint_improvers": int(((scores.candidate_id != "parent") & (scores.discovery_galaxy_gain > 0) & (scores.discovery_cluster_gain > 0)).sum()),
            "all_nonparent_RXJ1347_raw_improvers": int(((raw_scores.candidate_id != "parent") & (raw_scores.gain_vs_parent > 0)).sum()),
            "all_nonparent_three_domain_discovery_improvers": int(((scores.candidate_id != "parent") & (scores.discovery_galaxy_gain > 0) & (scores.discovery_cluster_gain > 0) & (scores.RXJ1347_raw_gain > 0)).sum()),
            "best_RXJ1347_candidate": raw_scores.sort_values("heldout_exact_RMS_arcsec").iloc[0].to_dict(),
            "two_grid_coordinates_collapse_to_effective_log_shift": effective_shift_fits,
            "selected_direction": "inward" if float(selected["radial_scale"]) < 1.0 else "outward",
            "no_formula_promoted": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    make_figure(scores, partitions, raw_scores, selected_id, output / protocol["outputs"]["figure"])
    summary = (
        "# P0554 radial flux-remap result\n\n"
        f"The discovery rule selected `{selected_id}` (f={selected['route_fraction']}, lambda={selected['radial_scale']}). "
        f"Its galaxy/cluster discovery gains were {100*discovery_gains['galaxy']:+.3f}% and {100*discovery_gains['cluster']:+.3f}%; "
        f"formula-holdout gains were {100*holdout_gains['galaxy']:+.3f}% and {100*holdout_gains['cluster']:+.3f}%. "
        f"RXJ1347 raw pair-holdout gain was {100*raw_gain:+.3f}%. Full transfer gate: {transfer_gate}.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
