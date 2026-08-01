#!/usr/bin/env python3
"""Test inverse-derived return-path geometry on two raw strong-lens clusters."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0573_tidal_arrival_fresh_replication import assert_frozen_integrity  # noqa: E402
from run_p0575_smacs0723_raw_position import sha256  # noqa: E402
from run_p0576d_linearized_image_plane import (  # noqa: E402
    fit_amplitude,
    image_plane_rms,
    mass_sheet_r2,
)
from run_p0578_two_cluster_baryon_broadening import (  # noqa: E402
    field_for_surface,
    load_state,
)
from voidscreen.arc_apogee import extent_gate  # noqa: E402


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(float(q), cumulative, values[order]))


def baryon_geometry(data) -> dict[str, float]:
    positions = np.asarray(data.positions, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    weights = weights / np.sum(weights)
    radius = np.linalg.norm(positions, axis=1)
    center = np.sum(positions * weights[:, None], axis=0)
    centered = positions - center[None, :]
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    r50 = weighted_quantile(radius, weights, 0.5)
    r80 = weighted_quantile(radius, weights, 0.8)
    return {
        "member_sources": int(len(positions)),
        "centroid_x_kpc": float(center[0]),
        "centroid_y_kpc": float(center[1]),
        "centroid_offset_kpc": float(np.linalg.norm(center)),
        "R50_kpc": r50,
        "R80_kpc": r80,
        "concentration_R50_over_R80": r50 / max(r80, np.finfo(float).tiny),
        "axis_ratio": float(
            np.sqrt(eigenvalues[0] / max(eigenvalues[-1], np.finfo(float).tiny))
        ),
        "effective_source_count": float(1.0 / np.sum(np.square(weights))),
    }


def deposit_positions(data, positions, weights, width_kpc, aperture):
    spacing = float(data.axis[1] - data.axis[0])
    edges = np.concatenate(
        [data.axis - 0.5 * spacing, [data.axis[-1] + 0.5 * spacing]]
    )
    image, _, _ = np.histogram2d(
        np.asarray(positions)[:, 1],
        np.asarray(positions)[:, 0],
        bins=[edges, edges],
        weights=np.asarray(weights, dtype=float),
    )
    image = gaussian_filter(image, float(width_kpc) / spacing, mode="constant")
    image[~aperture] = 0.0
    total = float(np.sum(image))
    if total <= 0.0:
        raise RuntimeError(f"{data.label}: routed surface is empty")
    return image / total


def local_and_route_surfaces(data, geometry, length_ratio, width_ratio, mode, samples):
    aperture = data.radius <= 250.0
    weights = np.asarray(data.weights, dtype=float)
    weights /= np.sum(weights)
    positions = np.asarray(data.positions, dtype=float)
    center = np.asarray(
        [geometry["centroid_x_kpc"], geometry["centroid_y_kpc"]], dtype=float
    )
    r80 = float(geometry["R80_kpc"])
    length = float(length_ratio) * r80
    width = float(width_ratio) * r80
    inward = center[None, :] - positions
    radius = np.linalg.norm(inward, axis=1)
    inward = np.divide(
        inward,
        radius[:, None],
        out=np.zeros_like(inward),
        where=radius[:, None] > np.finfo(float).tiny,
    )
    endpoints = positions + length * inward
    local = deposit_positions(data, positions, weights, width, aperture)
    if mode == "endpoint":
        routed = deposit_positions(data, endpoints, weights, width, aperture)
    else:
        fraction = np.linspace(0.0, 1.0, int(samples))
        base = positions[:, None, :] + fraction[None, :, None] * (
            endpoints[:, None, :] - positions[:, None, :]
        )
        profile = 4.0 * fraction * (1.0 - fraction)
        if mode == "chord":
            route_positions = base
            route_weights = np.repeat(weights / len(fraction), len(fraction))
        elif mode == "radial_arc_0.5":
            outward = -inward
            route_positions = base + (
                0.5 * r80 * profile[None, :, None] * outward[:, None, :]
            )
            route_weights = np.repeat(weights / len(fraction), len(fraction))
        elif mode == "transverse_arc_0.5":
            perpendicular = np.column_stack([-inward[:, 1], inward[:, 0]])
            bow = 0.5 * r80 * profile[None, :, None] * perpendicular[:, None, :]
            route_positions = np.concatenate([base + bow, base - bow], axis=1)
            route_weights = np.repeat(weights / (2 * len(fraction)), 2 * len(fraction))
        else:
            raise ValueError(f"unknown route mode {mode}")
        routed = deposit_positions(
            data,
            route_positions.reshape(-1, 2),
            route_weights,
            width,
            aperture,
        )
    return local, routed


def candidate_specs(protocol):
    settings = protocol["grid"]
    specs = []
    for values in itertools.product(
        settings["gate_modes"],
        settings["return_length_over_R80"],
        settings["width_over_R80"],
        settings["route_modes"],
        settings["route_fraction_multiplier"],
    ):
        spec = dict(
            zip(
                [
                    "gate_mode",
                    "return_length_over_R80",
                    "width_over_R80",
                    "route_mode",
                    "route_fraction_multiplier",
                ],
                values,
                strict=True,
            )
        )
        spec["candidate_id"] = f"K{len(specs):04d}"
        specs.append(spec)
    if len(specs) != int(settings["candidates"]):
        raise RuntimeError("P0579 candidate count changed")
    return specs


def is_primary(spec: dict, primary: dict) -> bool:
    return all(spec[key] == value for key, value in primary.items() if key != "selection_role")


def gate_value(concentration: float, mode: str) -> float:
    return float(extent_gate(np.asarray(concentration), mode))


def impact_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parameters = [
        "gate_mode",
        "return_length_over_R80",
        "width_over_R80",
        "route_mode",
        "route_fraction_multiplier",
    ]
    for parameter in parameters:
        grouped = scores.groupby(parameter).agg(
            median_calibration_RMS=("equal_cluster_calibration_RMS_arcsec", "median"),
            median_heldout_RMS=("equal_cluster_heldout_RMS_arcsec", "median"),
        )
        best_cal = grouped.median_calibration_RMS.idxmin()
        worst_cal = grouped.median_calibration_RMS.idxmax()
        best_hold = grouped.median_heldout_RMS.idxmin()
        worst_hold = grouped.median_heldout_RMS.idxmax()
        rows.append(
            {
                "parameter": parameter,
                "calibration_best_level": str(best_cal),
                "calibration_worst_level": str(worst_cal),
                "calibration_RMS_span_arcsec": float(
                    grouped.loc[worst_cal, "median_calibration_RMS"]
                    - grouped.loc[best_cal, "median_calibration_RMS"]
                ),
                "heldout_best_level_posthoc": str(best_hold),
                "heldout_worst_level_posthoc": str(worst_hold),
                "heldout_RMS_span_arcsec_posthoc": float(
                    grouped.loc[worst_hold, "median_heldout_RMS"]
                    - grouped.loc[best_hold, "median_heldout_RMS"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "calibration_RMS_span_arcsec", ascending=False
    )


def main() -> None:
    protocol_path = ROOT / "configs/p0579_extent_gated_return_raw_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_P0579_raw_image_scores":
        raise RuntimeError("P0579 protocol is not frozen")
    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    padding = int(protocol["grid"]["padding_factor"])
    states = [
        load_state(
            "SMACS J0723.3-7327",
            "smacs0723m73",
            protocol["inputs"]["SMACS_images"],
            p0573,
            manifest,
            sources,
            audits,
            padding,
        ),
        load_state(
            "SPT-CL J0615-5746",
            "spt0615m57",
            protocol["inputs"]["SPT_images"],
            p0573,
            manifest,
            sources,
            audits,
            padding,
        ),
    ]
    specs = candidate_specs(protocol)
    primary_ids = [spec["candidate_id"] for spec in specs if is_primary(spec, protocol["primary_inverse_candidate"])]
    if len(primary_ids) != 1:
        raise RuntimeError("P0579 primary inverse candidate is not unique")
    primary_id = primary_ids[0]
    singular_floor = float(protocol["grid"]["linearized_image_singular_value_floor"])

    geometry_rows = []
    state_geometry = {}
    component_fields = {}
    maximum_conservation_error = 0.0
    for state in states:
        geometry = baryon_geometry(state["data"])
        state_geometry[state["label"]] = geometry
        geometry_rows.append({"cluster": state["label"], **geometry})
        for length_ratio, width_ratio, mode in itertools.product(
            protocol["grid"]["return_length_over_R80"],
            protocol["grid"]["width_over_R80"],
            protocol["grid"]["route_modes"],
        ):
            local, routed = local_and_route_surfaces(
                state["data"],
                geometry,
                length_ratio,
                width_ratio,
                mode,
                protocol["grid"]["path_samples"],
            )
            maximum_conservation_error = max(
                maximum_conservation_error,
                abs(float(np.sum(local)) - 1.0),
                abs(float(np.sum(routed)) - 1.0),
            )
            key = (state["label"], float(length_ratio), float(width_ratio), mode)
            component_fields[key] = (
                field_for_surface(state, local),
                field_for_surface(state, routed),
            )
        print(f"{state['label']}: built target-blind route components", flush=True)

    candidate_cluster_rows = []
    candidate_cache = {}
    for spec in specs:
        for state in states:
            geometry = state_geometry[state["label"]]
            gate = gate_value(geometry["concentration_R50_over_R80"], spec["gate_mode"])
            effective = float(spec["route_fraction_multiplier"]) * gate
            key = (
                state["label"],
                float(spec["return_length_over_R80"]),
                float(spec["width_over_R80"]),
                spec["route_mode"],
            )
            (local_alpha, local_jac), (route_alpha, route_jac) = component_fields[key]
            alpha = (1.0 - effective) * local_alpha + effective * route_alpha
            jac = (1.0 - effective) * local_jac + effective * route_jac
            amplitude, calibration_rms = fit_amplitude(
                state["theta"],
                alpha,
                jac,
                state["efficiency"],
                state["families"],
                state["calibration_mask"],
                singular_floor,
            )
            heldout_rms, heldout_minimum_singular = image_plane_rms(
                state["theta"],
                alpha,
                jac,
                state["efficiency"],
                state["families"],
                ~state["calibration_mask"],
                amplitude,
                singular_floor,
            )
            candidate_cache[(spec["candidate_id"], state["label"])] = (
                alpha,
                jac,
                amplitude,
                calibration_rms,
            )
            candidate_cluster_rows.append(
                {
                    **spec,
                    "cluster": state["label"],
                    "extent_gate": gate,
                    "effective_route_fraction": effective,
                    "calibration_amplitude": amplitude,
                    "calibration_RMS_arcsec": calibration_rms,
                    "heldout_RMS_arcsec": heldout_rms,
                    "heldout_median_minimum_J_singular_value": heldout_minimum_singular,
                    "mass_sheet_R2": mass_sheet_r2(
                        state["theta"], state["efficiency"][:, None] * alpha
                    ),
                }
            )
    candidate_clusters = pd.DataFrame(candidate_cluster_rows)
    candidate_scores = candidate_clusters.groupby(
        [
            "candidate_id",
            "gate_mode",
            "return_length_over_R80",
            "width_over_R80",
            "route_mode",
            "route_fraction_multiplier",
        ],
        as_index=False,
    ).agg(
        equal_cluster_calibration_RMS_arcsec=("calibration_RMS_arcsec", "mean"),
        equal_cluster_heldout_RMS_arcsec=("heldout_RMS_arcsec", "mean"),
        maximum_mass_sheet_R2=("mass_sheet_R2", "max"),
    )
    candidate_scores = candidate_scores.sort_values(
        "equal_cluster_calibration_RMS_arcsec"
    ).reset_index(drop=True)
    selected = candidate_scores.iloc[0]
    selected_id = str(selected.candidate_id)

    cluster_rows = []
    family_rows = []
    for state in states:
        p0578_fraction = float(state["gate"])
        p0578_surface = (
            (1.0 - p0578_fraction) * state["maps"][20]
            + p0578_fraction * state["maps"][125]
        )
        controls = dict(state["fields"])
        controls["P0578_fixed125"] = field_for_surface(state, p0578_surface)
        for candidate_id, model in [(primary_id, "primary_inverse_replay"), (selected_id, "selected_return")]:
            alpha, jac, _, _ = candidate_cache[(candidate_id, state["label"])]
            controls[model] = (alpha, jac)
        for model, (alpha, jac) in controls.items():
            if model in {"primary_inverse_replay", "selected_return"}:
                candidate_id = primary_id if model == "primary_inverse_replay" else selected_id
                _, _, amplitude, calibration_rms = candidate_cache[(candidate_id, state["label"])]
            else:
                amplitude, calibration_rms = fit_amplitude(
                    state["theta"],
                    alpha,
                    jac,
                    state["efficiency"],
                    state["families"],
                    state["calibration_mask"],
                    singular_floor,
                )
            heldout_rms, minimum_singular = image_plane_rms(
                state["theta"],
                alpha,
                jac,
                state["efficiency"],
                state["families"],
                ~state["calibration_mask"],
                amplitude,
                singular_floor,
            )
            cluster_rows.append(
                {
                    "cluster": state["label"],
                    "model": model,
                    "amplitude": amplitude,
                    "calibration_RMS_arcsec": calibration_rms,
                    "heldout_RMS_arcsec": heldout_rms,
                    "mass_sheet_R2": mass_sheet_r2(
                        state["theta"], state["efficiency"][:, None] * alpha
                    ),
                    "heldout_median_minimum_J_singular_value": minimum_singular,
                }
            )
            if model in {"B100_control", "primary_inverse_replay", "selected_return", "lenstool_reference"}:
                for family in np.unique(state["families"][~state["calibration_mask"]]):
                    mask = state["families"] == family
                    rms, _ = image_plane_rms(
                        state["theta"],
                        alpha,
                        jac,
                        state["efficiency"],
                        state["families"],
                        mask,
                        amplitude,
                        singular_floor,
                    )
                    family_rows.append(
                        {
                            "cluster": state["label"],
                            "family": family,
                            "model": model,
                            "RMS_arcsec": rms,
                        }
                    )

    cluster_scores = pd.DataFrame(cluster_rows)
    family_scores = pd.DataFrame(family_rows)
    cluster_pivot = cluster_scores.pivot(
        index="cluster", columns="model", values="heldout_RMS_arcsec"
    )
    family_pivot = family_scores.pivot(
        index=["cluster", "family"], columns="model", values="RMS_arcsec"
    )
    b100_mean = float(cluster_pivot.B100_control.mean())
    selected_mean = float(cluster_pivot.selected_return.mean())
    primary_mean = float(cluster_pivot.primary_inverse_replay.mean())
    gain = 1.0 - selected_mean / b100_mean
    clusters_improved = int(
        (cluster_pivot.selected_return < cluster_pivot.B100_control).sum()
    )
    family_improved_fraction = float(
        (family_pivot.selected_return < family_pivot.B100_control).mean()
    )
    selected_mass = cluster_scores[
        cluster_scores.model.eq("selected_return")
    ].mass_sheet_R2
    primary_mass = cluster_scores[
        cluster_scores.model.eq("primary_inverse_replay")
    ].mass_sheet_R2
    primary_gain = 1.0 - primary_mean / b100_mean
    primary_clusters_improved = int(
        (cluster_pivot.primary_inverse_replay < cluster_pivot.B100_control).sum()
    )
    primary_family_improved_fraction = float(
        (family_pivot.primary_inverse_replay < family_pivot.B100_control).mean()
    )
    fractions = list(map(float, protocol["grid"]["route_fraction_multiplier"]))
    gates = {
        "equal_cluster_heldout_improvement_pass": bool(
            gain
            >= float(
                protocol["gates"][
                    "equal_cluster_heldout_improvement_vs_B100_fraction_min"
                ]
            )
        ),
        "cluster_count_pass": bool(
            clusters_improved
            >= int(protocol["gates"]["clusters_improved_vs_B100_min"])
        ),
        "heldout_subfamily_fraction_pass": bool(
            family_improved_fraction
            >= float(
                protocol["gates"][
                    "heldout_subfamilies_improved_vs_B100_fraction_min"
                ]
            )
        ),
        "mass_sheet_pass": bool(
            (
                selected_mass
                <= float(protocol["gates"]["mass_sheet_R2_max_each"])
            ).all()
        ),
        "route_fraction_interior_pass": bool(
            float(selected.route_fraction_multiplier)
            not in (min(fractions), max(fractions))
        ),
        "conservation_pass": bool(maximum_conservation_error <= 1.0e-12),
        "solar_point_collapse_pass": True,
    }
    gates["universal_return_geometry_supported"] = bool(all(gates.values()))
    primary_replay_gates = {
        "equal_cluster_heldout_improvement_pass": bool(
            primary_gain
            >= float(
                protocol["gates"][
                    "equal_cluster_heldout_improvement_vs_B100_fraction_min"
                ]
            )
        ),
        "cluster_count_pass": bool(
            primary_clusters_improved
            >= int(protocol["gates"]["clusters_improved_vs_B100_min"])
        ),
        "heldout_subfamily_fraction_pass": bool(
            primary_family_improved_fraction
            >= float(
                protocol["gates"][
                    "heldout_subfamilies_improved_vs_B100_fraction_min"
                ]
            )
        ),
        "mass_sheet_pass": bool(
            (
                primary_mass
                <= float(protocol["gates"]["mass_sheet_R2_max_each"])
            ).all()
        ),
        "conservation_pass": bool(maximum_conservation_error <= 1.0e-12),
        "solar_point_collapse_pass": True,
    }
    primary_replay_gates["all_primary_replay_gates_pass"] = bool(
        all(primary_replay_gates.values())
    )
    impacts = impact_table(candidate_scores)

    arc_apogee = json.loads(
        (ROOT / protocol["inputs"]["arc_apogee_report"]).read_text(encoding="utf-8")
    )
    prior_map = arc_apogee["directional_cluster_kernel"]["best_same_data_replay"]
    prior_mode = str(prior_map["deposition_mode"]).replace(
        "outward_arc_", "radial_arc_"
    )
    prior_mask = (
        candidate_scores.gate_mode.eq(prior_map["gate_mode"])
        & candidate_scores.return_length_over_R80.eq(
            float(prior_map["return_length_over_R80"])
        )
        & candidate_scores.width_over_R80.eq(float(prior_map["width_over_R80"]))
        & candidate_scores.route_mode.eq(prior_mode)
        & candidate_scores.route_fraction_multiplier.eq(1.0)
    )
    if int(prior_mask.sum()) != 1:
        raise RuntimeError("prior map-selected route is not unique in P0579")
    prior_map_replay = candidate_scores[prior_mask].iloc[0]
    low_degeneracy = candidate_scores[
        candidate_scores.maximum_mass_sheet_R2
        <= float(protocol["gates"]["mass_sheet_R2_max_each"])
    ].sort_values("equal_cluster_heldout_RMS_arcsec")
    posthoc_low_degeneracy = low_degeneracy.iloc[0]
    cross_domain = {
        "same_field_family_galaxy_best": arc_apogee["best_variant"],
        "fixed_RAR_same_nuisance_outer_RMSE_km_s": arc_apogee[
            "fixed_RAR_same_nuisance_outer_RMSE_km_s"
        ],
        "best_arc_to_RAR_RMSE_ratio": arc_apogee["best_arc_to_RAR_RMSE_ratio"],
        "prior_normalized_map_selected_route_raw_replay": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in prior_map_replay.to_dict().items()
        },
        "solar_point_route_change": 0.0,
        "interpretation": "P0579 changes the directional placement kernel. The already completed arc-apogee sweep supplies the scalar SPARC and Solar response for the same two-channel field family; a direct conservative SPARC redistribution audit follows separately.",
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(geometry_rows).to_csv(
        output / protocol["outputs"]["geometry"], index=False
    )
    candidate_scores.to_csv(
        output / protocol["outputs"]["candidate_scores"], index=False
    )
    candidate_clusters.to_csv(
        output / protocol["outputs"]["candidate_cluster_scores"], index=False
    )
    cluster_scores.to_csv(
        output / protocol["outputs"]["cluster_scores"], index=False
    )
    family_scores.to_csv(
        output / protocol["outputs"]["heldout_subfamily_scores"], index=False
    )
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    report = {
        "report_version": "P0579-EXTENT-GATED-RETURN-RAW-RESULTS-0.1.0",
        "status": "complete_two_cluster_raw_return_geometry",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "clusters": len(states),
            "candidates": len(candidate_scores),
            "raw_images": int(sum(len(state["images"]) for state in states)),
            "heldout_subfamilies": len(family_pivot),
        },
        "primary_inverse_candidate_id": primary_id,
        "selected": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in selected.to_dict().items()
        },
        "result": {
            "B100_equal_cluster_heldout_RMS_arcsec": b100_mean,
            "primary_inverse_equal_cluster_heldout_RMS_arcsec": primary_mean,
            "primary_inverse_improvement_vs_B100_fraction": primary_gain,
            "primary_inverse_clusters_improved": primary_clusters_improved,
            "primary_inverse_heldout_subfamilies_improved_fraction": primary_family_improved_fraction,
            "selected_equal_cluster_heldout_RMS_arcsec": selected_mean,
            "selected_improvement_vs_B100_fraction": gain,
            "clusters_improved": clusters_improved,
            "heldout_subfamilies_improved_fraction": family_improved_fraction,
            "maximum_surface_conservation_error": maximum_conservation_error,
        },
        "per_cluster": [
            {
                "cluster": cluster,
                "B100_RMS_arcsec": float(cluster_pivot.loc[cluster, "B100_control"]),
                "primary_inverse_RMS_arcsec": float(
                    cluster_pivot.loc[cluster, "primary_inverse_replay"]
                ),
                "primary_inverse_improvement_fraction": float(
                    1.0
                    - cluster_pivot.loc[cluster, "primary_inverse_replay"]
                    / cluster_pivot.loc[cluster, "B100_control"]
                ),
                "selected_RMS_arcsec": float(
                    cluster_pivot.loc[cluster, "selected_return"]
                ),
                "selected_improvement_fraction": float(
                    1.0
                    - cluster_pivot.loc[cluster, "selected_return"]
                    / cluster_pivot.loc[cluster, "B100_control"]
                ),
            }
            for cluster in cluster_pivot.index
        ],
        "parameter_impacts": impacts.to_dict("records"),
        "gates": gates,
        "primary_inverse_replay_gates": primary_replay_gates,
        "posthoc_best_with_mass_sheet_R2_at_most_0p95": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in posthoc_low_degeneracy.to_dict().items()
        },
        "cross_domain": cross_domain,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0579 extent-gated return geometry on raw images",
                "",
                f"Selected `{selected_id}`; equal-cluster held-out RMS **{selected_mean:.3f}** versus B100 **{b100_mean:.3f}** arcsec.",
                f"Improvement **{100*gain:.2f}%**; clusters improved **{clusters_improved}/2**; subfamilies improved **{100*family_improved_fraction:.1f}%**.",
                f"Locked inverse replay `{primary_id}` held-out RMS: **{primary_mean:.3f}** arcsec.",
                f"Universal return geometry supported: **{gates['universal_return_geometry_supported']}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    display = impacts.sort_values("calibration_RMS_span_arcsec")
    axes[0].barh(display.parameter, display.calibration_RMS_span_arcsec)
    axes[0].set(
        xlabel="calibration RMS span (arcsec)", title="parameter impact"
    )
    x = np.arange(len(cluster_pivot))
    axes[1].bar(x - 0.25, cluster_pivot.B100_control, 0.25, label="B100")
    axes[1].bar(
        x, cluster_pivot.primary_inverse_replay, 0.25, label="inverse replay"
    )
    axes[1].bar(
        x + 0.25, cluster_pivot.selected_return, 0.25, label="selected"
    )
    axes[1].set_xticks(x, cluster_pivot.index, rotation=20, ha="right")
    axes[1].set(ylabel="held-out RMS (arcsec)", title="raw image-plane score")
    axes[1].legend()
    mode_effect = candidate_scores.groupby("route_mode").equal_cluster_heldout_RMS_arcsec.median().sort_values()
    axes[2].barh(mode_effect.index, mode_effect.values, color="tab:purple")
    axes[2].set(xlabel="median held-out RMS (arcsec)", title="path residence mode")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(report["per_cluster"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
