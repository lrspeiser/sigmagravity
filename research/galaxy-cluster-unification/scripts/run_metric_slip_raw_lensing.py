#!/usr/bin/env python3
"""Fit one universal metric slip on raw cluster training images and transfer it."""

from __future__ import annotations

import hashlib
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

from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    RawLens,
    near_bound,
    score,
    spec_for,
)
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    aggregate_system_scores,
    json_safe,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)

from voidscreen.metric_slip import (  # noqa: E402
    extra_force_lensing_ratio,
    metric_slip_eta,
    metric_slip_lensing_acceleration,
)
from voidscreen.phenomenology import fixed_rar_enhancement  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_name(index: int) -> str:
    return f"metric_slip_grid_{index:02d}"


def fixed_rar_dynamics(gbar, a_dagger: float) -> np.ndarray:
    gbar = np.asarray(gbar, dtype=float)
    return gbar * fixed_rar_enhancement(gbar, float(a_dagger))


def build_fields(
    anchors: pd.DataFrame,
    local_protocol: dict,
    slip_values: list[float],
    *,
    cutoff_kpc: float,
    a_dagger: float,
) -> tuple[dict[str, RadialDeflectionField], pd.DataFrame]:
    radius_grid = np.geomspace(0.1, float(cutoff_kpc), 4096)
    anchor_radius = anchors["radius_kpc"].to_numpy(float)
    anchor_gbar = np.power(10.0, anchors["log_gbar"].to_numpy(float))
    gbar_grid = loglog_interpolate_with_tails(
        radius_grid,
        anchor_radius,
        anchor_gbar,
        outer_slope=-2.0,
    )
    gdyn_grid = fixed_rar_dynamics(gbar_grid, a_dagger)

    def lookup(values: np.ndarray):
        return lambda radius: np.exp(
            np.interp(np.log(radius), np.log(radius_grid), np.log(values))
        )

    scale = float(
        local_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    impact_kpc = impact_arcsec * scale
    alpha_baryon = spherical_deflection_radians(
        impact_kpc,
        lookup(gbar_grid),
        maximum_radius_kpc=float(cutoff_kpc),
        integration_points=800,
    )
    alpha_dynamics = spherical_deflection_radians(
        impact_kpc,
        lookup(gdyn_grid),
        maximum_radius_kpc=float(cutoff_kpc),
        integration_points=800,
    )

    fields = {}
    profiles = []
    sample_radius = np.geomspace(1.0, min(1000.0, 0.9 * float(cutoff_kpc)), 180)
    sample_gbar = loglog_interpolate_with_tails(
        sample_radius,
        anchor_radius,
        anchor_gbar,
        outer_slope=-2.0,
    )
    sample_gdyn = fixed_rar_dynamics(sample_gbar, a_dagger)
    for index, slip in enumerate(slip_values):
        factor = extra_force_lensing_ratio(slip)
        alpha = alpha_baryon + factor * (alpha_dynamics - alpha_baryon)
        fields[model_name(index)] = RadialDeflectionField(impact_arcsec, alpha)
        sample_glens = metric_slip_lensing_acceleration(sample_gbar, sample_gdyn, slip)
        profiles.append(
            pd.DataFrame(
                {
                    "slip_s": float(slip),
                    "cutoff_kpc": float(cutoff_kpc),
                    "radius_kpc": sample_radius,
                    "gbar_m_s2": sample_gbar,
                    "gdyn_m_s2": sample_gdyn,
                    "glens_m_s2": sample_glens,
                    "eta_Psi_over_Phi": metric_slip_eta(sample_gbar, sample_gdyn, slip),
                }
            )
        )
    return fields, pd.concat(profiles, ignore_index=True)


def finite_rms(value: dict) -> float:
    score_value = value.get("equal_system_radial_RMS_arcsec")
    return float(score_value) if score_value is not None else math.inf


def aggregate_for(
    system_scores: dict,
    labels: list[str],
    slip: float,
    stage: str,
) -> dict:
    return aggregate_system_scores(
        [system_scores[label][f"{slip:g}"][stage] for label in labels]
    )


def choose_slip(
    system_scores: dict,
    labels: list[str],
    slip_values: list[float],
) -> tuple[float, list[dict]]:
    rows = []
    for slip in slip_values:
        aggregate = aggregate_for(system_scores, labels, slip, "training")
        rows.append(
            {
                "slip_s": slip,
                "eligible_for_selection": bool(
                    aggregate["all_roots_converged"]
                    and math.isfinite(finite_rms(aggregate))
                ),
                **aggregate,
            }
        )
    eligible = [row for row in rows if row["eligible_for_selection"]]
    if not eligible:
        raise RuntimeError("No slip-grid point has complete finite training roots")
    chosen = min(
        eligible,
        key=lambda row: (
            finite_rms(row),
            abs(float(row["slip_s"])),
        ),
    )
    return float(chosen["slip_s"]), rows


def fit_grid_for_system(
    system: dict,
    base_protocol: dict,
    protocol: dict,
    catalog: pd.DataFrame,
    tian: pd.DataFrame,
    system_index: int,
) -> tuple[dict, list[pd.DataFrame], list[dict], pd.DataFrame]:
    local = system_protocol(base_protocol, system)
    local["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    images = load_system_images(catalog, system)
    training, heldout = predictive_split(images)
    anchors = load_anchors(tian, system["label"])
    slip_values = list(map(float, protocol["metric_slip"]["grid"]))
    fields, profiles = build_fields(
        anchors,
        local,
        slip_values,
        cutoff_kpc=float(protocol["field_closure"]["primary_maximum_radius_kpc"]),
        a_dagger=float(protocol["matter_law"]["a_dagger_m_s2"]),
    )
    lens = RawLens(local, fields)
    scores = {}
    predictions = []
    geometry = []
    previous = None
    for grid_index, slip in enumerate(slip_values):
        name = model_name(grid_index)
        print(f"system={system['label']} slip={slip:g}", flush=True)
        fitted = lens.fit(
            name,
            training,
            starts=int(protocol["optimization"]["starts_per_grid_point"]),
            seed=int(protocol["optimization"]["random_seed"])
            + 1000 * system_index
            + grid_index,
            initial_override=previous,
        )
        previous = fitted["result"].x
        train_prediction = lens.exact_predictions(
            name,
            fitted["result"].x,
            fitted["sources"],
            training,
            stage="training",
        )
        hold_prediction = lens.exact_predictions(
            name,
            fitted["result"].x,
            fitted["sources"],
            heldout,
            stage="heldout",
        )
        for table in [train_prediction, hold_prediction]:
            table.insert(0, "system", system["system"])
            table.insert(1, "system_label", system["label"])
            table.insert(2, "slip_s", slip)
            table.insert(
                3,
                "cutoff_kpc",
                float(protocol["field_closure"]["primary_maximum_radius_kpc"]),
            )
            predictions.append(table)
        scores[f"{slip:g}"] = {
            "training": score(
                train_prediction,
                lens.sigma,
                free_parameters=len(fitted["result"].x),
            ),
            "heldout": score(hold_prediction, lens.sigma),
            "geometry_at_boundary": near_bound(name, fitted["result"].x),
        }
        spec = spec_for(name)
        geometry.append(
            {
                "system": system["system"],
                "system_label": system["label"],
                "slip_s": slip,
                "cutoff_kpc": float(
                    protocol["field_closure"]["primary_maximum_radius_kpc"]
                ),
                **dict(zip(spec.labels, fitted["result"].x, strict=True)),
            }
        )
    profiles.insert(0, "system", system["system"])
    profiles.insert(1, "system_label", system["label"])
    return scores, predictions, geometry, profiles


def fit_selected_system(
    system: dict,
    base_protocol: dict,
    protocol: dict,
    catalog: pd.DataFrame,
    tian: pd.DataFrame,
    slip: float,
    *,
    cutoff_kpc: float,
    seed: int,
    starts: int,
    stage_prefix: str,
) -> tuple[dict, list[pd.DataFrame], dict, pd.DataFrame]:
    local = system_protocol(base_protocol, system)
    local["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    images = load_system_images(catalog, system)
    training, heldout = predictive_split(images)
    fit_rows = training if len(heldout) else images
    anchors = load_anchors(tian, system["label"])
    fields, profile = build_fields(
        anchors,
        local,
        [slip],
        cutoff_kpc=cutoff_kpc,
        a_dagger=float(protocol["matter_law"]["a_dagger_m_s2"]),
    )
    lens = RawLens(local, fields)
    name = model_name(0)
    fitted = lens.fit(name, fit_rows, starts=starts, seed=seed)
    train_prediction = lens.exact_predictions(
        name,
        fitted["result"].x,
        fitted["sources"],
        fit_rows,
        stage=f"{stage_prefix}_training",
    )
    predictions = [train_prediction]
    if len(heldout):
        heldout_prediction = lens.exact_predictions(
            name,
            fitted["result"].x,
            fitted["sources"],
            heldout,
            stage=f"{stage_prefix}_heldout",
        )
        predictions.append(heldout_prediction)
        heldout_score = score(heldout_prediction, lens.sigma)
    else:
        heldout_score = {"status": "no within-family holdout"}
    for table in predictions:
        table.insert(0, "system", system["system"])
        table.insert(1, "system_label", system["label"])
        table.insert(2, "slip_s", slip)
        table.insert(3, "cutoff_kpc", cutoff_kpc)
    result = {
        "training": score(
            train_prediction,
            lens.sigma,
            free_parameters=len(fitted["result"].x),
        ),
        "heldout": heldout_score,
        "geometry_at_boundary": near_bound(name, fitted["result"].x),
    }
    spec = spec_for(name)
    geometry = {
        "system": system["system"],
        "system_label": system["label"],
        "slip_s": slip,
        "cutoff_kpc": cutoff_kpc,
        **dict(zip(spec.labels, fitted["result"].x, strict=True)),
    }
    profile.insert(0, "system", system["system"])
    profile.insert(1, "system_label", system["label"])
    return result, predictions, geometry, profile


def radial_tian_diagnostic(tian: pd.DataFrame, slip: float, a_dagger: float) -> dict:
    frame = tian.copy()
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    gdyn = fixed_rar_dynamics(gbar, a_dagger)
    glens = metric_slip_lensing_acceleration(gbar, gdyn, slip)
    frame["predicted_log_gobs"] = np.log10(glens)
    frame["residual_dex"] = frame["predicted_log_gobs"] - frame["log_gobs"]
    per_system = frame.groupby("system")["residual_dex"].apply(
        lambda values: float(np.mean(np.square(values)))
    )
    return {
        "systems": int(frame["system"].nunique()),
        "points": len(frame),
        "equal_system_RMSE_dex": float(np.sqrt(per_system.mean())),
        "point_RMSE_dex": float(np.sqrt(np.mean(np.square(frame["residual_dex"])))),
        "mean_residual_dex": float(frame["residual_dex"].mean()),
        "median_predicted_to_observed": float(
            np.median(np.power(10.0, frame["residual_dex"]))
        ),
    }


def solar_diagnostic(slip: float, a_dagger: float) -> dict:
    solar_radius_m = 6.957e8
    au_m = 149597870700.0
    radius_m = np.geomspace(1.6 * solar_radius_m, 8.43 * au_m, 800)
    gm = 1.32712440018e20
    gbar = gm / np.square(radius_m)
    gdyn = fixed_rar_dynamics(gbar, a_dagger)
    eta = metric_slip_eta(gbar, gdyn, slip)
    return {
        "maximum_abs_eta_minus_one_limb_to_Saturn": float(np.max(np.abs(eta - 1.0))),
        "Earth_eta_minus_one": float(np.interp(au_m, radius_m, eta - 1.0)),
        "Saturn_eta_minus_one": float(np.interp(8.43 * au_m, radius_m, eta - 1.0)),
    }


def comparator_aggregate(base_report: dict, labels: list[str], model: str) -> dict:
    names_by_label = {value: key for key, value in base_report["system_labels"].items()}
    return aggregate_system_scores(
        [
            base_report["system_scores"][names_by_label[label]][model]["heldout"]
            for label in labels
        ]
    )


def make_figure(report: dict, grid: pd.DataFrame, output: Path) -> None:
    selected = float(report["selection"]["selected_slip_s"])
    validation = report["cluster_split"]["cross_cluster_validation_labels"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    aggregate = grid[grid["scope"].isin(["selection_training", "validation_heldout"])]
    for scope, block in aggregate.groupby("scope", sort=False):
        axes[0].plot(block["slip_s"], block["equal_system_radial_RMS_arcsec"], marker="o", label=scope)
    axes[0].axvline(selected, color="black", linestyle="--", label=f"selected s={selected:g}")
    axes[0].set(xlabel="universal slip s", ylabel="equal-system radial RMS (arcsec)", title="Slip selection and transfer")
    axes[0].legend(fontsize=8)

    labels = []
    values = []
    for label in validation:
        labels.extend([f"{label}\nzero", f"{label}\nselected", f"{label}\nhalo"])
        values.extend(
            [
                report["system_scores"][label]["0"]["heldout"]["exact_radial_RMS_arcsec"],
                report["system_scores"][label][f"{selected:g}"]["heldout"]["exact_radial_RMS_arcsec"],
                report["comparators"]["per_system_compact_halo_RMS_arcsec"][label],
            ]
        )
    axes[1].bar(labels, values)
    axes[1].set(title="Cross-cluster held-out images", ylabel="radial RMS (arcsec)")
    axes[1].tick_params(axis="x", rotation=25)

    slip_values = grid[grid["scope"].eq("selection_training")]["slip_s"].to_numpy(float)
    axes[2].plot(slip_values, 1.0 + 0.5 * slip_values, marker="o")
    axes[2].axvline(selected, color="black", linestyle="--")
    axes[2].set(
        xlabel="universal slip s",
        ylabel="extra-force lensing / dynamics",
        title="Physical meaning of the slip",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "configs/metric_slip_raw_lensing_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_metric_slip_scores":
        raise RuntimeError("metric-slip protocol was not frozen before scoring")
    galaxy_report_path = ROOT / protocol["inputs"]["galaxy_matter_report"]
    galaxy_report = json.loads(galaxy_report_path.read_text())
    if galaxy_report["advanced_to_slip"] != ["fixed_RAR"]:
        raise RuntimeError("lensing stage requires the frozen galaxy-stage survivor list")

    base_protocol_path = ROOT / protocol["inputs"]["base_multicluster_protocol"]
    base_protocol = json.loads(base_protocol_path.read_text())
    base_report_path = ROOT / protocol["inputs"]["base_multicluster_report"]
    base_report = json.loads(base_report_path.read_text())
    catalog_path = ROOT / protocol["inputs"]["image_catalog"]
    catalog = pd.read_csv(catalog_path)
    tian_path = ROOT / protocol["inputs"]["baryonic_profile"]
    tian = pd.read_csv(
        tian_path,
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    systems_by_label = {item["label"]: item for item in base_protocol["systems"]}
    selection_labels = list(protocol["cluster_split"]["slip_selection_labels"])
    validation_labels = list(protocol["cluster_split"]["cross_cluster_validation_labels"])
    core_labels = selection_labels + validation_labels
    slip_values = list(map(float, protocol["metric_slip"]["grid"]))

    system_scores = {}
    prediction_tables = []
    geometry_rows = []
    profile_tables = []
    for system_index, label in enumerate(core_labels):
        scores, predictions, geometry, profiles = fit_grid_for_system(
            systems_by_label[label],
            base_protocol,
            protocol,
            catalog,
            tian,
            system_index,
        )
        system_scores[label] = scores
        prediction_tables.extend(predictions)
        geometry_rows.extend(geometry)
        profile_tables.append(profiles)

    selected_s, selection_scan = choose_slip(system_scores, selection_labels, slip_values)
    validation_scan = []
    all_training_scan = []
    for slip in slip_values:
        validation_scan.append(
            {"slip_s": slip, **aggregate_for(system_scores, validation_labels, slip, "heldout")}
        )
        all_training_scan.append(
            {"slip_s": slip, **aggregate_for(system_scores, core_labels, slip, "training")}
        )
    selected_validation = aggregate_for(
        system_scores, validation_labels, selected_s, "heldout"
    )
    zero_validation = aggregate_for(system_scores, validation_labels, 0.0, "heldout")

    individual_best = {}
    leave_one_cluster_out = {}
    for label in core_labels:
        own_s, own_scan = choose_slip(system_scores, [label], slip_values)
        individual_best[label] = {
            "training_selected_s": own_s,
            "heldout_at_training_selected_s": system_scores[label][f"{own_s:g}"]["heldout"],
        }
        training_labels = [name for name in core_labels if name != label]
        loco_s, _ = choose_slip(system_scores, training_labels, slip_values)
        leave_one_cluster_out[label] = {
            "selected_s_from_other_clusters": loco_s,
            "heldout": system_scores[label][f"{loco_s:g}"]["heldout"],
        }

    far_scores = {}
    far_cutoff = float(protocol["field_closure"]["far_tail_control_maximum_radius_kpc"])
    for index, label in enumerate(core_labels):
        result, predictions, geometry, profile = fit_selected_system(
            systems_by_label[label],
            base_protocol,
            protocol,
            catalog,
            tian,
            selected_s,
            cutoff_kpc=far_cutoff,
            seed=int(protocol["optimization"]["random_seed"]) + 9000 + index,
            starts=max(4, int(protocol["optimization"]["starts_per_grid_point"])),
            stage_prefix="far_tail_control",
        )
        far_scores[label] = result
        prediction_tables.extend(predictions)
        geometry_rows.append(geometry)
        profile_tables.append(profile)
    far_validation = aggregate_system_scores(
        [far_scores[label]["heldout"] for label in validation_labels]
    )

    stress_scores = {}
    primary_cutoff = float(protocol["field_closure"]["primary_maximum_radius_kpc"])
    for index, label in enumerate(protocol["cluster_split"]["stress_test_labels"]):
        result, predictions, geometry, profile = fit_selected_system(
            systems_by_label[label],
            base_protocol,
            protocol,
            catalog,
            tian,
            selected_s,
            cutoff_kpc=primary_cutoff,
            seed=int(protocol["optimization"]["random_seed"]) + 10000 + index,
            starts=int(protocol["optimization"]["starts_selected_stress_test"]),
            stage_prefix="stress_test",
        )
        stress_scores[label] = result
        prediction_tables.extend(predictions)
        geometry_rows.append(geometry)
        profile_tables.append(profile)

    halo_validation = comparator_aggregate(
        base_report, validation_labels, "GR_plus_cluster_halo"
    )
    p2_validation = comparator_aggregate(base_report, validation_labels, "curvature_power_p2")
    mond_validation = comparator_aggregate(base_report, validation_labels, "fixed_simple_MOND")
    halo_per_system = {}
    names_by_label = {value: key for key, value in base_report["system_labels"].items()}
    for label in validation_labels:
        halo_per_system[label] = base_report["system_scores"][names_by_label[label]][
            "GR_plus_cluster_halo"
        ]["heldout"]["exact_radial_RMS_arcsec"]

    radial_selected = radial_tian_diagnostic(
        tian,
        selected_s,
        float(protocol["matter_law"]["a_dagger_m_s2"]),
    )
    radial_zero = radial_tian_diagnostic(
        tian,
        0.0,
        float(protocol["matter_law"]["a_dagger_m_s2"]),
    )
    solar = solar_diagnostic(
        selected_s,
        float(protocol["matter_law"]["a_dagger_m_s2"]),
    )
    tail_change = abs(
        finite_rms(far_validation) / finite_rms(selected_validation) - 1.0
    )
    halo_ratio = finite_rms(selected_validation) / finite_rms(halo_validation)
    gates = protocol["advance_gates"]
    gate_audit = {
        "galaxy_matter_law_pass": galaxy_report["models"]["fixed_RAR"]["SPARC"][
            "outer_holdout"
        ]["RMSE_km_s"]
        <= float(gates["galaxy_outer_RMSE_relative_to_RAR_max"])
        * float(galaxy_report["references"]["fixed_RAR_outer_RMSE_km_s"]),
        "selected_s_not_grid_boundary_pass": selected_s
        not in {min(slip_values), max(slip_values)},
        "selection_all_roots_pass": aggregate_for(
            system_scores, selection_labels, selected_s, "training"
        )["all_roots_converged"],
        "cross_cluster_all_roots_pass": selected_validation["all_roots_converged"],
        "cross_cluster_absolute_RMS_pass": finite_rms(selected_validation)
        <= float(gates["cross_cluster_equal_system_heldout_RMS_arcsec_max"]),
        "cross_cluster_compact_halo_ratio": halo_ratio,
        "cross_cluster_compact_halo_ratio_pass": halo_ratio
        <= float(gates["cross_cluster_to_compact_halo_RMS_ratio_max"]),
        "cross_cluster_improves_zero_slip_pass": finite_rms(selected_validation)
        < finite_rms(zero_validation),
        "far_tail_fractional_RMS_change": tail_change,
        "far_tail_robustness_pass": tail_change
        <= float(gates["far_tail_fractional_RMS_change_max"]),
        "Solar_System_eta_pass": solar["maximum_abs_eta_minus_one_limb_to_Saturn"]
        <= float(gates["Solar_System_eta_minus_one_max"]),
    }
    gate_audit["all_gates_pass"] = all(
        value for key, value in gate_audit.items() if key.endswith("_pass")
    )

    grid_rows = []
    for scope, rows in [
        ("selection_training", selection_scan),
        ("validation_heldout", validation_scan),
        ("all_core_training", all_training_scan),
    ]:
        for row in rows:
            grid_rows.append({"scope": scope, **row})
    grid_frame = pd.DataFrame(grid_rows)
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed galaxy-locked universal metric-slip test",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "galaxy_matter_report": sha256(galaxy_report_path),
            "base_multicluster_protocol": sha256(base_protocol_path),
            "base_multicluster_report": sha256(base_report_path),
            "image_catalog": sha256(catalog_path),
            "baryonic_profile": sha256(tian_path),
        },
        "matter_law": {
            **protocol["matter_law"],
            "SPARC": galaxy_report["models"]["fixed_RAR"]["SPARC"],
            "BCG_dynamics": galaxy_report["models"]["fixed_RAR"]["BCG_dynamics"],
        },
        "cluster_split": protocol["cluster_split"],
        "selection": {
            "selected_slip_s": selected_s,
            "extra_force_lensing_to_dynamics_ratio": extra_force_lensing_ratio(selected_s),
            "selection_training_aggregate": aggregate_for(
                system_scores, selection_labels, selected_s, "training"
            ),
            "selection_heldout_aggregate": aggregate_for(
                system_scores, selection_labels, selected_s, "heldout"
            ),
        },
        "cross_cluster_validation": {
            "selected_slip": selected_validation,
            "zero_slip": zero_validation,
            "far_tail_control": far_validation,
        },
        "system_scores": system_scores,
        "individual_cluster_training_optima": individual_best,
        "leave_one_cluster_out": leave_one_cluster_out,
        "stress_tests": stress_scores,
        "comparators": {
            "compact_halo_validation": halo_validation,
            "curvature_power_p2_validation": p2_validation,
            "fixed_simple_MOND_validation": mond_validation,
            "per_system_compact_halo_RMS_arcsec": halo_per_system,
        },
        "derived_Tian_lensing_diagnostic": {
            "selected_slip": radial_selected,
            "zero_slip": radial_zero,
        },
        "Solar_System": solar,
        "gate_audit": gate_audit,
        "verdict": {
            "universal_metric_slip_survives": gate_audit["all_gates_pass"],
            "same_object_cluster_matter_lensing_test_available": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    grid_frame.to_csv(ROOT / protocol["outputs"]["grid_scores"], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(geometry_rows).to_csv(
        ROOT / protocol["outputs"]["geometry"], index=False
    )
    pd.concat(profile_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["radial_profiles"], index=False
    )
    make_figure(report, grid_frame, ROOT / protocol["outputs"]["figure"])
    lines = [
        "# Galaxy-locked universal metric slip",
        "",
        f"Selected shared slip: **s={selected_s:g}** "
        f"(extra-force lensing/dynamics={extra_force_lensing_ratio(selected_s):.3f}).",
        "",
        "| score | radial RMS (arcsec) | roots |",
        "|---|---:|---|",
        f"| validation, selected slip | {finite_rms(selected_validation):.3f} | {selected_validation['all_roots_converged']} |",
        f"| validation, zero slip | {finite_rms(zero_validation):.3f} | {zero_validation['all_roots_converged']} |",
        f"| validation, compact halo | {finite_rms(halo_validation):.3f} | {halo_validation['all_roots_converged']} |",
        f"| validation, far-tail control | {finite_rms(far_validation):.3f} | {far_validation['all_roots_converged']} |",
        "",
        f"Universal slip survivor: **{gate_audit['all_gates_pass']}**.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
