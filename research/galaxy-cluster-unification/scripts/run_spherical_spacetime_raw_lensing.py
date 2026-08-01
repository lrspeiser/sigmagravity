#!/usr/bin/env python3
"""Transfer the frozen spherical-spacetime candidate to raw cluster images."""

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

from run_rxj2129_raw_theory_lensing import RawLens, near_bound, score, spec_for  # noqa: E402
from run_spherical_spacetime_cavity import json_safe, model_acceleration  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    aggregate_system_scores,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


MODEL_NAME = "closed_sphere_candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_field(
    model: str,
    parameters: list[float],
    anchors: pd.DataFrame,
    local: dict,
    cutoff_kpc: float,
    maximum_x: float,
) -> tuple[RadialDeflectionField, pd.DataFrame]:
    radius_grid = np.geomspace(0.1, cutoff_kpc, 4096)
    anchor_radius = anchors["radius_kpc"].to_numpy(float)
    anchor_gbar = np.power(10.0, anchors["log_gbar"].to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius_grid, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    acceleration = model_acceleration(
        model, gbar, radius_grid, parameters, maximum_x
    )

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(acceleration)))

    scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    maximum_impact_arcsec = min(500.0, 0.95 * cutoff_kpc / scale)
    impact_arcsec = np.geomspace(0.05, maximum_impact_arcsec, 700)
    physical_alpha = spherical_deflection_radians(
        impact_arcsec * scale,
        lookup,
        maximum_radius_kpc=cutoff_kpc,
        integration_points=800,
    )
    field = RadialDeflectionField(impact_arcsec, physical_alpha)
    sample_radius = np.geomspace(1.0, min(600.0, 0.9 * cutoff_kpc), 180)
    sample_gbar = loglog_interpolate_with_tails(
        sample_radius, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    sample_acceleration = model_acceleration(
        model, sample_gbar, sample_radius, parameters, maximum_x
    )
    profile = pd.DataFrame(
        {
            "radius_kpc": sample_radius,
            "gbar_m_s2": sample_gbar,
            "predicted_acceleration_m_s2": sample_acceleration,
            "enhancement": sample_acceleration / sample_gbar,
        }
    )
    return field, profile


def fit_system(
    system: dict,
    base_protocol: dict,
    protocol: dict,
    catalog: pd.DataFrame,
    tian: pd.DataFrame,
    model: str,
    parameters: list[float],
    cutoff_kpc: float,
    maximum_x: float,
    seed: int,
    starts: int,
) -> tuple[dict, list[pd.DataFrame], dict, pd.DataFrame]:
    local = system_protocol(base_protocol, system)
    local["optimization"]["maximum_function_evaluations"] = int(
        protocol["raw_lensing"]["maximum_function_evaluations"]
    )
    images = load_system_images(catalog, system)
    training, heldout = predictive_split(images)
    fit_rows = training if len(heldout) else images
    anchors = load_anchors(tian, system["label"])
    field, profile = build_field(
        model, parameters, anchors, local, cutoff_kpc, maximum_x
    )
    lens = RawLens(local, {MODEL_NAME: field})
    fitted = lens.fit(MODEL_NAME, fit_rows, starts=starts, seed=seed)
    train_prediction = lens.exact_predictions(
        MODEL_NAME,
        fitted["result"].x,
        fitted["sources"],
        fit_rows,
        stage="training",
    )
    predictions = [train_prediction]
    if len(heldout):
        hold_prediction = lens.exact_predictions(
            MODEL_NAME,
            fitted["result"].x,
            fitted["sources"],
            heldout,
            stage="heldout",
        )
        predictions.append(hold_prediction)
        hold_score = score(hold_prediction, lens.sigma)
    else:
        hold_score = {"status": "no within-family holdout"}
    for table in predictions:
        table.insert(0, "system", system["system"])
        table.insert(1, "system_label", system["label"])
        table.insert(2, "sphere_matter_model", model)
        table.insert(3, "cutoff_kpc", cutoff_kpc)
    result = {
        "training": score(
            train_prediction, lens.sigma, free_parameters=len(fitted["result"].x)
        ),
        "heldout": hold_score,
        "geometry_at_boundary": near_bound(MODEL_NAME, fitted["result"].x),
    }
    spec = spec_for(MODEL_NAME)
    geometry = {
        "system": system["system"],
        "system_label": system["label"],
        "model": model,
        "cutoff_kpc": cutoff_kpc,
        **dict(zip(spec.labels, fitted["result"].x, strict=True)),
    }
    profile.insert(0, "system", system["system"])
    profile.insert(1, "system_label", system["label"])
    profile.insert(2, "cutoff_kpc", cutoff_kpc)
    return result, predictions, geometry, profile


def comparator_aggregate(base_report: dict, labels: list[str], model: str) -> dict:
    names = {label: system for system, label in base_report["system_labels"].items()}
    return aggregate_system_scores(
        [base_report["system_scores"][names[label]][model]["heldout"] for label in labels]
    )


def make_figure(report: dict, output: Path) -> None:
    cutoff = report["cutoff_sensitivity"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(
        [float(value) for value in cutoff],
        [cutoff[value]["equal_system_radial_RMS_arcsec"] for value in cutoff],
        marker="o",
    )
    axes[0].set(
        xlabel="field integration cutoff (kpc)",
        ylabel="validation radial RMS (arcsec)",
        title="Closed-sphere cutoff sensitivity",
    )
    labels = ["sphere", "baryons", "MOND", "compact halo"]
    values = [
        report["cross_cluster_validation"]["equal_system_radial_RMS_arcsec"],
        report["comparators"]["baryons_GR"]["equal_system_radial_RMS_arcsec"],
        report["comparators"]["fixed_simple_MOND"]["equal_system_radial_RMS_arcsec"],
        report["comparators"]["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"],
    ]
    axes[1].bar(labels, values)
    axes[1].set(ylabel="validation radial RMS (arcsec)", title="Unseen raw cluster images")
    axes[1].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "configs/spherical_spacetime_cavity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    galaxy_path = ROOT / protocol["outputs"]["galaxy_report"]
    galaxy = json.loads(galaxy_path.read_text(encoding="utf-8"))
    advanced = list(galaxy["advanced_to_raw_lensing"])
    if advanced:
        candidate = min(
            advanced,
            key=lambda name: galaxy["models"][name]["SPARC"]["outer_holdout"]["RMSE_km_s"],
        )
        transfer_status = "galaxy-qualified"
    else:
        eligible = [
            name
            for name, result in galaxy["models"].items()
            if result["eligible_to_advance"] and result["cluster_domain"]["pass"]
        ]
        candidate = min(
            eligible,
            key=lambda name: galaxy["models"][name]["SPARC"]["outer_holdout"]["RMSE_km_s"],
        )
        transfer_status = "post-failure diagnostic required by frozen rule"
    parameters = galaxy["models"][candidate]["parameter_vector"]
    maximum_x = float(protocol["domain"]["maximum_closed_sphere_x"])
    base_path = ROOT / protocol["inputs"]["base_multicluster_protocol"]
    base_protocol = json.loads(base_path.read_text(encoding="utf-8"))
    base_report_path = ROOT / protocol["inputs"]["base_multicluster_report"]
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    catalog_path = ROOT / protocol["inputs"]["image_catalog"]
    catalog = pd.read_csv(catalog_path)
    tian_path = ROOT / protocol["inputs"]["baryonic_profile"]
    tian = pd.read_csv(
        tian_path,
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    labels = (
        protocol["raw_lensing"]["selection_labels"]
        + protocol["raw_lensing"]["validation_labels"]
        + protocol["raw_lensing"]["stress_labels"]
    )
    systems = {system["label"]: system for system in base_protocol["systems"]}
    primary_cutoff = float(protocol["raw_lensing"]["maximum_radius_kpc"])
    scores = {}
    prediction_tables = []
    geometry_rows = []
    profile_tables = []
    for index, label in enumerate(labels):
        print(f"raw spherical spacetime system={label}", flush=True)
        result, predictions, geometry, profile = fit_system(
            systems[label],
            base_protocol,
            protocol,
            catalog,
            tian,
            candidate,
            parameters,
            primary_cutoff,
            maximum_x,
            20260804 + 100 * index,
            int(protocol["raw_lensing"]["starts_per_model"]),
        )
        scores[label] = result
        prediction_tables.extend(predictions)
        geometry_rows.append(geometry)
        profile_tables.append(profile)

    validation_labels = protocol["raw_lensing"]["validation_labels"]
    validation = aggregate_system_scores([scores[label]["heldout"] for label in validation_labels])
    selection = aggregate_system_scores(
        [scores[label]["training"] for label in protocol["raw_lensing"]["selection_labels"]]
    )
    stress = {label: scores[label] for label in protocol["raw_lensing"]["stress_labels"]}
    cutoff_scores = {f"{primary_cutoff:g}": validation}
    for cutoff_index, cutoff in enumerate([600.0, 1000.0, 2000.0]):
        local_scores = []
        for system_index, label in enumerate(validation_labels):
            print(f"cutoff={cutoff:g} system={label}", flush=True)
            result, predictions, geometry, profile = fit_system(
                systems[label],
                base_protocol,
                protocol,
                catalog,
                tian,
                candidate,
                parameters,
                cutoff,
                maximum_x,
                20261804 + 100 * cutoff_index + system_index,
                4,
            )
            local_scores.append(result["heldout"])
            prediction_tables.extend(predictions)
            geometry_rows.append(geometry)
            profile_tables.append(profile)
        cutoff_scores[f"{cutoff:g}"] = aggregate_system_scores(local_scores)
    cutoff_scores = dict(sorted(cutoff_scores.items(), key=lambda item: float(item[0])))
    comparators = {
        name: comparator_aggregate(base_report, validation_labels, name)
        for name in ["baryons_GR", "fixed_simple_MOND", "GR_plus_cluster_halo"]
    }
    finite_cutoffs = [
        value["equal_system_radial_RMS_arcsec"]
        for value in cutoff_scores.values()
        if value["equal_system_radial_RMS_arcsec"] is not None
    ]
    cutoff_fractional_span = (
        (max(finite_cutoffs) - min(finite_cutoffs)) / validation["equal_system_radial_RMS_arcsec"]
        if finite_cutoffs
        else math.inf
    )
    report = {
        "report_version": "SPHERICAL-SPACETIME-RAW-LENSING-0.1.0",
        "status": "completed fixed-parameter raw-lensing transfer",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "input_hashes": {
            "galaxy_report": sha256(galaxy_path),
            "base_protocol": sha256(base_path),
            "base_report": sha256(base_report_path),
            "catalog": sha256(catalog_path),
            "baryonic_profile": sha256(tian_path),
        },
        "candidate": candidate,
        "parameters": galaxy["models"][candidate]["parameters"],
        "transfer_status": transfer_status,
        "selection_training": selection,
        "cross_cluster_validation": validation,
        "system_scores": scores,
        "stress_tests": stress,
        "cutoff_sensitivity": cutoff_scores,
        "cutoff_fractional_RMS_span": cutoff_fractional_span,
        "comparators": comparators,
        "verdict": {
            "galaxy_gate_passed": bool(advanced),
            "raw_lensing_better_than_compact_halo": bool(
                validation["equal_system_radial_RMS_arcsec"]
                < comparators["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"]
            ),
            "cutoff_robust_within_20_percent": bool(cutoff_fractional_span <= 0.2),
            "spherical_spacetime_candidate_survives": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output = (ROOT / protocol["outputs"]["raw_lensing_report"]).parent
    (ROOT / protocol["outputs"]["raw_lensing_report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["raw_lensing_predictions"], index=False
    )
    pd.DataFrame(geometry_rows).to_csv(output / "raw_lensing_geometry.csv", index=False)
    pd.concat(profile_tables, ignore_index=True).to_csv(output / "raw_lensing_profiles.csv", index=False)
    make_figure(report, output / "raw_lensing.png")
    summary_path = ROOT / protocol["outputs"]["summary"]
    summary_base = summary_path.read_text(encoding="utf-8").split(
        "## Frozen post-failure raw transfer", maxsplit=1
    )[0].rstrip()
    summary_section = [
        "## Frozen post-failure raw transfer",
        "",
        "The least-bad exact candidate was transferred without refitting its curvature",
        f"radius. Unseen raw-image RMS is **{validation['equal_system_radial_RMS_arcsec']:.3f} arcsec**, "
        f"versus {comparators['baryons_GR']['equal_system_radial_RMS_arcsec']:.3f} for baryons,",
        f"{comparators['fixed_simple_MOND']['equal_system_radial_RMS_arcsec']:.3f} for fixed simple MOND, "
        f"and {comparators['GR_plus_cluster_halo']['equal_system_radial_RMS_arcsec']:.3f} for the compact halo. "
        f"The score changes by only {100.0 * cutoff_fractional_span:.2f}% across",
        "600--3,000-kpc cutoffs. Candidate survives: **False**.",
    ]
    summary_path.write_text(
        summary_base + "\n\n" + "\n".join(summary_section) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "candidate": candidate,
        "transfer_status": transfer_status,
        "validation_RMS_arcsec": validation["equal_system_radial_RMS_arcsec"],
        "halo_RMS_arcsec": comparators["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"],
        "cutoff_fractional_span": cutoff_fractional_span,
    }), indent=2))


if __name__ == "__main__":
    main()
