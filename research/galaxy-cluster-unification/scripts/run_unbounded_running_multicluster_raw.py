#!/usr/bin/env python3
"""Locked unbounded-running laws on multiple raw CLASH image catalogs."""

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
from astropy.cosmology import FlatLambdaCDM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    RawLens,
    near_bound,
    score,
    spec_for,
)
from voidscreen.phenomenology import simple_mond_enhancement  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.unbounded_running import predict_running_acceleration  # noqa: E402


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def system_protocol(protocol: dict, system: dict) -> dict:
    cosmology = FlatLambdaCDM(
        H0=float(protocol["cosmology"]["H0_km_s_Mpc"]),
        Om0=float(protocol["cosmology"]["Omega_m"]),
    )
    scale = float(
        cosmology.kpc_proper_per_arcmin(float(system["lens_redshift"])).value / 60.0
    )
    return {
        "raw_lensing_inputs": {
            "position_sigma_arcsec_per_coordinate": float(system["position_sigma_arcsec"])
        },
        "cosmology_and_coordinates": {
            "lens_redshift": float(system["lens_redshift"]),
            "reference_source_redshift": float(protocol["cosmology"]["reference_source_redshift"]),
            "H0_km_s_Mpc": float(protocol["cosmology"]["H0_km_s_Mpc"]),
            "Omega_m": float(protocol["cosmology"]["Omega_m"]),
            "center_ra_deg": float(system["center_ra_deg"]),
            "center_dec_deg": float(system["center_dec_deg"]),
            "angular_scale_kpc_per_arcsec": scale,
        },
        "optimization": {
            "maximum_function_evaluations": int(protocol["optimization"]["maximum_function_evaluations"])
        },
    }


def load_system_images(catalog: pd.DataFrame, system: dict) -> pd.DataFrame:
    selected = catalog[
        catalog.system.eq(system["system"])
        & catalog.metric_neutral_likelihood_row.astype(bool)
    ].copy()
    selected["image_id"] = selected.image_id.astype(str)
    selected["source_family"] = selected.family_id.astype(int)
    selected["source_redshift"] = selected.spectroscopic_redshift.astype(float)
    cosine = math.cos(math.radians(float(system["center_dec_deg"])))
    selected["x_arcsec"] = (
        selected.ra_deg.astype(float) - float(system["center_ra_deg"])
    ) * 3600.0 * cosine
    selected["y_arcsec"] = (
        selected.dec_deg.astype(float) - float(system["center_dec_deg"])
    ) * 3600.0
    selected["radius_arcsec"] = np.hypot(selected.x_arcsec, selected.y_arcsec)
    selected = selected.sort_values(["source_family", "image_id"]).reset_index(drop=True)
    if len(selected) != int(system["images"]):
        raise RuntimeError(f"image count changed for {system['system']}")
    if selected.source_family.nunique() != int(system["families"]):
        raise RuntimeError(f"family count changed for {system['system']}")
    if not np.allclose(
        selected.position_sigma_axis_1_arcsec.astype(float),
        float(system["position_sigma_arcsec"]),
    ):
        raise RuntimeError(f"position error changed for {system['system']}")
    return selected


def predictive_split(images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout_indices = []
    for _, group in images.groupby("source_family", sort=True):
        if len(group) >= 3:
            heldout_indices.append(group.sort_values("image_id").index[-1])
    heldout = images.loc[heldout_indices].copy().sort_values(["source_family", "image_id"])
    training = images.drop(index=heldout_indices).copy().sort_values(["source_family", "image_id"])
    if len(heldout) and training.groupby("source_family").size().min() < 2:
        raise RuntimeError("predictive split left fewer than two training images")
    return training.reset_index(drop=True), heldout.reset_index(drop=True)


def load_anchors(tian: pd.DataFrame, label: str) -> pd.DataFrame:
    anchors = tian[tian.system.eq(label)].copy().sort_values("radius_kpc")
    if len(anchors) < 3:
        raise RuntimeError(f"{label} has fewer than three baryonic anchors")
    return anchors


def acceleration(model: str, radius, anchors: pd.DataFrame, protocol: dict) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(radius, dtype=float)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    if model in {"baryons_GR", "GR_plus_cluster_halo"}:
        predicted = gbar
    elif model == "fixed_simple_MOND":
        predicted = gbar * simple_mond_enhancement(
            gbar, float(protocol["comparators"][model]["a0_m_s2"])
        )
    elif model in protocol["models"]:
        specification = protocol["models"][model]
        predicted = predict_running_acceleration(
            gbar,
            radius,
            specification["running_model"],
            specification["parameters"],
        )["predicted_acceleration_m_s2"]
    else:
        raise ValueError(model)
    return gbar, predicted


def build_field(model: str, anchors: pd.DataFrame, protocol: dict, local_protocol: dict, cutoff_kpc: float):
    radius_grid = np.geomspace(0.1, float(cutoff_kpc), 4096)
    _, predicted = acceleration(model, radius_grid, anchors, protocol)

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(predicted)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    impact_kpc = impact_arcsec * float(
        local_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    physical_alpha = spherical_deflection_radians(
        impact_kpc,
        lookup,
        maximum_radius_kpc=float(cutoff_kpc),
        integration_points=800,
    )
    field = RadialDeflectionField(impact_arcsec, physical_alpha)
    sample_radius = np.geomspace(1.0, min(1000.0, float(cutoff_kpc) * 0.9), 180)
    sample_gbar, sample_predicted = acceleration(model, sample_radius, anchors, protocol)
    profile = pd.DataFrame(
        {
            "model": model,
            "cutoff_kpc": float(cutoff_kpc),
            "radius_kpc": sample_radius,
            "gbar_m_s2": sample_gbar,
            "predicted_acceleration_m_s2": sample_predicted,
            "physical_deflection_arcsec_before_distance_ratio": field.reduced_alpha_arcsec(
                sample_radius
                / float(local_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]),
                1.0,
            ),
        }
    )
    return field, profile


def aggregate_system_scores(rows: list[dict]) -> dict:
    finite = [row for row in rows if np.isfinite(row["exact_radial_RMS_arcsec"])]
    if not finite:
        return {"systems": len(rows), "all_roots_converged": False, "equal_system_radial_RMS_arcsec": None}
    return {
        "systems": len(rows),
        "images": int(sum(row["images"] for row in rows)),
        "all_roots_converged": bool(all(row["all_roots_converged"] for row in rows)),
        "equal_system_radial_RMS_arcsec": float(np.sqrt(np.mean([row["exact_radial_RMS_arcsec"] ** 2 for row in finite]))),
        "median_system_radial_RMS_arcsec": float(np.median([row["exact_radial_RMS_arcsec"] for row in finite])),
        "pooled_coordinate_chi2": float(sum(row["coordinate_chi2"] for row in finite)),
        "pooled_degrees_of_freedom": int(sum(row["degrees_of_freedom"] for row in finite)),
        "pooled_reduced_chi2": float(sum(row["coordinate_chi2"] for row in finite) / max(1, sum(row["degrees_of_freedom"] for row in finite))),
    }


def make_figure(report: dict, output: Path) -> None:
    models = list(report["primary_aggregate"])
    systems = list(report["unseen_raw_observable_systems"])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    width = 0.15
    x = np.arange(len(systems))
    for index, model in enumerate(models):
        values = [report["system_scores"][system][model]["heldout"]["exact_radial_RMS_arcsec"] for system in systems]
        axes[0].bar(x + (index - 2) * width, values, width, label=model)
    axes[0].set_xticks(x, [report["system_labels"][system] for system in systems], rotation=25)
    axes[0].set(ylabel="held-out radial RMS (arcsec)", title="Previously unscored raw image likelihoods")
    axes[0].legend(fontsize=8)
    aggregate = [report["primary_aggregate"][model]["equal_system_radial_RMS_arcsec"] for model in models]
    sensitivity = [report["cutoff_sensitivity_aggregate"].get(model, {}).get("equal_system_radial_RMS_arcsec", np.nan) for model in models]
    xx = np.arange(len(models))
    axes[1].bar(xx - 0.18, aggregate, 0.36, label="isolated tail to 1 Gpc")
    axes[1].bar(xx + 0.18, sensitivity, 0.36, label="tail truncated at 3 Mpc")
    axes[1].set_xticks(xx, [name.replace("curvature_", "c:").replace("fixed_simple_", "") for name in models], rotation=25)
    axes[1].set(ylabel="equal-system held-out RMS (arcsec)", title="Far-field closure sensitivity")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "configs/unbounded_running_multicluster_raw_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_multicluster_raw_scores":
        raise RuntimeError("multi-cluster protocol was not frozen")
    image_path = ROOT / protocol["inputs"]["image_catalog"]
    catalog = pd.read_csv(image_path)
    tian = pd.read_csv(
        ROOT / protocol["baryonic_profile"]["input"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    models = ["baryons_GR", "fixed_simple_MOND", *protocol["models"], "GR_plus_cluster_halo"]
    primary_cutoff = float(protocol["photon_and_environment_closure"]["primary_isolated_tail_cutoff_kpc"])
    sensitivity_cutoff = float(protocol["photon_and_environment_closure"]["cluster_environment_sensitivity_cutoff_kpc"])
    predictions = []
    parameter_rows = []
    profile_rows = []
    system_scores = {}
    cutoff_scores = {}
    unseen_systems = []
    system_labels = {}

    for system_index, system in enumerate(protocol["systems"]):
        print(f"system={system['label']}", flush=True)
        local = system_protocol(protocol, system)
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        anchors = load_anchors(tian, system["label"])
        system_scores[system["system"]] = {}
        cutoff_scores[system["system"]] = {}
        system_labels[system["system"]] = system["label"]
        if system["predictive_status"] == "raw_observable_unseen_same_bridge_system":
            unseen_systems.append(system["system"])

        fields = {}
        for model in models:
            field_model = "baryons_GR" if model == "GR_plus_cluster_halo" else model
            if field_model not in fields:
                fields[field_model], profile = build_field(field_model, anchors, protocol, local, primary_cutoff)
                profile.insert(0, "system", system["system"])
                profile_rows.append(profile)
        lens = RawLens(local, fields)

        for model_index, model in enumerate(models):
            fit_rows = training if len(heldout) else images
            print(f"  fit model={model} training={len(fit_rows)} heldout={len(heldout)}", flush=True)
            fitted = lens.fit(
                model,
                fit_rows,
                starts=int(protocol["optimization"]["multi_starts"]),
                seed=int(protocol["optimization"]["random_seed"]) + 100 * system_index + model_index,
            )
            train_prediction = lens.exact_predictions(model, fitted["result"].x, fitted["sources"], fit_rows, stage="training")
            train_prediction.insert(0, "system", system["system"])
            train_prediction.insert(1, "cutoff_kpc", primary_cutoff)
            predictions.append(train_prediction)
            if len(heldout):
                hold_prediction = lens.exact_predictions(model, fitted["result"].x, fitted["sources"], heldout, stage="heldout")
                hold_prediction.insert(0, "system", system["system"])
                hold_prediction.insert(1, "cutoff_kpc", primary_cutoff)
                predictions.append(hold_prediction)
                heldout_score = score(hold_prediction, lens.sigma)
            else:
                heldout_score = {"status": "no within-family holdout"}
            system_scores[system["system"]][model] = {
                "training": score(train_prediction, lens.sigma, free_parameters=len(fitted["result"].x)),
                "heldout": heldout_score,
                "geometry_at_boundary": near_bound(model, fitted["result"].x),
            }
            spec = spec_for(model)
            parameter_rows.append({"system": system["system"], "model": model, "cutoff_kpc": primary_cutoff, **dict(zip(spec.labels, fitted["result"].x, strict=True))})

        if len(heldout):
            sensitivity_fields = {}
            for candidate in protocol["models"]:
                sensitivity_fields[candidate], profile = build_field(candidate, anchors, protocol, local, sensitivity_cutoff)
                profile.insert(0, "system", system["system"])
                profile_rows.append(profile)
            sensitivity_lens = RawLens(local, sensitivity_fields)
            for candidate_index, candidate in enumerate(protocol["models"]):
                print(f"  cutoff sensitivity model={candidate}", flush=True)
                fitted = sensitivity_lens.fit(
                    candidate,
                    training,
                    starts=int(protocol["optimization"]["multi_starts"]),
                    seed=int(protocol["optimization"]["random_seed"]) + 100 * system_index + 20 + candidate_index,
                )
                hold_prediction = sensitivity_lens.exact_predictions(candidate, fitted["result"].x, fitted["sources"], heldout, stage="heldout_cutoff_sensitivity")
                hold_prediction.insert(0, "system", system["system"])
                hold_prediction.insert(1, "cutoff_kpc", sensitivity_cutoff)
                predictions.append(hold_prediction)
                cutoff_scores[system["system"]][candidate] = score(hold_prediction, sensitivity_lens.sigma)
                spec = spec_for(candidate)
                parameter_rows.append({"system": system["system"], "model": candidate, "cutoff_kpc": sensitivity_cutoff, **dict(zip(spec.labels, fitted["result"].x, strict=True))})

    primary_aggregate = {}
    for model in models:
        rows = [system_scores[name][model]["heldout"] for name in unseen_systems]
        primary_aggregate[model] = aggregate_system_scores(rows)
    cutoff_aggregate = {}
    for model in protocol["models"]:
        rows = [cutoff_scores[name][model] for name in unseen_systems]
        cutoff_aggregate[model] = aggregate_system_scores(rows)

    gates = protocol["advance_gates"]
    audits = {}
    for model in protocol["models"]:
        primary = primary_aggregate[model]
        truncated = cutoff_aggregate[model]
        ratio = primary["equal_system_radial_RMS_arcsec"] / primary_aggregate["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"]
        cutoff_change = abs(truncated["equal_system_radial_RMS_arcsec"] / primary["equal_system_radial_RMS_arcsec"] - 1.0)
        no_boundary = all(
            not any(system_scores[name][model]["geometry_at_boundary"].values())
            for name in unseen_systems
        )
        audits[model] = {
            "all_roots_converged": primary["all_roots_converged"],
            "absolute_RMS_pass": primary["equal_system_radial_RMS_arcsec"] <= gates["equal_system_heldout_radial_RMS_arcsec_max"],
            "beats_simple_MOND": primary["equal_system_radial_RMS_arcsec"] < primary_aggregate["fixed_simple_MOND"]["equal_system_radial_RMS_arcsec"],
            "compact_halo_RMS_ratio": ratio,
            "within_compact_halo_ratio_gate": ratio <= gates["candidate_to_compact_halo_equal_system_RMS_ratio_max"],
            "no_geometry_parameter_at_boundary": no_boundary,
            "cutoff_fractional_change": cutoff_change,
            "cutoff_robustness_pass": cutoff_change <= gates["primary_to_3Mpc_cutoff_RMS_fractional_change_max"],
        }
        audits[model]["all_gates_pass"] = all(
            value for key, value in audits[model].items() if key.endswith("pass") or key in {"all_roots_converged", "beats_simple_MOND", "no_geometry_parameter_at_boundary"}
        )

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed multi-cluster raw-observable transfer",
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(config_path)},
        "input_hashes": {"image_catalog": sha256(image_path), "baryonic_profile": sha256(ROOT / protocol["baryonic_profile"]["input"])},
        "claim_boundary": protocol["pre_score_disclosure"],
        "unseen_raw_observable_systems": unseen_systems,
        "system_labels": system_labels,
        "system_scores": system_scores,
        "primary_aggregate": primary_aggregate,
        "cutoff_sensitivity_scores": cutoff_scores,
        "cutoff_sensitivity_aggregate": cutoff_aggregate,
        "gate_audit": audits,
        "verdict": {"survivors": [name for name, audit in audits.items() if audit["all_gates_pass"]]},
    }
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(ROOT / protocol["outputs"]["predictions"], index=False)
    pd.DataFrame(parameter_rows).to_csv(ROOT / protocol["outputs"]["parameters"], index=False)
    pd.concat(profile_rows, ignore_index=True).to_csv(ROOT / protocol["outputs"]["radial_profiles"], index=False)
    make_figure(report, ROOT / protocol["outputs"]["figure"])
    ranking = sorted(primary_aggregate, key=lambda name: primary_aggregate[name]["equal_system_radial_RMS_arcsec"])
    lines = ["# Unbounded running: multi-cluster raw-image transfer", "", "Four previously unscored raw image likelihoods are predictive transfers on systems that were present in the derived-field bridge; they are not external-system validation.", "", "| rank | model | equal-system held-out RMS (arcsec) | pooled reduced chi2 | 3 Mpc sensitivity RMS |", "|---:|---|---:|---:|---:|"]
    for index, model in enumerate(ranking, 1):
        primary = primary_aggregate[model]
        sensitivity = cutoff_aggregate.get(model, {}).get("equal_system_radial_RMS_arcsec")
        label = f"{sensitivity:.3f}" if sensitivity is not None else "not run"
        lines.append(f"| {index} | {model} | {primary['equal_system_radial_RMS_arcsec']:.3f} | {primary['pooled_reduced_chi2']:.2f} | {label} |")
    lines += ["", f"Survivors: **{', '.join(report['verdict']['survivors']) or 'none'}**."]
    (ROOT / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((ROOT / protocol["outputs"]["summary"]).read_text())


if __name__ == "__main__":
    main()
