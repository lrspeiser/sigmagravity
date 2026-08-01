#!/usr/bin/env python3
"""Run the frozen curl-free member-tidal metric pilot."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_metric_slip_raw_lensing import build_fields as build_metric_fields
from run_rxj2129_raw_theory_lensing import (
    RawLens,
    near_bound,
    pseudo_elliptical_deflection,
    score,
    shear_deflection,
    spec_for,
)
from run_unbounded_running_multicluster_raw import (
    aggregate_system_scores,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from run_unbounded_running_spatial_vector import load_member_table
from voidscreen.tidal_metric import TidalCorrectionField, build_tidal_correction_field


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class SystemContext:
    system: dict
    local_protocol: dict
    training: pd.DataFrame
    heldout: pd.DataFrame
    members: pd.DataFrame
    fields: dict
    correction: TidalCorrectionField
    initial_geometry: np.ndarray
    extra_alpha: object


class MemberTidalLens(RawLens):
    """Locked scalar metric slip plus one precomputed curl-free tensor correction."""

    def __init__(self, protocol, fields, correction, coupling):
        super().__init__(protocol, fields)
        self.correction = correction
        self.coupling = float(coupling)

    def alpha(self, model, parameters, x_arcsec, y_arcsec, source_redshift):
        x = np.asarray(x_arcsec, dtype=float)
        y = np.asarray(y_arcsec, dtype=float)
        ratio = self.distance_ratio(source_redshift)
        distance_scale = ratio / self.distance_ratio_ref
        q, phi, cx, cy, gamma1, gamma2 = parameters
        scalar_field = self.fields["scalar_slip"]
        base_x, base_y = pseudo_elliptical_deflection(
            x,
            y,
            lambda radius: scalar_field.reduced_alpha_arcsec(radius, ratio),
            axis_ratio=q,
            phi_radian=phi,
            center_x_arcsec=cx,
            center_y_arcsec=cy,
        )
        correction_x, correction_y = self.correction.alpha_arcsec(x, y)
        shear_x, shear_y = shear_deflection(
            x, y, gamma1 * distance_scale, gamma2 * distance_scale
        )
        return (
            base_x + self.coupling * ratio * correction_x + shear_x,
            base_y + self.coupling * ratio * correction_y + shear_y,
        )


def model_name(coupling: float) -> str:
    sign = "m" if coupling < 0 else "p"
    return f"member_tidal_{sign}{abs(coupling):.3f}".replace(".", "d")


def geometry_initials(path: Path, slip: float, cutoff_kpc: float) -> dict[str, np.ndarray]:
    table = pd.read_csv(path)
    table = table[
        np.isclose(table.slip_s.astype(float), slip)
        & np.isclose(table.cutoff_kpc.astype(float), cutoff_kpc)
    ]
    labels = spec_for("fixed").labels
    result = {}
    for system, block in table.groupby("system", sort=False):
        if len(block) != 1:
            raise RuntimeError(f"ambiguous locked scalar geometry for {system}")
        result[str(system)] = block.iloc[0][list(labels)].to_numpy(float)
    return result


def correction_for(
    members: pd.DataFrame,
    local: dict,
    fields: dict,
    protocol: dict,
    softening_kpc: float,
) -> tuple[TidalCorrectionField, object]:
    scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    softening_arcsec = float(softening_kpc) / scale
    baryon = fields["baryon"]
    scalar = fields["scalar_slip"]

    def extra_alpha(radius_arcsec):
        return scalar.reduced_alpha_arcsec(radius_arcsec, 1.0) - baryon.reduced_alpha_arcsec(
            radius_arcsec, 1.0
        )

    numerical = protocol["numerics"]
    field = build_tidal_correction_field(
        members.x_arcsec.to_numpy(float),
        members.y_arcsec.to_numpy(float),
        members.normalized_light_weight.to_numpy(float),
        softening_arcsec=softening_arcsec,
        extra_alpha_arcsec=extra_alpha,
        half_width_arcsec=float(numerical["field_half_width_arcsec"]),
        pixels_per_axis=int(numerical["field_pixels_per_axis"]),
        polar_mean_radii=int(numerical["polar_mean_radii"]),
        polar_mean_azimuths=int(numerical["polar_mean_azimuths"]),
        subtract_circular_mean=bool(
            protocol["environment_tensor"].get("subtract_circular_mean", True)
        ),
    )
    return field, extra_alpha


def build_contexts(protocol: dict, *, softening_kpc: float) -> tuple[list[SystemContext], list[dict], dict]:
    base_path = ROOT / protocol["inputs"]["base_multicluster_protocol"]
    member_path = ROOT / protocol["inputs"]["member_system_protocol"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    member_protocol = json.loads(member_path.read_text(encoding="utf-8"))
    catalog_path = ROOT / protocol["inputs"]["image_catalog"]
    baryonic_path = ROOT / protocol["inputs"]["baryonic_profile"]
    catalog = pd.read_csv(catalog_path)
    tian = pd.read_csv(
        baryonic_path,
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    metric_protocol = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_protocol"]).read_text(encoding="utf-8")
    )
    slip = float(protocol["weak_field_equation"]["locked_slip_s"])
    cutoff = float(metric_protocol["field_closure"]["primary_maximum_radius_kpc"])
    a_dagger = float(metric_protocol["matter_law"]["a_dagger_m_s2"])
    initials = geometry_initials(
        ROOT / "results/metric_slip_raw_lensing/geometry.csv", slip, cutoff
    )
    contexts = []
    audits = []
    hashes = {
        "protocol": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "base_multicluster_protocol": sha256(base_path),
        "member_system_protocol": sha256(member_path),
        "image_catalog": sha256(catalog_path),
        "baryonic_profile": sha256(baryonic_path),
        "metric_slip_report": sha256(ROOT / protocol["inputs"]["metric_slip_report"]),
    }
    by_label = {item["label"]: item for item in member_protocol["systems"]}
    labels = protocol["cluster_split"]["selection_labels"] + protocol["cluster_split"]["validation_labels"]
    for label in labels:
        system = by_label[label]
        print(f"build map system={label} softening={softening_kpc:g} kpc", flush=True)
        local = system_protocol(base, system)
        local["optimization"]["maximum_function_evaluations"] = int(
            protocol["optimization"]["maximum_function_evaluations"]
        )
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        anchors = load_anchors(tian, label)
        raw_fields, _ = build_metric_fields(
            anchors,
            local,
            [-2.0, slip],
            cutoff_kpc=cutoff,
            a_dagger=a_dagger,
        )
        fields = {"baryon": raw_fields["metric_slip_grid_00"], "scalar_slip": raw_fields["metric_slip_grid_01"]}
        catalog_file = ROOT / system["member_catalog"]
        members = load_member_table(catalog_file, system)
        hashes[f"members_{label}"] = sha256(catalog_file)
        correction, extra_alpha = correction_for(
            members, local, fields, protocol, softening_kpc
        )
        audits.append(
            {
                "system": system["system"],
                "system_label": label,
                "members": len(members),
                "softening_kpc": float(softening_kpc),
                "softening_arcsec": float(softening_kpc)
                / float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]),
                **correction.audit,
            }
        )
        contexts.append(
            SystemContext(
                system,
                local,
                training,
                heldout,
                members,
                fields,
                correction,
                initials[system["system"]],
                extra_alpha,
            )
        )
    return contexts, audits, hashes


def fit_context(context: SystemContext, coupling: float, *, starts: int, seed: int) -> dict:
    lens = MemberTidalLens(
        context.local_protocol, context.fields, context.correction, coupling
    )
    name = model_name(coupling)
    fitted = lens.fit(
        name,
        context.training,
        starts=starts,
        seed=seed,
        initial_override=context.initial_geometry,
    )
    training_predictions = lens.exact_predictions(
        name,
        fitted["result"].x,
        fitted["sources"],
        context.training,
        stage="training",
    )
    heldout_predictions = lens.exact_predictions(
        name,
        fitted["result"].x,
        fitted["sources"],
        context.heldout,
        stage="heldout",
    )
    for table in (training_predictions, heldout_predictions):
        table.insert(0, "system", context.system["system"])
        table.insert(1, "system_label", context.system["label"])
        table.insert(2, "tensor_t", float(coupling))
    return {
        "lens": lens,
        "fit": fitted,
        "training_predictions": training_predictions,
        "heldout_predictions": heldout_predictions,
        "training": score(training_predictions, lens.sigma, free_parameters=len(fitted["result"].x)),
        "heldout": score(heldout_predictions, lens.sigma),
        "geometry_at_boundary": near_bound(name, fitted["result"].x),
    }


def run_selection_grid(protocol: dict, contexts: list[SystemContext]):
    labels = set(protocol["cluster_split"]["selection_labels"])
    selected_contexts = [context for context in contexts if context.system["label"] in labels]
    primary = list(map(float, protocol["tensor_coupling"]["primary_grid"]))
    extended = list(map(float, protocol["tensor_coupling"]["diagnostic_extended_grid"]))
    grid = sorted(set(primary + extended))
    rows = []
    fits = {}
    predictions = []
    geometry = []
    base_seed = int(protocol["optimization"]["random_seed"])
    starts = int(protocol["optimization"]["starts_per_selection_grid_point"])
    for coupling_index, coupling in enumerate(grid):
        system_scores = []
        for system_index, context in enumerate(selected_contexts):
            print(f"selection system={context.system['label']} t={coupling:g}", flush=True)
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=base_seed + coupling_index * 100 + system_index,
            )
            fits[(context.system["label"], coupling)] = fitted
            predictions.extend([fitted["training_predictions"], fitted["heldout_predictions"]])
            system_scores.append(fitted["training"])
            rows.append(
                {
                    "row_type": "system",
                    "grid_role": "primary" if coupling in primary else "extended_diagnostic",
                    "tensor_t": coupling,
                    "system": context.system["system"],
                    "system_label": context.system["label"],
                    "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_training_roots": fitted["training"]["all_roots_converged"],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
            geometry.append(
                {
                    "stage": "selection_grid",
                    "system": context.system["system"],
                    "system_label": context.system["label"],
                    "tensor_t": coupling,
                    **dict(zip(spec_for(model_name(coupling)).labels, fitted["fit"]["result"].x, strict=True)),
                }
            )
        aggregate = aggregate_system_scores(system_scores)
        rows.append(
            {
                "row_type": "aggregate",
                "grid_role": "primary" if coupling in primary else "extended_diagnostic",
                "tensor_t": coupling,
                "system": "equal_system",
                "system_label": "selection",
                "training_exact_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
                "heldout_exact_RMS_arcsec": None,
                "all_training_roots": aggregate["all_roots_converged"],
                "all_heldout_roots": None,
            }
        )
    primary_rows = [
        row
        for row in rows
        if row["row_type"] == "aggregate"
        and row["grid_role"] == "primary"
        and row["all_training_roots"]
        and row["training_exact_RMS_arcsec"] is not None
    ]
    if not primary_rows:
        raise RuntimeError("no eligible primary tensor coupling")
    choice = min(
        primary_rows,
        key=lambda row: (row["training_exact_RMS_arcsec"], abs(row["tensor_t"])),
    )
    return float(choice["tensor_t"]), pd.DataFrame(rows), fits, predictions, geometry


def run_validation(protocol, contexts, selected_t):
    validation_labels = set(protocol["cluster_split"]["validation_labels"])
    validation = [context for context in contexts if context.system["label"] in validation_labels]
    couplings = [selected_t] if selected_t == 0.0 else [0.0, selected_t]
    results = {}
    predictions = []
    geometry = []
    base_seed = int(protocol["optimization"]["random_seed"]) + 50000
    starts = int(protocol["optimization"]["starts_selected_validation"])
    for coupling_index, coupling in enumerate(couplings):
        scores = []
        for system_index, context in enumerate(validation):
            print(f"validation system={context.system['label']} t={coupling:g}", flush=True)
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=base_seed + coupling_index * 100 + system_index,
            )
            results[(context.system["label"], coupling)] = fitted
            scores.append(fitted["heldout"])
            predictions.extend([fitted["training_predictions"], fitted["heldout_predictions"]])
            geometry.append(
                {
                    "stage": "validation",
                    "system": context.system["system"],
                    "system_label": context.system["label"],
                    "tensor_t": coupling,
                    **dict(zip(spec_for(model_name(coupling)).labels, fitted["fit"]["result"].x, strict=True)),
                }
            )
        results[("aggregate", coupling)] = aggregate_system_scores(scores)
    return results, predictions, geometry


def fixed_source_local_rms(lens, name, parameters, sources, rows):
    residuals = []
    for family, group in rows.groupby("source_family", sort=True):
        x = group.x_arcsec.to_numpy(float)
        y = group.y_arcsec.to_numpy(float)
        redshift = float(group.source_redshift.median())
        beta_x, beta_y = lens.ray_shooting(name, parameters, x, y, redshift)
        delta_beta = np.column_stack([beta_x, beta_y]) - sources[int(family)]
        jacobian = lens.jacobian(name, parameters, x, y, redshift)
        inverse = np.asarray([np.linalg.pinv(item, rcond=1.0e-9) for item in jacobian])
        residuals.append(np.einsum("nij,nj->ni", inverse, delta_beta))
    residual = np.vstack(residuals)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def run_randomizations(protocol, contexts, validation_results, selected_t):
    labels = set(protocol["cluster_split"]["validation_labels"])
    validation = [context for context in contexts if context.system["label"] in labels]
    count = int(protocol["controls"]["member_angle_randomizations"])
    seed = int(protocol["optimization"]["random_seed"]) + 90000
    rng = np.random.default_rng(seed)
    actual_per_system = {}
    for context in validation:
        fitted = validation_results[(context.system["label"], selected_t)]
        actual_per_system[context.system["label"]] = fixed_source_local_rms(
            fitted["lens"],
            model_name(selected_t),
            fitted["fit"]["result"].x,
            fitted["fit"]["sources"],
            context.heldout,
        )
    actual_aggregate = float(np.sqrt(np.mean(np.square(list(actual_per_system.values())))))
    rows = []
    for index in range(count):
        per_system = {}
        for context in validation:
            radius = np.hypot(context.members.x_arcsec, context.members.y_arcsec).to_numpy(float)
            angle = rng.uniform(0.0, 2.0 * np.pi, len(context.members))
            randomized = context.members.copy()
            randomized["x_arcsec"] = radius * np.cos(angle)
            randomized["y_arcsec"] = radius * np.sin(angle)
            correction, _ = correction_for(
                randomized,
                context.local_protocol,
                context.fields,
                protocol,
                float(protocol["environment_tensor"]["primary_softening_kpc"]),
            )
            lens = MemberTidalLens(
                context.local_protocol, context.fields, correction, selected_t
            )
            fitted = validation_results[(context.system["label"], selected_t)]
            per_system[context.system["label"]] = fixed_source_local_rms(
                lens,
                model_name(selected_t),
                fitted["fit"]["result"].x,
                fitted["fit"]["sources"],
                context.heldout,
            )
        aggregate = float(np.sqrt(np.mean(np.square(list(per_system.values())))))
        rows.append(
            {
                "randomization": index,
                "aggregate_local_RMS_arcsec": aggregate,
                **{f"{key}_local_RMS_arcsec": value for key, value in per_system.items()},
            }
        )
    p_value = float(
        (1 + sum(row["aggregate_local_RMS_arcsec"] <= actual_aggregate for row in rows))
        / (1 + count)
    )
    return pd.DataFrame(rows), actual_per_system, actual_aggregate, p_value


def run_softening_sensitivity(protocol, selected_t):
    records = []
    audits = []
    for softening_index, softening in enumerate(protocol["environment_tensor"]["softening_sensitivity_kpc"]):
        contexts, local_audits, _ = build_contexts(protocol, softening_kpc=float(softening))
        audits.extend(local_audits)
        labels = set(protocol["cluster_split"]["validation_labels"])
        validation = [context for context in contexts if context.system["label"] in labels]
        scores = []
        for system_index, context in enumerate(validation):
            fitted = fit_context(
                context,
                selected_t,
                starts=int(protocol["optimization"]["starts_selected_validation"]),
                seed=int(protocol["optimization"]["random_seed"]) + 120000 + softening_index * 100 + system_index,
            )
            scores.append(fitted["heldout"])
            records.append(
                {
                    "softening_kpc": float(softening),
                    "system": context.system["system"],
                    "system_label": context.system["label"],
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_roots_converged": fitted["heldout"]["all_roots_converged"],
                }
            )
        aggregate = aggregate_system_scores(scores)
        records.append(
            {
                "softening_kpc": float(softening),
                "system": "equal_system",
                "system_label": "validation",
                "heldout_exact_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
                "all_roots_converged": aggregate["all_roots_converged"],
            }
        )
    return pd.DataFrame(records), audits


def make_figure(report, grid, randomizations, output):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    aggregate = grid[grid.row_type.eq("aggregate")].sort_values("tensor_t")
    primary = aggregate[aggregate.grid_role.eq("primary")]
    extended = aggregate[aggregate.grid_role.eq("extended_diagnostic")]
    primary_ok = primary[primary.all_training_roots.astype(bool)]
    extended_ok = extended[extended.all_training_roots.astype(bool)]
    failed = aggregate[~aggregate.all_training_roots.astype(bool)]
    axes[0].plot(primary_ok.tensor_t, primary_ok.training_exact_RMS_arcsec, "o-", label="primary, converged")
    axes[0].scatter(extended_ok.tensor_t, extended_ok.training_exact_RMS_arcsec, marker="x", label="sign-reversal, converged")
    axes[0].scatter(failed.tensor_t, failed.training_exact_RMS_arcsec, marker="x", color="#bb3333", label="incomplete roots")
    axes[0].axvline(report["selection"]["selected_t"], color="black", linestyle="--")
    axes[0].set(xlabel="universal tensor coupling t", ylabel="selection training RMS (arcsec)", title="Frozen coupling scan")
    axes[0].legend(fontsize=8)

    labels = ["scalar slip", "member tensor", "compact halo"]
    values = [
        report["validation"]["zero_tensor"]["equal_system_radial_RMS_arcsec"],
        report["validation"]["selected_tensor"]["equal_system_radial_RMS_arcsec"],
        report["comparators"]["compact_halo_validation_RMS_arcsec"],
    ]
    axes[1].bar(labels, values, color=["#888888", "#2f78b7", "#d18f31"])
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(ylabel="held-out validation RMS (arcsec)", title="Transfer to different clusters")

    if report["randomization_control"]["degenerate_because_selected_t_zero"]:
        axes[2].axis("off")
        axes[2].text(
            0.5,
            0.5,
            "Randomization control is degenerate\nselected t = 0, so every member map\nmakes the identical prediction",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[2].set_title("Does the observed layout matter?")
    else:
        axes[2].hist(randomizations.aggregate_local_RMS_arcsec, bins=10, color="#bbbbbb", edgecolor="white")
        axes[2].axvline(report["randomization_control"]["actual_local_RMS_arcsec"], color="#2f78b7", linewidth=2, label="actual member map")
        axes[2].set(xlabel="fixed-source local RMS (arcsec)", ylabel="randomized maps", title="Does the observed layout matter?")
        axes[2].legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    protocol_path = ROOT / "configs/member_tidal_metric_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / "results/member_tidal_metric"
    output.mkdir(parents=True, exist_ok=True)
    primary_softening = float(protocol["environment_tensor"]["primary_softening_kpc"])
    contexts, map_audits, hashes = build_contexts(protocol, softening_kpc=primary_softening)
    selected_t, grid, selection_fits, predictions, geometry = run_selection_grid(protocol, contexts)
    validation, validation_predictions, validation_geometry = run_validation(protocol, contexts, selected_t)
    predictions.extend(validation_predictions)
    geometry.extend(validation_geometry)
    randomizations, actual_random_systems, actual_random_rms, random_p = run_randomizations(
        protocol, contexts, validation, selected_t
    )
    sensitivity, sensitivity_audits = run_softening_sensitivity(protocol, selected_t)
    map_audits.extend(sensitivity_audits)

    metric_report = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    selected_validation = validation[("aggregate", selected_t)]
    zero_validation = validation[("aggregate", 0.0)]
    halo_rms = float(
        metric_report["comparators"]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"]
    )
    improvement = 1.0 - float(selected_validation["equal_system_radial_RMS_arcsec"]) / float(
        zero_validation["equal_system_radial_RMS_arcsec"]
    )
    halo_ratio = float(selected_validation["equal_system_radial_RMS_arcsec"]) / halo_rms
    sensitivity_aggregate = sensitivity[sensitivity.system.eq("equal_system")]
    sensitivity_change = float(
        np.max(
            np.abs(
                sensitivity_aggregate.heldout_exact_RMS_arcsec.to_numpy(float)
                / float(selected_validation["equal_system_radial_RMS_arcsec"])
                - 1.0
            )
        )
    )
    primary_grid = list(map(float, protocol["tensor_coupling"]["primary_grid"]))
    gates = protocol["advance_gates"]
    edge_max = max(row["maximum_edge_Q_eigenvalue"] for row in map_audits if row["softening_kpc"] == primary_softening)
    curl_max = max(row["normalized_curl_RMS"] for row in map_audits)
    gate_audit = {
        "selected_t_not_primary_grid_boundary_pass": selected_t not in {min(primary_grid), max(primary_grid)},
        "validation_all_roots_converged_pass": bool(selected_validation["all_roots_converged"]),
        "validation_RMS_improvement_over_scalar_slip_fraction": improvement,
        "validation_RMS_improvement_pass": improvement
        >= float(gates["validation_RMS_improvement_over_scalar_slip_fraction_min"]),
        "validation_to_compact_halo_RMS_ratio": halo_ratio,
        "validation_to_compact_halo_pass": halo_ratio
        <= float(gates["validation_to_compact_halo_RMS_ratio_max"]),
        "actual_member_map_randomization_p": random_p,
        "actual_member_map_randomization_pass": random_p
        <= float(gates["actual_member_map_randomization_p_max"]),
        "softening_maximum_fractional_RMS_change": sensitivity_change,
        "softening_sensitivity_pass": sensitivity_change
        <= float(gates["softening_sensitivity_max_fractional_RMS_change"]),
        "maximum_primary_edge_Q_eigenvalue": edge_max,
        "edge_Q_pass": edge_max <= float(gates["maximum_edge_Q_eigenvalue"]),
        "maximum_normalized_curl_RMS": curl_max,
        "curl_pass": curl_max <= float(gates["maximum_normalized_curl_RMS"]),
    }
    gate_audit["all_gates_pass"] = bool(
        all(value for key, value in gate_audit.items() if key.endswith("_pass"))
    )
    extended_aggregate = grid[
        grid.row_type.eq("aggregate") & grid.grid_role.eq("extended_diagnostic")
    ]
    extended_eligible = extended_aggregate[
        extended_aggregate.all_training_roots.astype(bool)
        & extended_aggregate.training_exact_RMS_arcsec.notna()
    ]
    extended_best = extended_eligible.nsmallest(1, "training_exact_RMS_arcsec").iloc[0]
    xmm_report = json.loads(
        (ROOT / protocol["inputs"]["xmm_response_report"]).read_text(encoding="utf-8")
    )
    report = {
        "report_version": "MEMBER-TIDAL-METRIC-RESULTS-0.1.0",
        "status": "complete",
        "protocol": protocol["protocol_version"],
        "input_hashes": hashes,
        "equation": protocol["weak_field_equation"],
        "selection": {
            "selected_t": selected_t,
            "primary_softening_kpc": primary_softening,
            "selection_training_equal_system_RMS_arcsec": float(
                grid[
                    grid.row_type.eq("aggregate") & np.isclose(grid.tensor_t.astype(float), selected_t)
                ].iloc[0].training_exact_RMS_arcsec
            ),
            "extended_sign_reversal_best_t": float(extended_best.tensor_t),
            "extended_sign_reversal_best_training_RMS_arcsec": float(
                extended_best.training_exact_RMS_arcsec
            ),
            "extended_grid_is_qualifying": False,
        },
        "validation": {
            "selected_tensor": selected_validation,
            "zero_tensor": zero_validation,
            "per_system": {
                label: {
                    "selected_tensor": validation[(label, selected_t)]["heldout"],
                    "zero_tensor": validation[(label, 0.0)]["heldout"],
                }
                for label in protocol["cluster_split"]["validation_labels"]
            },
        },
        "comparators": {
            "locked_scalar_slip_report_RMS_arcsec": metric_report["cross_cluster_validation"]["selected_slip"]["equal_system_radial_RMS_arcsec"],
            "compact_halo_validation_RMS_arcsec": halo_rms,
            "locked_fixed_RAR_galaxy_outer_RMSE_km_s": metric_report["matter_law"]["locked_galaxy_outer_RMSE_km_s"],
        },
        "randomization_control": {
            "actual_local_RMS_arcsec": actual_random_rms,
            "actual_per_system_local_RMS_arcsec": actual_random_systems,
            "randomizations": len(randomizations),
            "one_sided_p_value": random_p,
            "degenerate_because_selected_t_zero": selected_t == 0.0,
        },
        "softening_sensitivity": {
            "results": sensitivity_aggregate.to_dict(orient="records"),
            "maximum_fractional_validation_RMS_change": sensitivity_change,
        },
        "map_audit": {
            "maximum_primary_edge_Q_eigenvalue": edge_max,
            "maximum_all_map_normalized_curl_RMS": curl_max,
        },
        "data_readiness": {
            "XMM_X4_response_status": xmm_report["status"],
            "accepted_X5_gas_map_available": False,
            "member_tensor_is_gas_inclusive": False,
        },
        "gate_audit": gate_audit,
        "verdict": {
            "member_tidal_metric_survives": gate_audit["all_gates_pass"],
            "full_gas_inclusive_tensor_test_completed": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }

    grid.to_csv(output / "grid_scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(geometry).to_csv(output / "geometry.csv", index=False)
    pd.DataFrame(map_audits).to_csv(output / "map_audit.csv", index=False)
    randomizations.to_csv(output / "randomizations.csv", index=False)
    sensitivity.to_csv(output / "softening_sensitivity.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(report, grid, randomizations, output / "member_tidal_metric.png")
    summary = f"""# Member-tidal metric pilot

The frozen primary grid selected **t = {selected_t:g}**.  On the two held-out
validation clusters the tensor model has an equal-system image-position RMS of
**{selected_validation['equal_system_radial_RMS_arcsec']:.3f} arcsec**, versus
**{zero_validation['equal_system_radial_RMS_arcsec']:.3f} arcsec** for the same
fixed-RAR scalar-slip model with the tensor turned off and **{halo_rms:.3f}
arcsec** for the compact dark-halo comparator.

That is a **{100.0 * improvement:.1f}%** change relative to scalar slip and a
tensor/halo RMS ratio of **{halo_ratio:.3f}**.  The observed member layout has a
one-sided angle-randomization p-value of **{random_p:.4f}** in the fixed-source
control; because the selected coupling is zero, that control is degenerate and
all randomized maps are identical.  The full frozen gate result is
**{'PASS' if gate_audit['all_gates_pass'] else 'FAIL'}**.

This pilot uses observed member positions and light weights, not a completed gas
map.  XMM response construction passes, but the X5 temperature/density posterior
and projected gas surface-density map remain unfinished.  Therefore this result
tests only the member-driven directional term and cannot decide the full
gas-inclusive tensor idea.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report["gate_audit"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
