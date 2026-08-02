#!/usr/bin/env python3
"""Audit exact-root solver basins for the frozen P0657 fold-one fit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit
from run_clash_stellar_morphology_response import MorphologyLens
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields
from run_p0607_component_direction_raw_lensing import fixed_geometry
from run_p0644_spent_rxj2129_accumulated_tensor import make_field, read_json
from run_p0645_fair_geometry_cv_accumulated_tensor import stratified_folds
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images

DEFAULT_CONFIG = ROOT / "configs" / "p0658_exact_root_basin_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def starting_offsets(search: dict) -> list[tuple[float, float]]:
    offsets = [(0.0, 0.0)]
    angles = int(search["nonzero_radius_angles"])
    for radius in search["offset_radii_arcsec"]:
        value = float(radius)
        if value == 0.0:
            continue
        for index in range(angles):
            angle = 2.0 * np.pi * index / angles
            offsets.append((value * np.cos(angle), value * np.sin(angle)))
    if len(offsets) != int(search["starts_per_image"]):
        raise RuntimeError("root-start grid no longer matches the frozen count")
    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0658_audit":
        raise RuntimeError("P0658 protocol is not frozen")
    inputs = protocol["inputs"]
    target = protocol["fixed_target"]
    search = protocol["search"]
    p0644 = read_json(ROOT / inputs["P0644_protocol"])
    p0645 = read_json(ROOT / inputs["P0645_protocol"])
    p0657 = read_json(ROOT / inputs["P0657_protocol"])
    p0657_report = read_json(ROOT / inputs["P0657_report"])
    stored_folds = pd.read_csv(ROOT / inputs["P0657_fold_scores"])
    stored_fold = stored_folds[stored_folds.fold == int(target["CV_fold"])].iloc[0]
    if not np.isclose(stored_fold.optimizer_cost, float(target["expected_optimizer_cost"])):
        raise RuntimeError("stored P0657 fold cost changed")
    if p0657_report["CV_summary"]["CV_roots"] != 13:
        raise RuntimeError("P0657 root outcome changed")

    raw_protocol = read_json(ROOT / p0644["inputs"]["raw_protocol"])
    p0601 = read_json(ROOT / p0644["inputs"]["P0601_protocol"])
    images = load_images(raw_protocol)
    training, _ = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(anchors, raw_protocol, p0601["constants"])
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / p0644["inputs"]["P0601_parameters"])
    folds = list(stratified_folds(training, int(p0645["cross_validation"]["folds"])))
    fold, fit_images, validation_images = next(
        item for item in folds if item[0] == int(target["CV_fold"])
    )

    candidate = p0657["candidate"]
    field_protocol = copy.deepcopy(p0644)
    field_protocol["candidate"].update(
        {
            "coherence_length_kpc": candidate["coherence_length_kpc"],
            "accumulation_power": candidate["accumulation_power"],
            "a0_m_s2": candidate["a0_m_s2"],
            "stellar_mass_fraction": candidate["stellar_mass_fraction"],
            "gas_mass_fraction": candidate["gas_mass_fraction"],
            "common_smoothing_kpc": candidate["common_physical_smoothing_kpc"],
            "mismatch_mode": candidate["mismatch_mode"],
            "closure": candidate["closure"],
            "taper_inner_arcsec": candidate["taper_inner_arcsec"],
            "support_radius_arcsec": candidate["support_radius_arcsec"],
            "computational_padding_arcsec": candidate["computational_padding_arcsec"],
        }
    )
    field, map_audits = make_field(
        field_protocol, raw_protocol, anchors, parent, baryons, images
    )
    lens = MorphologyLens(
        raw_protocol,
        {"P0599_potential_shape": parent},
        parent="P0599_potential_shape",
        morphology=field,
        fraction=float(candidate["field_amplitude"]),
    )
    frozen_cv = p0657["cross_validation"]
    fitted = exact_fit(
        lens,
        fit_images,
        validation_images,
        initial=initial,
        starts=int(frozen_cv["geometry_refit_starts_per_fold"]),
        seed=int(frozen_cv["random_seed"]) + fold,
    )
    if not np.isclose(
        fitted["optimizer_cost"], float(target["expected_optimizer_cost"]), rtol=1e-9
    ):
        raise RuntimeError("recomputed P0657 fold fit does not match the frozen score")
    original = fitted["heldout_prediction"].copy()
    original_failed = original[~original.root_converged].image_id.astype(str).tolist()
    if original_failed != list(target["original_failed_images"]):
        raise RuntimeError("recomputed original failed-image list changed")

    tolerance = float(target["root_acceptance_closure_arcsec_max"])
    radius_limit = float(target["local_correspondence_radius_arcsec"])
    solver_tolerance = float(search["solver_tolerance"])
    maximum_evaluations = int(search["maximum_function_evaluations"])
    attempts = []
    offsets = starting_offsets(search)
    profiled_sources = lens.profiled_residuals(
        MODEL, fitted["parameters"], fit_images
    )[1]
    for row in validation_images.itertuples(index=False):
        family_source = profiled_sources[int(row.source_family)]
        observed = np.array([row.x_arcsec, row.y_arcsec], dtype=float)
        redshift = float(row.source_redshift)

        def equation(
            theta, *, current_redshift=redshift, current_source=family_source
        ):
            beta_x, beta_y = lens.ray_shooting(
                MODEL,
                fitted["parameters"],
                np.array([theta[0]]),
                np.array([theta[1]]),
                current_redshift,
            )
            return np.array(
                [beta_x[0] - current_source[0], beta_y[0] - current_source[1]]
            )

        def derivative(theta, *, current_redshift=redshift):
            return lens.jacobian(
                MODEL,
                fitted["parameters"],
                np.array([theta[0]]),
                np.array([theta[1]]),
                current_redshift,
            )[0]

        lower = observed - radius_limit
        upper = observed + radius_limit
        for start_index, offset in enumerate(offsets):
            start = observed + np.asarray(offset)
            method_results = []
            for method in ("hybr", "lm"):
                result = root(
                    equation,
                    start,
                    jac=derivative,
                    method=method,
                    tol=solver_tolerance,
                    options={"maxfev": maximum_evaluations}
                    if method == "hybr"
                    else {"maxiter": maximum_evaluations},
                )
                method_results.append(
                    (f"scipy_root_{method}_with_lens_jacobian", result.x, result.success)
                )
            for method in ("trf", "dogbox"):
                result = least_squares(
                    equation,
                    np.clip(start, lower, upper),
                    jac=derivative,
                    bounds=(lower, upper),
                    method=method,
                    max_nfev=maximum_evaluations,
                    ftol=solver_tolerance,
                    xtol=solver_tolerance,
                    gtol=solver_tolerance,
                )
                method_results.append(
                    (
                        f"scipy_least_squares_{method}_bounded_with_lens_jacobian",
                        result.x,
                        result.success,
                    )
                )
            for method, solution, success in method_results:
                solution = np.asarray(solution, dtype=float)
                closure = float(np.linalg.norm(equation(solution)))
                displacement = float(np.linalg.norm(solution - observed))
                finite = bool(np.all(np.isfinite(solution)) and np.isfinite(closure))
                accepted = bool(
                    finite and closure <= tolerance and displacement <= radius_limit
                )
                attempts.append(
                    {
                        "image_id": str(row.image_id),
                        "source_family": int(row.source_family),
                        "start_index": start_index,
                        "start_offset_x_arcsec": float(offset[0]),
                        "start_offset_y_arcsec": float(offset[1]),
                        "algorithm": method,
                        "reported_success": bool(success),
                        "solution_x_arcsec": float(solution[0]),
                        "solution_y_arcsec": float(solution[1]),
                        "closure_arcsec": closure,
                        "displacement_from_observed_arcsec": displacement,
                        "accepted": accepted,
                    }
                )
    attempt_frame = pd.DataFrame(attempts)
    root_rows = []
    deduplication = float(search["deduplication_radius_arcsec"])
    for image_id, block in attempt_frame[attempt_frame.accepted].groupby("image_id"):
        accepted_solutions = []
        for row in block.sort_values(
            ["displacement_from_observed_arcsec", "closure_arcsec"]
        ).itertuples(index=False):
            point = np.array([row.solution_x_arcsec, row.solution_y_arcsec])
            if any(np.linalg.norm(point - existing) <= deduplication for existing in accepted_solutions):
                continue
            accepted_solutions.append(point)
            root_rows.append(
                {
                    "image_id": image_id,
                    "solution_x_arcsec": row.solution_x_arcsec,
                    "solution_y_arcsec": row.solution_y_arcsec,
                    "closure_arcsec": row.closure_arcsec,
                    "displacement_from_observed_arcsec": row.displacement_from_observed_arcsec,
                    "algorithm": row.algorithm,
                    "start_index": row.start_index,
                }
            )
    root_frame = pd.DataFrame(root_rows)
    recovery_rows = []
    for row in validation_images.itertuples(index=False):
        image_id = str(row.image_id)
        block = attempt_frame[attempt_frame.image_id == image_id]
        accepted = block[block.accepted]
        original_row = original[original.image_id.astype(str) == image_id].iloc[0]
        recovery_rows.append(
            {
                "image_id": image_id,
                "original_converged": bool(original_row.root_converged),
                "original_closure_arcsec": float(original_row.source_plane_closure_arcsec),
                "attempts": len(block),
                "accepted_attempts": len(accepted),
                "distinct_local_roots": int(
                    0 if root_frame.empty else np.sum(root_frame.image_id == image_id)
                ),
                "minimum_audited_closure_arcsec": float(block.closure_arcsec.min()),
                "minimum_accepted_displacement_arcsec": float(
                    accepted.displacement_from_observed_arcsec.min()
                    if not accepted.empty
                    else np.inf
                ),
                "recovered_by_audit": bool(not original_row.root_converged and not accepted.empty),
            }
        )
    recovery_frame = pd.DataFrame(recovery_rows)
    recovered_failures = recovery_frame[
        ~recovery_frame.original_converged & recovery_frame.recovered_by_audit
    ].image_id.tolist()
    unrecovered_failures = recovery_frame[
        ~recovery_frame.original_converged & ~recovery_frame.recovered_by_audit
    ].image_id.tolist()
    report = {
        "report_version": "P0658-EXACT-ROOT-BASIN-AUDIT-RESULTS-1.0.0",
        "status": "numerical_solver_limitation"
        if recovered_failures
        else "local_topology_failure_supported",
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "reproduced_optimizer_cost": float(fitted["optimizer_cost"]),
        "coverage": {
            "validation_images": len(validation_images),
            "starts_per_image": len(offsets),
            "algorithms": 4,
            "attempts_per_image": 4 * len(offsets),
            "total_attempts": len(attempt_frame),
            "gravity_parameters_changed": 0,
            "geometry_parameters_changed": 0,
            "source_positions_changed": 0,
        },
        "root_acceptance": {
            "closure_arcsec_max": tolerance,
            "local_correspondence_radius_arcsec": radius_limit,
        },
        "recoveries": recovery_frame.to_dict(orient="records"),
        "originally_failed_images_recovered": recovered_failures,
        "originally_failed_images_unrecovered": unrecovered_failures,
        "universal_rescore_required_before_any_candidate_change": bool(recovered_failures),
        "candidate_selected_or_advanced": False,
        "field_audit": field.audit,
        "map_audits": map_audits,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    attempt_frame.to_csv(output / "root_attempts.csv", index=False)
    root_frame.to_csv(output / "distinct_local_roots.csv", index=False)
    recovery_frame.to_csv(output / "recovery_summary.csv", index=False)
    original.to_csv(output / "original_fold_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    minimum = (
        attempt_frame.groupby(["image_id", "algorithm"], as_index=False)
        .closure_arcsec.min()
        .pivot(index="image_id", columns="algorithm", values="closure_arcsec")
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    minimum.plot(kind="bar", logy=True, ax=axis)
    axis.axhline(tolerance, color="black", linestyle="--", label="acceptance threshold")
    axis.set(ylabel="minimum source-plane closure (arcsec)", title="P0657 fold-1 root basin audit")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "exact_root_basin_audit.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps(
            {
                "status": report["status"],
                "recovered": recovered_failures,
                "unrecovered": unrecovered_failures,
                "recoveries": report["recoveries"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
