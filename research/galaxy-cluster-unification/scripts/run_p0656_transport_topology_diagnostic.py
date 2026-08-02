#!/usr/bin/env python3
"""Describe how tested transport fields alter local lens topology."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field
from run_clash_stellar_morphology_response import MorphologyLens
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields
from run_p0607_component_direction_raw_lensing import fixed_geometry
from run_p0644_spent_rxj2129_accumulated_tensor import make_field, read_json
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import load_baryonic_anchors, load_images

DEFAULT_CONFIG = ROOT / "configs" / "p0656_transport_topology_diagnostic.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_jacobian(function, x: float, y: float, step: float) -> np.ndarray:
    plus_x = function(x + step, y)
    minus_x = function(x - step, y)
    plus_y = function(x, y + step)
    minus_y = function(x, y - step)
    return np.array(
        [
            [
                (float(plus_x[0]) - float(minus_x[0])) / (2.0 * step),
                (float(plus_y[0]) - float(minus_y[0])) / (2.0 * step),
            ],
            [
                (float(plus_x[1]) - float(minus_x[1])) / (2.0 * step),
                (float(plus_y[1]) - float(minus_y[1])) / (2.0 * step),
            ],
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0656_diagnostic":
        raise RuntimeError("P0656 protocol is not frozen")

    p0644 = read_json(ROOT / protocol["inputs"]["P0644_protocol"])
    p0654_report = read_json(ROOT / protocol["inputs"]["P0654_report"])
    p0655_report = read_json(ROOT / protocol["inputs"]["P0655_report"])
    if p0654_report["gate_results"]["CV_roots"]:
        raise RuntimeError("P0654 topology failure is absent")
    if p0655_report["gate_results"]["CV_roots"]:
        raise RuntimeError("P0655 topology failure is absent")
    raw_protocol = read_json(ROOT / p0644["inputs"]["raw_protocol"])
    p0601 = read_json(ROOT / p0644["inputs"]["P0601_protocol"])
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    labels = {str(value): "training" for value in training.image_id}
    labels.update({str(value): "spent_heldout" for value in heldout.image_id})
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(anchors, raw_protocol, p0601["constants"])
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    fixed = fixed_geometry(ROOT / p0644["inputs"]["P0601_parameters"])
    formula = protocol["fixed_formula"]

    fields = {}
    map_audits = {}
    for specification in protocol["fields"]:
        field_protocol = copy.deepcopy(p0644)
        field_protocol["candidate"].update(
            {
                "coherence_length_kpc": formula["coherence_length_kpc"],
                "accumulation_power": formula["accumulation_power"],
                "a0_m_s2": formula["a0_m_s2"],
                "stellar_mass_fraction": formula["stellar_mass_fraction"],
                "gas_mass_fraction": formula["gas_mass_fraction"],
                "common_smoothing_kpc": formula["common_physical_smoothing_kpc"],
                "mismatch_mode": formula["mismatch_mode"],
                "closure": specification["closure"],
                "transport_steps": formula["transport_steps"],
                "taper_inner_arcsec": formula["taper_inner_arcsec"],
                "support_radius_arcsec": formula["support_radius_arcsec"],
                "computational_padding_arcsec": specification[
                    "computational_padding_arcsec"
                ],
            }
        )
        field, audits = make_field(
            field_protocol, raw_protocol, anchors, parent, baryons, images
        )
        fields[specification["id"]] = field
        map_audits[specification["id"]] = audits

    diagnostic = protocol["diagnostic"]
    step = float(diagnostic["finite_difference_step_arcsec"])
    image_rows = []
    for field_id, field in fields.items():
        lens = MorphologyLens(
            raw_protocol,
            {"P0599_potential_shape": parent},
            parent="P0599_potential_shape",
            morphology=field,
            fraction=float(formula["field_amplitude"]),
        )
        for image in images.itertuples(index=False):
            x = float(image.x_arcsec)
            y = float(image.y_arcsec)
            redshift = float(image.source_redshift)
            ratio = lens.distance_ratio(redshift)

            def correction(query_x, query_y, *, current_field=field, current_ratio=ratio):
                return current_field.alpha_arcsec(
                    query_x, query_y, distance_ratio=current_ratio
                )

            def full_alpha(
                query_x, query_y, *, current_lens=lens, current_redshift=redshift
            ):
                return current_lens.alpha(
                    MODEL, fixed, query_x, query_y, current_redshift
                )

            correction_x, correction_y = correction(x, y)
            correction_x = float(correction_x)
            correction_y = float(correction_y)
            correction_gradient = alpha_jacobian(correction, x, y, step)
            full_gradient = alpha_jacobian(full_alpha, x, y, step)
            radius = max(float(np.hypot(x, y)), np.finfo(float).tiny)
            radial_x, radial_y = x / radius, y / radius
            radial = correction_x * radial_x + correction_y * radial_y
            tangential = -correction_x * radial_y + correction_y * radial_x
            convergence = 0.5 * np.trace(correction_gradient)
            shear_one = 0.5 * (correction_gradient[0, 0] - correction_gradient[1, 1])
            shear_two = 0.5 * (correction_gradient[0, 1] + correction_gradient[1, 0])
            rotation = 0.5 * (correction_gradient[1, 0] - correction_gradient[0, 1])
            image_rows.append(
                {
                    "field": field_id,
                    "image_id": str(image.image_id),
                    "split": labels[str(image.image_id)],
                    "source_family": int(image.source_family),
                    "source_redshift": redshift,
                    "x_arcsec": x,
                    "y_arcsec": y,
                    "correction_x_arcsec": correction_x,
                    "correction_y_arcsec": correction_y,
                    "correction_magnitude_arcsec": float(
                        np.hypot(correction_x, correction_y)
                    ),
                    "correction_radial_arcsec": radial,
                    "correction_tangential_arcsec": tangential,
                    "correction_convergence": convergence,
                    "correction_shear": float(np.hypot(shear_one, shear_two)),
                    "correction_rotation": rotation,
                    "correction_gradient_spectral_norm": float(
                        np.linalg.svd(correction_gradient, compute_uv=False)[0]
                    ),
                    "fixed_full_lens_mapping_determinant": float(
                        np.linalg.det(np.eye(2) - full_gradient)
                    ),
                }
            )
    image_frame = pd.DataFrame(image_rows)

    grid_axis = np.arange(
        float(diagnostic["shared_grid_min_arcsec"]),
        float(diagnostic["shared_grid_max_arcsec"])
        + 0.5 * float(diagnostic["shared_grid_spacing_arcsec"]),
        float(diagnostic["shared_grid_spacing_arcsec"]),
    )
    yy, xx = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid_vectors = {}
    for field_id, field in fields.items():
        alpha_x, alpha_y = field.alpha_arcsec(xx, yy, distance_ratio=1.0)
        grid_vectors[field_id] = np.stack([alpha_x.ravel(), alpha_y.ravel()], axis=1)
    pair_rows = []
    for left_id, right_id in combinations(fields, 2):
        left = grid_vectors[left_id]
        right = grid_vectors[right_id]
        left_rms = float(np.sqrt(np.mean(np.sum(left * left, axis=1))))
        right_rms = float(np.sqrt(np.mean(np.sum(right * right, axis=1))))
        dot = float(np.sum(left * right))
        correlation = dot / max(
            float(np.sqrt(np.sum(left * left) * np.sum(right * right))),
            np.finfo(float).tiny,
        )
        pair_rows.append(
            {
                "left": left_id,
                "right": right_id,
                "vector_cosine_correlation": correlation,
                "left_RMS_arcsec": left_rms,
                "right_RMS_arcsec": right_rms,
                "difference_RMS_over_left": float(
                    np.sqrt(np.mean(np.sum(np.square(right - left), axis=1)))
                    / max(left_rms, np.finfo(float).tiny)
                ),
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    summaries = []
    for field_id, block in image_frame.groupby("field", sort=False):
        determinant = block.fixed_full_lens_mapping_determinant.to_numpy(float)
        summaries.append(
            {
                "field": field_id,
                "image_correction_RMS_arcsec": float(
                    np.sqrt(np.mean(np.square(block.correction_magnitude_arcsec)))
                ),
                "image_tangential_RMS_arcsec": float(
                    np.sqrt(np.mean(np.square(block.correction_tangential_arcsec)))
                ),
                "correction_convergence_RMS": float(
                    np.sqrt(np.mean(np.square(block.correction_convergence)))
                ),
                "correction_shear_RMS": float(
                    np.sqrt(np.mean(np.square(block.correction_shear)))
                ),
                "correction_gradient_spectral_norm_max": float(
                    block.correction_gradient_spectral_norm.max()
                ),
                "minimum_absolute_fixed_mapping_determinant": float(
                    np.min(np.abs(determinant))
                ),
                "negative_fixed_mapping_determinants": int(np.sum(determinant < 0.0)),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    focus = image_frame[image_frame.image_id.isin(["6b", "3b"])].copy()
    report = {
        "report_version": "P0656-TRANSPORT-TOPOLOGY-DIAGNOSTIC-RESULTS-1.0.0",
        "status": "descriptive_only",
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "coverage": {
            "fields": len(fields),
            "spent_image_positions": len(images),
            "field_image_rows": len(image_frame),
            "pairwise_field_comparisons": len(pair_frame),
            "fitted_parameters": 0,
        },
        "field_summaries": summary_frame.to_dict(orient="records"),
        "failed_image_focus": focus.to_dict(orient="records"),
        "pairwise_field_comparisons": pair_frame.to_dict(orient="records"),
        "field_audits": {field_id: field.audit for field_id, field in fields.items()},
        "map_audits": map_audits,
        "candidate_selected_or_advanced": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    image_frame.to_csv(output / "image_local_topology.csv", index=False)
    pair_frame.to_csv(output / "pairwise_field_comparisons.csv", index=False)
    summary_frame.to_csv(output / "field_summaries.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for field_id, block in image_frame.groupby("field", sort=False):
        axes[0].scatter(
            block.correction_magnitude_arcsec,
            block.correction_gradient_spectral_norm,
            label=field_id,
            alpha=0.75,
        )
        axes[1].scatter(
            block.correction_convergence,
            block.fixed_full_lens_mapping_determinant,
            label=field_id,
            alpha=0.75,
        )
    axes[0].set(
        xlabel="correction magnitude (arcsec)",
        ylabel="correction gradient spectral norm",
        title="Strength versus local distortion",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="correction convergence",
        ylabel="fixed full lens-map determinant",
        title="Local topology proxy",
    )
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "transport_topology_diagnostic.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"summaries": report["field_summaries"], "focus": report["failed_image_focus"]}, indent=2))


if __name__ == "__main__":
    main()
