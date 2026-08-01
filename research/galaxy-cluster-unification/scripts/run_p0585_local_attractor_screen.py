#!/usr/bin/env python3
"""Screen baryon-derived multi-attractor endpoint fields on RX J2129."""

from __future__ import annotations

import hashlib
import itertools
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

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, load_sources  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0583b_signed_endpoint_amplitude import json_safe  # noqa: E402
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    load_baryonic_anchors,
    load_images,
    score,
)
from voidscreen.arc_apogee import extent_gate  # noqa: E402
from voidscreen.route_template import (  # noqa: E402
    conservative_local_attractor_route_template,
    conservative_route_template,
    weighted_radius,
)
from voidscreen.stellar_morphology_lensing import (  # noqa: E402
    build_stellar_morphology_deflection_field,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_specs(protocol: dict) -> list[dict]:
    grid = protocol["structural_grid"]
    specs = [
        {
            "candidate_id": "L0000",
            "kind": "global",
            "local_mix": 0.0,
            "softening_over_R80": float("nan"),
            "distance_power": float("nan"),
        }
    ]
    for mix, softening, power in itertools.product(
        grid["local_mix"],
        grid["softening_over_R80"],
        grid["distance_power"],
    ):
        specs.append(
            {
                "candidate_id": f"L{len(specs):04d}",
                "kind": "local_attractor",
                "local_mix": float(mix),
                "softening_over_R80": float(softening),
                "distance_power": float(power),
            }
        )
    if len(specs) != int(grid["candidates"]):
        raise RuntimeError("P0585 candidate count changed")
    return specs


def impact_table(scores: pd.DataFrame) -> pd.DataFrame:
    local = scores[
        scores.kind.eq("local_attractor")
        & scores.epsilon.gt(0.0)
        & scores.heldout_all_roots.astype(bool)
    ]
    rows = []
    for parameter in ("local_mix", "softening_over_R80", "distance_power"):
        grouped = local.groupby(parameter).heldout_RMS_arcsec.median()
        finite = grouped[np.isfinite(grouped)]
        rows.append(
            {
                "parameter": parameter,
                "best_level": str(finite.idxmin()),
                "worst_level": str(finite.idxmax()),
                "median_RMS_span_arcsec": float(finite.max() - finite.min()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_RMS_span_arcsec", ascending=False)


def main() -> None:
    protocol_path = ROOT / "configs/p0585_local_attractor_screen_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0584_before_local_attractor_scores":
        raise RuntimeError("P0585 protocol is not frozen")
    prior = json.loads(
        (ROOT / protocol["inputs"]["p0584_report"]).read_text(encoding="utf-8")
    )
    if prior["winner_including_zero"]["travel_mode"] != "tanh_no_cross":
        raise RuntimeError("P0584 no-cross trigger changed")
    p0581 = json.loads(
        (ROOT / protocol["inputs"]["p0581_protocol"]).read_text(encoding="utf-8")
    )
    p0583 = json.loads(
        (ROOT / protocol["inputs"]["p0583_protocol"]).read_text(encoding="utf-8")
    )
    rxj = json.loads(
        (ROOT / p0583["inputs"]["rxj_route_protocol"]).read_text(encoding="utf-8")
    )
    raw_protocol = json.loads(
        (ROOT / rxj["raw_inputs"]["raw_protocol"]).read_text(encoding="utf-8")
    )
    images = load_images(raw_protocol)
    _, heldout = split_images(images, raw_protocol)
    members = load_sources(rxj, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    parent_protocol = json.loads(
        (ROOT / rxj["raw_inputs"]["parent_protocol"]).read_text(encoding="utf-8")
    )
    parent_scores = pd.read_csv(ROOT / rxj["raw_inputs"]["parent_scores"])
    parent_id = str(rxj["raw_inputs"]["parent_candidate"])
    parent_row = parent_scores[parent_scores.candidate_id.eq(parent_id)].iloc[0]
    parents = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    parent, _ = raw_field(
        parents[parent_id], float(parent_row.universal_q), anchors, raw_protocol, 1.2e-10
    )
    baryons = baryon_field(anchors, raw_protocol)
    parameter_frame = pd.read_csv(ROOT / protocol["inputs"]["p0583_parameters"])
    scalar = parameter_frame[parameter_frame.variant.eq("scalar_baseline")].set_index(
        "parameter"
    )
    parameters = np.asarray([float(scalar.loc[name, "value"]) for name in FIXED_LABELS])
    old_predictions = pd.read_csv(ROOT / protocol["inputs"]["p0583_predictions"])
    scalar_predictions = old_predictions[old_predictions.variant.eq("scalar_baseline")]
    sources = {
        int(family): block[["source_x_arcsec", "source_y_arcsec"]]
        .iloc[0]
        .to_numpy(float)
        for family, block in scalar_predictions.groupby("source_family")
    }

    translation = p0581["field_translation"]
    scale = float(
        raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    xy = members[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = members.base_weight.to_numpy(float)
    weights /= np.sum(weights)
    radius_kpc = np.linalg.norm(xy, axis=1) * scale
    r50 = weighted_radius(radius_kpc, weights, 0.5)
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    concentration = r50 / r80
    route_fraction = float(extent_gate(concentration, "cluster_logistic"))
    axis = np.arange(
        -float(translation["grid_half_width_arcsec"]),
        float(translation["grid_half_width_arcsec"]) + 0.5,
        float(translation["grid_spacing_arcsec"]),
    )
    return_scale = 0.36 * r80 / scale
    smoothing = 0.23 * r80 / scale

    def carrier_alpha(radius_arcsec):
        return parent.reduced_alpha_arcsec(
            radius_arcsec, 1.0
        ) - baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    specs = candidate_specs(protocol)
    score_rows = []
    prediction_rows = []
    audit_rows = []
    for spec in specs:
        if spec["kind"] == "global":
            route_map, route_audit = conservative_route_template(
                axis,
                xy,
                weights,
                routing_fraction=route_fraction,
                return_scale=return_scale,
                radius_exponent=0.0,
                reference_radius=100.0 / scale,
                smoothing=smoothing,
                travel_mode="tanh_no_cross",
            )
        else:
            route_map, route_audit = conservative_local_attractor_route_template(
                axis,
                xy,
                weights,
                routing_fraction=route_fraction,
                return_scale=return_scale,
                smoothing=smoothing,
                softening=float(spec["softening_over_R80"]) * r80 / scale,
                distance_power=float(spec["distance_power"]),
                local_mix=float(spec["local_mix"]),
                travel_mode="tanh_no_cross",
            )
        field = build_stellar_morphology_deflection_field(
            axis,
            route_map,
            carrier_alpha,
            contrast_cap=20.0,
            contrast_mode="tanh",
            contrast_strength=1.0,
            annulus_width_arcsec=float(translation["annulus_width_arcsec"]),
            taper_inner_arcsec=float(translation["taper_inner_arcsec"]),
            support_radius_arcsec=float(translation["support_radius_arcsec"]),
            radial_samples=2048,
            circular_radii=512,
            circular_azimuths=720,
        )
        audit_rows.append(
            {
                **spec,
                "R50_kpc": r50,
                "R80_kpc": r80,
                "route_fraction": route_fraction,
                "route_normalization_error": route_audit["normalization_error"],
                "median_travel_arcsec": route_audit["median_travel"],
                "maximum_travel_arcsec": route_audit["maximum_travel"],
                "source_weight_crossing_destination": float(
                    route_audit.get(
                        "source_weight_crossing_target",
                        route_audit.get("source_weight_crossing_center", 0.0),
                    )
                ),
                **field.audit,
            }
        )
        for epsilon in protocol["positive_amplitudes"]:
            lens = MorphologyLens(
                raw_protocol,
                {MODEL: parent},
                parent=MODEL,
                morphology=field,
                fraction=float(epsilon),
            )
            predictions = lens.exact_predictions(
                MODEL, parameters, sources, heldout, stage="heldout"
            )
            metrics = score(predictions, lens.sigma)
            predictions["candidate_id"] = spec["candidate_id"]
            predictions["epsilon"] = float(epsilon)
            prediction_rows.append(predictions)
            score_rows.append(
                {
                    **spec,
                    "epsilon": float(epsilon),
                    "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                    "heldout_converged_roots": metrics["converged_roots"],
                    "heldout_all_roots": metrics["all_roots_converged"],
                    "unit_correction_RMS_arcsec": float(
                        field.audit["raw_correction_RMS_arcsec"]
                    ),
                }
            )
        print(spec["candidate_id"], flush=True)

    candidates = pd.DataFrame(specs)
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    audits = pd.DataFrame(audit_rows)
    impacts = impact_table(scores)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_specs"], index=False)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    complete = scores[scores.heldout_all_roots.astype(bool)]
    winner = complete.sort_values("heldout_RMS_arcsec").iloc[0].to_dict()
    global_best = complete[complete.kind.eq("global")].sort_values(
        "heldout_RMS_arcsec"
    ).iloc[0].to_dict()
    local_best = complete[complete.kind.eq("local_attractor")].sort_values(
        "heldout_RMS_arcsec"
    ).iloc[0].to_dict()
    report = {
        "report_version": "P0585-LOCAL-ATTRACTOR-SCREEN-RESULTS-0.1.0",
        "status": "complete_opened_data_local_attractor_screen",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            key: sha256(ROOT / value) for key, value in protocol["inputs"].items()
        },
        "coverage": {
            "candidates": len(candidates),
            "amplitudes": len(protocol["positive_amplitudes"]),
            "scores": len(scores),
            "heldout_images": len(heldout),
        },
        "winner": winner,
        "global_best": global_best,
        "local_best": local_best,
        "local_improvement_over_global_fraction": float(
            (global_best["heldout_RMS_arcsec"] - local_best["heldout_RMS_arcsec"])
            / global_best["heldout_RMS_arcsec"]
        ),
        "parameter_impacts": impacts.to_dict("records"),
        "field_audit": {
            "maximum_route_normalization_error": float(
                audits.route_normalization_error.max()
            ),
            "maximum_annular_convergence_mean_fraction": float(
                audits.maximum_annular_convergence_mean_fraction.max()
            ),
            "maximum_normalized_curl_RMS": float(audits.normalized_curl_RMS.max()),
            "maximum_source_weight_crossing_destination": float(
                audits.source_weight_crossing_destination.max()
            ),
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0585 local-attractor screen",
                "",
                f"Winner: **{winner['candidate_id']} at epsilon={winner['epsilon']}**.",
                f"Winner RMS: **{winner['heldout_RMS_arcsec']:.4f} arcsec**.",
                f"Local improvement over global: **{100*report['local_improvement_over_global_fraction']:.3f}%**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    best_candidates = (
        complete.groupby("candidate_id", as_index=False)
        .heldout_RMS_arcsec.min()
        .sort_values("heldout_RMS_arcsec")
    )
    axes[0].plot(np.arange(len(best_candidates)), best_candidates.heldout_RMS_arcsec)
    axes[0].set(xlabel="candidate rank", ylabel="best complete RMS", title="33 destination fields")
    local_scores = scores[scores.kind.eq("local_attractor")]
    for mix, block in local_scores.groupby("local_mix"):
        display = block.groupby("epsilon", as_index=False).heldout_RMS_arcsec.median()
        axes[1].plot(display.epsilon, display.heldout_RMS_arcsec, "o-", label=f"mu={mix}")
    axes[1].set(xlabel="epsilon", ylabel="median RMS", title="local-attractor response")
    axes[1].legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
