#!/usr/bin/env python3
"""Diagnose the sign and scale of P0583's failed RX J2129 endpoint field."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import (  # noqa: E402
    MODEL,
    baryon_field,
    load_sources,
)
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0581_locked_endpoint_exact_root import endpoint_field  # noqa: E402
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    load_baryonic_anchors,
    load_images,
    score,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> None:
    protocol_path = ROOT / "configs/p0583b_signed_endpoint_amplitude_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0583_before_signed_amplitude_scores":
        raise RuntimeError("P0583B protocol is not frozen")
    p0583_protocol = json.loads(
        (ROOT / protocol["inputs"]["p0583_protocol"]).read_text(encoding="utf-8")
    )
    p0583_report = json.loads(
        (ROOT / protocol["inputs"]["p0583_report"]).read_text(encoding="utf-8")
    )
    failed = next(
        row
        for row in p0583_report["scores"]
        if row["variant"] == "K0338_tanh20_candidate"
    )
    if failed["heldout_RMS_arcsec"] <= 10.0:
        raise RuntimeError("P0583 failure trigger changed")
    p0581 = json.loads(
        (ROOT / protocol["inputs"]["p0581_protocol"]).read_text(encoding="utf-8")
    )
    rxj_protocol = json.loads(
        (ROOT / p0583_protocol["inputs"]["rxj_route_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    raw_protocol = json.loads(
        (ROOT / rxj_protocol["raw_inputs"]["raw_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    images = load_images(raw_protocol)
    _, heldout = split_images(images, raw_protocol)
    sources_catalog = load_sources(rxj_protocol, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    parent_protocol = json.loads(
        (ROOT / rxj_protocol["raw_inputs"]["parent_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    parent_scores = pd.read_csv(ROOT / rxj_protocol["raw_inputs"]["parent_scores"])
    parent_id = str(rxj_protocol["raw_inputs"]["parent_candidate"])
    parent_row = parent_scores[parent_scores.candidate_id.eq(parent_id)].iloc[0]
    parent_specs = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    parent, _ = raw_field(
        parent_specs[parent_id],
        float(parent_row.universal_q),
        anchors,
        raw_protocol,
        1.2e-10,
    )
    baryons = baryon_field(anchors, raw_protocol)
    context = SimpleNamespace(
        local=raw_protocol,
        members=sources_catalog,
        parent=parent,
        baryons=baryons,
    )
    tanh_variant = next(
        row
        for row in p0583_protocol["variants"]
        if row["label"] == "K0338_tanh20_candidate"
    )
    field, audit = endpoint_field(
        p0581,
        context,
        {
            **p0583_protocol["locked_formula"],
            "contrast_mode": tanh_variant["contrast_mode"],
            "contrast_cap": tanh_variant["contrast_cap"],
            "variant": tanh_variant["label"],
        },
    )

    parameters_frame = pd.read_csv(ROOT / protocol["inputs"]["p0583_parameters"])
    scalar_parameters = parameters_frame[
        parameters_frame.variant.eq("scalar_baseline")
    ].set_index("parameter")
    parameters = np.asarray(
        [float(scalar_parameters.loc[name, "value"]) for name in FIXED_LABELS]
    )
    prior_predictions = pd.read_csv(ROOT / protocol["inputs"]["p0583_predictions"])
    scalar_predictions = prior_predictions[
        prior_predictions.variant.eq("scalar_baseline")
    ]
    source_positions = {
        int(family): block[["source_x_arcsec", "source_y_arcsec"]]
        .iloc[0]
        .to_numpy(float)
        for family, block in scalar_predictions.groupby("source_family")
    }

    score_rows = []
    prediction_rows = []
    for epsilon in protocol["signed_amplitudes"]:
        lens = MorphologyLens(
            raw_protocol,
            {MODEL: parent},
            parent=MODEL,
            morphology=field,
            fraction=float(epsilon),
        )
        predictions = lens.exact_predictions(
            MODEL,
            parameters,
            source_positions,
            heldout,
            stage="heldout",
        )
        metrics = score(predictions, lens.sigma)
        predictions["epsilon"] = float(epsilon)
        prediction_rows.append(predictions)
        score_rows.append(
            {
                "epsilon": float(epsilon),
                "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                "heldout_converged_roots": metrics["converged_roots"],
                "heldout_all_roots": metrics["all_roots_converged"],
                "correction_field_RMS_arcsec": abs(float(epsilon))
                * float(audit["raw_correction_RMS_arcsec"]),
            }
        )
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    complete = scores[scores.heldout_all_roots.astype(bool)]
    winner = complete.sort_values("heldout_RMS_arcsec").iloc[0].to_dict()
    zero = scores[scores.epsilon.eq(0.0)].iloc[0]
    positive = scores[scores.epsilon.gt(0.0)].sort_values("heldout_RMS_arcsec").iloc[0]
    negative = scores[scores.epsilon.lt(0.0)].sort_values("heldout_RMS_arcsec").iloc[0]
    local = predictions[predictions.epsilon.isin([-0.025, 0.0, 0.025])].pivot(
        index="image_id",
        columns="epsilon",
        values=[
            "predicted_x_arcsec",
            "predicted_y_arcsec",
            "delta_x_arcsec",
            "delta_y_arcsec",
            "root_converged",
        ],
    )
    alignment_rows = []
    for image_id, row in local.iterrows():
        both_sides = bool(row[("root_converged", -0.025)]) and bool(
            row[("root_converged", 0.025)]
        )
        if both_sides:
            derivative = np.asarray(
                [
                    (
                        float(row[("predicted_x_arcsec", 0.025)])
                        - float(row[("predicted_x_arcsec", -0.025)])
                    )
                    / 0.05,
                    (
                        float(row[("predicted_y_arcsec", 0.025)])
                        - float(row[("predicted_y_arcsec", -0.025)])
                    )
                    / 0.05,
                ]
            )
            residual = np.asarray(
                [
                    float(row[("delta_x_arcsec", 0.0)]),
                    float(row[("delta_y_arcsec", 0.0)]),
                ]
            )
            squared_residual_derivative = float(2.0 * np.dot(residual, derivative))
        else:
            squared_residual_derivative = float("nan")
        alignment_rows.append(
            {
                "image_id": str(image_id),
                "both_signed_roots_converged": both_sides,
                "positive_root_converged": bool(row[("root_converged", 0.025)]),
                "first_order_squared_residual_derivative": squared_residual_derivative,
                "positive_direction_locally_improves": bool(
                    both_sides and squared_residual_derivative < 0.0
                ),
            }
        )
    aligned = [
        row for row in alignment_rows if row["both_signed_roots_converged"]
    ]
    local_alignment = {
        "images_with_both_signed_roots": len(aligned),
        "images_improved_to_first_order_by_positive_route": int(
            sum(row["positive_direction_locally_improves"] for row in aligned)
        ),
        "image_lost_at_positive_0p025": "+".join(
            row["image_id"]
            for row in alignment_rows
            if not row["positive_root_converged"]
        ),
        "per_image": alignment_rows,
    }
    if float(winner["epsilon"]) > 0.0:
        disposition = "positive_amplitude_minimum"
    elif float(winner["epsilon"]) < 0.0:
        disposition = "negative_amplitude_minimum_wrong_sign"
    else:
        disposition = "zero_amplitude_minimum_no_route_information"
    report = {
        "report_version": "P0583B-SIGNED-ENDPOINT-AMPLITUDE-RESULTS-0.1.0",
        "status": "complete_opened_data_failure_forensics",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            key: sha256(ROOT / value) for key, value in protocol["inputs"].items()
        },
        "coverage": {
            "signed_amplitudes": len(scores),
            "heldout_images": len(heldout),
            "geometry_parameters_refit": 0,
            "gravity_parameters_refit": 0,
        },
        "best_amplitude": winner,
        "zero_amplitude": zero.to_dict(),
        "best_positive_amplitude": positive.to_dict(),
        "best_negative_amplitude": negative.to_dict(),
        "local_directional_alignment": local_alignment,
        "disposition": disposition,
        "scores": scores.to_dict("records"),
        "unit_field_audit": audit,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0583B signed endpoint amplitude forensics",
                "",
                f"Best epsilon: **{float(winner['epsilon']):.3f}**.",
                f"Best RMS: **{float(winner['heldout_RMS_arcsec']):.3f} arcsec**.",
                f"Disposition: **{disposition}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    figure, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(scores.epsilon, scores.heldout_RMS_arcsec, "o-")
    ax.axvline(0.0, color="black", lw=1)
    ax.set(
        xlabel="signed endpoint amplitude epsilon",
        ylabel="held-out RMS (arcsec)",
        title="RX J2129 at fixed scalar geometry",
    )
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
