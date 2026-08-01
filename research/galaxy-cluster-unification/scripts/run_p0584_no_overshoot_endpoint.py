#!/usr/bin/env python3
"""Test whether inner-member center overshoot causes P0583's angular failure."""

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

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, load_sources  # noqa: E402
from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_clash_stellar_morphology_response import MorphologyLens  # noqa: E402
from run_p0581_locked_endpoint_exact_root import endpoint_field  # noqa: E402
from run_p0583b_signed_endpoint_amplitude import json_safe  # noqa: E402
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    load_baryonic_anchors,
    load_images,
    score,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol_path = ROOT / "configs/p0584_no_overshoot_endpoint_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0583B_before_no_overshoot_scores":
        raise RuntimeError("P0584 protocol is not frozen")
    previous = json.loads(
        (ROOT / protocol["inputs"]["p0583b_report"]).read_text(encoding="utf-8")
    )
    if previous["disposition"] != "zero_amplitude_minimum_no_route_information":
        raise RuntimeError("P0583B trigger changed")
    p0581 = json.loads(
        (ROOT / protocol["inputs"]["p0581_protocol"]).read_text(encoding="utf-8")
    )
    p0583_protocol = json.loads(
        (ROOT / protocol["inputs"]["p0583_protocol"]).read_text(encoding="utf-8")
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
    source_catalog = load_sources(rxj_protocol, raw_protocol)
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
        members=source_catalog,
        parent=parent,
        baryons=baryons,
    )
    parameter_frame = pd.read_csv(ROOT / protocol["inputs"]["p0583_parameters"])
    scalar_parameters = parameter_frame[
        parameter_frame.variant.eq("scalar_baseline")
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
    audit_rows = []
    for travel_mode in protocol["travel_modes"]:
        field, audit = endpoint_field(
            p0581,
            context,
            {
                **protocol["locked_formula"],
                "travel_mode": travel_mode,
                "variant": travel_mode,
            },
        )
        audit_rows.append({"travel_mode": travel_mode, **audit})
        for epsilon in protocol["positive_amplitudes"]:
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
            predictions["travel_mode"] = travel_mode
            predictions["epsilon"] = float(epsilon)
            prediction_rows.append(predictions)
            score_rows.append(
                {
                    "travel_mode": travel_mode,
                    "epsilon": float(epsilon),
                    "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                    "heldout_converged_roots": metrics["converged_roots"],
                    "heldout_all_roots": metrics["all_roots_converged"],
                    "unit_correction_RMS_arcsec": float(
                        audit["raw_correction_RMS_arcsec"]
                    ),
                    "scaled_correction_RMS_arcsec": float(epsilon)
                    * float(audit["raw_correction_RMS_arcsec"]),
                }
            )
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    audits = pd.DataFrame(audit_rows)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)

    complete = scores[scores.heldout_all_roots.astype(bool)]
    winner = complete.sort_values("heldout_RMS_arcsec").iloc[0].to_dict()
    best_by_mode = []
    for mode, block in scores.groupby("travel_mode"):
        local = block[block.heldout_all_roots.astype(bool)].sort_values(
            "heldout_RMS_arcsec"
        )
        best_by_mode.append(local.iloc[0].to_dict())
    positive_complete = complete[complete.epsilon.gt(0.0)]
    best_positive = (
        positive_complete.sort_values("heldout_RMS_arcsec").iloc[0].to_dict()
        if len(positive_complete)
        else None
    )
    report = {
        "report_version": "P0584-NO-OVERSHOOT-ENDPOINT-RESULTS-0.1.0",
        "status": "complete_opened_data_no_overshoot_forensics",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            key: sha256(ROOT / value) for key, value in protocol["inputs"].items()
        },
        "coverage": {
            "travel_modes": len(protocol["travel_modes"]),
            "positive_amplitudes": len(protocol["positive_amplitudes"]),
            "fields": len(audits),
            "scores": len(scores),
            "heldout_images": len(heldout),
        },
        "winner_including_zero": winner,
        "best_positive_complete": best_positive,
        "best_by_mode": best_by_mode,
        "travel_audits": audits[
            [
                "travel_mode",
                "sources_crossing_center",
                "source_weight_crossing_center",
                "maximum_travel_arcsec",
                "median_travel_arcsec",
                "raw_correction_RMS_arcsec",
            ]
        ].to_dict("records"),
        "scores": scores.to_dict("records"),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0584 no-overshoot endpoint forensics",
                "",
                f"Winner including zero: **{winner['travel_mode']} at epsilon={winner['epsilon']}**.",
                f"Winner RMS: **{winner['heldout_RMS_arcsec']:.3f} arcsec**.",
                f"Best positive complete: **{best_positive}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    figure, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for mode, block in scores.groupby("travel_mode"):
        display = block.copy()
        display.loc[~display.heldout_all_roots.astype(bool), "heldout_RMS_arcsec"] = np.nan
        ax.plot(display.epsilon, display.heldout_RMS_arcsec, "o-", label=mode)
    ax.set(
        xlabel="positive endpoint amplitude epsilon",
        ylabel="held-out RMS (arcsec; complete roots only)",
        title="RX J2129 no-overshoot screen",
    )
    ax.legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
