#!/usr/bin/env python3
"""One-at-a-time raw route sensitivity around the P0605 failure."""

from __future__ import annotations

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

from run_p0601_frozen_potential_raw_lensing import fit_model, json_safe  # noqa: E402
from run_p0605_strict_route_raw_lensing import build_fields  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    load_baryonic_anchors,
    load_images,
    spec_for,
)


def variants(protocol: dict) -> list[dict]:
    base = dict(protocol["base_route"])
    output = [
        {
            "variant_id": "no_route",
            "changed_parameter": "fraction_max",
            "changed_level": 0.0,
            **base,
            "fraction_max": 0.0,
        }
    ]
    for value in protocol["one_at_a_time"]["fraction_max"]:
        output.append(
            {
                "variant_id": f"fraction_{value:g}",
                "changed_parameter": "fraction_max",
                "changed_level": float(value),
                **base,
                "fraction_max": float(value),
            }
        )
    for parameter in ("length_over_R80", "width_over_R80"):
        for value in protocol["one_at_a_time"][parameter]:
            candidate = dict(base)
            candidate[parameter] = float(value)
            output.append(
                {
                    "variant_id": f"{parameter}_{value:g}",
                    "changed_parameter": parameter,
                    "changed_level": float(value),
                    **candidate,
                }
            )
    if len(output) != protocol["one_at_a_time"]["candidate_count"]:
        raise RuntimeError("P0606 variant count changed")
    return output


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0606_raw_route_sensitivity_protocol.json").read_text()
    )
    parent = json.loads((ROOT / protocol["inputs"]["P0605_protocol"]).read_text())
    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_protocol"]).read_text())
    images = load_images(raw_protocol)
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    training = images[~images.image_id.isin(heldout_ids)].copy()
    heldout = images[images.image_id.isin(heldout_ids)].copy()
    anchors = load_baryonic_anchors(raw_protocol)
    previous = pd.read_csv(ROOT / protocol["inputs"]["initial_parameters"])
    block = previous[previous.model.eq("P0599_potential_shape")].set_index("parameter")
    initial = block.loc[list(spec_for("fixed").labels), "value"].to_numpy(float)
    rows, prediction_frames = [], []
    for index, spec in enumerate(variants(protocol)):
        local_parent = json.loads(json.dumps(parent))
        local_parent["selected_route"] = {
            key: spec[key]
            for key in (
                "fraction_max",
                "length_over_R80",
                "radius_exponent",
                "width_over_R80",
                "shape_gate",
                "source_acceleration_gate_power",
            )
        }
        fields, _, diagnostic = build_fields(anchors, raw_protocol, local_parent)
        row, predictions, _ = fit_model(
            spec["variant_id"],
            fields["strict_route_P0599"],
            raw_protocol,
            training,
            heldout,
            initial,
            starts=int(protocol["validation"]["optimization_starts_per_variant"]),
            seed=21292026 + 60600 + index,
        )
        row.update(spec)
        row["effective_route_fraction_RXJ2129"] = diagnostic["effective_route_fraction"]
        rows.append(row)
        prediction_frames.append(predictions)
    scores = pd.DataFrame(rows).sort_values("training_RMS_arcsec").reset_index(drop=True)
    impacts = []
    base = scores[scores.variant_id.eq("fraction_0.2")]
    for parameter in ("fraction_max", "length_over_R80", "width_over_R80"):
        family = pd.concat(
            [scores[scores.changed_parameter.eq(parameter)], base], ignore_index=True
        ).drop_duplicates("variant_id")
        finite = family[
            np.isfinite(family.training_RMS_arcsec)
            & np.isfinite(family.heldout_RMS_arcsec)
        ]
        best = finite.sort_values("training_RMS_arcsec").iloc[0]
        impacts.append(
            {
                "parameter": parameter,
                "variants": len(family),
                "failed_exact_root_variants": int(len(family) - len(finite)),
                "training_RMS_span_arcsec": float(
                    finite.training_RMS_arcsec.max() - finite.training_RMS_arcsec.min()
                ),
                "spent_heldout_RMS_span_arcsec": float(
                    finite.heldout_RMS_arcsec.max() - finite.heldout_RMS_arcsec.min()
                ),
                "best_training_variant": best.variant_id,
                "best_training_RMS_arcsec": float(best.training_RMS_arcsec),
                "corresponding_spent_heldout_RMS_arcsec": float(best.heldout_RMS_arcsec),
            }
        )
    impacts = pd.DataFrame(impacts).sort_values("training_RMS_span_arcsec", ascending=False)
    selected = scores.iloc[0]
    no_route = scores[scores.variant_id.eq("no_route")].iloc[0]
    report = {
        "report_version": "P0606-RAW-ROUTE-SENSITIVITY-RESULTS-0.1.0",
        "status": "complete_posthoc_spent_raw_route_response",
        "coverage": {
            "variants": len(scores),
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "optimization_starts_per_variant": protocol["validation"]["optimization_starts_per_variant"],
        },
        "training_selected_variant": selected.to_dict(),
        "no_route_reference": no_route.to_dict(),
        "parameter_impacts": impacts.to_dict("records"),
        "strict_interpretation": {
            "heldout_is_fresh": False,
            "selection_uses_training_only": True,
            "one_start_global_optimum_claimed": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )
    finite = scores[np.isfinite(scores.training_RMS_arcsec) & np.isfinite(scores.heldout_RMS_arcsec)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for parameter, group in finite.groupby("changed_parameter", sort=False):
        axes[0].scatter(group.training_RMS_arcsec, group.heldout_RMS_arcsec, label=parameter)
    axes[0].scatter([no_route.training_RMS_arcsec], [no_route.heldout_RMS_arcsec], marker="*", s=140, color="black", label="no route")
    axes[0].set(xlabel="training RMS (arcsec)", ylabel="spent held-out RMS (arcsec)", title="Raw route sensitivity")
    axes[0].legend(fontsize=7)
    display = impacts.sort_values("training_RMS_span_arcsec")
    axes[1].barh(display.parameter, display.training_RMS_span_arcsec, color="#1261A0")
    axes[1].set(xlabel="training RMS response span (arcsec)", title="Which route parameter matters?")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0606 raw route sensitivity\n\n"
        f"Training-selected variant: **{selected.variant_id}**, training RMS "
        f"**{selected.training_RMS_arcsec:.4f} arcsec**, spent held-out RMS "
        f"**{selected.heldout_RMS_arcsec:.4f} arcsec**.\n\n"
        "This is a local response study on spent data, not validation.\n"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
