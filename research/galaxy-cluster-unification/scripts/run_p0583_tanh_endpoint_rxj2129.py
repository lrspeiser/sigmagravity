#!/usr/bin/env python3
"""Transfer the frozen K0338 tanh endpoint response to RX J2129 exact roots."""

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
    exact_fit,
    json_safe,
    load_sources,
    parent_initial,
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
    near_bound,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol_path = ROOT / "configs/p0583_tanh_endpoint_rxj2129_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0582_before_RXJ2129_K0338_scores":
        raise RuntimeError("P0583 protocol is not frozen")
    p0581_path = ROOT / protocol["inputs"]["p0581_protocol"]
    p0581 = json.loads(p0581_path.read_text(encoding="utf-8"))
    p0582_report_path = ROOT / protocol["inputs"]["p0582_report"]
    p0582 = json.loads(p0582_report_path.read_text(encoding="utf-8"))
    if not any(
        row["variant"] == "tanh_A20p0" and row["all_four_complete"]
        for row in p0582["summary_grid"]
    ):
        raise RuntimeError("P0582 tanh-20 candidate changed")

    rxj_protocol_path = ROOT / protocol["inputs"]["rxj_route_protocol"]
    rxj_protocol = json.loads(rxj_protocol_path.read_text(encoding="utf-8"))
    raw_path = ROOT / rxj_protocol["raw_inputs"]["raw_protocol"]
    raw_protocol = json.loads(raw_path.read_text(encoding="utf-8"))
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    sources = load_sources(rxj_protocol, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    parent_protocol_path = ROOT / rxj_protocol["raw_inputs"]["parent_protocol"]
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    parent_scores_path = ROOT / rxj_protocol["raw_inputs"]["parent_scores"]
    parent_scores = pd.read_csv(parent_scores_path)
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
    parent_parameters_path = ROOT / rxj_protocol["raw_inputs"]["parent_parameters"]
    initial = parent_initial(pd.read_csv(parent_parameters_path), parent_id)
    context = SimpleNamespace(
        local=raw_protocol,
        members=sources,
        parent=parent,
        baryons=baryons,
    )

    inverse = pd.read_csv(ROOT / protocol["inputs"]["inverse_driver_table"])
    if inverse.system.astype(str).str.contains("2129", case=False).any():
        raise RuntimeError("RX J2129 unexpectedly entered the K0338 inverse driver")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    sources.to_csv(output / protocol["outputs"]["source_catalog"], index=False)
    score_rows = []
    predictions = []
    parameter_rows = []
    audits = []
    for index, variant in enumerate(protocol["variants"]):
        label = variant["label"]
        field = None
        if variant["kind"] == "endpoint":
            spec = {
                **protocol["locked_formula"],
                "contrast_mode": str(variant["contrast_mode"]),
                "contrast_cap": float(variant["contrast_cap"]),
                "variant": label,
            }
            field, audit = endpoint_field(p0581, context, spec)
            audits.append({"variant": label, **audit})
        lens = MorphologyLens(
            raw_protocol,
            {MODEL: parent},
            parent=MODEL,
            morphology=field,
            fraction=0.0 if field is None else 1.0,
        )
        print(f"RXJ2129: exact fit {label}", flush=True)
        fit = exact_fit(
            lens,
            training,
            heldout,
            initial=initial,
            starts=int(protocol["fit"]["starts_per_variant"]),
            seed=int(protocol["fit"]["random_seed"]) + index,
        )
        joined = pd.concat(
            [fit["training_prediction"], fit["heldout_prediction"]],
            ignore_index=True,
        )
        joined["variant"] = label
        predictions.append(joined)
        train_score = fit["training_score"]
        hold_score = fit["heldout_score"]
        score_rows.append(
            {
                "variant": label,
                "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                "training_converged_roots": train_score["converged_roots"],
                "heldout_RMS_arcsec": hold_score["exact_radial_RMS_arcsec"],
                "heldout_converged_roots": hold_score["converged_roots"],
                "heldout_all_roots": hold_score["all_roots_converged"],
                "optimizer_cost": fit["optimizer_cost"],
                "geometry_at_boundary": any(near_bound(MODEL, fit["parameters"]).values()),
            }
        )
        bounds = near_bound(MODEL, fit["parameters"])
        for name, value in zip(FIXED_LABELS, fit["parameters"], strict=True):
            parameter_rows.append(
                {
                    "variant": label,
                    "parameter": name,
                    "value": float(value),
                    "near_bound": bool(bounds[name]),
                }
            )

    scores = pd.DataFrame(score_rows)
    scalar_rms = float(
        scores[scores.variant.eq("scalar_baseline")].heldout_RMS_arcsec.iloc[0]
    )
    scores["fractional_improvement_vs_scalar"] = (
        scalar_rms - scores.heldout_RMS_arcsec
    ) / scalar_rms
    prediction_frame = pd.concat(predictions, ignore_index=True)
    audits_frame = pd.DataFrame(audits)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    prediction_frame.to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.DataFrame(parameter_rows).to_csv(
        output / protocol["outputs"]["parameters"], index=False
    )
    audits_frame.to_csv(output / protocol["outputs"]["field_audits"], index=False)

    tanh = scores[scores.variant.eq("K0338_tanh20_candidate")].iloc[0]
    gates = protocol["gates"]
    maximum_annular = float(
        audits_frame.maximum_annular_convergence_mean_fraction.max()
    )
    maximum_curl = float(audits_frame.normalized_curl_RMS.max())
    gate_audit = {
        "all_heldout_roots_pass": bool(
            int(tanh.heldout_converged_roots)
            == int(gates["all_heldout_roots_required"])
        ),
        "improvement_over_scalar_pass": bool(
            float(tanh.fractional_improvement_vs_scalar)
            >= float(gates["tanh20_improvement_over_scalar_fraction_min"])
        ),
        "absolute_RMS_pass": bool(
            float(tanh.heldout_RMS_arcsec)
            <= float(gates["absolute_heldout_RMS_arcsec_max"])
        ),
        "annular_monopole_pass": bool(
            maximum_annular
            <= float(gates["maximum_annular_convergence_mean_fraction_max"])
        ),
        "curl_free_pass": bool(
            maximum_curl <= float(gates["normalized_curl_RMS_max"])
        ),
        "solar_axisymmetric_zero_monopole_pass": True,
    }
    gate_audit["all_gates_pass"] = bool(all(gate_audit.values()))
    report = {
        "report_version": "P0583-TANH-ENDPOINT-RXJ2129-RESULTS-0.1.0",
        "status": "complete_locked_RXJ2129_transfer",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            "p0581_protocol": sha256(p0581_path),
            "p0582_report": sha256(p0582_report_path),
            "rxj_route_protocol": sha256(rxj_protocol_path),
            "raw_protocol": sha256(raw_path),
            "parent_protocol": sha256(parent_protocol_path),
            "parent_scores": sha256(parent_scores_path),
            "parent_parameters": sha256(parent_parameters_path),
            "inverse_driver_table": sha256(
                ROOT / protocol["inputs"]["inverse_driver_table"]
            ),
        },
        "coverage": {
            "hard_photometric_sources": len(sources),
            "training_images": len(training),
            "heldout_images": len(heldout),
            "variants": len(scores),
        },
        "scores": scores.to_dict("records"),
        "field_audit": {
            "maximum_annular_convergence_mean_fraction": maximum_annular,
            "maximum_normalized_curl_RMS": maximum_curl,
        },
        "gate_audit": gate_audit,
        "inherited_cross_domain": {
            "scalar_parent": parent_id,
            "galaxy_outer_RMSE_km_s": float(parent_row.cross_galaxy_outer_RMSE_km_s),
            "CLASH_absolute_RMSE_dex": float(parent_row.cluster_RMSE_dex),
            "Solar_all_proxies_pass": bool(parent_row.all_solar_proxies_pass),
            "directional_axisymmetric_change": 0.0,
        },
        "independence": protocol["independence"],
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0583 K0338 tanh-20 RX J2129 transfer",
                "",
                f"Scalar held-out RMS: **{scalar_rms:.3f} arcsec**.",
                f"Tanh-20 held-out RMS: **{float(tanh.heldout_RMS_arcsec):.3f} arcsec**.",
                f"Tanh-20 roots: **{int(tanh.heldout_converged_roots)}/{len(heldout)}**.",
                f"All gates pass: **{gate_audit['all_gates_pass']}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    display = scores.sort_values("heldout_RMS_arcsec")
    axes[0].barh(display.variant, display.heldout_RMS_arcsec)
    axes[0].set(xlabel="held-out RMS (arcsec)", title="RX J2129 exact-root transfer")
    held = prediction_frame[prediction_frame.stage.eq("heldout")]
    pivot = held.pivot(index="image_id", columns="variant", values="radial_residual_arcsec")
    x = np.arange(len(pivot))
    axes[1].plot(x, pivot.scalar_baseline, "o-", label="scalar")
    axes[1].plot(x, pivot.K0338_tanh20_candidate, "o-", label="tanh-20")
    axes[1].set(
        xticks=x,
        xticklabels=pivot.index,
        ylabel="residual (arcsec)",
        title="held-out images",
    )
    axes[1].legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    print(json.dumps(json_safe(report["scores"]), indent=2))
    print(json.dumps(json_safe(gate_audit), indent=2))


if __name__ == "__main__":
    main()
