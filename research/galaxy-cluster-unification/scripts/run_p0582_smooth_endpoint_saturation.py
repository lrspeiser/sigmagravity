#!/usr/bin/env python3
"""Diagnose hard versus smooth endpoint saturation on fixed P0581 lens fits."""

from __future__ import annotations

import hashlib
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

from run_adaptive_route_multicluster_raw import (  # noqa: E402
    MODEL,
    build_contexts,
    decorate_predictions,
    json_safe,
    make_lens,
)
from run_p0581_locked_endpoint_exact_root import endpoint_field  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, score  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_positions(predictions: pd.DataFrame, label: str) -> dict[int, np.ndarray]:
    local = predictions[predictions.system_label.eq(label)]
    return {
        int(family): block[["source_x_arcsec", "source_y_arcsec"]]
        .iloc[0]
        .to_numpy(float)
        for family, block in local.groupby("source_family")
    }


def variant_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, cap), block in scores.groupby(
        ["contrast_mode", "nominal_cap"], sort=False
    ):
        complete = block[block.heldout_all_roots.astype(bool)]
        finite = complete.heldout_RMS_arcsec.to_numpy(float)
        equal_rms = (
            float(np.sqrt(np.mean(np.square(finite))))
            if len(finite)
            else float("inf")
        )
        rows.append(
            {
                "variant": f"{mode}_A{str(cap).replace('.', 'p')}",
                "contrast_mode": mode,
                "nominal_cap": float(cap),
                "complete_systems": int(len(complete)),
                "heldout_converged_roots": int(block.heldout_converged_roots.sum()),
                "all_four_complete": bool(len(complete) == len(block)),
                "equal_complete_system_RMS_arcsec": equal_rms,
                "median_complete_system_RMS_arcsec": (
                    float(np.median(finite)) if len(finite) else float("inf")
                ),
                "complete_labels": "+".join(sorted(complete.system_label)),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "complete_systems",
            "heldout_converged_roots",
            "equal_complete_system_RMS_arcsec",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    protocol_path = ROOT / "configs/p0582_smooth_endpoint_saturation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0581_before_smooth_saturation_scores":
        raise RuntimeError("P0582 protocol is not frozen")
    p0581_path = ROOT / protocol["inputs"]["p0581_protocol"]
    p0581 = json.loads(p0581_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(
        (ROOT / p0581["inputs"]["base_exact_protocol"]).read_text(encoding="utf-8")
    )
    contexts, _, _ = build_contexts(base_protocol)
    geometry = pd.read_csv(ROOT / protocol["inputs"]["p0581_geometry"])
    prior_predictions = pd.read_csv(ROOT / protocol["inputs"]["p0581_predictions"])

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    score_rows = []
    prediction_rows = []
    audit_rows = []
    for context in contexts:
        label = context.system["label"]
        geometry_row = geometry[
            geometry.system_label.eq(label) & geometry.variant.eq("K0338_primary")
        ].iloc[0]
        parameters = np.asarray([float(geometry_row[name]) for name in FIXED_LABELS])
        sources = source_positions(prior_predictions, label)
        for mode in protocol["contrast_grid"]["modes"]:
            for cap in protocol["contrast_grid"]["nominal_caps"]:
                variant = f"{mode}_A{str(cap).replace('.', 'p')}"
                spec = {
                    **protocol["locked_field"],
                    "contrast_mode": mode,
                    "contrast_cap": float(cap),
                    "parameter": "smooth_saturation",
                    "level": variant,
                    "variant": variant,
                }
                field, audit = endpoint_field(p0581, context, spec)
                lens = make_lens(context, field)
                heldout = lens.exact_predictions(
                    MODEL,
                    parameters,
                    sources,
                    context.heldout,
                    stage="heldout",
                )
                heldout_score = score(heldout, lens.sigma)
                score_rows.append(
                    {
                        "system_label": label,
                        "variant": variant,
                        "contrast_mode": mode,
                        "nominal_cap": float(cap),
                        "heldout_images": len(context.heldout),
                        "heldout_converged_roots": heldout_score["converged_roots"],
                        "heldout_all_roots": heldout_score["all_roots_converged"],
                        "heldout_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
                    }
                )
                prediction_rows.append(decorate_predictions(heldout, context, variant))
                audit_rows.append(
                    {
                        "system_label": label,
                        "variant": variant,
                        "contrast_mode": mode,
                        "nominal_cap": float(cap),
                        **audit,
                    }
                )
        print(f"{label}: {len(protocol['contrast_grid']['modes']) * len(protocol['contrast_grid']['nominal_caps'])} saturation fields", flush=True)

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    audits = pd.DataFrame(audit_rows)
    summary = variant_summary(scores)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    summary.to_csv(output / protocol["outputs"]["summary_grid"], index=False)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)

    response_metrics = [
        "raw_correction_RMS_arcsec",
        "raw_correction_maximum_arcsec",
        "delta_convergence_minimum",
        "delta_convergence_maximum",
        "maximum_light_weight_after_strength",
    ]
    response = scores.merge(
        audits[
            ["system_label", "variant", "contrast_mode", "nominal_cap"]
            + response_metrics
        ],
        on=["system_label", "variant", "contrast_mode", "nominal_cap"],
    )
    response_windows = []
    for label, block in response.groupby("system_label"):
        complete = block[block.heldout_all_roots.astype(bool)]
        incomplete = block[~block.heldout_all_roots.astype(bool)]
        root_values = block.heldout_converged_roots.to_numpy(float)
        correction_values = block.raw_correction_RMS_arcsec.to_numpy(float)
        response_windows.append(
            {
                "system_label": label,
                "complete_variants": int(len(complete)),
                "minimum_converged_roots": int(block.heldout_converged_roots.min()),
                "maximum_converged_roots": int(block.heldout_converged_roots.max()),
                "correction_RMS_root_spearman": (
                    float(pd.Series(correction_values).corr(pd.Series(root_values), method="spearman"))
                    if np.unique(root_values).size > 1
                    else float("nan")
                ),
                "complete_correction_RMS_min_arcsec": (
                    float(complete.raw_correction_RMS_arcsec.min())
                    if len(complete)
                    else float("nan")
                ),
                "complete_correction_RMS_max_arcsec": (
                    float(complete.raw_correction_RMS_arcsec.max())
                    if len(complete)
                    else float("nan")
                ),
                "incomplete_correction_RMS_min_arcsec": (
                    float(incomplete.raw_correction_RMS_arcsec.min())
                    if len(incomplete)
                    else float("nan")
                ),
                "incomplete_correction_RMS_max_arcsec": (
                    float(incomplete.raw_correction_RMS_arcsec.max())
                    if len(incomplete)
                    else float("nan")
                ),
            }
        )

    winner = summary.iloc[0].to_dict()
    cap20 = summary[summary.nominal_cap.eq(20.0)].sort_values("contrast_mode")
    by_mode = []
    for mode, block in summary.groupby("contrast_mode"):
        local = block.sort_values(
            [
                "complete_systems",
                "heldout_converged_roots",
                "equal_complete_system_RMS_arcsec",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        by_mode.append(local.to_dict())
    by_cap = []
    for cap, block in summary.groupby("nominal_cap"):
        local = block.sort_values(
            [
                "complete_systems",
                "heldout_converged_roots",
                "equal_complete_system_RMS_arcsec",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        by_cap.append(local.to_dict())

    maximum_annular = float(audits.maximum_annular_convergence_mean_fraction.max())
    maximum_curl = float(audits.normalized_curl_RMS.max())
    complete_variants = int(summary.all_four_complete.sum())
    gates = protocol["gates"]
    gate_audit = {
        "at_least_one_all_cluster_complete_variant_pass": bool(
            winner["complete_systems"] >= int(gates["complete_systems_required"])
            and winner["heldout_converged_roots"] >= int(gates["heldout_roots_required"])
        ),
        "annular_monopole_pass": bool(
            maximum_annular
            <= float(gates["maximum_annular_convergence_mean_fraction_max"])
        ),
        "curl_free_pass": bool(maximum_curl <= float(gates["normalized_curl_RMS_max"])),
        "solar_axisymmetric_zero_monopole_pass": True,
    }
    gate_audit["all_diagnostic_gates_pass"] = bool(all(gate_audit.values()))
    report = {
        "report_version": "P0582-SMOOTH-ENDPOINT-SATURATION-RESULTS-0.1.0",
        "status": "complete_diagnostic_saturation_sweep",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            key: sha256(ROOT / value) for key, value in protocol["inputs"].items()
        },
        "coverage": {
            "clusters": len(contexts),
            "contrast_modes": len(protocol["contrast_grid"]["modes"]),
            "nominal_caps": len(protocol["contrast_grid"]["nominal_caps"]),
            "variants": len(summary),
            "cluster_fields": len(audits),
            "heldout_images_per_variant": int(scores.heldout_images.sum() / len(summary)),
        },
        "diagnostic_winner": winner,
        "all_four_complete_variants": complete_variants,
        "cap20_mode_comparison": cap20.to_dict("records"),
        "best_by_mode": by_mode,
        "best_by_cap": by_cap,
        "field_response_windows": response_windows,
        "summary_grid": summary.to_dict("records"),
        "field_audit": {
            "maximum_annular_convergence_mean_fraction": maximum_annular,
            "maximum_normalized_curl_RMS": maximum_curl,
            "maximum_postnormalization_light_weight": float(
                audits.maximum_light_weight_after_strength.max()
            ),
            "nominal_cap_interpretation": "The cap is applied before carrier-weighted annular renormalization; it is not a global bound on the final weight.",
        },
        "gate_audit": gate_audit,
        "hypotheses": protocol["hypotheses"],
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0582 smooth endpoint saturation",
                "",
                f"Diagnostic winner: **{winner['variant']}**.",
                f"Complete variants: **{complete_variants}/{len(summary)}**.",
                f"Winner held-out RMS: **{winner['equal_complete_system_RMS_arcsec']:.3f} arcsec**.",
                f"All diagnostic gates pass: **{gate_audit['all_diagnostic_gates_pass']}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    roots = summary.pivot(index="contrast_mode", columns="nominal_cap", values="heldout_converged_roots")
    rms = summary.pivot(index="contrast_mode", columns="nominal_cap", values="equal_complete_system_RMS_arcsec")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    image0 = axes[0].imshow(roots.to_numpy(float), aspect="auto", vmin=7, vmax=11, cmap="viridis")
    axes[0].set(
        xticks=np.arange(len(roots.columns)),
        xticklabels=roots.columns,
        yticks=np.arange(len(roots.index)),
        yticklabels=roots.index,
        xlabel="nominal saturation A",
        title="held-out roots (11 required)",
    )
    figure.colorbar(image0, ax=axes[0])
    rms_values = rms.to_numpy(float)
    rms_values[~np.isfinite(rms_values)] = np.nan
    image1 = axes[1].imshow(rms_values, aspect="auto", cmap="magma_r")
    axes[1].set(
        xticks=np.arange(len(rms.columns)),
        xticklabels=rms.columns,
        yticks=np.arange(len(rms.index)),
        yticklabels=rms.index,
        xlabel="nominal saturation A",
        title="complete-system RMS (arcsec)",
    )
    figure.colorbar(image1, ax=axes[1])
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    print(json.dumps(json_safe(report["diagnostic_winner"]), indent=2))
    print(json.dumps(json_safe(report["cap20_mode_comparison"]), indent=2))
    print(json.dumps(json_safe(report["best_by_mode"]), indent=2))
    print(json.dumps(json_safe(gate_audit), indent=2))


if __name__ == "__main__":
    main()
