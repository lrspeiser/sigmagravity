#!/usr/bin/env python3
"""Run the frozen P0558 measured central-mass anchor proxy-tensor diagnostic."""

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

from run_cpr0_accept_clash_bcg_stellar import load_clash_bcg_properties  # noqa: E402
from run_member_tidal_metric import build_contexts, fit_context  # noqa: E402
from run_p0557_baryon_proxy_tidal import (  # noqa: E402
    build_candidate_context,
    json_safe,
    mixed_catalog,
    prepare_component_catalogs,
    sha256,
)
from run_unbounded_running_multicluster_raw import aggregate_system_scores  # noqa: E402


def mass_anchors(protocol: dict) -> pd.DataFrame:
    table = load_clash_bcg_properties(ROOT / protocol["inputs"]["tian_table"])
    table = table.set_index("cluster").loc[protocol["systems"]].reset_index()
    rows = []
    for row in table.itertuples(index=False):
        star = float(row.stellar_mass_1e11_msun)
        gas = float(row.gas_mass_1e11_msun)
        gas_error = float(row.gas_mass_error_1e11_msun)
        fraction = gas / (star + gas)
        fraction_low = max(gas - gas_error, 0.0) / (
            1.1 * star + max(gas - gas_error, 0.0)
        )
        fraction_high = (gas + gas_error) / (0.9 * star + gas + gas_error)
        rows.append(
            {
                "system_label": row.cluster,
                "anchor_radius_kpc": float(row.central_radius_kpc),
                "BCG_stellar_mass_1e11_msun": star,
                "BCG_stellar_mass_sigma_1e11_msun": 0.1 * star,
                "gas_mass_1e11_msun": gas,
                "gas_mass_sigma_1e11_msun": gas_error,
                "nominal_gas_fraction": fraction,
                "conservative_gas_fraction_low": fraction_low,
                "conservative_gas_fraction_high": fraction_high,
            }
        )
    return pd.DataFrame(rows)


def variant_for(anchor: pd.Series, model: dict) -> dict:
    gas = float(model["gas_mass_multiplier"]) * float(anchor.gas_mass_1e11_msun)
    star = float(anchor.BCG_stellar_mass_1e11_msun)
    fraction = gas / (star + gas)
    return {
        "variant_id": model["model_id"],
        "member_fraction": 0.0,
        "star_fraction": 1.0 - fraction,
        "gas_fraction": fraction,
        "gas_transform": model["gas_transform"],
    }


def rms(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    config_path = ROOT / "configs/p0558_mass_anchor_proxy_tidal_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_before_any_"):
        raise RuntimeError("P0558 protocol is not prospectively frozen")
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    member = json.loads(
        (ROOT / protocol["inputs"]["member_tidal_protocol"]).read_text()
    )
    member["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    contexts, _, input_hashes = build_contexts(
        member, softening_kpc=float(protocol["locked_field"]["softening_kpc"])
    )
    anchors = mass_anchors(protocol)
    anchor_lookup = anchors.set_index("system_label")
    catalogs, _ = prepare_component_catalogs(p0557, contexts)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(output / protocol["outputs"]["mass_anchors"], index=False)
    operator = {"operator_id": "contrast", "subtract_circular_mean": True}
    starts = int(protocol["optimization"]["starts_per_fit"])
    seed = int(protocol["optimization"]["random_seed"])
    score_rows = []
    audit_rows = []
    predictions = []
    aggregates = {}
    for model_index, model in enumerate(protocol["models"]):
        model_id = model["model_id"]
        coupling = float(model["tensor_t"])
        heldout_scores = []
        validation_subset_scores = []
        for system_index, base in enumerate(contexts):
            label = base.system["label"]
            if label not in protocol["systems"]:
                continue
            if coupling == 0.0:
                context = base
                gas_fraction = 0.0
            else:
                variant = variant_for(anchor_lookup.loc[label], model)
                gas_fraction = float(variant["gas_fraction"])
                catalog = mixed_catalog(catalogs[label], variant)
                context = build_candidate_context(
                    base,
                    catalog,
                    p0557,
                    model_id,
                    operator,
                    pixels_per_axis=int(protocol["locked_field"]["pixels_per_axis"]),
                    softening_kpc=float(protocol["locked_field"]["softening_kpc"]),
                    audit_rows=audit_rows,
                    stage="p0558_exact",
                )
            print(
                f"P0558 exact: {label} {model_id} f_gas={gas_fraction:.4f} t={coupling:g}",
                flush=True,
            )
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=seed + model_index * 100 + system_index,
            )
            heldout_scores.append(fitted["heldout"])
            if label in {"MACS1115", "MACS1931"}:
                validation_subset_scores.append(fitted["heldout"])
            predictions.extend(
                [fitted["training_predictions"], fitted["heldout_predictions"]]
            )
            score_rows.append(
                {
                    "row_type": "system",
                    "model_id": model_id,
                    "role": model["role"],
                    "system_label": label,
                    "tensor_t": coupling,
                    "gas_transform": model["gas_transform"],
                    "gas_mass_multiplier": float(model["gas_mass_multiplier"]),
                    "effective_gas_fraction": gas_fraction,
                    "heldout_exact_RMS_arcsec": fitted["heldout"][
                        "exact_radial_RMS_arcsec"
                    ],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
        aggregate = aggregate_system_scores(heldout_scores)
        subset = aggregate_system_scores(validation_subset_scores)
        aggregates[model_id] = {"all_four": aggregate, "validation_subset": subset}
        score_rows.append(
            {
                "row_type": "aggregate",
                "model_id": model_id,
                "role": model["role"],
                "system_label": "all_four",
                "tensor_t": coupling,
                "gas_transform": model["gas_transform"],
                "gas_mass_multiplier": float(model["gas_mass_multiplier"]),
                "effective_gas_fraction": None,
                "heldout_exact_RMS_arcsec": aggregate[
                    "equal_system_radial_RMS_arcsec"
                ],
                "all_heldout_roots": aggregate["all_roots_converged"],
            }
        )
        score_rows.append(
            {
                "row_type": "aggregate",
                "model_id": model_id,
                "role": model["role"],
                "system_label": "validation_subset",
                "tensor_t": coupling,
                "gas_transform": model["gas_transform"],
                "gas_mass_multiplier": float(model["gas_mass_multiplier"]),
                "effective_gas_fraction": None,
                "heldout_exact_RMS_arcsec": subset[
                    "equal_system_radial_RMS_arcsec"
                ],
                "all_heldout_roots": subset["all_roots_converged"],
            }
        )
    scores = pd.DataFrame(score_rows)
    systems = scores[scores.row_type.eq("system")]
    zero = systems[systems.model_id.eq("zero")].set_index("system_label")
    primary = systems[systems.model_id.eq("measured_sqrt")].set_index("system_label")
    scores["improvement_fraction_vs_zero"] = np.nan
    for index, row in scores.iterrows():
        if row["row_type"] == "system":
            reference = float(zero.loc[row["system_label"], "heldout_exact_RMS_arcsec"])
        else:
            reference = float(
                aggregates["zero"][
                    "all_four" if row["system_label"] == "all_four" else "validation_subset"
                ]["equal_system_radial_RMS_arcsec"]
            )
        if row["heldout_exact_RMS_arcsec"] is not None:
            scores.loc[index, "improvement_fraction_vs_zero"] = 1.0 - float(
                row["heldout_exact_RMS_arcsec"]
            ) / reference
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output / protocol["outputs"]["tensor_audits"], index=False
    )
    metric = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text()
    )
    halo = float(
        metric["comparators"]["compact_halo_validation"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    zero_all = float(aggregates["zero"]["all_four"]["equal_system_radial_RMS_arcsec"])
    primary_all = float(
        aggregates["measured_sqrt"]["all_four"]["equal_system_radial_RMS_arcsec"]
    )
    primary_subset = float(
        aggregates["measured_sqrt"]["validation_subset"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    improvement = 1.0 - primary_all / zero_all
    all_improve = bool(
        (primary.heldout_exact_RMS_arcsec < zero.heldout_exact_RMS_arcsec).all()
    )
    all_roots = bool(primary.all_heldout_roots.astype(bool).all())
    gates = protocol["advance_gates"]
    gate_audit = {
        "primary_all_heldout_roots_converged": all_roots,
        "primary_all_four_systems_improve": all_improve,
        "primary_equal_system_RMS_improvement_fraction": improvement,
        "primary_improvement_pass": improvement
        >= float(gates["primary_equal_system_RMS_improvement_fraction_min"]),
        "validation_subset_to_compact_halo_RMS_ratio": primary_subset / halo,
        "compact_halo_ratio_pass": primary_subset / halo
        <= float(gates["validation_subset_to_compact_halo_RMS_ratio_max"]),
    }
    report = {
        "report_version": "P0558-MASS-ANCHOR-PROXY-TIDAL-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "input_hashes": input_hashes,
        "mass_anchors": anchors.to_dict("records"),
        "scores": scores.to_dict("records"),
        "primary": {
            "zero_all_four_RMS_arcsec": zero_all,
            "measured_sqrt_all_four_RMS_arcsec": primary_all,
            "improvement_fraction": improvement,
            "all_four_systems_improve": all_improve,
            "all_heldout_roots_converged": all_roots,
        },
        "comparators": {
            "validation_subset_measured_sqrt_RMS_arcsec": primary_subset,
            "compact_halo_validation_RMS_arcsec": halo,
            "ratio": primary_subset / halo,
        },
        "gate_audit": gate_audit,
        "verdict": {
            "all_advancement_gates_pass": bool(
                gate_audit["primary_all_heldout_roots_converged"]
                and gate_audit["primary_all_four_systems_improve"]
                and gate_audit["primary_improvement_pass"]
                and gate_audit["compact_halo_ratio_pass"]
            ),
            "formula_promoted": False,
            "X5_projected_gas_mass_still_required": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    plot = scores[
        (scores.row_type == "aggregate") & (scores.system_label == "all_four")
    ].sort_values("heldout_exact_RMS_arcsec")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(plot.model_id, plot.heldout_exact_RMS_arcsec)
    axes[0].set(xlabel="four-cluster held-out exact RMS (arcsec)", title="Measured mass-anchor variants")
    axes[1].bar(anchors.system_label, 100.0 * anchors.nominal_gas_fraction)
    axes[1].set(ylabel="central gas fraction (%)", title="Published central baryon anchors")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    (output / protocol["outputs"]["summary"]).write_text(
        f"""# P0558 measured mass-anchor proxy tensor

The primary measured-mass sqrt-X-ray tensor changes the four-cluster held-out
exact RMS from {zero_all:.3f} to {primary_all:.3f} arcsec
({100.0 * improvement:+.3f}%). It improves all four systems: {all_improve}.
Every primary held-out root converges: {all_roots}. No formula is promoted.
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["primary"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
