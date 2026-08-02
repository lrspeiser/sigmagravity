#!/usr/bin/env python3
"""Fit and globally audit the frozen P0685 RX J2129 lens."""

from __future__ import annotations

import argparse
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

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256
from run_p0672_spent_rxj2129_absolute_raw_topology import (
    AbsoluteGridLens,
    PhysicalDeflectionGrid,
    exact_fit,
    global_topology,
    load_images,
    near_bound_count,
    split_images,
    topology_summary,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0686_locked_path_qumond_raw_topology.json"
MODEL = "locked_path_qumond_P0685"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0685_field_or_P0686_raw_lens_score":
        raise RuntimeError("P0686 protocol is not frozen before its parent result")
    parent_path = ROOT / protocol["anticipated_parent"]
    parent = read_json(parent_path)
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0685 parent did not pass its frozen field gates")

    raw = read_json(ROOT / protocol["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    field_path = ROOT / protocol["field_input"]
    with np.load(field_path) as data:
        axis_arcsec = data["axis_kpc"].astype(float) / scale
        grid = PhysicalDeflectionGrid(
            axis_arcsec,
            data["candidate_alpha_x_physical_arcsec"].astype(float),
            data["candidate_alpha_y_physical_arcsec"].astype(float),
        )
    lens = AbsoluteGridLens(raw, {MODEL: grid})
    fit = exact_fit(lens, MODEL, training, heldout, protocol, seed_offset=0)
    prediction = pd.concat(
        [fit["training_prediction"], fit["heldout_prediction"]],
        ignore_index=True,
    )
    fit_score = {
        "model": MODEL,
        "training_RMS_arcsec": fit["training_score"]["exact_radial_RMS_arcsec"],
        "training_roots_converged": fit["training_score"]["converged_roots"],
        "heldout_RMS_arcsec": fit["heldout_score"]["exact_radial_RMS_arcsec"],
        "heldout_roots_converged": fit["heldout_score"]["converged_roots"],
        "optimizer_cost": fit["optimizer_cost"],
        "nuisance_parameters_near_bound": near_bound_count(fit["parameters"]),
    }
    parameter_rows = [
        {
            "model": MODEL,
            "parameter": label,
            "value": float(value),
            "lower": float(lower),
            "upper": float(upper),
        }
        for label, value, lower, upper in zip(
            AbsoluteGridLens.labels,
            fit["parameters"],
            AbsoluteGridLens.lower,
            AbsoluteGridLens.upper,
            strict=True,
        )
    ]

    settings = protocol["global_topology"]
    roots, assignments, families, critical_maps = global_topology(
        lens,
        MODEL,
        fit,
        images,
        settings,
    )
    topology = topology_summary(families)
    comparator = read_json(ROOT / protocol["compact_halo_comparator_report"])
    compact_halo = float(
        comparator["model_scores"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"]
    )
    heldout_rms = float(fit_score["heldout_RMS_arcsec"])
    halo_ratio = heldout_rms / compact_halo if np.isfinite(heldout_rms) else float("inf")
    accounting = protocol["models"]
    gates = protocol["predeclared_progression_gates"]
    gate_results = {
        "P0685_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0685_all_progression_gates_pass"]),
        "training_roots": int(fit_score["training_roots_converged"])
        == int(gates["candidate_training_roots_converged"]),
        "heldout_roots": int(fit_score["heldout_roots_converged"])
        == int(gates["candidate_heldout_roots_converged"]),
        "training_absolute_RMS": float(fit_score["training_RMS_arcsec"])
        <= float(gates["candidate_training_RMS_arcsec_max"]),
        "heldout_absolute_RMS": heldout_rms <= float(gates["candidate_heldout_RMS_arcsec_max"]),
        "compact_halo_comparison": halo_ratio
        <= float(gates["candidate_to_compact_halo_heldout_RMS_ratio_max"]),
        "no_missing_multiplicity": topology["missing_multiplicity_families"]
        <= int(gates["candidate_missing_multiplicity_families_max"]),
        "observable_surplus": topology["potentially_observable_surplus_families"]
        <= int(gates["candidate_potentially_observable_surplus_families_max"]),
        "acceptable_multiplicity": topology["exact_or_demagnified_only_families"]
        >= int(gates["candidate_exact_or_demagnified_only_families_min"]),
        "parity_diversity": topology["parity_diverse_families"]
        >= int(gates["candidate_parity_diverse_families_min"]),
        "critical_curves": topology["critical_curve_present_families"]
        >= int(gates["candidate_critical_curve_present_families_min"]),
        "nuisance_bounds": int(fit_score["nuisance_parameters_near_bound"])
        <= int(gates["candidate_nuisance_parameters_near_bound_max"]),
        "no_fitted_gravity": int(accounting["gravity_parameters_fit_to_RXJ2129"])
        == int(gates["gravity_parameters_fit_to_RXJ2129"]),
        "no_fitted_photon_amplitude": int(accounting["photon_amplitudes_fit_to_RXJ2129"])
        == int(gates["photon_amplitudes_fit_to_RXJ2129"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([fit_score]).to_csv(output / "fit_scores.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "nuisance_parameters.csv", index=False)
    prediction.to_csv(output / "exact_predictions.csv", index=False)
    roots.to_csv(output / "global_roots.csv", index=False)
    assignments.to_csv(output / "global_assignments.csv", index=False)
    families.to_csv(output / "family_topology.csv", index=False)
    report = {
        "report_version": "P0686-LOCKED-PATH-QUMOND-RAW-TOPOLOGY-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_resolution_and_baryon_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "field_sha256": sha256(field_path),
        "coverage": {
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "source_families": int(images.source_family.nunique()),
            "ordinary_nuisance_parameters": 4,
            "profiled_source_coordinates": 14,
            "gravity_parameters": int(accounting["gravity_parameters_fit_to_RXJ2129"]),
            "photon_amplitudes": int(accounting["photon_amplitudes_fit_to_RXJ2129"]),
        },
        "fit_score": fit_score,
        "comparisons": {
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "candidate_to_compact_halo_heldout_RMS_ratio": halo_ratio,
            "published_multi_halo_reference_RMS_arcsec": 0.29,
        },
        "topology": topology,
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(
        ["training", "spent heldout"],
        [fit_score["training_roots_converged"], fit_score["heldout_roots_converged"]],
    )
    axes[0].scatter(
        ["training", "spent heldout"],
        [len(training), len(heldout)],
        color="black",
        label="required",
    )
    axes[0].set(ylabel="exact roots recovered", title="Root convergence")
    axes[0].legend()
    class_counts = families.multiplicity_classification.value_counts()
    axes[1].bar(class_counts.index, class_counts.values)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="families", title="Candidate multiplicity")
    determinant = critical_maps[1]
    half = float(settings["critical_grid_half_width_arcsec"])
    image = axes[2].imshow(
        np.sign(determinant).T,
        origin="lower",
        extent=[-half, half, -half, half],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[2].set(title="Family 1 Jacobian sign", xlabel="x (arcsec)", ylabel="y (arcsec)")
    figure.colorbar(image, ax=axes[2], shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / "p0686_locked_path_qumond_raw_topology.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0686 locked path-QUMOND raw topology

- Status: **{"PASS" if all_pass else "FAIL"}**.
- Training / spent-heldout exact-root RMS: **{fit_score["training_RMS_arcsec"]:.4g} / {heldout_rms:.4g} arcsec**.
- Candidate/compact-halo heldout RMS ratio: **{halo_ratio:.4g}**.
- Missing / exact-or-demagnified / observable-surplus families: **{topology["missing_multiplicity_families"]} / {topology["exact_or_demagnified_only_families"]} / {topology["potentially_observable_surplus_families"]}**.
- Parity-diverse / critical-curve families: **{topology["parity_diverse_families"]} / {topology["critical_curve_present_families"]}** of 7.
- Failed frozen gates: **{", ".join(failed) if failed else "none"}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
