#!/usr/bin/env python3
"""Qualify the spent compact-halo derivatives with its exact Hessian."""

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
from run_p0679_compact_halo_derivative_convergence import (
    direct_derivatives,
    halo_model,
    relative_rms,
    rms,
    selected_parameters,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0680_compact_halo_exact_hessian.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0680_hessian_metric":
        raise RuntimeError("P0680 protocol is not frozen")
    failure_parent = read_json(ROOT / protocol["failure_parent"])
    decomposition_parent = read_json(ROOT / protocol["decomposition_parent"])
    raw = read_json(ROOT / protocol["raw_lensing_protocol"])
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    parameter_path = ROOT / protocol["parameter_table"]
    parameter_rows, parameters = selected_parameters(
        parameter_path,
        protocol["parameter_selection"],
    )
    model, kwargs = halo_model(parameters)

    points = protocol["evaluation_points"]
    axis_kpc = np.linspace(
        float(points["axis_minimum_kpc"]),
        float(points["axis_maximum_kpc"]),
        int(points["grid_cells"]),
    )
    axis_arcsec = axis_kpc / scale
    x, y = np.meshgrid(axis_arcsec, axis_arcsec, indexing="ij")
    radius_kpc = np.hypot(x, y) * scale
    lower, upper = (float(value) for value in points["strong_lens_radius_kpc"])
    mask = (radius_kpc >= lower) & (radius_kpc <= upper)
    point_x = x[mask]
    point_y = y[mask]

    f_xx, f_xy, f_yx, f_yy = (
        np.asarray(values, dtype=float)
        for values in model.hessian(point_x, point_y, kwargs, diff=None)
    )
    exact_kappa = 0.5 * (f_xx + f_yy)
    exact_curl = f_yx - f_xy
    exact_determinant = (1.0 - f_xx) * (1.0 - f_yy) - f_xy * f_yx
    exact_normalized_curl = rms(exact_curl) / max(
        rms(2.0 * exact_kappa),
        np.finfo(float).tiny,
    )
    direct_step = float(protocol["hessian"]["comparison_direct_central_difference_step_arcsec"])
    direct_kappa, direct_curl, direct_determinant = direct_derivatives(
        model,
        kwargs,
        point_x,
        point_y,
        direct_step,
    )
    kappa_difference = relative_rms(exact_kappa, direct_kappa)
    determinant_difference = relative_rms(exact_determinant, direct_determinant)
    finite = bool(
        all(
            np.all(np.isfinite(values))
            for values in (
                f_xx,
                f_xy,
                f_yx,
                f_yy,
                exact_kappa,
                exact_curl,
                exact_determinant,
                direct_kappa,
                direct_curl,
                direct_determinant,
            )
        )
    )

    gates = protocol["predeclared_integrity_gates"]
    p0679_failed = [
        name for name, passed in failure_parent["gate_results"].items() if not passed
    ]
    p0678_failed = [
        name for name, passed in decomposition_parent["gate_results"].items() if not passed
    ]
    p0679_smallest_curl = float(
        failure_parent["metrics"]["direct_step_normalized_curl_RMS"]["0.01"]
    )
    gate_results = {
        "P0679_failed": failure_parent["status"] == gates["P0679_status"],
        "P0679_exact_failure": p0679_failed == [gates["P0679_failed_gate_exactly"]],
        "P0679_smallest_curl": p0679_smallest_curl
        <= float(gates["P0679_smallest_direct_step_normalized_curl_RMS_max"]),
        "P0678_failed": decomposition_parent["status"] == gates["P0678_status"],
        "P0678_exact_failure": p0678_failed == [gates["P0678_failed_gate_exactly"]],
        "point_count": len(point_x) == int(gates["evaluation_points"]),
        "finite": finite is bool(gates["exact_hessian_all_finite"]),
        "exact_symmetry": exact_normalized_curl
        <= float(gates["exact_hessian_normalized_curl_RMS_max"]),
        "kappa_agreement": kappa_difference
        <= float(gates["exact_vs_direct_0p01_kappa_relative_RMS_difference_max"]),
        "determinant_agreement": determinant_difference
        <= float(
            gates[
                "exact_vs_direct_0p01_jacobian_determinant_relative_RMS_difference_max"
            ]
        ),
        "no_candidate_fit": not bool(gates["new_candidate_formula_fit"]),
        "no_raw_root_score": not bool(gates["new_raw_image_root_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "evaluation_points": len(point_x),
        "exact_hessian_normalized_curl_RMS": exact_normalized_curl,
        "direct_0p01_normalized_curl_RMS": rms(direct_curl)
        / max(rms(2.0 * direct_kappa), np.finfo(float).tiny),
        "exact_vs_direct_0p01_kappa_relative_RMS_difference": kappa_difference,
        "exact_vs_direct_0p01_jacobian_determinant_relative_RMS_difference": determinant_difference,
        "exact_negative_jacobian_points": int(np.sum(exact_determinant < 0.0)),
        "direct_negative_jacobian_points": int(np.sum(direct_determinant < 0.0)),
        "exact_kappa_RMS": rms(exact_kappa),
        "exact_jacobian_determinant_RMS": rms(exact_determinant),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "x_arcsec": point_x,
            "y_arcsec": point_y,
            "radius_kpc": np.hypot(point_x, point_y) * scale,
            "f_xx": f_xx,
            "f_xy": f_xy,
            "f_yx": f_yx,
            "f_yy": f_yy,
            "exact_kappa": exact_kappa,
            "exact_curl": exact_curl,
            "exact_jacobian_determinant": exact_determinant,
            "direct_0p01_kappa": direct_kappa,
            "direct_0p01_curl": direct_curl,
            "direct_0p01_jacobian_determinant": direct_determinant,
        }
    )
    point_path = output / protocol["outputs"]["point_table"]
    table.to_csv(point_path, index=False)
    report = {
        "report_version": "P0680-COMPACT-HALO-EXACT-HESSIAN-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_integrity_gates_pass": all_pass,
        "P0678_decomposition_numerically_qualified": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "parameter_table_sha256": sha256(parameter_path),
        "point_table_sha256": sha256(point_path),
        "parameter_rows": len(parameter_rows),
        "metrics": metrics,
        "gate_results": gate_results,
        "new_candidate_formula_fit": False,
        "new_raw_image_root_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(direct_kappa, exact_kappa, s=14, alpha=0.7)
    limits = [float(min(np.min(direct_kappa), np.min(exact_kappa))), float(max(np.max(direct_kappa), np.max(exact_kappa)))]
    axes[0].plot(limits, limits, color="black", linestyle="--")
    axes[0].set(
        xlabel="direct 0.01 arcsec kappa",
        ylabel="exact-Hessian kappa",
        title="Convergence agreement",
    )
    axes[1].scatter(direct_determinant, exact_determinant, s=14, alpha=0.7)
    limits = [
        float(min(np.min(direct_determinant), np.min(exact_determinant))),
        float(max(np.max(direct_determinant), np.max(exact_determinant))),
    ]
    axes[1].plot(limits, limits, color="black", linestyle="--")
    axes[1].set(
        xlabel="direct 0.01 arcsec determinant",
        ylabel="exact-Hessian determinant",
        title="Jacobian agreement",
    )
    figure.tight_layout()
    figure.savefig(output / "p0680_compact_halo_exact_hessian.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0680 compact-halo exact Hessian

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Exact-Hessian / direct-0.01 normalized curl: **{exact_normalized_curl:.3g} / {metrics['direct_0p01_normalized_curl_RMS']:.3g}**.
- Exact-vs-direct kappa relative RMS difference: **{kappa_difference:.3g}**.
- Exact-vs-direct Jacobian-determinant relative RMS difference: **{determinant_difference:.3g}**.
- Exact/direct negative-Jacobian points: **{metrics['exact_negative_jacobian_points']} / {metrics['direct_negative_jacobian_points']}** of {len(point_x)}.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- P0678 decomposition numerically qualified: **{'yes' if all_pass else 'no'}**.
- New formula/root score/sealed outcome: **no / no / no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
