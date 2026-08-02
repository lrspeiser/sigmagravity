#!/usr/bin/env python3
"""Final direct-step convergence audit against the exact NIE Hessian."""

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

DEFAULT_CONFIG = ROOT / "configs" / "p0681_compact_halo_hessian_richardson.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0681_step_metric":
        raise RuntimeError("P0681 protocol is not frozen")
    parent = read_json(ROOT / protocol["failure_parent"])
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

    rows = []
    all_values = [f_xx, f_xy, f_yx, f_yy, exact_kappa, exact_determinant]
    for step in protocol["direct_steps_arcsec"]:
        kappa, curl, determinant = direct_derivatives(
            model,
            kwargs,
            point_x,
            point_y,
            float(step),
        )
        all_values.extend([kappa, curl, determinant])
        rows.append(
            {
                "step_arcsec": float(step),
                "kappa_exact_relative_RMS": relative_rms(kappa, exact_kappa),
                "determinant_exact_relative_RMS": relative_rms(
                    determinant,
                    exact_determinant,
                ),
                "normalized_curl_RMS": rms(curl)
                / max(rms(2.0 * kappa), np.finfo(float).tiny),
                "negative_jacobian_points": int(np.sum(determinant < 0.0)),
                "all_finite": bool(
                    np.all(np.isfinite(kappa))
                    and np.all(np.isfinite(curl))
                    and np.all(np.isfinite(determinant))
                ),
            }
        )
    table = pd.DataFrame(rows)
    steps = table.step_arcsec.to_numpy(float)
    kappa_errors = table.kappa_exact_relative_RMS.to_numpy(float)
    determinant_errors = table.determinant_exact_relative_RMS.to_numpy(float)
    curl_values = table.normalized_curl_RMS.to_numpy(float)
    finite = bool(all(np.all(np.isfinite(values)) for values in all_values))

    gates = protocol["predeclared_integrity_gates"]
    parent_failed = {
        name for name, passed in parent["gate_results"].items() if not passed
    }
    gate_results = {
        "P0680_failed": parent["status"] == gates["P0680_status"],
        "P0680_exact_failures": parent_failed == set(gates["P0680_failed_gates_exact"]),
        "P0680_exact_symmetry": float(
            parent["metrics"]["exact_hessian_normalized_curl_RMS"]
        )
        <= float(gates["P0680_exact_hessian_normalized_curl_RMS_max"]),
        "P0680_negative_points": int(
            parent["metrics"]["exact_negative_jacobian_points"]
        )
        == int(gates["P0680_exact_negative_jacobian_points"]),
        "point_count": len(point_x) == int(gates["evaluation_points"]),
        "step_count": len(table) == int(gates["direct_step_count"]),
        "steps_decrease": bool(np.all(np.diff(steps) < 0.0))
        is bool(gates["direct_steps_strictly_decreasing"]),
        "kappa_error_decreases": bool(np.all(np.diff(kappa_errors) < 0.0))
        is bool(gates["kappa_exact_relative_RMS_strictly_decreasing"]),
        "determinant_error_decreases": bool(
            np.all(np.diff(determinant_errors) < 0.0)
        )
        is bool(gates["determinant_exact_relative_RMS_strictly_decreasing"]),
        "smallest_kappa_agreement": kappa_errors[-1]
        <= float(gates["smallest_step_kappa_exact_relative_RMS_max"]),
        "smallest_determinant_agreement": determinant_errors[-1]
        <= float(gates["smallest_step_determinant_exact_relative_RMS_max"]),
        "smallest_curl": curl_values[-1]
        <= float(gates["smallest_step_normalized_curl_RMS_max"]),
        "negative_point_stability": table.negative_jacobian_points.eq(
            int(gates["negative_jacobian_point_count_stable_at_all_steps"])
        ).all(),
        "finite": finite is bool(gates["all_derivatives_finite"]),
        "no_candidate_fit": not bool(gates["new_candidate_formula_fit"]),
        "no_raw_root_score": not bool(gates["new_raw_image_root_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    kappa_order = float(
        np.log(kappa_errors[0] / kappa_errors[-1]) / np.log(steps[0] / steps[-1])
    )
    determinant_order = float(
        np.log(determinant_errors[0] / determinant_errors[-1])
        / np.log(steps[0] / steps[-1])
    )
    metrics = {
        "evaluation_points": len(point_x),
        "exact_hessian_normalized_curl_RMS": exact_normalized_curl,
        "direct_step_kappa_exact_relative_RMS": {
            f"{row.step_arcsec:g}": float(row.kappa_exact_relative_RMS)
            for row in table.itertuples()
        },
        "direct_step_determinant_exact_relative_RMS": {
            f"{row.step_arcsec:g}": float(row.determinant_exact_relative_RMS)
            for row in table.itertuples()
        },
        "direct_step_normalized_curl_RMS": {
            f"{row.step_arcsec:g}": float(row.normalized_curl_RMS)
            for row in table.itertuples()
        },
        "observed_kappa_convergence_order": kappa_order,
        "observed_determinant_convergence_order": determinant_order,
        "exact_negative_jacobian_points": int(np.sum(exact_determinant < 0.0)),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / protocol["outputs"]["step_table"]
    table.to_csv(table_path, index=False)
    report = {
        "report_version": "P0681-COMPACT-HALO-HESSIAN-RICHARDSON-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_integrity_gates_pass": all_pass,
        "P0678_decomposition_numerically_qualified": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "parameter_table_sha256": sha256(parameter_path),
        "step_table_sha256": sha256(table_path),
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
    axes[0].loglog(steps, kappa_errors, marker="o", label="kappa")
    axes[0].loglog(steps, determinant_errors, marker="s", label="determinant")
    axes[0].invert_xaxis()
    axes[0].axhline(1e-6, color="black", linestyle="--", label="agreement gate")
    axes[0].set(
        xlabel="central-difference step (arcsec)",
        ylabel="relative RMS vs exact Hessian",
        title="Exact-Hessian convergence",
    )
    axes[0].legend()
    axes[1].loglog(steps, curl_values, marker="o")
    axes[1].invert_xaxis()
    axes[1].axhline(2e-9, color="black", linestyle="--", label="curl gate")
    axes[1].set(
        xlabel="central-difference step (arcsec)",
        ylabel="normalized curl RMS",
        title="Curl convergence",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "p0681_compact_halo_hessian_richardson.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0681 compact-halo Hessian step convergence

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Kappa exact-relative RMS at 0.01 / 0.005 / 0.002 / 0.001 arcsec: **{' / '.join(f'{value:.3g}' for value in kappa_errors)}**.
- Determinant exact-relative RMS: **{' / '.join(f'{value:.3g}' for value in determinant_errors)}**.
- Normalized curl RMS: **{' / '.join(f'{value:.3g}' for value in curl_values)}**.
- Observed kappa / determinant convergence order: **{kappa_order:.3f} / {determinant_order:.3f}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- P0678 decomposition numerically qualified: **{'yes' if all_pass else 'no'}**.
- New formula/root score/sealed outcome: **no / no / no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
