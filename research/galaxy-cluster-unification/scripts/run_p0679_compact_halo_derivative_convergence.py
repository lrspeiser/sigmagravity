#!/usr/bin/env python3
"""Audit compact-halo derivatives across sampled grids and direct steps."""

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
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import param_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.required_field_decomposition import (
    convergence_and_jacobian_determinant,
    sign_change_cells,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0679_compact_halo_derivative_convergence.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rms(values) -> float:
    data = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(data * data)))


def relative_rms(first, second) -> float:
    return rms(np.asarray(first) - np.asarray(second)) / max(
        rms(second),
        np.finfo(float).tiny,
    )


def selected_parameters(path: Path, selection: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    table = pd.read_csv(path)
    rows = table[
        table.stage.eq(selection["stage"]) & table.model.eq(selection["model"])
    ].copy()
    return rows, {str(row.parameter): float(row.value) for row in rows.itertuples()}


def halo_model(parameters: dict[str, float]):
    phi = parameters["position_angle_phi_radian"]
    q = parameters["axis_ratio_q"]
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    model = LensModel(lens_model_list=["NIE"])
    kwargs = [
        {
            "theta_E": parameters["theta_E_ref_arcsec"],
            "e1": e1,
            "e2": e2,
            "s_scale": 10.0 ** parameters["log10_core_arcsec"],
            "center_x": parameters["center_x_arcsec"],
            "center_y": parameters["center_y_arcsec"],
        }
    ]
    return model, kwargs


def alpha(model, kwargs, x, y):
    x_values, y_values = model.alpha(
        np.asarray(x, dtype=float).ravel(),
        np.asarray(y, dtype=float).ravel(),
        kwargs,
    )
    shape = np.asarray(x).shape
    return np.asarray(x_values).reshape(shape), np.asarray(y_values).reshape(shape)


def normalized_curl(curl, convergence, mask) -> float:
    selected = np.asarray(mask, dtype=bool)
    return rms(np.asarray(curl)[selected]) / max(
        rms(2.0 * np.asarray(convergence)[selected]),
        np.finfo(float).tiny,
    )


def direct_derivatives(model, kwargs, x, y, step):
    ax_xp, ay_xp = alpha(model, kwargs, x + step, y)
    ax_xm, ay_xm = alpha(model, kwargs, x - step, y)
    ax_yp, ay_yp = alpha(model, kwargs, x, y + step)
    ax_ym, ay_ym = alpha(model, kwargs, x, y - step)
    dax_dx = (ax_xp - ax_xm) / (2.0 * step)
    day_dx = (ay_xp - ay_xm) / (2.0 * step)
    dax_dy = (ax_yp - ax_ym) / (2.0 * step)
    day_dy = (ay_yp - ay_ym) / (2.0 * step)
    convergence = 0.5 * (dax_dx + day_dy)
    curl = day_dx - dax_dy
    determinant = (1.0 - dax_dx) * (1.0 - day_dy) - dax_dy * day_dx
    return convergence, curl, determinant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0679_derivative_metric":
        raise RuntimeError("P0679 protocol is not frozen")
    parent = read_json(ROOT / protocol["failure_parent"])
    parameter_path = ROOT / protocol["parameter_table"]
    parameter_rows, parameters = selected_parameters(
        parameter_path,
        protocol["parameter_selection"],
    )
    model, kwargs = halo_model(parameters)
    raw = read_json(ROOT / protocol["raw_lensing_protocol"])
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    domain = protocol["domain"]
    lower_kpc, upper_kpc = (float(value) for value in domain["strong_lens_radius_kpc"])
    axis_minimum = float(domain["axis_minimum_kpc"])
    axis_maximum = float(domain["axis_maximum_kpc"])

    grid_rows = []
    grid_fields = {}
    for cells in protocol["nested_grid_cells"]:
        axis_kpc = np.linspace(axis_minimum, axis_maximum, int(cells))
        axis_arcsec = axis_kpc / scale
        x, y = np.meshgrid(axis_arcsec, axis_arcsec, indexing="ij")
        radius_kpc = np.hypot(x, y) * scale
        mask = (radius_kpc >= lower_kpc) & (radius_kpc <= upper_kpc)
        alpha_x, alpha_y = alpha(model, kwargs, x, y)
        spacing = float(axis_arcsec[1] - axis_arcsec[0])
        convergence, curl, determinant = convergence_and_jacobian_determinant(
            alpha_x,
            alpha_y,
            spacing,
        )
        curl_metric = normalized_curl(curl, convergence, mask)
        grid_rows.append(
            {
                "cells": int(cells),
                "spacing_arcsec": spacing,
                "strong_lens_samples": int(np.sum(mask)),
                "normalized_curl_RMS": curl_metric,
                "mean_convergence": float(np.mean(convergence[mask])),
                "convergence_RMS": rms(convergence[mask]),
                "jacobian_determinant_RMS": rms(determinant[mask]),
                "critical_sign_change_cells": sign_change_cells(determinant),
                "all_finite": bool(
                    np.all(np.isfinite(convergence))
                    and np.all(np.isfinite(curl))
                    and np.all(np.isfinite(determinant))
                ),
            }
        )
        grid_fields[int(cells)] = (convergence, curl, determinant)
    grid_table = pd.DataFrame(grid_rows)

    axis_kpc = np.linspace(axis_minimum, axis_maximum, 33)
    axis_arcsec = axis_kpc / scale
    grid_x, grid_y = np.meshgrid(axis_arcsec, axis_arcsec, indexing="ij")
    radius_kpc = np.hypot(grid_x, grid_y) * scale
    point_mask = (radius_kpc >= lower_kpc) & (radius_kpc <= upper_kpc)
    point_x = grid_x[point_mask]
    point_y = grid_y[point_mask]
    step_rows = []
    direct_fields = {}
    for step in protocol["direct_central_difference_steps_arcsec"]:
        convergence, curl, determinant = direct_derivatives(
            model,
            kwargs,
            point_x,
            point_y,
            float(step),
        )
        direct_fields[float(step)] = (convergence, curl, determinant)
        step_rows.append(
            {
                "step_arcsec": float(step),
                "points": len(point_x),
                "normalized_curl_RMS": rms(curl)
                / max(rms(2.0 * convergence), np.finfo(float).tiny),
                "convergence_RMS": rms(convergence),
                "jacobian_determinant_RMS": rms(determinant),
                "all_finite": bool(
                    np.all(np.isfinite(convergence))
                    and np.all(np.isfinite(curl))
                    and np.all(np.isfinite(determinant))
                ),
            }
        )
    step_table = pd.DataFrame(step_rows)
    steps = [float(value) for value in protocol["direct_central_difference_steps_arcsec"]]
    penultimate = direct_fields[steps[-2]]
    smallest = direct_fields[steps[-1]]
    kappa_stability = relative_rms(penultimate[0], smallest[0])
    determinant_stability = relative_rms(penultimate[2], smallest[2])

    grid_curls = grid_table.normalized_curl_RMS.to_numpy(float)
    step_curls = step_table.normalized_curl_RMS.to_numpy(float)
    gates = protocol["predeclared_integrity_gates"]
    parent_failed = [name for name, passed in parent["gate_results"].items() if not passed]
    all_finite = bool(grid_table.all_finite.all() and step_table.all_finite.all())
    gate_results = {
        "P0678_failed": parent["status"] == gates["P0678_status"],
        "P0678_exact_failure": parent_failed == [gates["P0678_failed_gate_exactly"]],
        "grid_count": len(grid_table) == int(gates["nested_grid_count"]),
        "grid_cells": grid_table.cells.tolist() == gates["nested_grid_cells_exact"],
        "step_count": len(step_table) == int(gates["direct_step_count"]),
        "steps_decrease": bool(np.all(np.diff(steps) < 0.0))
        is bool(gates["direct_steps_strictly_decreasing"]),
        "grid_curl_decreases": bool(np.all(np.diff(grid_curls) < 0.0))
        is bool(gates["grid_normalized_curl_strictly_decreasing"]),
        "grid_improvement": grid_curls[0] / max(grid_curls[-1], np.finfo(float).tiny)
        >= float(gates["grid_257_curl_improvement_factor_vs_33_min"]),
        "direct_smallest_curl": step_curls[-1]
        <= float(gates["direct_smallest_step_normalized_curl_RMS_max"]),
        "direct_threshold_count": int(
            np.sum(step_curls <= gates["direct_smallest_step_normalized_curl_RMS_max"])
        )
        >= int(gates["direct_steps_below_original_P0678_threshold_min"]),
        "kappa_step_stability": kappa_stability
        <= float(gates["smallest_two_step_kappa_relative_RMS_difference_max"]),
        "determinant_step_stability": determinant_stability
        <= float(
            gates["smallest_two_step_jacobian_determinant_relative_RMS_difference_max"]
        ),
        "finite": all_finite is bool(gates["all_derivatives_finite"]),
        "no_candidate_fit": not bool(gates["new_candidate_formula_fit"]),
        "no_raw_root_score": not bool(gates["new_raw_image_root_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "P0678_coarse_normalized_curl_RMS": parent["metrics"][
            "compact_halo_normalized_curl_RMS"
        ],
        "nested_grid_normalized_curl_RMS": {
            str(int(row.cells)): float(row.normalized_curl_RMS)
            for row in grid_table.itertuples()
        },
        "grid_257_curl_improvement_factor_vs_33": float(
            grid_curls[0] / max(grid_curls[-1], np.finfo(float).tiny)
        ),
        "direct_step_normalized_curl_RMS": {
            f"{row.step_arcsec:g}": float(row.normalized_curl_RMS)
            for row in step_table.itertuples()
        },
        "direct_steps_below_original_P0678_threshold": int(
            np.sum(step_curls <= gates["direct_smallest_step_normalized_curl_RMS_max"])
        ),
        "smallest_two_step_kappa_relative_RMS_difference": kappa_stability,
        "smallest_two_step_jacobian_determinant_relative_RMS_difference": determinant_stability,
        "direct_smallest_step_critical_points_negative_determinant": int(
            np.sum(smallest[2] < 0.0)
        ),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    grid_table.to_csv(output / protocol["outputs"]["grid_table"], index=False)
    step_table.to_csv(output / protocol["outputs"]["step_table"], index=False)
    report = {
        "report_version": "P0679-COMPACT-HALO-DERIVATIVE-CONVERGENCE-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_integrity_gates_pass": all_pass,
        "P0678_decomposition_numerically_qualified": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "parameter_table_sha256": sha256(parameter_path),
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
    axes[0].loglog(
        grid_table.spacing_arcsec,
        grid_table.normalized_curl_RMS,
        marker="o",
    )
    axes[0].invert_xaxis()
    axes[0].axhline(1e-5, color="black", linestyle="--", label="P0678 gate")
    axes[0].set(
        xlabel="sampled-grid spacing (arcsec)",
        ylabel="normalized curl RMS",
        title="Sampled-grid convergence",
    )
    axes[0].legend()
    axes[1].loglog(
        step_table.step_arcsec,
        step_table.normalized_curl_RMS,
        marker="o",
    )
    axes[1].invert_xaxis()
    axes[1].axhline(1e-5, color="black", linestyle="--", label="P0678 gate")
    axes[1].set(
        xlabel="direct central-difference step (arcsec)",
        ylabel="normalized curl RMS",
        title="Direct analytic-field derivatives",
    )
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "p0679_compact_halo_derivative_convergence.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0679 compact-halo derivative convergence

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Nested-grid curl at 33 / 65 / 129 / 257 cells: **{' / '.join(f'{value:.3g}' for value in grid_curls)}**.
- 257-vs-33 curl improvement: **{metrics['grid_257_curl_improvement_factor_vs_33']:.3g}x**.
- Direct curl at 0.5 / 0.2 / 0.1 / 0.05 / 0.02 / 0.01 arcsec: **{' / '.join(f'{value:.3g}' for value in step_curls)}**.
- Smallest-two-step kappa / determinant relative RMS difference: **{kappa_stability:.3g} / {determinant_stability:.3g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- P0678 decomposition numerically qualified: **{'yes' if all_pass else 'no'}**.
- New formula/root score/sealed outcome: **no / no / no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
