#!/usr/bin/env python3
"""Run the frozen P0702 parameter-free mass-planarity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0635_map_geometry_sensitivity import build_density
from voidscreen.mass_planarity import baryonic_mass_planarity, planarity_blended_coherence

DEFAULT_CONFIG = ROOT / "configs" / "p0702_mass_planarity_math_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian(axis: np.ndarray, widths: list[float]) -> np.ndarray:
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.exp(-0.5 * sum((q / w) ** 2 for q, w in zip((x, y, z), widths, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_P0702_formula_field_or_outcome_score":
        raise RuntimeError("P0702 protocol is not frozen")

    axis = np.linspace(-6.0, 6.0, 49)
    spacing = float(axis[1] - axis[0])
    cases = {
        name: baryonic_mass_planarity(gaussian(axis, widths), spacing)
        for name, widths in (
            ("sheet", protocol["synthetic_cases"]["sheet_widths"]),
            ("filament", protocol["synthetic_cases"]["filament_widths"]),
            ("ball", protocol["synthetic_cases"]["ball_widths"]),
        )
    }
    rotated = baryonic_mass_planarity(
        np.transpose(gaussian(axis, protocol["synthetic_cases"]["sheet_widths"]), (1, 2, 0)),
        spacing,
    )
    shifted = baryonic_mass_planarity(
        np.roll(gaussian(axis, protocol["synthetic_cases"]["sheet_widths"]), (2, -1, 1), axis=(0, 1, 2)),
        spacing,
    )
    controller = np.linspace(0.0, 1.0, 101)
    endpoint_zero = planarity_blended_coherence(controller, 0.0)
    endpoint_one = planarity_blended_coherence(controller, 1.0)

    with np.load(ROOT / "results/p0635_ddo154_map_commissioning/baryonic_maps.npz") as data:
        galaxy_axis = data["axis_kpc"].astype(float)
        galaxy_density = build_density(
            data["gas_surface_density_solar_kpc2"],
            data["stellar_surface_density_solar_kpc2"],
            galaxy_axis,
            0.15,
            0.30,
        )
    galaxy = baryonic_mass_planarity(galaxy_density, float(galaxy_axis[1] - galaxy_axis[0]))
    with np.load(ROOT / "results/p0670_spent_rxj2129_absolute_3d_map_build/rxj2129_absolute_baryons_3d.npz") as data:
        cluster_axis = data["axis_kpc"].astype(float)
        cluster_density = data["stellar_volume_density_kg_m3"] + data["gas_volume_density_kg_m3"]
    cluster = baryonic_mass_planarity(cluster_density, float(cluster_axis[1] - cluster_axis[0]))

    gates = protocol["predeclared_gates"]
    metrics = {
        "synthetic_planarity": {name: value.planarity for name, value in cases.items()},
        "axis_permutation_difference": abs(cases["sheet"].planarity - rotated.planarity),
        "translation_difference": abs(cases["sheet"].planarity - shifted.planarity),
        "zero_planarity_endpoint_error": float(np.max(np.abs(endpoint_zero - controller))),
        "unit_planarity_endpoint_error": float(np.max(np.abs(endpoint_one - 1.0))),
        "spent_DDO154": {
            "eigenvalues_kpc2": galaxy.eigenvalues.tolist(),
            "planarity": galaxy.planarity,
        },
        "spent_RXJ2129": {
            "eigenvalues_kpc2": cluster.eigenvalues.tolist(),
            "planarity": cluster.planarity,
        },
    }
    all_values = [case.planarity for case in cases.values()] + [galaxy.planarity, cluster.planarity]
    gate_results = {
        "planarity_range": min(all_values) >= gates["planarity_min"] and max(all_values) <= gates["planarity_max"],
        "sheet_identified": cases["sheet"].planarity >= gates["sheet_planarity_min"],
        "filament_rejected": cases["filament"].planarity <= gates["filament_planarity_max"],
        "ball_rejected": cases["ball"].planarity <= gates["ball_planarity_max"],
        "axis_permutation_invariance": metrics["axis_permutation_difference"] <= gates["axis_permutation_difference_max"],
        "translation_invariance": metrics["translation_difference"] <= gates["translation_difference_max"],
        "zero_endpoint": metrics["zero_planarity_endpoint_error"] <= gates["zero_planarity_endpoint_error_max"],
        "unit_endpoint": metrics["unit_planarity_endpoint_error"] <= gates["unit_planarity_endpoint_error_max"],
        "effective_coherence_bounds": bool(np.min(endpoint_zero) >= 0.0 and np.max(endpoint_one) <= 1.0),
        "no_new_constants": protocol["equation"]["new_universal_constants"] == gates["new_universal_constants"],
        "no_per_object_gravity": protocol["equation"]["per_object_gravity_parameters"] == gates["per_object_gravity_parameters"],
    }
    passed = all(gate_results.values())
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0702-MASS-PLANARITY-MATH-AUDIT-RESULTS-1.0.0",
        "status": "pass" if passed else "fail",
        "all_math_gates_pass": passed,
        "candidate_advanced_to_spent_joint_screen": passed,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/mass_planarity.py"),
        "metrics": metrics,
        "gate_results": gate_results,
        "failed_gates": [name for name, value in gate_results.items() if not value],
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = f"""# P0702 mass-planarity math audit

- Status: **{'PASS' if passed else 'FAIL'}**.
- Synthetic sheet / filament / ball planarity: **{cases['sheet'].planarity:.6g} / {cases['filament'].planarity:.6g} / {cases['ball'].planarity:.6g}**.
- Axis-permutation / translation differences: **{metrics['axis_permutation_difference']:.3g} / {metrics['translation_difference']:.3g}**.
- Spent DDO154 / RX J2129 planarity diagnostics: **{galaxy.planarity:.6g} / {cluster.planarity:.6g}**.
- New universal constants / per-object gravity settings: **0 / 0**.
- Advanced to a frozen spent joint screen: **{'yes' if passed else 'no'}**.
- Sealed P0633/P0640 outcomes opened: **no**.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
