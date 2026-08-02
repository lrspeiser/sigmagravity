#!/usr/bin/env python3
"""Test a scale-free baryonic multipole gate for 3D tensor activation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import sha256
from run_p0666_zero_slip_3d_photon_deflection import gaussian_3d

from voidscreen.metric_lensing_3d import constitutive_tensor_components_3d
from voidscreen.multipole_activation_3d import (
    baryonic_multipole_gate_3d,
    exact_multipole_gated_activation_3d,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0667_multipole_gated_3d_activation.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_sigma(result, total_density):
    return float(np.sum(result.sigma * total_density) / np.sum(total_density))


def relative(first, second):
    return abs(float(first) / max(abs(float(second)), np.finfo(float).tiny) - 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0667_score":
        raise RuntimeError("P0667 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    failed_parent = [name for name, passed in parent["gate_results"].items() if not passed]
    if failed_parent != ["radial_activation_null"]:
        raise RuntimeError("P0666 diagnostic state changed")

    fixture = protocol["synthetic_fixture"]
    cells = int(fixture["grid_cells"])
    half_width = float(fixture["half_width"])
    axis = np.linspace(-half_width, half_width, cells)
    spacing = float(axis[1] - axis[0])
    star_mass = float(fixture["stellar_mass"])
    gas_mass = float(fixture["gas_mass"])
    star_sigma = float(fixture["stellar_gaussian_sigma"])
    gas_sigma = float(fixture["gas_gaussian_sigma"])
    offset = float(fixture["offset"])
    stars = gaussian_3d(axis, (0.0, 0.0, 0.0), star_sigma, star_mass, spacing)
    radial_gas = gaussian_3d(axis, (0.0, 0.0, 0.0), gas_sigma, gas_mass, spacing)
    offset_gas = gaussian_3d(axis, (offset, 0.0, 0.0), gas_sigma, gas_mass, spacing)
    activation_kwargs = {
        "gravitational_constant": 1.0,
        "a0": float(fixture["dimensionless_a0"]),
        "coherence_length": float(fixture["dimensionless_coherence_length"]),
        "coherence_power": 2.0,
    }
    radial = exact_multipole_gated_activation_3d(
        stars,
        radial_gas,
        spacing,
        **activation_kwargs,
    )
    displaced = exact_multipole_gated_activation_3d(
        stars,
        offset_gas,
        spacing,
        **activation_kwargs,
    )
    radial_weighted = weighted_sigma(radial, stars + radial_gas)
    offset_weighted = weighted_sigma(displaced, stars + offset_gas)
    parent_offset = float(parent["metrics"]["offset_sigma_mass_weighted_mean"])
    retained_fraction = offset_weighted / max(parent_offset, np.finfo(float).tiny)

    rotated = exact_multipole_gated_activation_3d(
        np.swapaxes(stars, 0, 1),
        np.swapaxes(offset_gas, 0, 1),
        spacing,
        **activation_kwargs,
    )
    rotation_error = relative(
        weighted_sigma(rotated, np.swapaxes(stars + offset_gas, 0, 1)),
        offset_weighted,
    )
    exchanged = exact_multipole_gated_activation_3d(
        offset_gas,
        stars,
        spacing,
        **activation_kwargs,
    )
    exchange_error = relative(weighted_sigma(exchanged, stars + offset_gas), offset_weighted)

    translation = tuple(int(value) for value in fixture["translation_cells"])
    translated_stars = ndimage.shift(
        stars,
        translation,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    translated_gas = ndimage.shift(
        offset_gas,
        translation,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    translated = exact_multipole_gated_activation_3d(
        translated_stars,
        translated_gas,
        spacing,
        **activation_kwargs,
    )
    translation_error = relative(
        weighted_sigma(translated, translated_stars + translated_gas),
        offset_weighted,
    )
    scaled_gate = baryonic_multipole_gate_3d(
        stars,
        offset_gas,
        float(fixture["scale_factor"]) * spacing,
    )
    scale_error = relative(scaled_gate.gate, displaced.multipole.gate)

    direct_tensor = constitutive_tensor_components_3d(
        displaced.sigma,
        displaced.local.transport_direction,
    )
    reversed_tensor = constitutive_tensor_components_3d(
        displaced.sigma,
        tuple(-component for component in displaced.local.transport_direction),
    )
    reversal_numerator = np.sqrt(
        sum(
            float(np.mean((left - right) ** 2))
            for left, right in zip(direct_tensor, reversed_tensor, strict=True)
        )
    )
    reversal_denominator = np.sqrt(
        sum(float(np.mean(component**2)) for component in direct_tensor)
    )
    reversal_error = float(
        reversal_numerator / max(reversal_denominator, np.finfo(float).tiny)
    )
    sigma_bounded = bool(
        np.all(np.isfinite(displaced.sigma))
        and np.min(displaced.sigma) >= 0.0
        and np.max(displaced.sigma) <= 1.0
    )
    minimum_eigenvalue = float(np.min(displaced.minimum_eigenvalue_proxy))
    point_mass_gates = {
        key: value
        for key, value in parent["gate_results"].items()
        if key
        in {
            "point_mass_median",
            "point_mass_p95",
            "linear_mass_scaling",
            "rotation_covariance",
            "curl_free_deflection",
            "surface_to_volume_mass",
            "zero_slip",
            "no_fitted_photon_amplitude",
        }
    }
    gates = protocol["predeclared_progression_gates"]
    candidate = protocol["candidate"]
    gate_results = {
        "P0666_diagnostic_state": failed_parent == ["radial_activation_null"],
        "radial_multipole_null": radial.multipole.gate
        <= gates["radial_cocentered_multipole_gate_max"],
        "radial_activation_null": radial_weighted
        <= gates["radial_cocentered_3D_sigma_mass_weighted_mean_max"],
        "offset_multipole_present": displaced.multipole.gate
        >= gates["offset_multipole_gate_min"],
        "offset_activation_present": offset_weighted
        >= gates["offset_3D_sigma_mass_weighted_mean_min"],
        "offset_signal_retained": retained_fraction
        >= gates["offset_signal_retained_fraction_vs_P0666_min"],
        "rotation_covariance": rotation_error
        <= gates["rotation_covariance_relative_error_max"],
        "component_exchange": exchange_error
        <= gates["component_exchange_relative_error_max"],
        "translation_covariance": translation_error
        <= gates["translation_covariance_relative_error_max"],
        "scale_covariance": scale_error
        <= gates["scale_covariance_multipole_gate_relative_error_max"],
        "bounded_sigma": sigma_bounded
        is bool(gates["sigma_finite_and_in_closed_unit_interval"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "direction_reversal": reversal_error
        <= gates["direction_reversal_tensor_relative_error_max"],
        "P0666_photon_gates": bool(all(point_mass_gates.values()))
        is bool(gates["P0666_point_mass_photon_gates_remain_passed"]),
        "no_new_constants": int(candidate["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(candidate["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "spent_lensing_untouched": not bool(gates["spent_lensing_outcomes_opened"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "radial_multipole_gate": radial.multipole.gate,
        "radial_sigma_mass_weighted_mean": radial_weighted,
        "offset_multipole_gate": displaced.multipole.gate,
        "offset_dipole_squared": displaced.multipole.dipole_squared,
        "offset_quadrupole_squared": displaced.multipole.quadrupole_squared,
        "offset_sigma_mass_weighted_mean": offset_weighted,
        "offset_signal_retained_fraction_vs_P0666": retained_fraction,
        "rotation_covariance_relative_error": rotation_error,
        "component_exchange_relative_error": exchange_error,
        "translation_covariance_relative_error": translation_error,
        "scale_covariance_multipole_gate_relative_error": scale_error,
        "minimum_constitutive_eigenvalue_proxy": minimum_eigenvalue,
        "direction_reversal_tensor_relative_error": reversal_error,
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0667-MULTIPOLE-GATED-3D-ACTIVATION-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_registered_map_audit": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(ROOT / "src/voidscreen/multipole_activation_3d.py"),
        "metrics": metrics,
        "gate_results": gate_results,
        "spent_RXJ2129_lensing_outcomes_opened": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(
        ["radial", "offset"],
        [radial.multipole.gate, displaced.multipole.gate],
        color=["#3274a1", "#d95f02"],
    )
    axes[0].set_yscale("symlog", linthresh=1e-15)
    axes[0].set_ylabel("multipole gate")
    axes[0].set_title("Exact radial structural null")
    axes[1].bar(
        ["P0666 local", "P0667 gated"],
        [parent_offset, offset_weighted],
        color=["#777777", "#55a868"],
    )
    axes[1].set_ylabel("offset mass-weighted sigma")
    axes[1].set_title("Retained displaced signal")
    figure.suptitle("P0667 multipole-gated 3D activation")
    figure.tight_layout()
    figure.savefig(output / "p0667_multipole_gated_activation.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0667 multipole-gated 3D activation

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Radial multipole gate / weighted sigma: **{radial.multipole.gate:.3e} / {radial_weighted:.3e}**.
- Offset multipole gate / weighted sigma: **{displaced.multipole.gate:.6g} / {offset_weighted:.6g}**.
- Offset signal retained from P0666: **{retained_fraction:.3%}**.
- Rotation/exchange/translation errors: **{rotation_error:.3e} / {exchange_error:.3e} / {translation_error:.3e}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Spent and sealed lensing outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
