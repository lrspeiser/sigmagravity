#!/usr/bin/env python3
"""Run the frozen P0703 spent planarity-blended joint screen."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p0699_spent_local_vector_coherence_joint_screen as parent_runner
from voidscreen.coherent_monopole import coherent_monopole_potential
from voidscreen.field_solvers import boundary_mask, laplacian, normalized_residual_rms, solve_poisson_dirichlet
from voidscreen.local_vector_coherence import (
    baryonic_vector_coherence,
    base_boundary_relative_mismatch,
    coherence_gated_source_potential,
    hybrid_coherence_routing_potential,
)
from voidscreen.mass_planarity import baryonic_mass_planarity, planarity_blended_coherence
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import (
    projected_baryonic_spectral_anisotropy,
    solve_source_conserving_baryonic_routing,
)

OVERLAY = ROOT / "configs" / "p0703_spent_planarity_blended_joint_screen.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_grid_rms(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    numerator = float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))
    denominator = float(np.sqrt(np.mean(right[mask] ** 2)))
    return numerator / max(denominator, np.finfo(float).tiny)


def solve_joint_field(
    density: np.ndarray,
    surface_density: np.ndarray,
    spacing: float,
    *,
    gravitational_constant: float,
    a0: float,
    light_speed: float,
    equation: dict,
):
    fraction, covariance, eigenvalues = projected_baryonic_spectral_anisotropy(surface_density, spacing)
    routing = solve_source_conserving_baryonic_routing(
        density,
        spacing,
        gravitational_constant=gravitational_constant,
        a0=a0,
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
        light_speed=light_speed,
    )
    local_potential = solve_poisson_dirichlet(
        routing.local_generator_source,
        spacing,
        routing.boundary_potential,
    )
    coherent = coherent_monopole_potential(
        density,
        routing.newtonian.potential,
        routing.newtonian.acceleration,
        spacing,
        a0=a0,
    )
    raw_vector_coherence = baryonic_vector_coherence(
        density,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    geometry = baryonic_mass_planarity(density, spacing)
    effective_coherence = planarity_blended_coherence(
        raw_vector_coherence.coherence,
        geometry.planarity,
    )
    vector_coherence = dataclasses.replace(
        raw_vector_coherence,
        coherence=effective_coherence,
    )
    base = coherence_gated_source_potential(
        coherent,
        routing.local_generator_source,
        effective_coherence,
        spacing,
    )
    joint = hybrid_coherence_routing_potential(
        base,
        local_potential,
        routing.field.potential,
        spacing,
        fraction,
    )
    interior = np.zeros(density.shape, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    expected_source = effective_coherence * coherent.equation_source + (
        1.0 - effective_coherence
    ) * routing.local_generator_source
    source_identity = relative_grid_rms(base.equation_source, expected_source, interior)
    expected_joint = base.potential + fraction * (routing.field.potential - local_potential)
    joint_identity = relative_grid_rms(joint.potential, expected_joint, interior)
    local_residual = normalized_residual_rms(
        laplacian(local_potential, spacing) - routing.local_generator_source,
        routing.local_generator_source,
    )
    edge = boundary_mask(density.shape)
    boundary_scale = max(float(np.max(np.abs(routing.boundary_potential[edge]))), np.finfo(float).tiny)
    routing_boundary = float(
        np.max(np.abs((routing.field.potential - local_potential)[edge])) / boundary_scale
    )
    arrays = (
        routing.newtonian.potential,
        *routing.newtonian.acceleration,
        local_potential,
        routing.field.potential,
        *routing.field.acceleration,
        coherent.potential,
        *coherent.acceleration,
        coherent.equation_source,
        raw_vector_coherence.coherence,
        effective_coherence,
        base.potential,
        *base.acceleration,
        base.equation_source,
        joint.potential,
        *joint.acceleration,
        joint.equation_source,
    )
    audit = {
        "newtonian_normalized_residual_RMS": routing.newtonian.normalized_residual_rms,
        "local_routing_component_normalized_residual_RMS": local_residual,
        "routed_component_normalized_residual_RMS": routing.field.normalized_residual_rms,
        "gated_component_normalized_residual_RMS": base.normalized_residual_rms,
        "gated_source_identity_relative_RMS": source_identity,
        "hybrid_potential_identity_relative_RMS": joint_identity,
        "base_coherent_boundary_relative_mismatch": base_boundary_relative_mismatch(base, coherent),
        "routing_correction_boundary_relative_mismatch": routing_boundary,
        "mass_planarity": geometry.planarity,
        "mass_covariance_eigenvalues": geometry.eigenvalues.tolist(),
        "raw_local_coherence_minimum": float(np.min(raw_vector_coherence.coherence)),
        "raw_local_coherence_median": float(np.median(raw_vector_coherence.coherence)),
        "raw_local_coherence_maximum": float(np.max(raw_vector_coherence.coherence)),
        "coherence_minimum": float(np.min(effective_coherence)),
        "coherence_median": float(np.median(effective_coherence)),
        "coherence_maximum": float(np.max(effective_coherence)),
        "raw_triangle_inequality_excess_maximum": raw_vector_coherence.maximum_triangle_inequality_excess,
        "normalized_acceleration_curl_RMS": normalized_acceleration_curl(joint.acceleration, spacing),
        "finite": bool(all(np.all(np.isfinite(array)) for array in arrays)),
    }
    return (
        joint,
        base,
        coherent,
        vector_coherence,
        routing,
        local_potential,
        fraction,
        covariance,
        eigenvalues,
        audit,
    )


def merge_protocol() -> tuple[dict, Path]:
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    parent_path = ROOT / overlay["parent_protocol"]
    protocol = copy.deepcopy(json.loads(parent_path.read_text(encoding="utf-8")))
    for key, value in overlay.items():
        if key in {"equation", "outputs"}:
            protocol[key].update(value)
        else:
            protocol[key] = value
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_protocol.json"
    resolved.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol, resolved


def main() -> None:
    protocol, resolved = merge_protocol()
    math_report = json.loads((ROOT / protocol["geometry_math_parent"]).read_text(encoding="utf-8"))
    if not math_report.get("all_math_gates_pass") or not math_report.get("candidate_advanced_to_spent_joint_screen"):
        raise RuntimeError("P0702 did not authorize the P0703 spent screen")
    parent_runner.MODEL = "mass_planarity_blended_projected_routing_P0703"
    parent_runner.solve_joint_field = solve_joint_field
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(__file__).resolve()), "--config", str(resolved)]
        parent_runner.main()
    finally:
        sys.argv = original_argv

    report_path = ROOT / protocol["outputs"]["directory"] / protocol["outputs"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_version"] = "P0703-SPENT-PLANARITY-BLENDED-JOINT-SCREEN-RESULTS-1.0.0"
    report["protocol_overlay_sha256"] = sha256(OVERLAY)
    report["resolved_protocol_sha256"] = sha256(resolved)
    report["wrapper_source_sha256"] = sha256(Path(__file__).resolve())
    report["planarity_operator_source_sha256"] = sha256(ROOT / "src/voidscreen/mass_planarity.py")
    report["geometry_math_parent_sha256"] = sha256(ROOT / protocol["geometry_math_parent"])
    report["candidate_advanced_to_sealed_outcomes"] = False
    report["next_if_pass"] = protocol["if_pass"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    galaxy = report["spent_DDO154"]
    cluster = report["spent_RXJ2129"]
    fit = cluster["fit"]
    topology = cluster["topology"]
    failed = report["failed_gates"]
    summary = f"""# P0703 spent mass-planarity blended joint screen

- Status: **{'PASS' if report['all_progression_gates_pass'] else 'FAIL'}**.
- DDO154 / RX J2129 mass planarity: **{galaxy['field_audit']['mass_planarity']:.6g} / {cluster['field']['mass_planarity']:.6g}**.
- DDO154 RMSE / weighted RMSE: **{galaxy['candidate_score']['RMSE_km_s']:.4g} / {galaxy['candidate_score']['weighted_RMSE_km_s']:.4g} km/s**.
- DDO154 ordinary / weighted algebraic-MOND ratios: **{galaxy['comparisons']['candidate_RMSE_to_algebraic_MOND_ratio']:.4g} / {galaxy['comparisons']['candidate_weighted_RMSE_to_algebraic_MOND_ratio']:.4g}**.
- RX J2129 training / heldout roots: **{fit['training_roots_converged']}/15 / {fit['heldout_roots_converged']}/7**.
- RX J2129 training / heldout RMS / compact-halo ratio: **{fit['training_RMS_arcsec']:.4g} / {fit['heldout_RMS_arcsec']:.4g} arcsec / {cluster['candidate_to_compact_halo_heldout_RMS_ratio']:.4g}**.
- Missing / surplus / parity / critical families: **{topology['missing_multiplicity_families']} / {topology['potentially_observable_surplus_families']} / {topology['parity_diverse_families']} / {topology['critical_curve_present_families']}**.
- Failed gates: **{', '.join(failed) if failed else 'none'}**.
- P0633/P0640 outcome data opened: **no**.
"""
    (report_path.parent / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
