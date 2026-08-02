#!/usr/bin/env python3
"""Run the frozen P0704 planarity blend of complementary spent endpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p0697_spent_coherent_monopole_joint_screen as parent_runner
from voidscreen.coherent_monopole import CoherentMonopoleSolution, HybridCoherentRoutingSolution, coherent_monopole_potential
from voidscreen.field_solvers import acceleration_from_potential, boundary_mask, laplacian, normalized_residual_rms, solve_poisson_dirichlet
from voidscreen.mass_planarity import baryonic_mass_planarity
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import projected_baryonic_spectral_anisotropy, solve_source_conserving_baryonic_routing

OVERLAY = ROOT / "configs" / "p0704_spent_planarity_endpoint_joint_screen.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_text(value: float | None) -> str:
    return "inf" if value is None else f"{value:.4g}"


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
    geometry = baryonic_mass_planarity(density, spacing)
    base_potential = geometry.planarity * coherent.potential + (1.0 - geometry.planarity) * local_potential
    correction = routing.field.potential - local_potential
    potential = base_potential + fraction * correction
    joint = HybridCoherentRoutingSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, spacing),
        equation_source=laplacian(potential, spacing),
        routing_correction_potential=correction,
        routing_fraction=fraction,
    )
    interior = np.zeros(density.shape, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    coherent_identity = relative_grid_rms(
        coherent.potential,
        routing.newtonian.potential + coherent.correction_potential,
        interior,
    )
    expected_joint = (
        geometry.planarity * coherent.potential
        + (1.0 - geometry.planarity) * local_potential
        + fraction * correction
    )
    joint_identity = relative_grid_rms(joint.potential, expected_joint, interior)
    local_residual = normalized_residual_rms(
        laplacian(local_potential, spacing) - routing.local_generator_source,
        routing.local_generator_source,
    )
    edge = boundary_mask(density.shape)
    boundary_scale = max(float(np.max(np.abs(routing.boundary_potential[edge]))), np.finfo(float).tiny)
    routing_boundary = float(np.max(np.abs(correction[edge])) / boundary_scale)
    arrays = (
        routing.newtonian.potential,
        *routing.newtonian.acceleration,
        local_potential,
        routing.field.potential,
        *routing.field.acceleration,
        coherent.potential,
        *coherent.acceleration,
        coherent.equation_source,
        joint.potential,
        *joint.acceleration,
        joint.equation_source,
    )
    audit = {
        "newtonian_normalized_residual_RMS": routing.newtonian.normalized_residual_rms,
        "local_routing_component_normalized_residual_RMS": local_residual,
        "routed_component_normalized_residual_RMS": routing.field.normalized_residual_rms,
        "coherent_potential_identity_relative_RMS": coherent_identity,
        "hybrid_potential_identity_relative_RMS": joint_identity,
        "routing_correction_boundary_relative_mismatch": routing_boundary,
        "mass_planarity": geometry.planarity,
        "mass_covariance_eigenvalues": geometry.eigenvalues.tolist(),
        "normalized_acceleration_curl_RMS": normalized_acceleration_curl(joint.acceleration, spacing),
        "finite": bool(all(np.all(np.isfinite(array)) for array in arrays)),
    }
    return joint, coherent, routing, local_potential, fraction, covariance, eigenvalues, audit


def merge_protocol() -> tuple[dict, Path]:
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(json.loads((ROOT / overlay["parent_protocol"]).read_text(encoding="utf-8")))
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
    geometry_parent = json.loads((ROOT / protocol["geometry_math_parent"]).read_text(encoding="utf-8"))
    cluster_parent = json.loads((ROOT / protocol["cluster_endpoint_parent"]).read_text(encoding="utf-8"))
    if not geometry_parent.get("all_math_gates_pass"):
        raise RuntimeError("P0702 planarity audit did not pass")
    cluster_gates = cluster_parent.get("gate_results", {})
    cluster_only = [
        "training_roots", "heldout_roots", "training_RMS", "heldout_RMS",
        "compact_halo_comparison", "no_missing_multiplicity", "observable_surplus",
        "acceptable_multiplicity", "parity_diversity", "critical_curves", "nuisance_bounds",
    ]
    if not all(cluster_gates.get(name, False) for name in cluster_only):
        raise RuntimeError("P0693 is not the registered topology-complete cluster endpoint")
    parent_runner.MODEL = protocol.get(
        "model_id", "mass_planarity_endpoint_projected_routing_P0704"
    )
    parent_runner.solve_joint_field = solve_joint_field
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(__file__).resolve()), "--config", str(resolved)]
        parent_runner.main()
    finally:
        sys.argv = original_argv

    report_path = ROOT / protocol["outputs"]["directory"] / protocol["outputs"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_version"] = "P0704-SPENT-PLANARITY-ENDPOINT-JOINT-SCREEN-RESULTS-1.0.0"
    report["protocol_overlay_sha256"] = sha256(OVERLAY)
    report["resolved_protocol_sha256"] = sha256(resolved)
    report["wrapper_source_sha256"] = sha256(Path(__file__).resolve())
    report["planarity_operator_source_sha256"] = sha256(ROOT / "src/voidscreen/mass_planarity.py")
    report["geometry_math_parent_sha256"] = sha256(ROOT / protocol["geometry_math_parent"])
    report["cluster_endpoint_parent_sha256"] = sha256(ROOT / protocol["cluster_endpoint_parent"])
    report["candidate_advanced_to_sealed_outcomes"] = False
    report["next_if_pass"] = protocol["if_pass"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    galaxy = report["spent_DDO154"]
    cluster = report["spent_RXJ2129"]
    fit = cluster["fit"]
    topology = cluster["topology"]
    failed = report["failed_gates"]
    summary = f"""# P0704 spent mass-planarity endpoint joint screen

- Status: **{'PASS' if report['all_progression_gates_pass'] else 'FAIL'}**.
- DDO154 / RX J2129 planarity: **{galaxy['field_audit']['mass_planarity']:.6g} / {cluster['field']['mass_planarity']:.6g}**.
- DDO154 RMSE / weighted RMSE: **{galaxy['candidate_score']['RMSE_km_s']:.4g} / {galaxy['candidate_score']['weighted_RMSE_km_s']:.4g} km/s**.
- DDO154 ordinary / weighted algebraic-MOND ratios: **{galaxy['comparisons']['candidate_RMSE_to_algebraic_MOND_ratio']:.4g} / {galaxy['comparisons']['candidate_weighted_RMSE_to_algebraic_MOND_ratio']:.4g}**.
- RX J2129 training / heldout roots: **{fit['training_roots_converged']}/15 / {fit['heldout_roots_converged']}/7**.
- RX J2129 training / heldout RMS / compact-halo ratio: **{metric_text(fit['training_RMS_arcsec'])} / {metric_text(fit['heldout_RMS_arcsec'])} arcsec / {metric_text(cluster['candidate_to_compact_halo_heldout_RMS_ratio'])}**.
- Missing / surplus / parity / critical families: **{topology['missing_multiplicity_families']} / {topology['potentially_observable_surplus_families']} / {topology['parity_diverse_families']} / {topology['critical_curve_present_families']}**.
- Near-bound nuisance parameters: **{fit['nuisance_parameters_near_bound']}**.
- Failed gates: **{', '.join(failed) if failed else 'none'}**.
- P0633/P0640 outcomes opened: **no**.
"""
    (report_path.parent / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
