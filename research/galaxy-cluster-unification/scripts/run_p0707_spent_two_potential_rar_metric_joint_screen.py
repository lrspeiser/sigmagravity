#!/usr/bin/env python3
"""Run the frozen P0707 spent two-potential RAR metric screen."""

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
from voidscreen.coherent_monopole import HybridCoherentRoutingSolution
from voidscreen.field_solvers import acceleration_from_potential, boundary_mask, laplacian, normalized_residual_rms, solve_poisson_dirichlet
from voidscreen.radial_path_potential import normalized_acceleration_curl
from voidscreen.source_routing_qumond import projected_baryonic_spectral_anisotropy, solve_source_conserving_baryonic_routing
from voidscreen.two_potential_metric import build_two_potential_metric, rar_coherent_monopole_potential

OVERLAY = ROOT / "configs" / "p0707_spent_two_potential_rar_metric_joint_screen.json"
ORIGINAL_PHOTON_DEFLECTION = parent_runner.photon_deflection_zero_slip
METRIC_SOLUTIONS = []


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
    coherent = rar_coherent_monopole_potential(
        density,
        routing.newtonian.potential,
        routing.newtonian.acceleration,
        spacing,
        a0=a0,
    )
    correction = routing.field.potential - local_potential
    time_potential = coherent.potential + fraction * correction
    weyl_potential = local_potential + fraction * correction
    metric = build_two_potential_metric(time_potential, weyl_potential, spacing)
    METRIC_SOLUTIONS.append(metric)
    joint = HybridCoherentRoutingSolution(
        potential=time_potential,
        acceleration=metric.time_acceleration,
        equation_source=laplacian(time_potential, spacing),
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
    time_identity = relative_grid_rms(joint.potential, coherent.potential + fraction * correction, interior)
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
        coherent.potential,
        *coherent.acceleration,
        joint.potential,
        *metric.time_acceleration,
        metric.spatial_potential,
        *metric.spatial_acceleration,
        metric.weyl_potential,
        *metric.weyl_acceleration,
    )
    audit = {
        "newtonian_normalized_residual_RMS": routing.newtonian.normalized_residual_rms,
        "local_routing_component_normalized_residual_RMS": local_residual,
        "routed_component_normalized_residual_RMS": routing.field.normalized_residual_rms,
        "coherent_potential_identity_relative_RMS": coherent_identity,
        "hybrid_potential_identity_relative_RMS": time_identity,
        "routing_correction_boundary_relative_mismatch": routing_boundary,
        "weyl_metric_identity_relative_RMS": metric.weyl_identity_relative_rms,
        "normalized_acceleration_curl_RMS": normalized_acceleration_curl(metric.time_acceleration, spacing),
        "normalized_weyl_acceleration_curl_RMS": normalized_acceleration_curl(metric.weyl_acceleration, spacing),
        "finite": bool(all(np.all(np.isfinite(array)) for array in arrays)),
    }
    return joint, coherent, routing, local_potential, fraction, covariance, eigenvalues, audit


def metric_photon_deflection(_time_acceleration, spacing, *, distance_ratio):
    if not METRIC_SOLUTIONS:
        raise RuntimeError("metric photon deflection requested before a metric solve")
    return ORIGINAL_PHOTON_DEFLECTION(
        METRIC_SOLUTIONS[-1].weyl_acceleration,
        spacing,
        distance_ratio=distance_ratio,
    )


def merge_protocol() -> tuple[dict, Path]:
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(json.loads((ROOT / overlay["parent_protocol"]).read_text(encoding="utf-8")))
    for key, value in overlay.items():
        if key in {"equation", "outputs", "nuisance_fit"}:
            protocol[key].update(value)
        else:
            protocol[key] = value
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_protocol.json"
    resolved.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol, resolved


def save_metric(path: Path, metric, axis_key: str, axis: np.ndarray) -> None:
    np.savez_compressed(
        path,
        **{
            axis_key: axis,
            "time_potential": metric.time_potential,
            "spatial_potential": metric.spatial_potential,
            "weyl_potential": metric.weyl_potential,
            "time_acceleration_x": metric.time_acceleration[0],
            "time_acceleration_y": metric.time_acceleration[1],
            "time_acceleration_z": metric.time_acceleration[2],
            "weyl_acceleration_x": metric.weyl_acceleration[0],
            "weyl_acceleration_y": metric.weyl_acceleration[1],
            "weyl_acceleration_z": metric.weyl_acceleration[2],
        },
    )


def main() -> None:
    METRIC_SOLUTIONS.clear()
    protocol, resolved = merge_protocol()
    metric_parent = json.loads((ROOT / protocol["metric_math_solar_parent"]).read_text(encoding="utf-8"))
    if not metric_parent.get("all_math_and_solar_gates_pass"):
        raise RuntimeError("P0706 did not authorize the spent two-potential screen")
    parent_runner.MODEL = protocol["model_id"]
    parent_runner.solve_joint_field = solve_joint_field
    parent_runner.photon_deflection_zero_slip = metric_photon_deflection
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(__file__).resolve()), "--config", str(resolved)]
        parent_runner.main()
    finally:
        sys.argv = original_argv
        parent_runner.photon_deflection_zero_slip = ORIGINAL_PHOTON_DEFLECTION
    if len(METRIC_SOLUTIONS) != 2:
        raise RuntimeError("expected one galaxy and one cluster metric solution")

    output = ROOT / protocol["outputs"]["directory"]
    with np.load(ROOT / protocol["galaxy_map_input"]) as data:
        galaxy_axis = data["axis_kpc"].astype(float)
    with np.load(ROOT / protocol["cluster_map_input"]) as data:
        cluster_axis = data["axis_kpc"].astype(float)
    galaxy_metric_path = output / protocol["outputs"]["galaxy_metric_field"]
    cluster_metric_path = output / protocol["outputs"]["cluster_metric_field"]
    save_metric(galaxy_metric_path, METRIC_SOLUTIONS[0], "axis_kpc", galaxy_axis)
    save_metric(cluster_metric_path, METRIC_SOLUTIONS[1], "axis_kpc", cluster_axis)

    report_path = output / protocol["outputs"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_version"] = "P0707-SPENT-TWO-POTENTIAL-RAR-METRIC-JOINT-SCREEN-RESULTS-1.0.0"
    report["protocol_overlay_sha256"] = sha256(OVERLAY)
    report["resolved_protocol_sha256"] = sha256(resolved)
    report["wrapper_source_sha256"] = sha256(Path(__file__).resolve())
    report["metric_operator_source_sha256"] = sha256(ROOT / "src/voidscreen/two_potential_metric.py")
    report["metric_math_solar_parent_sha256"] = sha256(ROOT / protocol["metric_math_solar_parent"])
    report["cluster_endpoint_parent_sha256"] = sha256(ROOT / protocol["cluster_endpoint_parent"])
    report["galaxy_metric_field_sha256"] = sha256(galaxy_metric_path)
    report["cluster_metric_field_sha256"] = sha256(cluster_metric_path)
    report["candidate_advanced_to_external_lock_robustness"] = bool(report["all_progression_gates_pass"])
    report["candidate_advanced_to_sealed_outcomes"] = False
    report["next_if_pass"] = protocol["if_pass"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    galaxy = report["spent_DDO154"]
    cluster = report["spent_RXJ2129"]
    fit = cluster["fit"]
    topology = cluster["topology"]
    failed = report["failed_gates"]
    summary = f"""# P0707 spent two-potential RAR metric joint screen

- Status: **{'PASS' if report['all_progression_gates_pass'] else 'FAIL'}**.
- DDO154 time-potential RMSE / weighted RMSE: **{galaxy['candidate_score']['RMSE_km_s']:.4g} / {galaxy['candidate_score']['weighted_RMSE_km_s']:.4g} km/s**.
- DDO154 ordinary / weighted algebraic-MOND ratios: **{galaxy['comparisons']['candidate_RMSE_to_algebraic_MOND_ratio']:.4g} / {galaxy['comparisons']['candidate_weighted_RMSE_to_algebraic_MOND_ratio']:.4g}**.
- RX J2129 Weyl training / heldout roots: **{fit['training_roots_converged']}/15 / {fit['heldout_roots_converged']}/7**.
- RX J2129 Weyl training / heldout RMS / compact-halo ratio: **{fit['training_RMS_arcsec']:.4g} / {fit['heldout_RMS_arcsec']:.4g} arcsec / {cluster['candidate_to_compact_halo_heldout_RMS_ratio']:.4g}**.
- Missing / surplus / parity / critical families: **{topology['missing_multiplicity_families']} / {topology['potentially_observable_surplus_families']} / {topology['parity_diverse_families']} / {topology['critical_curve_present_families']}**.
- Near-bound nuisance parameters: **{fit['nuisance_parameters_near_bound']}**.
- Failed gates: **{', '.join(failed) if failed else 'none'}**.
- P0706 math and Solar audit: **PASS**.
- P0633/P0640 outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
