#!/usr/bin/env python3
"""Run the frozen, non-promotable P0692 linear source-routing atlas."""

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

from voidscreen.field_solvers import boundary_mask
from voidscreen.metric_lensing_3d import (
    KPC_M,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.source_routing_qumond import (
    solve_linear_routing_mixture,
    solve_source_conserving_baryonic_routing,
)

G_SI = 6.67430e-11
DEFAULT_CONFIG = ROOT / "configs" / "p0692_spent_linear_routing_continuum_atlas.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_rms(x_values: np.ndarray, y_values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x_values[mask] ** 2 + y_values[mask] ** 2)))


def model_name(index: int, fraction: float) -> str:
    token = f"{fraction:.12g}".replace(".", "p")
    return f"linear_route_{index:02d}_f{token}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != (
        "frozen_before_any_P0692_intermediate_fraction_field_photon_fit_or_topology_score"
    ):
        raise RuntimeError("P0692 protocol is not frozen")

    parent_paths = {key: ROOT / value for key, value in protocol["parents"].items()}
    parents = {key: read_json(path) for key, path in parent_paths.items()}
    expected = protocol["predeclared_integrity_gates"]
    fractions = np.asarray(protocol["routing_fractions"], dtype=float)
    marker = float(protocol["registered_marker"]["fraction"])
    integrity = {
        "P0691_status": parents["P0691_result"].get("status") == expected["P0691_status"],
        "P0691_not_advanced": bool(
            parents["P0691_result"].get("candidate_advanced_to_real_2D_galaxy_and_robustness")
        )
        is bool(expected["P0691_candidate_advanced"]),
        "P0690_status": parents["P0690_result"].get("status") == expected["P0690_status"],
        "P0670_parent": bool(parents["P0670_map_result"].get("all_progression_gates_pass"))
        is bool(expected["P0670_all_progression_gates_pass"]),
        "fraction_range": bool(
            np.all(fractions >= float(expected["routing_fraction_min"]))
            and np.all(fractions <= float(expected["routing_fraction_max"]))
        ),
        "fractions_unique": len(np.unique(fractions)) == len(fractions),
        "fractions_strictly_increasing": bool(np.all(np.diff(fractions) > 0.0)),
        "registered_marker_present": bool(np.any(np.isclose(fractions, marker, rtol=0, atol=1e-15))),
        "no_selected_gravity_parameter": int(expected["gravity_parameters_selected_from_atlas"])
        == 0,
        "no_selected_photon_parameter": int(expected["photon_parameters_selected_from_atlas"])
        == 0,
        "sealed_targets_untouched": not bool(expected["sealed_target_outcomes_opened"]),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"P0692 integrity failure before scores: {integrity}")

    map_path = ROOT / protocol["map_input"]
    with np.load(map_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        density = data["stellar_volume_density_kg_m3"].astype(float) + data[
            "gas_volume_density_kg_m3"
        ].astype(float)
        map_a0 = float(data["a0_m_s2"])
    equation = protocol["equation"]
    a0 = float(equation["a0_m_s2"])
    if not np.isclose(a0, map_a0, rtol=0.0, atol=0.0):
        raise RuntimeError("P0692 a0 no longer matches the physical map")
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M

    print("solving shared local and routed source endpoints", flush=True)
    routing = solve_source_conserving_baryonic_routing(
        density,
        spacing_m,
        gravitational_constant=G_SI,
        a0=a0,
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
    )

    raw = read_json(ROOT / protocol["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    integrity["coverage"] = (
        len(training) == int(expected["training_images"])
        and len(heldout) == int(expected["spent_heldout_images"])
        and int(images.source_family.nunique()) == int(expected["source_families"])
    )
    if not integrity["coverage"]:
        raise RuntimeError("P0692 raw-image coverage changed")
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    compact_report = read_json(ROOT / protocol["compact_halo_comparator_report"])
    compact_halo = float(
        compact_report["model_scores"]["GR_plus_cluster_halo"]["heldout"]
        ["exact_radial_RMS_arcsec"]
    )
    x_kpc, y_kpc = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    annulus = (np.hypot(x_kpc, y_kpc) >= 15.8) & (np.hypot(x_kpc, y_kpc) <= 76.5)
    edge = boundary_mask(density.shape)
    boundary_scale = max(
        float(np.max(np.abs(routing.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    gates = protocol["diagnostic_viability_gates"]

    atlas_rows: list[dict] = []
    family_frames: list[pd.DataFrame] = []
    nuisance_rows: list[dict] = []
    for index, fraction in enumerate(fractions):
        model = model_name(index, float(fraction))
        print(
            f"P0692 fraction {index + 1}/{len(fractions)}: f={fraction:.12g}",
            flush=True,
        )
        solution = solve_linear_routing_mixture(routing, spacing_m, float(fraction))
        expected_source = (
            (1.0 - fraction) * routing.local_generator_source
            + fraction * routing.routed_source
        )
        identity_scale = max(
            float(np.sqrt(np.mean(np.square(expected_source)))),
            np.finfo(float).tiny,
        )
        identity_relative = float(
            np.sqrt(np.mean(np.square(solution.mixed_source - expected_source)))
            / identity_scale
        )
        deflection = photon_deflection_zero_slip(
            solution.field.acceleration,
            spacing_m,
            distance_ratio=1.0,
        )
        magnitude = np.hypot(deflection.alpha_x_arcsec, deflection.alpha_y_arcsec)
        field_median = float(np.median(magnitude[annulus]))
        field_rms = vector_rms(
            deflection.alpha_x_arcsec,
            deflection.alpha_y_arcsec,
            annulus,
        )
        field_curl = normalized_deflection_curl(
            deflection.alpha_x_arcsec,
            deflection.alpha_y_arcsec,
            float(axis_kpc[1] - axis_kpc[0]),
        )
        boundary_mismatch = float(
            np.max(np.abs(solution.field.potential[edge] - routing.boundary_potential[edge]))
            / boundary_scale
        )
        finite = bool(
            np.all(np.isfinite(solution.mixed_source))
            and np.all(np.isfinite(solution.field.potential))
            and all(np.all(np.isfinite(item)) for item in solution.field.acceleration)
            and np.all(np.isfinite(magnitude))
        )

        grid = PhysicalDeflectionGrid(
            axis_kpc / scale,
            deflection.alpha_x_arcsec,
            deflection.alpha_y_arcsec,
        )
        lens = AbsoluteGridLens(raw, {model: grid})
        fit = exact_fit(lens, model, training, heldout, protocol, seed_offset=0)
        training_rms = float(fit["training_score"]["exact_radial_RMS_arcsec"])
        heldout_rms = float(fit["heldout_score"]["exact_radial_RMS_arcsec"])
        training_roots = int(fit["training_score"]["converged_roots"])
        heldout_roots = int(fit["heldout_score"]["converged_roots"])
        nuisance_near_bound = near_bound_count(fit["parameters"])
        halo_ratio = heldout_rms / compact_halo if np.isfinite(heldout_rms) else float("inf")
        _, _, families, _ = global_topology(
            lens,
            model,
            fit,
            images,
            protocol["global_topology"],
        )
        topology = topology_summary(families)
        row_gates = {
            "mixed_source_identity": identity_relative
            <= float(gates["mixed_source_linear_identity_relative_RMS_max"]),
            "field_residual": solution.field.normalized_residual_rms
            <= float(gates["field_normalized_residual_RMS_max"]),
            "boundary": boundary_mismatch
            <= float(gates["boundary_maximum_relative_mismatch_max"]),
            "finite": finite
            is bool(gates["all_sources_potentials_accelerations_and_deflections_finite"]),
            "field_amplitude_lower": field_median
            >= float(gates["strong_lens_median_physical_deflection_arcsec_min"]),
            "field_amplitude_upper": field_median
            <= float(gates["strong_lens_median_physical_deflection_arcsec_max"]),
            "field_curl": field_curl <= float(gates["normalized_deflection_curl_RMS_max"]),
            "training_roots": training_roots == int(gates["training_roots_converged"]),
            "heldout_roots": heldout_roots == int(gates["heldout_roots_converged"]),
            "training_RMS": training_rms <= float(gates["training_RMS_arcsec_max"]),
            "heldout_RMS": heldout_rms <= float(gates["heldout_RMS_arcsec_max"]),
            "compact_halo_comparison": halo_ratio
            <= float(gates["candidate_to_compact_halo_heldout_RMS_ratio_max"]),
            "no_missing_multiplicity": topology["missing_multiplicity_families"]
            <= int(gates["missing_multiplicity_families_max"]),
            "observable_surplus": topology["potentially_observable_surplus_families"]
            <= int(gates["potentially_observable_surplus_families_max"]),
            "acceptable_multiplicity": topology["exact_or_demagnified_only_families"]
            >= int(gates["exact_or_demagnified_only_families_min"]),
            "parity_diversity": topology["parity_diverse_families"]
            >= int(gates["parity_diverse_families_min"]),
            "critical_curves": topology["critical_curve_present_families"]
            >= int(gates["critical_curve_present_families_min"]),
            "nuisance_bounds": nuisance_near_bound
            <= int(gates["nuisance_parameters_near_bound_max"]),
        }
        viable = bool(all(row_gates.values()))
        failed = [name for name, passed in row_gates.items() if not passed]
        atlas_rows.append(
            {
                "model": model,
                "routing_fraction": float(fraction),
                "registered_quadrupole_marker": bool(np.isclose(fraction, marker, rtol=0, atol=1e-15)),
                "field_normalized_residual_RMS": solution.field.normalized_residual_rms,
                "mixed_source_identity_relative_RMS": identity_relative,
                "boundary_maximum_relative_mismatch": boundary_mismatch,
                "strong_lens_median_physical_deflection_arcsec": field_median,
                "strong_lens_RMS_physical_deflection_arcsec": field_rms,
                "normalized_deflection_curl_RMS": field_curl,
                "training_RMS_arcsec": training_rms,
                "training_roots_converged": training_roots,
                "heldout_RMS_arcsec": heldout_rms,
                "heldout_roots_converged": heldout_roots,
                "candidate_to_compact_halo_heldout_RMS_ratio": halo_ratio,
                "optimizer_cost": float(fit["optimizer_cost"]),
                "nuisance_parameters_near_bound": nuisance_near_bound,
                **topology,
                "passes_all_diagnostic_viability_gates": viable,
                "failed_diagnostic_gates": ",".join(failed),
            }
        )
        family_copy = families.copy()
        family_copy.insert(0, "routing_fraction", float(fraction))
        family_copy.insert(0, "model", model)
        family_frames.append(family_copy)
        for label, value, lower, upper in zip(
            AbsoluteGridLens.labels,
            fit["parameters"],
            AbsoluteGridLens.lower,
            AbsoluteGridLens.upper,
            strict=True,
        ):
            nuisance_rows.append(
                {
                    "model": model,
                    "routing_fraction": float(fraction),
                    "parameter": label,
                    "value": float(value),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )
        print(
            f"f={fraction:.6g}: roots={training_roots}/15+{heldout_roots}/7 "
            f"missing={topology['missing_multiplicity_families']} "
            f"parity={topology['parity_diverse_families']}/7 viable={viable}",
            flush=True,
        )

    atlas = pd.DataFrame(atlas_rows)
    family_table = pd.concat(family_frames, ignore_index=True)
    nuisance_table = pd.DataFrame(nuisance_rows)
    viable_rows = atlas.loc[atlas.passes_all_diagnostic_viability_gates.astype(bool)]
    retire_sampled = len(viable_rows) == 0
    total_root_recovery = atlas.training_roots_converged + atlas.heldout_roots_converged
    best_root_count = int(total_root_recovery.max())
    best_root_fractions = atlas.loc[
        total_root_recovery.eq(best_root_count), "routing_fraction"
    ].astype(float).tolist()
    marker_row = atlas.loc[atlas.registered_quadrupole_marker.astype(bool)].iloc[0]
    p0691 = parents["P0691_result"]
    marker_reproduction = {
        "field_median_absolute_difference_arcsec": abs(
            float(marker_row.strong_lens_median_physical_deflection_arcsec)
            - float(p0691["field"]["strong_lens_median_physical_deflection_arcsec"])
        ),
        "training_root_difference": int(marker_row.training_roots_converged)
        - int(p0691["fit_score"]["training_roots_converged"]),
        "heldout_root_difference": int(marker_row.heldout_roots_converged)
        - int(p0691["fit_score"]["heldout_roots_converged"]),
        "missing_family_difference": int(marker_row.missing_multiplicity_families)
        - int(p0691["topology"]["missing_multiplicity_families"]),
        "parity_family_difference": int(marker_row.parity_diverse_families)
        - int(p0691["topology"]["parity_diverse_families"]),
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(output / protocol["outputs"]["atlas_table"], index=False)
    family_table.to_csv(output / protocol["outputs"]["family_table"], index=False)
    nuisance_table.to_csv(output / protocol["outputs"]["nuisance_table"], index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    axes[0, 0].plot(
        atlas.routing_fraction,
        atlas.strong_lens_median_physical_deflection_arcsec,
        marker="o",
    )
    axes[0, 0].axhspan(
        float(gates["strong_lens_median_physical_deflection_arcsec_min"]),
        float(gates["strong_lens_median_physical_deflection_arcsec_max"]),
        alpha=0.12,
        color="green",
    )
    axes[0, 0].set(title="Physical field amplitude", ylabel="median deflection (arcsec)")
    axes[0, 1].plot(atlas.routing_fraction, atlas.training_roots_converged, marker="o", label="training / 15")
    axes[0, 1].plot(atlas.routing_fraction, atlas.heldout_roots_converged, marker="s", label="heldout / 7")
    axes[0, 1].axhline(15, color="C0", linestyle=":")
    axes[0, 1].axhline(7, color="C1", linestyle=":")
    axes[0, 1].set(title="Exact observed-image roots", ylabel="recovered roots")
    axes[0, 1].legend()
    axes[1, 0].plot(atlas.routing_fraction, atlas.missing_multiplicity_families, marker="o", label="missing")
    axes[1, 0].plot(atlas.routing_fraction, atlas.potentially_observable_surplus_families, marker="s", label="observable surplus")
    axes[1, 0].plot(atlas.routing_fraction, 7 - atlas.parity_diverse_families, marker="^", label="lacking both parities")
    axes[1, 0].set(title="Topology failures", xlabel="routing fraction f", ylabel="families")
    axes[1, 0].legend()
    axes[1, 1].plot(atlas.routing_fraction, atlas.optimizer_cost, marker="o", color="C4", label="optimizer cost")
    axes[1, 1].set(title="Fit and boundary behavior", xlabel="routing fraction f", ylabel="optimizer cost")
    nuisance_axis = axes[1, 1].twinx()
    nuisance_axis.step(
        atlas.routing_fraction,
        atlas.nuisance_parameters_near_bound,
        where="mid",
        color="C3",
        label="near-bound nuisances",
    )
    nuisance_axis.set_ylabel("near-bound nuisances")
    for axis in axes.ravel():
        axis.axvline(marker, color="black", linestyle="--", linewidth=1, alpha=0.65)
        axis.grid(alpha=0.2)
    figure.suptitle("P0692 spent linear source-routing continuum (diagnostic only)")
    figure.tight_layout()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0692-SPENT-LINEAR-ROUTING-CONTINUUM-ATLAS-RESULTS-1.0.0",
        "status": "complete",
        "evidence_class": protocol["evidence_class"],
        "diagnostic_only": True,
        "candidate_advanced": False,
        "sampled_linear_family_retired": retire_sampled,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "operator_source_sha256": sha256(ROOT / "src/voidscreen/source_routing_qumond.py"),
        "map_sha256": sha256(map_path),
        "parent_sha256": {key: sha256(path) for key, path in parent_paths.items()},
        "integrity_gates": integrity,
        "routing_fraction_count": len(fractions),
        "viable_row_count": len(viable_rows),
        "viable_routing_fractions": viable_rows.routing_fraction.astype(float).tolist(),
        "best_total_observed_root_recovery": best_root_count,
        "best_total_observed_root_fractions": best_root_fractions,
        "minimum_missing_multiplicity_families": int(atlas.missing_multiplicity_families.min()),
        "maximum_parity_diverse_families": int(atlas.parity_diverse_families.max()),
        "maximum_exact_or_demagnified_only_families": int(
            atlas.exact_or_demagnified_only_families.max()
        ),
        "registered_marker_reproduction": marker_reproduction,
        "compact_halo_heldout_RMS_arcsec": compact_halo,
        "atlas_rows": atlas_rows,
        "decision_rule": protocol["decision_rule"],
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    report_path = output / protocol["outputs"]["report"]
    report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    viable_text = ", ".join(f"{value:.6g}" for value in viable_rows.routing_fraction) or "none"
    summary = f"""# P0692 spent linear source-routing continuum atlas

- Evidence class: **diagnostic only; no row may advance**.
- Frozen routing fractions: **{len(fractions)}**.
- Rows passing every viability gate: **{len(viable_rows)}** ({viable_text}).
- Best observed-root recovery: **{best_root_count}/22** at fractions **{best_root_fractions}**.
- Minimum missing-multiplicity families: **{int(atlas.missing_multiplicity_families.min())}/7**.
- Maximum parity-diverse families: **{int(atlas.parity_diverse_families.max())}/7**.
- Sampled linear family retired: **{'yes' if retire_sampled else 'no'}**.
- Candidate advanced: **no**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
