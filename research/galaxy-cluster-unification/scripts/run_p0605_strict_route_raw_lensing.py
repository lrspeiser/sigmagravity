#!/usr/bin/env python3
"""Replay the strict universal route with two frozen scalar closures on RX J2129."""

from __future__ import annotations

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

from run_arc_apogee_cross_domain import radius_at_mass_fraction  # noqa: E402
from run_p0601_frozen_potential_raw_lensing import fit_model, json_safe  # noqa: E402
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    load_baryonic_anchors,
    load_images,
    spec_for,
)
from voidscreen.arc_apogee import G_SI, M_SUN_KG  # noqa: E402
from voidscreen.arc_invariants import spherical_profile_invariants  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    radial_shape_activation,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.tensor_routing import redistributed_cumulative_mass_tensor  # noqa: E402
from voidscreen.unified import rar_acceleration  # noqa: E402


def build_fields(anchors: pd.DataFrame, raw_protocol: dict, protocol: dict):
    constants = protocol["constants"]
    route = protocol["selected_route"]
    radius = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    mass = np.maximum.accumulate(
        gbar * np.square(radius * KPC_M) / (G_SI * M_SUN_KG)
    )
    anchor_mass = np.maximum.accumulate(
        anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    )
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    concentration = r50 / r80
    shape = float(
        radial_shape_activation(
            concentration,
            midpoint=constants["shape_midpoint"],
            width=constants["shape_width"],
        )
    )
    source_g = float(np.interp(r80, radius, gbar))
    screen = float(
        low_acceleration_activation(
            source_g,
            a0_m_s2=constants["a0_m_s2"],
            power=route["source_acceleration_gate_power"],
        )
    )
    fraction = route["fraction_max"] * shape * screen
    routed, conservation_error = redistributed_cumulative_mass_tensor(
        radius,
        mass,
        r80=r80,
        length_over_r80=route["length_over_R80"],
        radius_exponent=route["radius_exponent"],
        width_over_r80=route["width_over_R80"],
        axis_ratio=1.0,
        bins=constants["radial_bins"],
    )
    effective_mass = (1.0 - fraction) * mass + fraction * routed
    g_route = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
    base = rar_acceleration(g_route, constants["a0_m_s2"])
    invariants = spherical_profile_invariants(radius, gbar)
    chi = invariants["potential_depth"]
    p = constants["potential_power"]
    threshold = constants["potential_threshold_chi"]
    potential_gate = chi**p / (chi**p + threshold**p)
    amplitude_fraction = (
        constants["potential_amplitude_A"] * screen * shape * potential_gate
    )
    accelerations = {
        "strict_route_RAR": base,
        "strict_route_P0599": base * (1.0 + amplitude_fraction),
    }
    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    fields = {}
    for name, acceleration in accelerations.items():
        def lookup(query_radius, values=acceleration):
            return np.exp(np.interp(np.log(query_radius), np.log(radius), np.log(values)))

        alpha = spherical_deflection_radians(
            impact_arcsec * scale,
            lookup,
            maximum_radius_kpc=1.0e6,
            integration_points=800,
        )
        fields[name] = RadialDeflectionField(impact_arcsec, alpha)
    indices = np.unique(np.geomspace(1, len(radius), 220).astype(int) - 1)
    profiles = []
    for name, acceleration in accelerations.items():
        profiles.append(
            pd.DataFrame(
                {
                    "model": name,
                    "radius_kpc": radius[indices],
                    "gbar_m_s2": gbar[indices],
                    "route_acceleration_m_s2": g_route[indices],
                    "effective_acceleration_m_s2": acceleration[indices],
                    "potential_depth_chi": chi[indices],
                    "potential_gate": potential_gate[indices],
                    "potential_amplitude_fraction": (
                        np.zeros_like(indices, dtype=float)
                        if name == "strict_route_RAR"
                        else amplitude_fraction[indices]
                    ),
                    "route_fraction": fraction,
                    "R50_kpc": r50,
                    "R80_kpc": r80,
                    "concentration_R50_over_R80": concentration,
                }
            )
        )
    diagnostic = {
        "R50_kpc": r50,
        "R80_kpc": r80,
        "concentration_R50_over_R80": concentration,
        "shape_gate": shape,
        "source_acceleration_gate": screen,
        "effective_route_fraction": fraction,
        "radial_mass_conservation_error": conservation_error,
    }
    return fields, pd.concat(profiles, ignore_index=True), diagnostic


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0605_strict_route_raw_lensing_protocol.json").read_text()
    )
    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_protocol"]).read_text())
    images = load_images(raw_protocol)
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    training = images[~images.image_id.isin(heldout_ids)].copy()
    heldout = images[images.image_id.isin(heldout_ids)].copy()
    fields, profiles, field_diagnostic = build_fields(
        load_baryonic_anchors(raw_protocol), raw_protocol, protocol
    )
    previous_parameters = pd.read_csv(
        ROOT / protocol["inputs"]["previous_raw_parameters"]
    )
    block = previous_parameters[
        previous_parameters.model.eq("P0599_potential_shape")
    ].set_index("parameter")
    initial = block.loc[list(spec_for("fixed").labels), "value"].to_numpy(float)
    score_rows, prediction_frames, parameter_frames = [], [], []
    for offset, name in enumerate(("strict_route_RAR", "strict_route_P0599")):
        row, predictions, parameters = fit_model(
            name,
            fields[name],
            raw_protocol,
            training,
            heldout,
            initial,
            starts=int(protocol["validation"]["optimization_starts"]),
            seed=21292026 + 60500 + offset,
        )
        score_rows.append(row)
        prediction_frames.append(predictions)
        parameter_frames.append(parameters)
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    previous = json.loads((ROOT / protocol["inputs"]["previous_raw_report"]).read_text())
    comparators = {
        "P0599_no_route_heldout_RMS_arcsec": next(
            row["heldout_RMS_arcsec"]
            for row in previous["scores"]
            if row["model"] == "P0599_potential_shape"
        ),
        "previous_locked_candidate_heldout_RMS_arcsec": 1.0642772678285497,
        "compact_one_halo_heldout_RMS_arcsec": 2.5361068843508456,
        "published_multi_halo_reference_RMS_arcsec": 0.29,
    }
    report = {
        "report_version": "P0605-STRICT-ROUTE-RAW-LENSING-RESULTS-0.1.0",
        "status": "complete_spent_raw_route_amplitude_decomposition",
        "coverage": {
            "models": 2,
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "optimization_starts_per_model": protocol["validation"]["optimization_starts"],
            "fitted_gravity_parameters": 0,
        },
        "selected_route": protocol["selected_route"],
        "field_diagnostic": field_diagnostic,
        "scores": scores.to_dict("records"),
        "comparators": comparators,
        "strict_interpretation": {
            "raw_data_are_fresh": False,
            "route_or_amplitude_fit_to_RXJ2129": False,
            "structural_geometry_parameters_fit": 6,
            "next_confirmation_requires_different_cluster": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    parameters.to_csv(output / protocol["outputs"]["parameters"], index=False)
    profiles.to_csv(output / protocol["outputs"]["profiles"], index=False)
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for name, group in profiles.groupby("model", sort=False):
        axes[0].loglog(group.radius_kpc, group.effective_acceleration_m_s2, label=name.replace("_", " "))
    axes[0].axvspan(15.8, 76.5, color="black", alpha=0.08, label="image annulus")
    axes[0].set(xlabel="radius (kpc)", ylabel="effective acceleration (m/s²)", title="Frozen routed fields")
    axes[0].legend(fontsize=7)
    finite_scores = scores[np.isfinite(scores.heldout_RMS_arcsec)]
    axes[1].bar(
        [name.replace("_", "\n") for name in finite_scores.model],
        finite_scores.heldout_RMS_arcsec,
        color=["#55A868", "#1261A0"][: len(finite_scores)],
    )
    axes[1].axhline(0.5, color="black", linestyle="--", label="0.5 arcsec gate")
    axes[1].axhline(comparators["P0599_no_route_heldout_RMS_arcsec"], color="gray", linestyle=":", label="P0599 no route")
    axes[1].set(ylabel="spent held-out RMS (arcsec)", title="Raw route/amplitude decomposition")
    axes[1].legend(fontsize=7)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    best = scores[np.isfinite(scores.heldout_RMS_arcsec)].sort_values("heldout_RMS_arcsec").iloc[0]
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0605 strict-route raw diagnostic\n\n"
        f"Best frozen closure: **{best.model}**, spent held-out RMS "
        f"**{best.heldout_RMS_arcsec:.4f} arcsec**, with "
        f"**{int(best.heldout_roots_converged)}/7** roots.\n\n"
        "This is a decomposition on spent RX J2129 data, not confirmation.\n"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
