#!/usr/bin/env python3
"""One-at-a-time raw-lensing response diagnostic around P0599."""

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
from voidscreen.unified import rar_acceleration  # noqa: E402


def variant_specs() -> list[dict]:
    base = {
        "family": "baseline",
        "A": 3.0,
        "threshold": 1.0e-6,
        "power": 4.0,
        "carrier": "shape",
        "eta": 0.0,
    }
    variants = [{"variant_id": "P0599_baseline", **base}]
    for value in (1.0, 2.0, 4.0):
        variants.append({"variant_id": f"amplitude_A{value:g}", **base, "family": "amplitude", "A": value})
    for value in (5.0e-7, 2.0e-6, 4.0e-6):
        variants.append({"variant_id": f"threshold_{value:.0e}", **base, "family": "threshold", "threshold": value})
    for value in (1.0, 2.0):
        variants.append({"variant_id": f"power_p{value:g}", **base, "family": "power", "power": value})
    for carrier in ("uniform", "path_normalized", "potential_gradient_normalized", "mass_growth_normalized"):
        variants.append({"variant_id": f"carrier_{carrier}", **base, "family": "carrier", "carrier": carrier})
    for eta in (-0.5, -0.25, 0.25, 0.5):
        variants.append({"variant_id": f"radial_eta_{eta:+.2f}", **base, "family": "radial_power", "carrier": "radial_power", "eta": eta})
    return variants


def prepare_profile(anchors: pd.DataFrame, constants: dict):
    radius = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(radius, anchor_radius, anchor_gbar, outer_slope=-2.0)
    invariants = spherical_profile_invariants(radius, gbar)
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
            power=constants["source_acceleration_gate_power"],
        )
    )
    return radius, gbar, invariants, r80, shape, screen


def normalized_at(values: np.ndarray, radius: np.ndarray, r80: float) -> np.ndarray:
    reference = float(np.interp(r80, radius, values))
    if not np.isfinite(reference) or abs(reference) < 1.0e-12:
        return np.ones_like(values)
    return values / reference


def carrier_values(spec: dict, radius, invariants, r80, shape, bounds):
    name = spec["carrier"]
    if name == "shape":
        values = np.full_like(radius, shape)
    elif name == "uniform":
        values = np.ones_like(radius)
    elif name == "path_normalized":
        path = invariants["potential_path_ratio"] / (1.0 + invariants["potential_path_ratio"])
        values = shape * normalized_at(path, radius, r80)
    elif name == "potential_gradient_normalized":
        gradient = -np.gradient(np.log(invariants["potential_depth"]), np.log(radius), edge_order=2)
        gradient = np.clip(gradient, 1.0e-4, None)
        values = shape * normalized_at(gradient, radius, r80)
    elif name == "mass_growth_normalized":
        growth = np.clip(invariants["enclosed_mass_log_slope"], 0.05, 3.0)
        values = shape * normalized_at(growth, radius, r80)
    elif name == "radial_power":
        values = shape * np.power(np.clip(radius / r80, 0.1, 10.0), spec["eta"])
    else:
        raise ValueError(name)
    return np.clip(values, float(bounds[0]), float(bounds[1]))


def build_field(spec, radius, gbar, invariants, r80, shape, screen, protocol, raw_protocol):
    constants = protocol["base_constants"]
    chi = invariants["potential_depth"]
    potential_gate = np.power(chi, spec["power"]) / (
        np.power(chi, spec["power"]) + spec["threshold"] ** spec["power"]
    )
    carrier = carrier_values(
        spec,
        radius,
        invariants,
        r80,
        shape,
        protocol["carrier_bounds"],
    )
    amplitude_fraction = spec["A"] * screen * potential_gate * carrier
    acceleration = rar_acceleration(gbar, constants["a0_m_s2"]) * (1.0 + amplitude_fraction)

    def lookup(query_radius):
        return np.exp(np.interp(np.log(query_radius), np.log(radius), np.log(acceleration)))

    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    alpha = spherical_deflection_radians(
        impact_arcsec * scale,
        lookup,
        maximum_radius_kpc=1.0e6,
        integration_points=800,
    )
    field = RadialDeflectionField(impact_arcsec, alpha)
    indices = np.unique(np.geomspace(1, len(radius), 180).astype(int) - 1)
    profile = pd.DataFrame(
        {
            "variant_id": spec["variant_id"],
            "family": spec["family"],
            "radius_kpc": radius[indices],
            "gbar_m_s2": gbar[indices],
            "effective_acceleration_m_s2": acceleration[indices],
            "potential_depth_chi": chi[indices],
            "potential_gate": potential_gate[indices],
            "carrier_weight": carrier[indices],
            "amplitude_fraction": amplitude_fraction[indices],
        }
    )
    return field, profile


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0602_raw_radial_structure_diagnostic_protocol.json").read_text()
    )
    raw_protocol = json.loads(
        (ROOT / "configs/rxj2129_raw_theory_lensing_protocol.json").read_text()
    )
    images = load_images(raw_protocol)
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    training = images[~images.image_id.isin(heldout_ids)].copy()
    heldout = images[images.image_id.isin(heldout_ids)].copy()
    anchors = load_baryonic_anchors(raw_protocol)
    radius, gbar, invariants, r80, shape, screen = prepare_profile(
        anchors, protocol["base_constants"]
    )
    specs = variant_specs()
    if len(specs) != protocol["candidate_count"]:
        raise RuntimeError("P0602 variant count changed")

    p0601_parameters = pd.read_csv(
        ROOT / "results/p0601_frozen_potential_raw_lensing/fitted_parameters.csv"
    )
    block = p0601_parameters[p0601_parameters.model.eq("P0599_potential_shape")].set_index("parameter")
    initial = block.loc[list(spec_for("fixed").labels), "value"].to_numpy(float)
    score_rows, profiles, predictions, parameters = [], [], [], []
    for index, spec in enumerate(specs):
        field, profile = build_field(
            spec, radius, gbar, invariants, r80, shape, screen, protocol, raw_protocol
        )
        row, prediction, parameter = fit_model(
            spec["variant_id"],
            field,
            raw_protocol,
            training,
            heldout,
            initial,
            starts=int(protocol["validation"]["optimization_starts_per_variant"]),
            seed=21292026 + 60200 + index,
        )
        row.update(spec)
        score_rows.append(row)
        profiles.append(profile)
        predictions.append(prediction)
        parameters.append(parameter)

    scores = pd.DataFrame(score_rows).sort_values("training_RMS_arcsec").reset_index(drop=True)
    selected = scores.iloc[0]
    baseline = scores[scores.variant_id.eq("P0599_baseline")].iloc[0]
    impact_rows = []
    for family, block in scores.groupby("family", sort=True):
        if family == "baseline":
            continue
        combined = pd.concat([block, scores[scores.variant_id.eq("P0599_baseline")]], ignore_index=True)
        finite = combined[
            np.isfinite(combined.training_RMS_arcsec)
            & np.isfinite(combined.heldout_RMS_arcsec)
        ].copy()
        best = finite.sort_values("training_RMS_arcsec").iloc[0]
        impact_rows.append(
            {
                "family": family,
                "variants_including_baseline": len(combined),
                "finite_variants_including_baseline": len(finite),
                "failed_exact_root_variants": int(len(combined) - len(finite)),
                "training_RMS_span_arcsec": float(finite.training_RMS_arcsec.max() - finite.training_RMS_arcsec.min()),
                "heldout_RMS_span_arcsec": float(finite.heldout_RMS_arcsec.max() - finite.heldout_RMS_arcsec.min()),
                "best_training_variant": best.variant_id,
                "best_training_RMS_arcsec": float(best.training_RMS_arcsec),
                "corresponding_spent_heldout_RMS_arcsec": float(best.heldout_RMS_arcsec),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values("training_RMS_span_arcsec", ascending=False)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.concat(profiles, ignore_index=True).to_csv(output / protocol["outputs"]["profiles"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.concat(parameters, ignore_index=True).to_csv(output / protocol["outputs"]["parameters"], index=False)
    impacts.to_csv(output / protocol["outputs"]["impacts"], index=False)

    annulus = (radius >= 15.8) & (radius <= 76.5)
    selected_profile = pd.concat(profiles, ignore_index=True)
    selected_profile = selected_profile[selected_profile.variant_id.eq(selected.variant_id)]
    report = {
        "report_version": "P0602-RAW-RADIAL-STRUCTURE-DIAGNOSTIC-RESULTS-0.1.0",
        "status": "complete_posthoc_spent_data_diagnostic",
        "coverage": {"variants": len(scores), "training_images": len(training), "spent_heldout_images": len(heldout)},
        "baseline": baseline.to_dict(),
        "training_selected_variant": selected.to_dict(),
        "parameter_impacts": impacts.to_dict(orient="records"),
        "field_context": {
            "r80_kpc": r80,
            "shape_gate": shape,
            "source_acceleration_gate": screen,
            "image_annulus_kpc": [15.8, 76.5],
        },
        "interpretation": {
            "selection_used_training_images_only": True,
            "heldout_is_fresh": False,
            "better_training_fit_can_be_claimed_as_prediction": False,
            "next_required_test": "freeze a physically motivated radial or two-dimensional carrier and replay it on another raw cluster",
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for family, block in scores.groupby("family", sort=False):
        axes[0].scatter(block.training_RMS_arcsec, block.heldout_RMS_arcsec, label=family, s=38)
    axes[0].scatter([baseline.training_RMS_arcsec], [baseline.heldout_RMS_arcsec], marker="*", s=160, color="black", label="P0599 baseline")
    axes[0].set(xlabel="training RMS (arcsec)", ylabel="spent held-out RMS (arcsec)", title="One-at-a-time raw response")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)
    axes[1].barh(impacts.family, impacts.training_RMS_span_arcsec, color="#1261A0")
    axes[1].set(xlabel="training RMS response span (arcsec)", title="Which ingredient changes the fit?")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    summary = (
        "# P0602 raw radial-structure diagnostic\n\n"
        f"Training-selected variant: **{selected.variant_id}**; training RMS "
        f"**{selected.training_RMS_arcsec:.4f} arcsec**; spent held-out RMS "
        f"**{selected.heldout_RMS_arcsec:.4f} arcsec**.\n\n"
        f"P0599 baseline: training **{baseline.training_RMS_arcsec:.4f}**, spent held-out "
        f"**{baseline.heldout_RMS_arcsec:.4f} arcsec**.\n\n"
        "This is a posthoc diagnostic, not validation.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary)
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
