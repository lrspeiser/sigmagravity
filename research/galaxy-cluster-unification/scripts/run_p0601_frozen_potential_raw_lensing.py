#!/usr/bin/env python3
"""Replay the frozen P0599 field on raw RX J2129 image positions."""

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
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    RawLens,
    load_baryonic_anchors,
    load_images,
    near_bound,
    score as raw_score,
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


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def build_fields(anchors: pd.DataFrame, raw_protocol: dict, constants: dict):
    radius = np.geomspace(0.1, 1.0e6, 4096)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius, anchor_radius, anchor_gbar, outer_slope=-2.0
    )
    invariants = spherical_profile_invariants(radius, gbar)

    anchor_mass = anchor_gbar * np.square(anchor_radius * KPC_M) / (G_SI * M_SUN_KG)
    anchor_mass = np.maximum.accumulate(anchor_mass)
    r50 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.5)
    r80 = radius_at_mass_fraction(anchor_radius, anchor_mass, 0.8)
    concentration = r50 / r80
    source_g = float(np.interp(r80, radius, gbar))
    shape = float(
        radial_shape_activation(
            concentration,
            midpoint=constants["shape_midpoint"],
            width=constants["shape_width"],
        )
    )
    screen = float(
        low_acceleration_activation(
            source_g,
            a0_m_s2=constants["a0_m_s2"],
            power=constants["source_acceleration_gate_power"],
        )
    )
    chi = invariants["potential_depth"]
    p = float(constants["potential_power"])
    threshold = float(constants["potential_threshold_chi"])
    potential_gate = np.power(chi, p) / (np.power(chi, p) + threshold**p)
    amplitude_fraction = float(constants["amplitude_A"]) * screen * shape * potential_gate
    fixed_rar = rar_acceleration(gbar, constants["a0_m_s2"])
    accelerations = {
        "fixed_RAR": fixed_rar,
        "P0599_potential_shape": fixed_rar * (1.0 + amplitude_fraction),
    }

    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    impact_arcsec = np.geomspace(0.05, 500.0, 700)
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

    sample_index = np.unique(np.geomspace(1, len(radius), 300).astype(int) - 1)
    profile_rows = []
    for name, acceleration in accelerations.items():
        for index in sample_index:
            profile_rows.append(
                {
                    "model": name,
                    "radius_kpc": radius[index],
                    "gbar_m_s2": gbar[index],
                    "effective_acceleration_m_s2": acceleration[index],
                    "potential_depth_chi": chi[index],
                    "potential_gate": potential_gate[index],
                    "amplitude_fraction": 0.0
                    if name == "fixed_RAR"
                    else amplitude_fraction[index],
                    "concentration_r50_over_r80": concentration,
                    "r50_kpc": r50,
                    "r80_kpc": r80,
                    "source_gbar_at_r80_m_s2": source_g,
                    "shape_gate": shape,
                    "source_acceleration_gate": screen,
                }
            )
    diagnostics = {
        "r50_kpc": r50,
        "r80_kpc": r80,
        "concentration_r50_over_r80": concentration,
        "shape_gate": shape,
        "source_gbar_at_r80_m_s2": source_g,
        "source_acceleration_gate": screen,
        "potential_depth_at_image_min": float(
            np.interp(
                raw_protocol["baryonic_inputs"]["strong_lens_impact_radius_range_kpc_expected"][0],
                radius,
                chi,
            )
        ),
        "potential_depth_at_image_max": float(
            np.interp(
                raw_protocol["baryonic_inputs"]["strong_lens_impact_radius_range_kpc_expected"][1],
                radius,
                chi,
            )
        ),
        "P0599_amplitude_fraction_at_image_min": float(
            np.interp(
                raw_protocol["baryonic_inputs"]["strong_lens_impact_radius_range_kpc_expected"][0],
                radius,
                amplitude_fraction,
            )
        ),
        "P0599_amplitude_fraction_at_image_max": float(
            np.interp(
                raw_protocol["baryonic_inputs"]["strong_lens_impact_radius_range_kpc_expected"][1],
                radius,
                amplitude_fraction,
            )
        ),
    }
    return fields, pd.DataFrame(profile_rows), diagnostics


def fit_model(
    display_name: str,
    field: RadialDeflectionField,
    raw_protocol: dict,
    training: pd.DataFrame,
    heldout: pd.DataFrame,
    initial: np.ndarray,
    *,
    starts: int,
    seed: int,
):
    internal_name = "fixed"
    lens = RawLens(raw_protocol, {internal_name: field})
    fit = lens.fit(
        internal_name,
        training,
        starts=starts,
        seed=seed,
        initial_override=initial,
    )
    training_predictions = lens.exact_predictions(
        internal_name,
        fit["result"].x,
        fit["sources"],
        training,
        stage="training",
    )
    heldout_predictions = lens.exact_predictions(
        internal_name,
        fit["result"].x,
        fit["sources"],
        heldout,
        stage="heldout",
    )
    predictions = pd.concat([training_predictions, heldout_predictions], ignore_index=True)
    predictions["model"] = display_name
    training_metrics = raw_score(training_predictions, lens.sigma, free_parameters=20)
    heldout_metrics = raw_score(heldout_predictions, lens.sigma)
    bound_flags = near_bound(internal_name, fit["result"].x)
    parameters = pd.DataFrame(
        {
            "model": display_name,
            "parameter": spec_for(internal_name).labels,
            "value": fit["result"].x,
            "near_bound": [bound_flags[name] for name in spec_for(internal_name).labels],
        }
    )
    row = {
        "model": display_name,
        "training_RMS_arcsec": training_metrics["exact_radial_RMS_arcsec"],
        "heldout_RMS_arcsec": heldout_metrics["exact_radial_RMS_arcsec"],
        "training_roots_converged": training_metrics["converged_roots"],
        "heldout_roots_converged": heldout_metrics["converged_roots"],
        "heldout_all_roots_converged": heldout_metrics["all_roots_converged"],
        "maximum_heldout_residual_arcsec": heldout_metrics["maximum_radial_residual_arcsec"],
        "heldout_reduced_chi2": heldout_metrics["reduced_chi2"],
        "optimizer_success": bool(fit["result"].success),
        "optimizer_cost": float(fit["result"].cost),
        "any_geometry_near_bound": bool(any(bound_flags.values())),
    }
    return row, predictions, parameters


def make_figure(
    images: pd.DataFrame,
    profiles: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
):
    colors = {"fixed_RAR": "#777777", "P0599_potential_shape": "#1261A0"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for model, block in profiles.groupby("model", sort=False):
        axes[0].loglog(
            block.radius_kpc,
            block.effective_acceleration_m_s2,
            color=colors[model],
            label=model.replace("_", " "),
        )
    scale = 3.741653570564318
    axes[0].axvspan(
        images.radius_arcsec.min() * scale,
        images.radius_arcsec.max() * scale,
        color="black",
        alpha=0.08,
        label="image radii",
    )
    axes[0].set(xlabel="radius (kpc)", ylabel="effective acceleration (m/s²)", title="Frozen radial fields")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)

    held = predictions[predictions.stage.eq("heldout")]
    observed = held.drop_duplicates("image_id")
    axes[1].scatter(observed.observed_x_arcsec, observed.observed_y_arcsec, c="black", s=28, label="observed")
    for model, block in held.groupby("model", sort=False):
        axes[1].scatter(
            block.predicted_x_arcsec,
            block.predicted_y_arcsec,
            color=colors[model],
            marker="x",
            s=42,
            label=model.replace("_", " "),
        )
    axes[1].set(xlabel="east offset (arcsec)", ylabel="north offset (arcsec)", title="Held-out image roots")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=7)

    axes[2].bar(
        [name.replace("_", "\n") for name in scores.model],
        scores.heldout_RMS_arcsec,
        color=[colors[name] for name in scores.model],
    )
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1, label="predeclared 0.5 arcsec gate")
    axes[2].set(ylabel="held-out radial RMS (arcsec)", title="Raw predictive score")
    axes[2].legend(fontsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    protocol_path = ROOT / "configs/p0601_frozen_potential_raw_lensing_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw_protocol = json.loads(
        (ROOT / protocol["inputs"]["raw_lensing_protocol"]).read_text(encoding="utf-8")
    )
    images = load_images(raw_protocol)
    heldout_ids = set(raw_protocol["predictive_split"]["heldout"])
    training = images[~images.image_id.isin(heldout_ids)].copy()
    heldout = images[images.image_id.isin(heldout_ids)].copy()
    if len(training) != 15 or len(heldout) != 7:
        raise RuntimeError("P0601 raw split changed")
    anchors = load_baryonic_anchors(raw_protocol)
    fields, profiles, field_diagnostics = build_fields(
        anchors, raw_protocol, protocol["constants"]
    )

    previous_parameters = pd.read_csv(ROOT / protocol["inputs"]["previous_raw_parameters"])
    previous_block = previous_parameters[
        previous_parameters.stage.eq("training")
        & previous_parameters.model.eq("locked_universal_candidate")
    ].set_index("parameter")
    initial = previous_block.loc[list(spec_for("fixed").labels), "value"].to_numpy(float)

    score_rows = []
    prediction_frames = []
    parameter_frames = []
    starts = int(protocol["validation"]["optimization_starts"])
    seed = int(raw_protocol["optimization"]["random_seed"]) + 60100
    for offset, name in enumerate(("fixed_RAR", "P0599_potential_shape")):
        row, predictions, parameters = fit_model(
            name,
            fields[name],
            raw_protocol,
            training,
            heldout,
            initial,
            starts=starts,
            seed=seed + offset,
        )
        score_rows.append(row)
        prediction_frames.append(predictions)
        parameter_frames.append(parameters)

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    previous_report = json.loads(
        (ROOT / protocol["inputs"]["previous_raw_report"]).read_text(encoding="utf-8")
    )
    previous_models = previous_report["model_scores"]
    comparators = {
        "previous_locked_candidate_heldout_RMS_arcsec": previous_models[
            "locked_universal_candidate"
        ]["heldout"]["exact_radial_RMS_arcsec"],
        "compact_one_halo_heldout_RMS_arcsec": previous_models["GR_plus_cluster_halo"][
            "heldout"
        ]["exact_radial_RMS_arcsec"],
        "prior_P0554_heldout_RMS_arcsec": 1.245,
        "published_multi_halo_reference_RMS_arcsec": 0.29,
    }
    new_row = scores.set_index("model").loc["P0599_potential_shape"]
    fixed_row = scores.set_index("model").loc["fixed_RAR"]
    gates = {
        "all_heldout_roots_converged": bool(new_row.heldout_all_roots_converged),
        "heldout_RMS_arcsec_max": bool(new_row.heldout_RMS_arcsec <= 0.5),
        "heldout_RMS_below_previous_locked_candidate": bool(
            new_row.heldout_RMS_arcsec
            < comparators["previous_locked_candidate_heldout_RMS_arcsec"]
        ),
        "no_geometry_parameter_near_bound": bool(not new_row.any_geometry_near_bound),
    }
    report = {
        "report_version": "P0601-FROZEN-POTENTIAL-RAW-LENSING-RESULTS-0.1.0",
        "status": "complete_frozen_single_cluster_raw_replay",
        "coverage": {
            "cluster": "RX J2129.7+0005",
            "training_images": len(training),
            "heldout_images": len(heldout),
            "source_families": int(images.source_family.nunique()),
            "optimization_starts_per_law": starts,
            "fitted_gravity_parameters": 0,
            "fitted_structural_geometry_parameters": 6,
        },
        "frozen_formula": protocol["frozen_laws"]["P0599_potential_shape"],
        "frozen_constants": protocol["constants"],
        "field_diagnostics": field_diagnostics,
        "scores": scores.to_dict(orient="records"),
        "comparators": comparators,
        "P0599_vs_fixed_RAR_heldout_RMS_ratio": float(
            new_row.heldout_RMS_arcsec / fixed_row.heldout_RMS_arcsec
        )
        if np.isfinite(fixed_row.heldout_RMS_arcsec)
        else None,
        "P0599_vs_previous_locked_candidate_heldout_RMS_ratio": float(
            new_row.heldout_RMS_arcsec
            / comparators["previous_locked_candidate_heldout_RMS_arcsec"]
        ),
        "advance_gate_audit": {**gates, "passes_all": bool(all(gates.values()))},
        "strict_interpretation": {
            "amplitude_or_gravity_parameter_fit_to_RXJ2129": False,
            "raw_image_positions_used": True,
            "independent_cluster_population_validation": False,
            "covariant_field_equation_derived": False,
            "publication_grade_dark_matter_comparator": False,
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
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(
        images,
        profiles,
        predictions,
        scores,
        output / protocol["outputs"]["figure"],
    )
    summary = (
        "# P0601 frozen potential raw-lensing replay\n\n"
        f"P0599 held-out RMS: **{new_row.heldout_RMS_arcsec:.4f} arcsec** "
        f"({int(new_row.heldout_roots_converged)}/7 roots).\n\n"
        f"Fixed RAR held-out RMS: **{fixed_row.heldout_RMS_arcsec:.4f} arcsec** "
        f"({int(fixed_row.heldout_roots_converged)}/7 roots).\n\n"
        f"All predeclared gates pass: **{all(gates.values())}**.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
