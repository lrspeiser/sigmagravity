#!/usr/bin/env python3
"""Test source-redshift scaling of the frozen P0607 angular route."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import MODEL, baryon_field, exact_fit, load_sources  # noqa: E402
from run_p0554_all_baryon_route_screen import prepare_xray_maps  # noqa: E402
from run_p0601_frozen_potential_raw_lensing import build_fields as build_p0599_fields, json_safe  # noqa: E402
from run_p0607_component_direction_raw_lensing import (  # noqa: E402
    component_fields,
    evaluate_fixed,
    fixed_geometry,
)
from run_rxj2129_member_geometry import split_images  # noqa: E402
from run_rxj2129_raw_theory_lensing import RawLens, load_baryonic_anchors, load_images  # noqa: E402
from voidscreen.baryon_morphology import map_attraction_directions  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TomographicRouteLens(RawLens):
    """Radial parent plus a route with generalized distance-ratio scaling."""

    def __init__(self, protocol, fields, *, parent, morphology, strength, gamma):
        super().__init__(protocol, fields)
        self.parent = parent
        self.morphology = morphology
        self.strength = float(strength)
        self.gamma = float(gamma)

    def alpha(self, model, parameters, x_arcsec, y_arcsec, source_redshift):
        base_x, base_y = super().alpha(
            self.parent, parameters, x_arcsec, y_arcsec, source_redshift
        )
        ratio = self.distance_ratio(float(source_redshift))
        relative = ratio / self.distance_ratio_ref
        effective_ratio = self.distance_ratio_ref * relative**self.gamma
        correction_x, correction_y = self.morphology.alpha_arcsec(
            x_arcsec, y_arcsec, distance_ratio=effective_ratio
        )
        return (
            base_x + self.strength * correction_x,
            base_y + self.strength * correction_y,
        )


def family_summary(predictions, variant_id):
    rows = []
    selected = predictions[predictions.variant_id.eq(variant_id)]
    for (stage, family, redshift), block in selected.groupby(
        ["stage", "source_family", "source_redshift"], sort=True
    ):
        finite = block[block.root_converged.astype(bool)]
        rows.append(
            {
                "variant_id": variant_id,
                "stage": stage,
                "source_family": int(family),
                "source_redshift": float(redshift),
                "images": len(block),
                "roots_converged": len(finite),
                "RMS_arcsec": float(np.sqrt(np.mean(np.square(finite.radial_residual_arcsec))))
                if len(finite) == len(block)
                else np.inf,
            }
        )
    return rows


def main() -> None:
    config_path = ROOT / "configs/p0608_route_redshift_tomography_protocol.json"
    protocol = read_json(config_path)
    if not protocol["status"].startswith("frozen_"):
        raise RuntimeError("P0608 protocol is not frozen")
    inputs = protocol["inputs"]
    p0607 = read_json(ROOT / inputs["P0607_report"])
    locked = p0607["fixed_geometry_training_selection"]["best_positive_route"]
    expected = protocol["formula"]
    if locked["component_id"] != protocol["selection"]["component_locked_from_P0607"]:
        raise RuntimeError("P0607 component changed")
    if not np.isclose(float(locked["angular_strength"]), float(expected["reference_strength"])):
        raise RuntimeError("P0607 route strength changed")

    raw_protocol = read_json(ROOT / inputs["raw_protocol"])
    p0601_protocol = read_json(ROOT / inputs["P0601_protocol"])
    p0607_protocol = read_json(ROOT / inputs["P0607_protocol"])
    source_protocol = read_json(ROOT / inputs["route_source_protocol"])
    screen_protocol = read_json(ROOT / inputs["component_screen_protocol"])
    acquisition = read_json(ROOT / inputs["component_acquisition_protocol"])
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    radial_fields, _, _ = build_p0599_fields(anchors, raw_protocol, p0601_protocol["constants"])
    parent = radial_fields["P0599_potential_shape"]
    baryons = baryon_field(anchors, raw_protocol)
    initial = fixed_geometry(ROOT / inputs["P0601_parameters"])
    sources = load_sources(source_protocol, raw_protocol)

    settings = screen_protocol["map_construction"]
    map_axis = np.arange(
        float(settings["axis_min_arcsec"]),
        float(settings["axis_max_arcsec"]) + 0.5 * float(settings["grid_spacing_arcsec"]),
        float(settings["grid_spacing_arcsec"]),
    )
    context = SimpleNamespace(label="RXJ2129", local=raw_protocol)
    _, gas_map, gas_audit = prepare_xray_maps(screen_protocol, acquisition, context, map_axis)
    xy = sources[["x_arcsec", "y_arcsec"]].to_numpy(float)
    scale = float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    route = p0607_protocol["route_geometry"]
    gas_direction, direction_audit = map_attraction_directions(
        map_axis,
        gas_map,
        xy,
        softening=float(route["direction_softening_kpc"]) / scale,
        distance_power=float(route["direction_distance_power"]),
    )
    fields, _, realized = component_fields(
        p0607_protocol,
        raw_protocol,
        sources,
        parent,
        baryons,
        {"gas": gas_direction},
    )
    gas_field = fields["gas"]

    unique_redshifts = np.sort(images.source_redshift.unique())
    geometry_lens = RawLens(raw_protocol, {MODEL: parent})
    reference_ratio = geometry_lens.distance_ratio_ref
    distance_rows = []
    for redshift in unique_redshifts:
        ratio = geometry_lens.distance_ratio(float(redshift))
        row = {
            "source_redshift": float(redshift),
            "distance_ratio": ratio,
            "relative_to_reference": ratio / reference_ratio,
        }
        for gamma in protocol["primary_gamma_grid"]:
            row[f"scale_gamma_{gamma:g}"] = (ratio / reference_ratio) ** float(gamma)
        distance_rows.append(row)
    distances = pd.DataFrame(distance_rows)

    strengths = [float(expected["reference_strength"])] + [
        float(value) for value in protocol["amplitude_leverage_diagnostics"]
    ]
    screen_rows, prediction_frames = [], []
    for strength in strengths:
        role = "primary_locked_strength" if np.isclose(strength, expected["reference_strength"]) else "amplitude_leverage_only"
        for gamma in protocol["primary_gamma_grid"]:
            variant_id = f"s{strength:g}__gamma{gamma:g}"
            lens = TomographicRouteLens(
                raw_protocol,
                {MODEL: parent},
                parent=MODEL,
                morphology=gas_field,
                strength=strength,
                gamma=float(gamma),
            )
            metrics, predictions = evaluate_fixed(
                lens, training, heldout, initial, variant_id
            )
            screen_rows.append(
                {
                    "variant_id": variant_id,
                    "role": role,
                    "angular_strength": strength,
                    "gamma": float(gamma),
                    **metrics,
                }
            )
            if not predictions.empty:
                prediction_frames.append(predictions)
            print(
                f"{variant_id}: train={metrics['training_RMS_arcsec']:.6f} "
                f"held={metrics['heldout_RMS_arcsec']:.6f}",
                flush=True,
            )
    screen = pd.DataFrame(screen_rows)
    primary = screen[
        screen.role.eq("primary_locked_strength")
        & screen.training_roots_converged.eq(len(training))
        & np.isfinite(screen.training_RMS_arcsec)
    ]
    selected = primary.sort_values(["training_RMS_arcsec", "gamma"]).iloc[0]
    standard = primary[primary.gamma.eq(float(expected["standard_single_plane_value"]))].iloc[0]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    family_rows = []
    for variant_id in {str(selected.variant_id), str(standard.variant_id)}:
        family_rows.extend(family_summary(predictions, variant_id))
    families = pd.DataFrame(family_rows)

    refit_rows = []
    for offset, (role, gamma) in enumerate(
        (("training_selected_gamma", float(selected.gamma)), ("standard_single_plane_gamma", 1.0))
    ):
        lens = TomographicRouteLens(
            raw_protocol,
            {MODEL: parent},
            parent=MODEL,
            morphology=gas_field,
            strength=float(expected["reference_strength"]),
            gamma=gamma,
        )
        fitted = exact_fit(
            lens,
            training,
            heldout,
            initial=initial,
            starts=int(protocol["selection"]["selected_gamma_exact_refit_starts"]),
            seed=int(protocol["selection"]["random_seed"]) + offset,
        )
        refit_rows.append(
            {
                "role": role,
                "gamma": gamma,
                "angular_strength": float(expected["reference_strength"]),
                "training_RMS_arcsec": fitted["training_score"]["exact_radial_RMS_arcsec"],
                "training_roots_converged": fitted["training_score"]["converged_roots"],
                "heldout_RMS_arcsec": fitted["heldout_score"]["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": fitted["heldout_score"]["converged_roots"],
                "optimizer_cost": fitted["optimizer_cost"],
            }
        )
    previous = read_json(ROOT / inputs["P0601_report"])
    previous_row = next(row for row in previous["scores"] if row["model"] == "P0599_potential_shape")
    refit_rows.insert(
        0,
        {
            "role": "P0599_no_route_reference",
            "gamma": np.nan,
            "angular_strength": 0.0,
            "training_RMS_arcsec": previous_row["training_RMS_arcsec"],
            "training_roots_converged": previous_row["training_roots_converged"],
            "heldout_RMS_arcsec": previous_row["heldout_RMS_arcsec"],
            "heldout_roots_converged": previous_row["heldout_roots_converged"],
            "optimizer_cost": previous_row["optimizer_cost"],
        },
    )
    refits = pd.DataFrame(refit_rows)

    primary_train_span = float(primary.training_RMS_arcsec.max() - primary.training_RMS_arcsec.min())
    primary_held_span = float(primary.heldout_RMS_arcsec.max() - primary.heldout_RMS_arcsec.min())
    leverage = []
    for strength, block in screen.groupby("angular_strength"):
        leverage.append(
            {
                "angular_strength": float(strength),
                "training_RMS_gamma_span_arcsec": float(block.training_RMS_arcsec.max() - block.training_RMS_arcsec.min()),
                "heldout_RMS_gamma_span_arcsec": float(block.heldout_RMS_arcsec.max() - block.heldout_RMS_arcsec.min()),
            }
        )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    distances.to_csv(output / protocol["outputs"]["distance_ratios"], index=False)
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    families.to_csv(output / protocol["outputs"]["family_scores"], index=False)
    refits.to_csv(output / protocol["outputs"]["refit_scores"], index=False)
    selected_refit = refits[refits.role.eq("training_selected_gamma")].iloc[0]
    standard_refit = refits[refits.role.eq("standard_single_plane_gamma")].iloc[0]
    report = {
        "report_version": "P0608-ROUTE-REDSHIFT-TOMOGRAPHY-RESULTS-0.1.0",
        "status": "complete_spent_source_redshift_arc_identifiability_test",
        "coverage": {
            "source_redshifts": len(unique_redshifts),
            "minimum_source_redshift": float(np.min(unique_redshifts)),
            "maximum_source_redshift": float(np.max(unique_redshifts)),
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "primary_gammas": len(protocol["primary_gamma_grid"]),
            "screen_variants": len(screen),
        },
        "locked_route": {
            "component": "gas",
            "angular_strength": float(expected["reference_strength"]),
            "R80_kpc": realized["R80_kpc"],
            "Chandra_exposure_ks": float(gas_audit["total_exposure_ks"]),
            "gas_map_centroid_arcsec": np.asarray(direction_audit["map_centroid"]).tolist(),
        },
        "distance_ratio_leverage": {
            "reference_distance_ratio": reference_ratio,
            "minimum_relative_ratio": float(distances.relative_to_reference.min()),
            "maximum_relative_ratio": float(distances.relative_to_reference.max()),
        },
        "training_selected": selected.to_dict(),
        "standard_single_plane_screen": standard.to_dict(),
        "primary_gamma_response": {
            "training_RMS_span_arcsec": primary_train_span,
            "heldout_RMS_span_arcsec": primary_held_span,
            "training_fractional_span": primary_train_span / float(primary.training_RMS_arcsec.min()),
            "heldout_fractional_span": primary_held_span / float(primary.heldout_RMS_arcsec.min()),
        },
        "amplitude_leverage": leverage,
        "exact_refits": refits.to_dict("records"),
        "interpretation": {
            "selected_gamma_differs_from_standard": bool(float(selected.gamma) != 1.0),
            "selected_gamma_beats_standard_training_after_refit": bool(selected_refit.training_RMS_arcsec < standard_refit.training_RMS_arcsec),
            "selected_gamma_beats_standard_spent_heldout_after_refit": bool(selected_refit.heldout_RMS_arcsec < standard_refit.heldout_RMS_arcsec),
            "gamma_identified": bool(primary_train_span > 0.01),
            "hidden_arc_height_identified": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    redshift_grid = np.linspace(float(unique_redshifts.min()), float(unique_redshifts.max()), 100)
    ratio_grid = np.asarray([geometry_lens.distance_ratio(value) for value in redshift_grid]) / reference_ratio
    for gamma in protocol["primary_gamma_grid"]:
        axes[0].plot(redshift_grid, ratio_grid ** float(gamma), label=f"gamma={gamma:g}")
    axes[0].scatter(distances.source_redshift, distances.relative_to_reference, color="black", s=18, label="standard beta")
    axes[0].set(xlabel="source redshift", ylabel="route scale / reference", title="Tomographic scaling family")
    axes[0].legend(fontsize=7)
    for strength, block in screen.groupby("angular_strength"):
        ordered = block.sort_values("gamma")
        axes[1].plot(ordered.gamma, ordered.training_RMS_arcsec, marker="o", label=f"s={strength:g}")
        axes[2].plot(ordered.gamma, ordered.heldout_RMS_arcsec, marker="o", label=f"s={strength:g}")
    axes[1].set(xlabel="redshift exponent gamma", ylabel="training RMS (arcsec)", title="Fixed-geometry training response")
    axes[2].set(xlabel="redshift exponent gamma", ylabel="spent held-out RMS (arcsec)", title="Held-out diagnostic")
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    (output / protocol["outputs"]["summary"]).write_text(
        "# P0608 source-redshift route tomography\n\n"
        f"At the locked gas-route strength s_theta={expected['reference_strength']}, training selected gamma={selected.gamma:g}. "
        f"The complete gamma grid changes training RMS by only {primary_train_span:.6f} arcsec and spent held-out RMS by {primary_held_span:.6f} arcsec.\n\n"
        f"After exact refits, selected/standard-gamma held-out RMS values are {selected_refit.heldout_RMS_arcsec:.6f}/{standard_refit.heldout_RMS_arcsec:.6f} arcsec. "
        "This is an identifiability test on spent data, not an arc-height measurement.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe({"selected": report["training_selected"], "response": report["primary_gamma_response"], "refits": report["exact_refits"], "interpretation": report["interpretation"]}), indent=2))


if __name__ == "__main__":
    main()
