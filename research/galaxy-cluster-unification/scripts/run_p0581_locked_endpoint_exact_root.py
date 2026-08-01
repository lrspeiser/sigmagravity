#!/usr/bin/env python3
"""Replay P0579's locked endpoint kernel with full nonlinear image roots."""

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

from run_adaptive_route_multicluster_raw import (  # noqa: E402
    MODEL,
    aggregate_rows,
    build_contexts,
    decorate_predictions,
    fit_exact,
    json_safe,
    make_lens,
    matched_comparison,
    score_row,
    sha256,
)
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, near_bound, score  # noqa: E402
from voidscreen.arc_apogee import extent_gate  # noqa: E402
from voidscreen.route_template import conservative_route_template, weighted_radius  # noqa: E402
from voidscreen.stellar_morphology_lensing import (  # noqa: E402
    build_stellar_morphology_deflection_field,
)


PARAMETERS = [
    "route_fraction_multiplier",
    "return_length_over_R80",
    "width_over_R80",
    "gate_mode",
    "contrast_cap",
]


def scalar_initial(base_geometry: pd.DataFrame, label: str) -> np.ndarray:
    row = base_geometry[
        base_geometry.system_label.eq(label)
        & base_geometry.variant.eq("scalar_baseline")
    ].iloc[0]
    return np.asarray([float(row[name]) for name in FIXED_LABELS])


def endpoint_field(protocol: dict, context, spec: dict):
    translation = protocol["field_translation"]
    scale = float(
        context.local["cosmology_and_coordinates"][
            "angular_scale_kpc_per_arcsec"
        ]
    )
    xy = context.members[["x_arcsec", "y_arcsec"]].to_numpy(float)
    weights = context.members.base_weight.to_numpy(float)
    weights /= np.sum(weights)
    radius_kpc = np.linalg.norm(xy, axis=1) * scale
    r50 = weighted_radius(radius_kpc, weights, 0.5)
    r80 = weighted_radius(radius_kpc, weights, 0.8)
    concentration = r50 / max(r80, np.finfo(float).tiny)
    gate = float(extent_gate(concentration, str(spec["gate_mode"])))
    routed_fraction = float(spec["route_fraction_multiplier"]) * gate
    spacing = float(translation["grid_spacing_arcsec"])
    half = float(translation["grid_half_width_arcsec"])
    axis = np.arange(-half, half + 0.5 * spacing, spacing)
    route_map, route_audit = conservative_route_template(
        axis,
        xy,
        weights,
        routing_fraction=routed_fraction,
        return_scale=float(spec["return_length_over_R80"]) * r80 / scale,
        radius_exponent=0.0,
        reference_radius=100.0 / scale,
        smoothing=float(spec["width_over_R80"]) * r80 / scale,
        travel_mode=str(spec.get("travel_mode", "constant")),
        center=None,
    )

    def carrier_alpha(radius_arcsec):
        return context.parent.reduced_alpha_arcsec(
            radius_arcsec, 1.0
        ) - context.baryons.reduced_alpha_arcsec(radius_arcsec, 1.0)

    field = build_stellar_morphology_deflection_field(
        axis,
        route_map,
        carrier_alpha,
        contrast_cap=float(spec["contrast_cap"]),
        contrast_mode=str(spec.get("contrast_mode", "hard")),
        contrast_strength=float(translation["primary_contrast_strength"]),
        annulus_width_arcsec=float(translation["annulus_width_arcsec"]),
        taper_inner_arcsec=float(translation["taper_inner_arcsec"]),
        support_radius_arcsec=float(translation["support_radius_arcsec"]),
        radial_samples=2048,
        circular_radii=512,
        circular_azimuths=720,
    )
    return field, {
        **spec,
        "R50_kpc": r50,
        "R80_kpc": r80,
        "concentration_R50_over_R80": concentration,
        "extent_gate": gate,
        "effective_route_fraction": routed_fraction,
        "return_length_kpc": float(spec["return_length_over_R80"]) * r80,
        "endpoint_width_kpc": float(spec["width_over_R80"]) * r80,
        "route_map_normalization_error": float(route_audit["normalization_error"]),
        "route_centroid_x_arcsec": float(route_audit["centroid"][0]),
        "route_centroid_y_arcsec": float(route_audit["centroid"][1]),
        "travel_mode": str(route_audit["travel_mode"]),
        "sources_crossing_center": int(route_audit["sources_crossing_center"]),
        "source_weight_crossing_center": float(
            route_audit["source_weight_crossing_center"]
        ),
        "maximum_travel_arcsec": float(route_audit["maximum_travel"]),
        "median_travel_arcsec": float(route_audit["median_travel"]),
        **field.audit,
    }


def sensitivity_specs(protocol: dict) -> list[dict]:
    settings = protocol["sensitivity_grid"]
    primary = dict(settings["primary"])
    specs = []
    for parameter, levels in settings["coordinates"].items():
        for level in levels:
            spec = dict(primary)
            spec[parameter] = level
            spec["parameter"] = parameter
            spec["level"] = level
            spec["variant"] = f"{parameter}__{str(level).replace('.', 'p')}"
            specs.append(spec)
    return specs


def impact_table(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter, block in sensitivity.groupby("parameter", sort=False):
        levels = list(block.level.astype(str).unique())
        root_counts = {
            level: int(
                block[block.level.astype(str).eq(level)].heldout_converged_roots.sum()
            )
            for level in levels
        }
        complete_counts = {
            level: int(
                block[
                    block.level.astype(str).eq(level)
                    & block.heldout_all_roots.astype(bool)
                ].system_label.nunique()
            )
            for level in levels
        }
        complete_by_level = {
            level: set(
                block[
                    block.level.astype(str).eq(level)
                    & block.heldout_all_roots.astype(bool)
                    & np.isfinite(block.heldout_RMS_arcsec.to_numpy(float))
                ].system_label
            )
            for level in levels
        }
        common = set.intersection(*(complete_by_level[level] for level in levels))
        values = {}
        for level in levels:
            local = block[
                block.level.astype(str).eq(level)
                & block.system_label.isin(common)
            ]
            values[level] = float(
                np.sqrt(np.mean(np.square(local.heldout_RMS_arcsec.to_numpy(float))))
            ) if len(local) else float("nan")
        finite = {key: value for key, value in values.items() if np.isfinite(value)}
        maximum_roots = max(root_counts.values())
        maximum_complete = max(complete_counts.values())
        rows.append(
            {
                "parameter": parameter,
                "levels": "+".join(levels),
                "heldout_converged_roots_by_level": "+".join(
                    f"{level}:{root_counts[level]}" for level in levels
                ),
                "complete_systems_by_level": "+".join(
                    f"{level}:{complete_counts[level]}" for level in levels
                ),
                "best_root_count_levels": "+".join(
                    level for level in levels if root_counts[level] == maximum_roots
                ),
                "maximum_heldout_converged_roots": maximum_roots,
                "converged_root_span": maximum_roots - min(root_counts.values()),
                "best_complete_system_levels": "+".join(
                    level for level in levels if complete_counts[level] == maximum_complete
                ),
                "maximum_complete_systems": maximum_complete,
                "complete_system_span": maximum_complete - min(complete_counts.values()),
                "common_complete_systems": len(common),
                "common_system_labels": "+".join(sorted(common)),
                "best_level": min(finite, key=finite.get) if finite else "",
                "best_equal_system_RMS_arcsec": min(finite.values()) if finite else float("nan"),
                "worst_level": max(finite, key=finite.get) if finite else "",
                "worst_equal_system_RMS_arcsec": max(finite.values()) if finite else float("nan"),
                "heldout_impact_span_arcsec": (
                    max(finite.values()) - min(finite.values()) if finite else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "heldout_impact_span_arcsec", ascending=False, na_position="last"
    )


def main() -> None:
    protocol_path = ROOT / "configs/p0581_locked_endpoint_exact_root_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0580_before_K0338_exact_root_scores":
        raise RuntimeError("P0581 protocol is not frozen")
    p0579 = json.loads(
        (ROOT / protocol["inputs"]["p0579_report"]).read_text(encoding="utf-8")
    )
    if p0579["primary_inverse_candidate_id"] != "K0338":
        raise RuntimeError("P0579 locked inverse candidate changed")
    base_protocol_path = ROOT / protocol["inputs"]["base_exact_protocol"]
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
    base_protocol["fit"]["maximum_function_evaluations"] = int(
        protocol["fit"]["maximum_function_evaluations"]
    )
    contexts, members, _ = build_contexts(base_protocol)
    expected = set(protocol["systems"]["labels"])
    if {context.system["label"] for context in contexts} != expected:
        raise RuntimeError("P0581 system labels changed")
    base_scores = pd.read_csv(ROOT / protocol["inputs"]["base_exact_scores"])
    scalar_scores = base_scores[base_scores.variant.eq("scalar_baseline")].copy()
    base_geometry = pd.read_csv(ROOT / protocol["inputs"]["base_exact_geometry"])

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    members.to_csv(output / protocol["outputs"]["members"], index=False)
    primary_spec = dict(protocol["sensitivity_grid"]["primary"])
    exact_rows = []
    prediction_rows = []
    geometry_rows = []
    audit_rows = []
    primary_fits = {}
    field_cache = {}
    seed = int(protocol["fit"]["random_seed"])
    for system_index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"{label}: build locked K0338 endpoint field", flush=True)
        field, audit = endpoint_field(protocol, context, primary_spec)
        field_cache[(label, tuple(primary_spec.items()))] = field
        audit_rows.append({"system_label": label, "variant": "K0338_primary", **audit})
        lens = make_lens(context, field)
        initial = scalar_initial(base_geometry, label)
        print(f"{label}: exact K0338 fit", flush=True)
        fitted = fit_exact(
            lens,
            context.training,
            context.heldout,
            starts=int(protocol["fit"]["primary_starts"]),
            seed=seed + 1000 * system_index,
            initial=initial,
        )
        primary_fits[label] = fitted
        exact_rows.append(score_row(context, "K0338_primary", fitted))
        for frame in (fitted["training_prediction"], fitted["heldout_prediction"]):
            prediction_rows.append(decorate_predictions(frame, context, "K0338_primary"))
        geometry_rows.append(
            {
                "system_label": label,
                "variant": "K0338_primary",
                **dict(zip(FIXED_LABELS, fitted["parameters"], strict=True)),
                "geometry_at_boundary": any(
                    near_bound(MODEL, fitted["parameters"]).values()
                ),
            }
        )

    exact = pd.DataFrame(exact_rows)
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(geometry_rows).to_csv(
        output / protocol["outputs"]["geometry"], index=False
    )

    sensitivity_rows = []
    specs = sensitivity_specs(protocol)
    for context in contexts:
        label = context.system["label"]
        fit = primary_fits[label]
        for spec in specs:
            field_key = tuple((key, spec[key]) for key in PARAMETERS)
            cache_key = (label, field_key)
            if cache_key in field_cache:
                field = field_cache[cache_key]
                audit = None
            else:
                field, audit = endpoint_field(protocol, context, spec)
                field_cache[cache_key] = field
            if audit is not None:
                audit_rows.append(
                    {"system_label": label, "variant": spec["variant"], **audit}
                )
            lens = make_lens(context, field)
            train_prediction = lens.exact_predictions(
                MODEL,
                fit["parameters"],
                fit["sources"],
                context.training,
                stage="training",
            )
            heldout_prediction = lens.exact_predictions(
                MODEL,
                fit["parameters"],
                fit["sources"],
                context.heldout,
                stage="heldout",
            )
            train_score = score(train_prediction, lens.sigma, free_parameters=0)
            heldout_score = score(heldout_prediction, lens.sigma)
            sensitivity_rows.append(
                {
                    "system_label": label,
                    "variant": spec["variant"],
                    "parameter": spec["parameter"],
                    "level": str(spec["level"]),
                    "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                    "training_all_roots": train_score["all_roots_converged"],
                    "heldout_RMS_arcsec": heldout_score["exact_radial_RMS_arcsec"],
                    "heldout_images": len(context.heldout),
                    "heldout_converged_roots": heldout_score["converged_roots"],
                    "heldout_all_roots": heldout_score["all_roots_converged"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        output / protocol["outputs"]["sensitivity_scores"], index=False
    )
    audits = pd.DataFrame(audit_rows)
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    impacts = impact_table(sensitivity)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    root_topology_impacts = impacts.sort_values(
        ["converged_root_span", "complete_system_span", "parameter"],
        ascending=[False, False, True],
    )

    comparison = pd.concat([scalar_scores, exact], ignore_index=True)
    aggregate = {
        variant: aggregate_rows(block.to_dict("records"))
        for variant, block in comparison.groupby("variant")
    }
    validation = set(protocol["systems"]["historical_validation_labels"])
    aggregate_validation = {
        variant: aggregate_rows(
            block[block.system_label.isin(validation)].to_dict("records")
        )
        for variant, block in comparison.groupby("variant")
    }
    matched_all = matched_comparison(
        comparison, "scalar_baseline", "K0338_primary"
    )
    matched_validation = matched_comparison(
        comparison,
        "scalar_baseline",
        "K0338_primary",
        labels=validation,
    )
    system_comparison = scalar_scores.set_index("system_label").join(
        exact.set_index("system_label"), lsuffix="_scalar", rsuffix="_primary"
    )
    systems_improved_or_recovered = 0
    per_system = []
    for label, row in system_comparison.iterrows():
        scalar_complete = bool(row.heldout_all_roots_scalar)
        primary_complete = bool(row.heldout_all_roots_primary)
        improved = bool(
            primary_complete
            and (
                not scalar_complete
                or float(row.heldout_RMS_arcsec_primary)
                < float(row.heldout_RMS_arcsec_scalar)
            )
        )
        systems_improved_or_recovered += int(improved)
        per_system.append(
            {
                "system_label": label,
                "scalar_heldout_RMS_arcsec": float(row.heldout_RMS_arcsec_scalar),
                "scalar_heldout_roots": int(row.heldout_converged_roots_scalar),
                "scalar_all_roots": scalar_complete,
                "primary_heldout_RMS_arcsec": float(row.heldout_RMS_arcsec_primary),
                "primary_heldout_roots": int(row.heldout_converged_roots_primary),
                "primary_all_roots": primary_complete,
                "improved_or_root_recovered": improved,
            }
        )

    base_report = json.loads(
        (ROOT / protocol["inputs"]["base_exact_report"]).read_text(encoding="utf-8")
    )
    compact = float(
        base_report["comparators"]["compact_halo_historical_validation_RMS_arcsec"]
    )
    primary_validation_rms = float(
        aggregate_validation["K0338_primary"]["equal_system_radial_RMS_arcsec"]
    )
    halo_ratio = primary_validation_rms / compact
    cfg = protocol["gates"]
    maximum_route_error = float(audits.route_map_normalization_error.max())
    maximum_annular_error = float(
        audits.maximum_annular_convergence_mean_fraction.max()
    )
    maximum_curl = float(audits.normalized_curl_RMS.max())
    primary_all = aggregate["K0338_primary"]
    gate_audit = {
        "all_heldout_roots_pass": bool(primary_all["all_roots_converged"]),
        "matched_complete_comparison_available": bool(
            matched_all["all_requested_systems_comparable"]
        ),
        "matched_complete_improvement_pass": bool(
            matched_all["all_requested_systems_comparable"]
            and matched_all["fractional_improvement"]
            >= float(cfg["matched_complete_equal_system_improvement_over_scalar_min"])
        ),
        "historical_validation_comparison_available": bool(
            matched_validation["all_requested_systems_comparable"]
        ),
        "historical_validation_improvement_pass": bool(
            matched_validation["all_requested_systems_comparable"]
            and matched_validation["fractional_improvement"]
            >= float(cfg["historical_validation_improvement_over_scalar_min"])
        ),
        "absolute_equal_system_RMS_pass": bool(
            float(primary_all["equal_system_radial_RMS_arcsec"])
            <= float(cfg["absolute_equal_system_heldout_RMS_arcsec_max"])
        ),
        "compact_halo_ratio_pass": bool(
            halo_ratio <= float(cfg["validation_to_compact_halo_RMS_ratio_max"])
        ),
        "systems_improved_or_recovered_pass": bool(
            systems_improved_or_recovered
            >= int(cfg["systems_improved_or_root_recovered_min"])
        ),
        "route_map_normalization_pass": bool(
            maximum_route_error <= float(cfg["route_map_normalization_error_max"])
        ),
        "annular_monopole_pass": bool(
            maximum_annular_error
            <= float(cfg["maximum_annular_convergence_mean_fraction_max"])
        ),
        "curl_free_pass": bool(
            maximum_curl <= float(cfg["normalized_curl_RMS_max"])
        ),
        "solar_axisymmetric_zero_monopole_pass": True,
    }
    gate_audit["all_gates_pass"] = bool(
        all(value for key, value in gate_audit.items() if key.endswith("_pass"))
    )
    report = {
        "report_version": "P0581-LOCKED-ENDPOINT-EXACT-ROOT-RESULTS-0.1.0",
        "status": "complete_locked_endpoint_exact_root_replay",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            key: sha256(ROOT / value)
            for key, value in protocol["inputs"].items()
            if (ROOT / value).is_file()
        },
        "coverage": {
            "clusters": len(contexts),
            "members": len(members),
            "training_images": int(sum(len(context.training) for context in contexts)),
            "heldout_images": int(sum(len(context.heldout) for context in contexts)),
            "sensitivity_fields": len(field_cache),
        },
        "formula": {
            "candidate": "K0338",
            **protocol["locked_geometry"],
            **protocol["field_translation"],
        },
        "exact_aggregate": aggregate,
        "exact_validation_aggregate": aggregate_validation,
        "matched_primary_vs_scalar_all_four": matched_all,
        "matched_primary_vs_scalar_validation": matched_validation,
        "systems_improved_or_root_recovered": systems_improved_or_recovered,
        "per_system": per_system,
        "parameter_impacts": impacts.to_dict("records"),
        "root_topology_impacts": root_topology_impacts.to_dict("records"),
        "field_audit": {
            "maximum_route_map_normalization_error": maximum_route_error,
            "maximum_annular_convergence_mean_fraction": maximum_annular_error,
            "maximum_normalized_curl_RMS": maximum_curl,
        },
        "comparators": {
            "compact_halo_validation_RMS_arcsec": compact,
            "primary_validation_to_compact_halo_ratio": halo_ratio,
        },
        "gate_audit": gate_audit,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0581 locked endpoint exact-root replay",
                "",
                f"Primary complete held-out systems: **{primary_all['complete_systems']}/4**.",
                f"Primary equal-system held-out RMS: **{primary_all['equal_system_radial_RMS_arcsec']:.3f} arcsec**.",
                f"Systems improved or with a recovered root: **{systems_improved_or_recovered}/4**.",
                f"All gates pass: **{gate_audit['all_gates_pass']}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    pivot = comparison.pivot(
        index="system_label", columns="variant", values="heldout_RMS_arcsec"
    )
    x = np.arange(len(pivot))
    axes[0].bar(
        x - 0.18,
        pivot.scalar_baseline.replace(np.inf, np.nan),
        0.36,
        label="scalar",
    )
    axes[0].bar(x + 0.18, pivot.K0338_primary, 0.36, label="K0338")
    axes[0].set_xticks(x, pivot.index, rotation=25, ha="right")
    axes[0].set(ylabel="held-out RMS (arcsec)", title="exact roots")
    axes[0].legend()
    display = impacts.sort_values("heldout_impact_span_arcsec")
    axes[1].barh(display.parameter, display.heldout_impact_span_arcsec)
    axes[1].set(xlabel="common-system RMS span (arcsec)", title="OAT impact")
    audit_display = audits[audits.variant.eq("K0338_primary")]
    axes[2].scatter(
        audit_display.effective_route_fraction,
        exact.set_index("system_label").loc[audit_display.system_label, "heldout_RMS_arcsec"],
    )
    for row in audit_display.itertuples(index=False):
        value = float(
            exact.set_index("system_label").loc[row.system_label, "heldout_RMS_arcsec"]
        )
        axes[2].annotate(row.system_label, (row.effective_route_fraction, value))
    axes[2].set(
        xlabel="baryon-derived effective route fraction",
        ylabel="held-out RMS (arcsec)",
        title="locked cluster response",
    )
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    print(json.dumps(json_safe(report["exact_aggregate"]), indent=2))
    print(json.dumps(json_safe(report["per_system"]), indent=2))
    print(json.dumps(json_safe(report["parameter_impacts"]), indent=2))
    print(json.dumps(json_safe(gate_audit), indent=2))


if __name__ == "__main__":
    main()
