#!/usr/bin/env python3
"""Select one universal lens parameter and transfer it across cluster holdouts."""

from __future__ import annotations

import hashlib
import json
import math
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

from run_rxj2129_raw_theory_lensing import RawLens, near_bound, score, spec_for  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    aggregate_system_scores,
    json_safe,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.one_parameter_lens import predict_one_parameter_acceleration  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_key(family: str, index: int, stage: str) -> str:
    return f"{stage}__{family}__{index:03d}"


def family_grid(protocol: dict) -> list[dict]:
    rows = []
    for family, specification in protocol["families"].items():
        for index, value in enumerate(specification["grid"]):
            rows.append(
                {
                    "model": model_key(family, index, "coarse"),
                    "family": family,
                    "parameter": float(value),
                    "grid_index": index,
                    "stage": "coarse",
                }
            )
    return rows


def acceleration_profile(
    family: str,
    parameter: float,
    radius: np.ndarray,
    anchors: pd.DataFrame,
    protocol: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius,
        anchor_radius,
        anchor_gbar,
        outer_slope=float(protocol["field"]["outer_baryonic_slope"]),
    )
    reference_radius = float(protocol["field"]["reference_radius_kpc"])
    reference_gbar = float(
        loglog_interpolate_with_tails(
            np.array([reference_radius]),
            anchor_radius,
            anchor_gbar,
            outer_slope=float(protocol["field"]["outer_baryonic_slope"]),
        )[0]
    )
    predicted = predict_one_parameter_acceleration(
        family,
        gbar,
        radius,
        parameter,
        a0_m_s2=float(protocol["field"]["a0_m_s2"]),
        reference_radius_kpc=reference_radius,
        gbar_at_reference_m_s2=reference_gbar,
    )
    return gbar, predicted, reference_gbar


def build_fields(
    definitions: list[dict],
    anchors: pd.DataFrame,
    local_protocol: dict,
    protocol: dict,
) -> tuple[dict[str, RadialDeflectionField], pd.DataFrame]:
    maximum_radius = float(protocol["field"]["maximum_radius_kpc"])
    radius_grid = np.geomspace(0.1, maximum_radius, 4096)
    scale = float(
        local_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
    )
    impact_arcsec = np.geomspace(0.05, 500.0, 700)
    impact_kpc = impact_arcsec * scale
    sample_radius = np.geomspace(1.0, min(1000.0, 0.9 * maximum_radius), 180)
    fields: dict[str, RadialDeflectionField] = {}
    profiles = []
    for definition in definitions:
        family = definition["family"]
        parameter = float(definition["parameter"])
        gbar, predicted, reference_gbar = acceleration_profile(
            family, parameter, radius_grid, anchors, protocol
        )

        def lookup(radius, values=predicted):
            return np.exp(
                np.interp(np.log(radius), np.log(radius_grid), np.log(values))
            )

        physical_alpha = spherical_deflection_radians(
            impact_kpc,
            lookup,
            maximum_radius_kpc=maximum_radius,
            integration_points=800,
        )
        fields[definition["model"]] = RadialDeflectionField(
            impact_arcsec, physical_alpha
        )
        sample_gbar, sample_predicted, _ = acceleration_profile(
            family, parameter, sample_radius, anchors, protocol
        )
        profiles.append(
            pd.DataFrame(
                {
                    "model": definition["model"],
                    "stage": definition["stage"],
                    "family": family,
                    "parameter": parameter,
                    "radius_kpc": sample_radius,
                    "gbar_m_s2": sample_gbar,
                    "predicted_acceleration_m_s2": sample_predicted,
                    "gbar_at_200kpc_m_s2": reference_gbar,
                }
            )
        )
    return fields, pd.concat(profiles, ignore_index=True)


def fit_definitions_for_system(
    system: dict,
    system_index: int,
    definitions: list[dict],
    base_protocol: dict,
    protocol: dict,
    catalog: pd.DataFrame,
    tian: pd.DataFrame,
    *,
    starts: int,
    seed_offset: int,
) -> tuple[dict, list[pd.DataFrame], list[dict], pd.DataFrame]:
    local = system_protocol(base_protocol, system)
    local["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    images = load_system_images(catalog, system)
    training, heldout = predictive_split(images)
    anchors = load_anchors(tian, system["label"])
    fields, profiles = build_fields(definitions, anchors, local, protocol)
    profiles.insert(0, "system", system["system"])
    profiles.insert(1, "system_label", system["label"])
    lens = RawLens(local, fields)
    scores = {}
    predictions = []
    geometry = []
    previous_by_family: dict[str, np.ndarray] = {}
    for definition_index, definition in enumerate(definitions):
        name = definition["model"]
        family = definition["family"]
        parameter = float(definition["parameter"])
        print(
            f"system={system['label']} stage={definition['stage']} "
            f"family={family} parameter={parameter:g}",
            flush=True,
        )
        fitted = lens.fit(
            name,
            training,
            starts=starts,
            seed=int(protocol["optimization"]["random_seed"])
            + seed_offset
            + 1000 * system_index
            + definition_index,
            initial_override=previous_by_family.get(family),
        )
        previous_by_family[family] = fitted["result"].x
        train_prediction = lens.exact_predictions(
            name,
            fitted["result"].x,
            fitted["sources"],
            training,
            stage=f"{definition['stage']}_training",
        )
        heldout_prediction = lens.exact_predictions(
            name,
            fitted["result"].x,
            fitted["sources"],
            heldout,
            stage=f"{definition['stage']}_heldout",
        )
        for table in (train_prediction, heldout_prediction):
            table.insert(0, "system", system["system"])
            table.insert(1, "system_label", system["label"])
            table.insert(2, "family", family)
            table.insert(3, "parameter", parameter)
            predictions.append(table)
        heldout_score = (
            score(heldout_prediction, lens.sigma)
            if not heldout_prediction.empty
            else {"status": "no within-family holdout"}
        )
        scores[name] = {
            "family": family,
            "parameter": parameter,
            "training": score(
                train_prediction,
                lens.sigma,
                free_parameters=len(fitted["result"].x),
            ),
            "heldout": heldout_score,
            "geometry_at_boundary": near_bound(name, fitted["result"].x),
        }
        spec = spec_for(name)
        geometry.append(
            {
                "system": system["system"],
                "system_label": system["label"],
                "stage": definition["stage"],
                "family": family,
                "parameter": parameter,
                **dict(zip(spec.labels, fitted["result"].x, strict=True)),
            }
        )
    return scores, predictions, geometry, profiles


def aggregate_definition(
    system_scores: dict,
    labels: list[str],
    definition: dict,
    stage: str = "heldout",
) -> dict:
    return aggregate_system_scores(
        [system_scores[label][definition["model"]][stage] for label in labels]
    )


def boundary_count(system_scores: dict, labels: list[str], model: str) -> int:
    return sum(
        sum(system_scores[label][model]["geometry_at_boundary"].values())
        for label in labels
    )


def select_definition(
    system_scores: dict,
    labels: list[str],
    definitions: list[dict],
    protocol: dict,
) -> tuple[dict, list[dict]]:
    rows = []
    for definition in definitions:
        aggregate = aggregate_definition(system_scores, labels, definition)
        nested = float(
            protocol["families"][definition["family"]]["nested_baryons_value"]
        )
        row = {
            **definition,
            "eligible": bool(
                aggregate["all_roots_converged"]
                and aggregate.get("equal_system_radial_RMS_arcsec") is not None
            ),
            "geometry_boundary_count": boundary_count(
                system_scores, labels, definition["model"]
            ),
            "distance_from_nested_baryons": abs(
                float(definition["parameter"]) - nested
            ),
            **aggregate,
        }
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("no one-parameter grid point retained every development root")
    chosen = min(
        eligible,
        key=lambda row: (
            float(row["equal_system_radial_RMS_arcsec"]),
            int(row["geometry_boundary_count"]),
            float(row["distance_from_nested_baryons"]),
            str(row["family"]),
        ),
    )
    return chosen, rows


def refinement_definitions(coarse_winner: dict, protocol: dict) -> list[dict]:
    family = coarse_winner["family"]
    grid = np.asarray(protocol["families"][family]["grid"], dtype=float)
    index = int(coarse_winner["grid_index"])
    lower_index = max(0, index - 1)
    upper_index = min(len(grid) - 1, index + 1)
    lower = float(grid[lower_index])
    upper = float(grid[upper_index])
    if lower == upper:
        return [
            {
                "model": model_key(family, 0, "refined"),
                "family": family,
                "parameter": lower,
                "grid_index": 0,
                "stage": "refined",
            }
        ]
    values = np.linspace(lower, upper, 9)
    return [
        {
            "model": model_key(family, index, "refined"),
            "family": family,
            "parameter": float(value),
            "grid_index": index,
            "stage": "refined",
        }
        for index, value in enumerate(values)
    ]


def make_figure(report: dict, coarse: pd.DataFrame, refined: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    family_best = (
        coarse[coarse.eligible]
        .sort_values("equal_system_radial_RMS_arcsec")
        .groupby("family", as_index=False)
        .first()
        .sort_values("equal_system_radial_RMS_arcsec")
    )
    axes[0].barh(
        family_best.family,
        family_best.equal_system_radial_RMS_arcsec,
    )
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="development held-out RMS (arcsec)",
        title="Best coarse point per one-parameter family",
    )
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].plot(
        refined.parameter,
        refined.equal_system_radial_RMS_arcsec,
        marker="o",
    )
    axes[1].axvline(
        float(report["selection"]["selected_parameter"]),
        color="black",
        linestyle="--",
    )
    axes[1].set(
        xlabel="universal parameter",
        ylabel="development held-out RMS (arcsec)",
        title=f"Refinement: {report['selection']['selected_family']}",
    )
    axes[1].grid(alpha=0.2)

    labels = ["selected", "baryons", "MOND", "fixed RAR", "compact halo"]
    validation = report["validation"]["selected_law"][
        "equal_system_radial_RMS_arcsec"
    ]
    comparators = report["comparators"]
    values = [
        validation,
        comparators["baryons_GR"]["equal_system_radial_RMS_arcsec"],
        comparators["fixed_simple_MOND"]["equal_system_radial_RMS_arcsec"],
        comparators["fixed_RAR_zero_slip"]["equal_system_radial_RMS_arcsec"],
        comparators["GR_plus_compact_cluster_halo"][
            "equal_system_radial_RMS_arcsec"
        ],
    ]
    axes[2].bar(labels, values)
    axes[2].set(
        ylabel="validation held-out RMS (arcsec)",
        title="Locked two-cluster validation",
    )
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "configs/one_parameter_multicluster_lens_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_one_parameter_scores":
        raise RuntimeError("one-parameter protocol was not frozen before scoring")
    base_protocol = load_json(ROOT / protocol["inputs"]["base_protocol"])
    image_path = ROOT / protocol["inputs"]["image_catalog"]
    baryon_path = ROOT / protocol["inputs"]["baryonic_profile"]
    catalog = pd.read_csv(image_path)
    tian = pd.read_csv(
        baryon_path,
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_gbar",
            "log_gobs",
            "err_log_gbar",
            "err_log_gobs",
        ],
    )
    systems = {system["label"]: system for system in base_protocol["systems"]}
    development_labels = protocol["selection_design"]["development_clusters"]
    validation_labels = protocol["selection_design"]["cluster_holdouts"]
    coarse_definitions = family_grid(protocol)
    system_scores: dict[str, dict] = {}
    prediction_tables: list[pd.DataFrame] = []
    geometry_rows: list[dict] = []
    profile_tables: list[pd.DataFrame] = []

    for system_index, label in enumerate(development_labels):
        scores, predictions, geometry, profiles = fit_definitions_for_system(
            systems[label],
            system_index,
            coarse_definitions,
            base_protocol,
            protocol,
            catalog,
            tian,
            starts=int(protocol["optimization"]["starts_per_coarse_point"]),
            seed_offset=0,
        )
        system_scores[label] = scores
        prediction_tables.extend(predictions)
        geometry_rows.extend(geometry)
        profile_tables.append(profiles)

    coarse_winner, coarse_rows = select_definition(
        system_scores, development_labels, coarse_definitions, protocol
    )
    refined_definitions = refinement_definitions(coarse_winner, protocol)
    refined_scores: dict[str, dict] = {}
    for system_index, label in enumerate(development_labels):
        scores, predictions, geometry, profiles = fit_definitions_for_system(
            systems[label],
            system_index,
            refined_definitions,
            base_protocol,
            protocol,
            catalog,
            tian,
            starts=int(protocol["optimization"]["starts_per_refined_point"]),
            seed_offset=100_000,
        )
        refined_scores[label] = scores
        prediction_tables.extend(predictions)
        geometry_rows.extend(geometry)
        profile_tables.append(profiles)
    refined_winner, refined_rows = select_definition(
        refined_scores, development_labels, refined_definitions, protocol
    )

    selected_definition = {
        "model": model_key(refined_winner["family"], 0, "validation"),
        "family": refined_winner["family"],
        "parameter": float(refined_winner["parameter"]),
        "grid_index": 0,
        "stage": "validation",
    }
    validation_scores: dict[str, dict] = {}
    for system_index, label in enumerate(validation_labels):
        scores, predictions, geometry, profiles = fit_definitions_for_system(
            systems[label],
            system_index,
            [selected_definition],
            base_protocol,
            protocol,
            catalog,
            tian,
            starts=int(protocol["optimization"]["starts_final_validation"]),
            seed_offset=200_000,
        )
        validation_scores[label] = scores
        prediction_tables.extend(predictions)
        geometry_rows.extend(geometry)
        profile_tables.append(profiles)
    validation_aggregate = aggregate_definition(
        validation_scores, validation_labels, selected_definition
    )

    metric_report = load_json(ROOT / "results/metric_slip_raw_lensing/report.json")
    spherical_report = load_json(
        ROOT / "results/spherical_spacetime_cavity/raw_lensing_report.json"
    )
    comparators = {
        "baryons_GR": spherical_report["comparators"]["baryons_GR"],
        "fixed_simple_MOND": spherical_report["comparators"]["fixed_simple_MOND"],
        "fixed_RAR_zero_slip": {
            "systems": 2,
            "images": 6,
            "all_roots_converged": True,
            "equal_system_radial_RMS_arcsec": 25.673185979905394,
        },
        "GR_plus_compact_cluster_halo": spherical_report["comparators"][
            "GR_plus_cluster_halo"
        ],
    }
    halo_rms = comparators["GR_plus_compact_cluster_halo"][
        "equal_system_radial_RMS_arcsec"
    ]
    selected_parameter = float(selected_definition["parameter"])
    refined_values = [float(item["parameter"]) for item in refined_definitions]
    parameter_at_boundary = bool(
        np.isclose(selected_parameter, min(refined_values))
        or np.isclose(selected_parameter, max(refined_values))
    )
    per_system_improves = {}
    for label in validation_labels:
        selected_rms = validation_scores[label][selected_definition["model"]][
            "heldout"
        ]["exact_radial_RMS_arcsec"]
        base_system_name = systems[label]["system"]
        baryon_rms = load_json(ROOT / protocol["inputs"]["base_report"])[
            "system_scores"
        ][base_system_name]["baryons_GR"]["heldout"]["exact_radial_RMS_arcsec"]
        per_system_improves[label] = bool(selected_rms < baryon_rms)

    gates = {
        "all_validation_roots_converged": bool(
            validation_aggregate["all_roots_converged"]
        ),
        "validation_absolute_RMS_pass": bool(
            validation_aggregate["equal_system_radial_RMS_arcsec"]
            <= float(
                protocol["advance_gates"]["validation_equal_system_RMS_arcsec_max"]
            )
        ),
        "compact_halo_RMS_ratio": float(
            validation_aggregate["equal_system_radial_RMS_arcsec"] / halo_rms
        ),
        "compact_halo_ratio_pass": bool(
            validation_aggregate["equal_system_radial_RMS_arcsec"] / halo_rms
            <= float(protocol["advance_gates"]["compact_halo_RMS_ratio_max"])
        ),
        "per_system_improves_over_baryons": per_system_improves,
        "both_validation_clusters_improve_over_baryons": bool(
            all(per_system_improves.values())
        ),
        "selected_parameter_at_refined_boundary": parameter_at_boundary,
        "selected_parameter_interior_pass": not parameter_at_boundary,
    }
    gates["all_advance_gates_pass"] = bool(
        gates["all_validation_roots_converged"]
        and gates["validation_absolute_RMS_pass"]
        and gates["compact_halo_ratio_pass"]
        and gates["both_validation_clusters_improve_over_baryons"]
        and gates["selected_parameter_interior_pass"]
    )

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed one-universal-parameter multi-cluster replay holdout",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "input_hashes": {
            "image_catalog": sha256(image_path),
            "baryonic_profile": sha256(baryon_path),
        },
        "parameter_accounting": protocol["parameter_policy"],
        "selection_design": protocol["selection_design"],
        "coarse_selection": {
            "families": len(protocol["families"]),
            "grid_points": len(coarse_definitions),
            "winner": coarse_winner,
        },
        "selection": {
            "selected_family": selected_definition["family"],
            "selected_parameter": selected_parameter,
            "equation": protocol["families"][selected_definition["family"]][
                "equation"
            ],
            "development_heldout": {
                key: refined_winner[key]
                for key in (
                    "systems",
                    "images",
                    "all_roots_converged",
                    "equal_system_radial_RMS_arcsec",
                    "median_system_radial_RMS_arcsec",
                    "pooled_reduced_chi2",
                )
            },
        },
        "validation": {
            "selected_law": validation_aggregate,
            "per_system": {
                label: validation_scores[label][selected_definition["model"]]
                for label in validation_labels
            },
        },
        "comparators": comparators,
        "gate_audit": gates,
        "verdict": {
            "one_parameter_law_survives": gates["all_advance_gates_pass"],
            "best_possible_within_frozen_family_set": True,
            "external_validation": False,
        },
        "claim_boundary": [
            protocol["selection_design"]["disclosure"],
            "Formula family and parameter were selected only with development-cluster held-out images; the two validation-cluster heldouts were not read by this run until the law was locked.",
            "The radial baryonic input is a sparse spherical reconstruction, and pseudo-ellipticity plus shear do not replace gas, BCG, ICL, and member-galaxy mass maps.",
            "Zero slip is an imposed photon closure, not a covariant derivation.",
            "The compact halo is a deliberately limited comparator, not a state-of-the-art many-halo cluster model.",
        ],
    }

    output_dir = (ROOT / protocol["outputs"]["report"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    coarse_frame = pd.DataFrame(coarse_rows)
    refined_frame = pd.DataFrame(refined_rows)
    coarse_frame.to_csv(ROOT / protocol["outputs"]["coarse_grid"], index=False)
    refined_frame.to_csv(ROOT / protocol["outputs"]["refined_grid"], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(geometry_rows).to_csv(
        ROOT / protocol["outputs"]["geometry"], index=False
    )
    pd.concat(profile_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["radial_profiles"], index=False
    )
    make_figure(
        report,
        coarse_frame,
        refined_frame,
        ROOT / protocol["outputs"]["figure"],
    )
    lines = [
        "# One-parameter multi-cluster lens result",
        "",
        f"Selected family: **{selected_definition['family']}**",
        f"Selected universal parameter: **{selected_parameter:.8g}**",
        f"Development held-out RMS: **{refined_winner['equal_system_radial_RMS_arcsec']:.3f} arcsec**",
        f"Validation held-out RMS: **{validation_aggregate['equal_system_radial_RMS_arcsec']:.3f} arcsec**",
        f"Compact-halo validation RMS: **{halo_rms:.3f} arcsec**",
        f"Survives every frozen gate: **{gates['all_advance_gates_pass']}**",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
