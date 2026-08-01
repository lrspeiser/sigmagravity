#!/usr/bin/env python3
"""Fit one screened tail parameter on clusters and test its Solar-System transfer."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_one_parameter_multicluster_lens import (  # noqa: E402
    aggregate_definition,
    boundary_count,
    fit_definitions_for_system,
    json_safe,
    load_json,
    model_key,
)
from voidscreen.solar_system_tail import (  # noqa: E402
    AU_M,
    M_SUN_KG,
    PARSEC_M,
    fractional_extra_force,
    secular_perihelion_precession_mas_per_century,
)


PLANETS = (
    ("Mercury", 0.38709893, 0.205630, 87.9691),
    ("Venus", 0.72333199, 0.006772, 224.701),
    ("Earth", 1.00000011, 0.016710, 365.25636),
    ("Mars", 1.52366231, 0.093412, 686.980),
    ("Jupiter", 5.20336301, 0.048393, 4332.589),
    ("Saturn", 9.53707032, 0.054151, 10759.22),
    ("Uranus", 19.19126393, 0.047168, 30688.5),
    ("Neptune", 30.06896348, 0.008586, 60182.0),
)
SOLAR_RADIUS_M = 695_700_000.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(values: list[float], stage: str, family: str) -> list[dict]:
    return [
        {
            "model": model_key(family, index, stage),
            "family": family,
            "parameter": float(value),
            "grid_index": index,
            "stage": stage,
        }
        for index, value in enumerate(values)
    ]


def select(
    scores_by_label: dict,
    labels: list[str],
    candidates: list[dict],
) -> tuple[dict, list[dict]]:
    rows = []
    for candidate in candidates:
        aggregate = aggregate_definition(scores_by_label, labels, candidate)
        row = {
            **candidate,
            "eligible": bool(
                aggregate["all_roots_converged"]
                and aggregate.get("equal_system_radial_RMS_arcsec") is not None
            ),
            "geometry_boundary_count": boundary_count(
                scores_by_label, labels, candidate["model"]
            ),
            "distance_from_nested_baryons": abs(float(candidate["parameter"])),
            **aggregate,
        }
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("no screened-tail point retained every development root")
    winner = min(
        eligible,
        key=lambda row: (
            float(row["equal_system_radial_RMS_arcsec"]),
            int(row["geometry_boundary_count"]),
            float(row["distance_from_nested_baryons"]),
        ),
    )
    return winner, rows


def solar_diagnostics(
    parameter: float,
    *,
    reference_radius_m: float,
    a0_m_s2: float,
) -> pd.DataFrame:
    rows = []
    for law, screened in (("unscreened_control", False), ("screened", True)):
        for planet, axis_au, eccentricity, period_days in PLANETS:
            axis_m = axis_au * AU_M
            fraction = float(
                fractional_extra_force(
                    axis_m,
                    source_mass_kg=M_SUN_KG,
                    parameter=parameter,
                    reference_radius_m=reference_radius_m,
                    a0_m_s2=a0_m_s2,
                    screened=screened,
                )
            )
            precession = secular_perihelion_precession_mas_per_century(
                semimajor_axis_m=axis_m,
                eccentricity=eccentricity,
                orbital_period_days=period_days,
                source_mass_kg=M_SUN_KG,
                parameter=parameter,
                reference_radius_m=reference_radius_m,
                a0_m_s2=a0_m_s2,
                screened=screened,
            )
            rows.append(
                {
                    "law": law,
                    "planet": planet,
                    "parameter": parameter,
                    "semimajor_axis_AU": axis_au,
                    "eccentricity": eccentricity,
                    "fractional_extra_force_at_semimajor_axis": fraction,
                    "supplementary_precession_mas_per_century": precession,
                }
            )
    return pd.DataFrame(rows)


def run_cluster_stage(
    *,
    labels: list[str],
    candidates: list[dict],
    systems: dict,
    base_protocol: dict,
    lens_protocol: dict,
    catalog: pd.DataFrame,
    tian: pd.DataFrame,
    starts: int,
    seed_offset: int,
) -> tuple[dict, list[pd.DataFrame], list[dict], list[pd.DataFrame]]:
    stage_scores = {}
    predictions = []
    geometry = []
    profiles = []
    for system_index, label in enumerate(labels):
        local_scores, local_predictions, local_geometry, local_profiles = (
            fit_definitions_for_system(
                systems[label],
                system_index,
                candidates,
                base_protocol,
                lens_protocol,
                catalog,
                tian,
                starts=starts,
                seed_offset=seed_offset,
            )
        )
        stage_scores[label] = local_scores
        predictions.extend(local_predictions)
        geometry.extend(local_geometry)
        profiles.append(local_profiles)
    return stage_scores, predictions, geometry, profiles


def main() -> None:
    config_path = ROOT / "configs/solar_screened_isothermal_protocol.json"
    protocol = load_json(config_path)
    if protocol["status"] != "frozen_before_screened_cluster_and_solar_scores":
        raise RuntimeError("screened Solar-System protocol is not frozen")
    lens_protocol = load_json(
        ROOT / protocol["inputs"]["base_one_parameter_protocol"]
    )
    lens_protocol["optimization"] = protocol["optimization"]
    lens_protocol["field"]["a0_m_s2"] = protocol["law"]["fixed_constants"][
        "a0_m_s2"
    ]
    lens_protocol["field"]["reference_radius_kpc"] = protocol["law"][
        "fixed_constants"
    ]["reference_radius_kpc"]
    lens_protocol["field"]["maximum_radius_kpc"] = protocol["law"][
        "fixed_constants"
    ]["maximum_lens_radius_kpc"]
    lens_protocol["field"]["gravitational_slip"] = protocol["law"][
        "fixed_constants"
    ]["gravitational_slip"]
    base_protocol = load_json(ROOT / protocol["inputs"]["base_protocol"])
    base_report = load_json(ROOT / protocol["inputs"]["base_report"])
    prior_report = load_json(ROOT / protocol["inputs"]["one_parameter_report"])
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
    family = protocol["law"]["family"]
    development_labels = protocol["selection_design"]["development_clusters"]
    validation_labels = protocol["selection_design"]["cluster_holdouts"]
    stress_labels = protocol["selection_design"]["stress_only"]

    all_predictions: list[pd.DataFrame] = []
    all_geometry: list[dict] = []
    all_profiles: list[pd.DataFrame] = []

    coarse_candidates = definitions(
        [float(value) for value in protocol["law"]["coarse_grid"]],
        "coarse",
        family,
    )
    coarse_scores, predictions, geometry, profiles = run_cluster_stage(
        labels=development_labels,
        candidates=coarse_candidates,
        systems=systems,
        base_protocol=base_protocol,
        lens_protocol=lens_protocol,
        catalog=catalog,
        tian=tian,
        starts=int(protocol["optimization"]["starts_per_coarse_point"]),
        seed_offset=0,
    )
    all_predictions.extend(predictions)
    all_geometry.extend(geometry)
    all_profiles.extend(profiles)
    coarse_winner, coarse_rows = select(
        coarse_scores, development_labels, coarse_candidates
    )

    coarse_values = np.asarray(protocol["law"]["coarse_grid"], dtype=float)
    coarse_index = int(coarse_winner["grid_index"])
    lower = float(coarse_values[max(0, coarse_index - 1)])
    upper = float(coarse_values[min(len(coarse_values) - 1, coarse_index + 1)])
    refined_values = np.linspace(
        lower, upper, int(protocol["law"]["refinement_points"])
    ).tolist()
    refined_candidates = definitions(refined_values, "refined", family)
    refined_scores, predictions, geometry, profiles = run_cluster_stage(
        labels=development_labels,
        candidates=refined_candidates,
        systems=systems,
        base_protocol=base_protocol,
        lens_protocol=lens_protocol,
        catalog=catalog,
        tian=tian,
        starts=int(protocol["optimization"]["starts_per_refined_point"]),
        seed_offset=100_000,
    )
    all_predictions.extend(predictions)
    all_geometry.extend(geometry)
    all_profiles.extend(profiles)
    refined_winner, refined_rows = select(
        refined_scores, development_labels, refined_candidates
    )

    selected_parameter = float(refined_winner["parameter"])
    selected_candidate = definitions([selected_parameter], "validation", family)[0]
    validation_scores, predictions, geometry, profiles = run_cluster_stage(
        labels=validation_labels,
        candidates=[selected_candidate],
        systems=systems,
        base_protocol=base_protocol,
        lens_protocol=lens_protocol,
        catalog=catalog,
        tian=tian,
        starts=int(protocol["optimization"]["starts_final_validation"]),
        seed_offset=200_000,
    )
    all_predictions.extend(predictions)
    all_geometry.extend(geometry)
    all_profiles.extend(profiles)
    validation = aggregate_definition(
        validation_scores, validation_labels, selected_candidate
    )

    stress_candidate = {
        **selected_candidate,
        "model": model_key(family, 0, "stress"),
        "stage": "stress",
    }
    stress_scores, predictions, geometry, profiles = run_cluster_stage(
        labels=stress_labels,
        candidates=[stress_candidate],
        systems=systems,
        base_protocol=base_protocol,
        lens_protocol=lens_protocol,
        catalog=catalog,
        tian=tian,
        starts=int(protocol["optimization"]["starts_final_validation"]),
        seed_offset=300_000,
    )
    all_predictions.extend(predictions)
    all_geometry.extend(geometry)
    all_profiles.extend(profiles)

    a0 = float(protocol["law"]["fixed_constants"]["a0_m_s2"])
    reference_radius_m = (
        float(protocol["law"]["fixed_constants"]["reference_radius_kpc"])
        * 1000.0
        * PARSEC_M
    )
    solar = solar_diagnostics(
        selected_parameter,
        reference_radius_m=reference_radius_m,
        a0_m_s2=a0,
    )
    mercury = solar[
        (solar.law == "screened") & (solar.planet == "Mercury")
    ].iloc[0]
    unscreened_mercury = solar[
        (solar.law == "unscreened_control") & (solar.planet == "Mercury")
    ].iloc[0]
    saturn_radius_m = PLANETS[5][1] * AU_M
    cassini_radii = np.geomspace(1.6 * SOLAR_RADIUS_M, saturn_radius_m, 4096)
    screened_force_fraction = fractional_extra_force(
        cassini_radii,
        source_mass_kg=M_SUN_KG,
        parameter=selected_parameter,
        reference_radius_m=reference_radius_m,
        a0_m_s2=a0,
        screened=True,
    )
    cassini_proxy = float(np.max(screened_force_fraction))

    mercury_constraint = protocol["published_constraints"][
        "Mercury_supplementary_precession_mas_per_century"
    ]
    mercury_margin = float(mercury_constraint["one_sigma"]) * float(
        mercury_constraint["gate_sigma_multiplier"]
    )
    mercury_pass = bool(
        abs(
            float(mercury["supplementary_precession_mas_per_century"])
            - float(mercury_constraint["central"])
        )
        <= mercury_margin
    )
    cassini_limit = float(
        protocol["published_constraints"]["Cassini_gamma_minus_one"][
            "phenomenological_fractional_force_proxy_max"
        ]
    )

    per_system_improves = {}
    validation_detail = {}
    model = selected_candidate["model"]
    for label in validation_labels:
        system_name = systems[label]["system"]
        selected_score = validation_scores[label][model]
        base_scores = base_report["system_scores"][system_name]
        validation_detail[label] = {
            **selected_score,
            "comparators": {
                name: base_scores[name]
                for name in (
                    "baryons_GR",
                    "fixed_simple_MOND",
                    "GR_plus_cluster_halo",
                )
            },
        }
        per_system_improves[label] = bool(
            selected_score["heldout"]["exact_radial_RMS_arcsec"]
            < base_scores["baryons_GR"]["heldout"]["exact_radial_RMS_arcsec"]
        )

    selected_at_boundary = bool(
        math.isclose(selected_parameter, min(refined_values))
        or math.isclose(selected_parameter, max(refined_values))
    )
    gates = {
        "Mercury_prediction_mas_per_century": float(
            mercury["supplementary_precession_mas_per_century"]
        ),
        "Mercury_allowed_absolute_margin_mas_per_century": mercury_margin,
        "Mercury_within_published_one_sigma_margin": mercury_pass,
        "Cassini_fractional_force_proxy": cassini_proxy,
        "Cassini_fractional_force_proxy_limit": cassini_limit,
        "Cassini_fractional_force_proxy_pass": bool(cassini_proxy <= cassini_limit),
        "all_validation_heldout_roots_converged": bool(
            validation["all_roots_converged"]
        ),
        "both_validation_clusters_improve_over_baryons": bool(
            all(per_system_improves.values())
        ),
        "per_system_improves_over_baryons": per_system_improves,
        "validation_RMS_pass": bool(
            validation["equal_system_radial_RMS_arcsec"]
            <= float(
                protocol["advance_gates"][
                    "validation_equal_system_RMS_arcsec_max"
                ]
            )
        ),
        "selected_parameter_at_refined_boundary": selected_at_boundary,
        "selected_parameter_interior_pass": not selected_at_boundary,
    }
    gates["all_advance_gates_pass"] = bool(
        gates["Mercury_within_published_one_sigma_margin"]
        and gates["Cassini_fractional_force_proxy_pass"]
        and gates["all_validation_heldout_roots_converged"]
        and gates["both_validation_clusters_improve_over_baryons"]
        and gates["validation_RMS_pass"]
        and gates["selected_parameter_interior_pass"]
    )

    stress_detail = {}
    for label in stress_labels:
        system_name = systems[label]["system"]
        selected_score = stress_scores[label][stress_candidate["model"]]
        base_scores = base_report["system_scores"][system_name]
        stress_detail[label] = {
            **selected_score,
            "comparators": {
                name: base_scores[name]
                for name in ("baryons_GR", "GR_plus_cluster_halo")
            },
        }

    compact_halo = prior_report["comparators"]["GR_plus_compact_cluster_halo"]
    post_result_diagnostics = {
        "prior_program_absolute_2_arcsec_gate_pass": bool(
            validation["equal_system_radial_RMS_arcsec"] <= 2.0
        ),
        "validation_RMS_ratio_to_compact_halo": float(
            validation["equal_system_radial_RMS_arcsec"]
            / compact_halo["equal_system_radial_RMS_arcsec"]
        ),
        "validation_pooled_chi2_ratio_to_compact_halo": float(
            validation["pooled_coordinate_chi2"]
            / compact_halo["pooled_coordinate_chi2"]
        ),
        "RXJ2129_heldout_RMS_ratio_to_compact_halo": float(
            stress_detail["RXJ2129"]["heldout"]["exact_radial_RMS_arcsec"]
            / stress_detail["RXJ2129"]["comparators"]["GR_plus_cluster_halo"]
            ["heldout"]["exact_radial_RMS_arcsec"]
        ),
        "interpretation": (
            "These are post-result continuity and stress diagnostics, not additions "
            "to the frozen screened-law gate."
        ),
    }

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed screened one-parameter cluster and Solar-System replay",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "input_hashes": {
            "image_catalog": sha256(image_path),
            "baryonic_profile": sha256(baryon_path),
        },
        "law": {
            **protocol["law"],
            "selected_parameter": selected_parameter,
        },
        "selection_design": protocol["selection_design"],
        "coarse_winner": coarse_winner,
        "refined_winner": refined_winner,
        "validation": {
            "aggregate": validation,
            "per_system": validation_detail,
        },
        "solar_system": {
            "source_convention": protocol["solar_source_convention"],
            "Mercury_unscreened_control_mas_per_century": float(
                unscreened_mercury["supplementary_precession_mas_per_century"]
            ),
            "Mercury_screened_mas_per_century": float(
                mercury["supplementary_precession_mas_per_century"]
            ),
            "published_constraints": protocol["published_constraints"],
        },
        "comparators": {
            "prior_unscreened_one_parameter_validation": prior_report["validation"][
                "selected_law"
            ],
            "baryons_GR": prior_report["comparators"]["baryons_GR"],
            "fixed_simple_MOND": prior_report["comparators"]["fixed_simple_MOND"],
            "fixed_RAR_zero_slip": prior_report["comparators"][
                "fixed_RAR_zero_slip"
            ],
            "GR_plus_compact_cluster_halo": prior_report["comparators"][
                "GR_plus_compact_cluster_halo"
            ],
        },
        "stress_tests": stress_detail,
        "gate_audit": gates,
        "post_result_diagnostics": post_result_diagnostics,
        "verdict": {
            "screened_one_parameter_law_survives_frozen_replay_gates": gates[
                "all_advance_gates_pass"
            ],
            "raw_ephemeris_validation_completed": False,
            "external_cluster_validation": False,
        },
        "claim_boundary": [
            protocol["selection_design"]["disclosure"],
            protocol["solar_source_convention"]["limitation"],
            "The acceleration screen uses fixed a0 and adds no fitted parameter, but its functional form was proposed after the unscreened Mercury conflict was known.",
            "Cassini gamma is not derived; the reported Solar light-propagation result remains a zero-slip phenomenological proxy.",
        ],
    }

    output_dir = (ROOT / protocol["outputs"]["report"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(coarse_rows).to_csv(
        ROOT / protocol["outputs"]["coarse_grid"], index=False
    )
    pd.DataFrame(refined_rows).to_csv(
        ROOT / protocol["outputs"]["refined_grid"], index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(all_geometry).to_csv(
        ROOT / protocol["outputs"]["geometry"], index=False
    )
    pd.concat(all_profiles, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["radial_profiles"], index=False
    )
    solar.to_csv(ROOT / protocol["outputs"]["solar_diagnostics"], index=False)

    summary = [
        "# Solar-screened one-parameter isothermal result",
        "",
        f"Selected universal lambda: **{selected_parameter:.8g}**",
        f"Development held-out RMS: **{refined_winner['equal_system_radial_RMS_arcsec']:.3f} arcsec**",
        f"Validation held-out RMS: **{validation['equal_system_radial_RMS_arcsec']:.3f} arcsec**",
        f"Mercury supplementary precession: **{float(mercury['supplementary_precession_mas_per_century']):.6g} mas/century**",
        f"Mercury allowed margin: **+/-{mercury_margin:.3f} mas/century**",
        f"Every frozen replay gate passes: **{gates['all_advance_gates_pass']}**",
        f"Earlier 2-arcsec absolute gate passes: **{post_result_diagnostics['prior_program_absolute_2_arcsec_gate_pass']}**",
        f"RXJ2129 RMS / compact-halo RMS: **{post_result_diagnostics['RXJ2129_heldout_RMS_ratio_to_compact_halo']:.3f}**",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
