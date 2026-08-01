#!/usr/bin/env python3
"""Build the residual-blind R1B same-system readiness gap ledger."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_same_system_pilot_targets.json"
OUTPUT_PATH = ROOT / "data/derived/r1_same_system_pilot_gap_ledger.csv"
REPORT_PATH = ROOT / "results/r1_same_system_pilot_gap/report.json"


def published_dynamics_records() -> list[dict]:
    velocity = pd.read_csv(ROOT / "data/derived/r1_published_bcg_velocity_profiles.csv")
    source_counts = (
        velocity.groupby(["system", "source_sample"]).size().rename("bins").reset_index()
    )
    selected = source_counts.sort_values(["system", "bins"], ascending=[True, False]).drop_duplicates("system")
    lens = pd.read_csv(ROOT / "data/derived/r1_strong_lens_image_observables.csv")
    rank = pd.read_csv(ROOT / "data/derived/r1_lensing_geometric_rank.csv").set_index("system")
    photometry = set(pd.read_csv(ROOT / "data/derived/r1_published_bcg_photometric_fits.csv")["system"])
    a383_calibration_path = ROOT / "results/r1_a383_gmos_calibrations/report.json"
    a383_acquisition_path = ROOT / "results/r1_a383_gemini_acquisition/report.json"
    ms2137_geometry_path = ROOT / "results/r1_ms2137_ppxf_geometry/report.json"
    ms2137_acquisition_path = ROOT / "results/r1_ms2137_muse_acquisition/report.json"
    a2537_calibration_path = ROOT / "results/r1_a2537_gmos_calibrations/report.json"
    a2537_acquisition_path = ROOT / "results/r1_a2537_gemini_acquisition/report.json"
    a383_calibration = json.loads(a383_calibration_path.read_text()) if a383_calibration_path.exists() else None
    a383_acquisition = json.loads(a383_acquisition_path.read_text()) if a383_acquisition_path.exists() else None
    ms2137_geometry = json.loads(ms2137_geometry_path.read_text()) if ms2137_geometry_path.exists() else None
    ms2137_acquisition = json.loads(ms2137_acquisition_path.read_text()) if ms2137_acquisition_path.exists() else None
    a2537_calibration = json.loads(a2537_calibration_path.read_text()) if a2537_calibration_path.exists() else None
    a2537_acquisition = json.loads(a2537_acquisition_path.read_text()) if a2537_acquisition_path.exists() else None
    records = []
    for _, row in selected.iterrows():
        system = str(row["system"])
        strict_lens = lens.loc[
            (lens["system"] == system) & lens["alternative_metric_likelihood_ready"].astype(bool)
        ]
        rank_row = rank.loc[system] if system in rank.index else None
        overlap = int(rank_row["strict_inner_image_positions"]) if rank_row is not None else 0
        structural_rank = int(rank_row["structural_radial_rank_upper_bound"]) if rank_row is not None else 0
        record = {
                "system": system,
                "candidate_class": "published_numerical_dynamics",
                "dynamics_source": str(row["source_sample"]),
                "reported_or_numerical_dynamics_points": int(row["bins"]),
                "numerical_dynamics_profile_local": True,
                "dynamics_covariance_local": False,
                "dynamics_internal_consistency_pass": True,
                "observable_lens_positions": int(len(lens.loc[lens["system"] == system])),
                "source_redshift_likelihood_rows": int(len(strict_lens)),
                "lensing_points_on_dynamics_support": overlap,
                "structural_radial_rank_upper_bound": structural_rank,
                "three_plus_three_structural_pass": overlap >= 3 and structural_rank >= 3,
                "declared_lens_position_error": len(strict_lens) > 0,
                "coordinate_covariance_independent_of_fitted_gr_residuals": False,
                "bcg_starlight_model_local": system in photometry,
                "complete_baryonic_forward_inputs": False,
                "strict_r1_ready": False,
                "primary_obstruction": (
                    "no observable position/redshift likelihood is local"
                    if len(strict_lens) == 0
                    else f"only {overlap} lensing points lie on the numerical dynamics support; at least 3 are required"
                ),
            }
        if system == "A383" and a383_calibration is not None:
            arc_rows = a383_calibration["arcs"]
            record.update({
                "candidate_class": "frozen_10p5arcsec_raw_dynamics_calibration_failed",
                "dynamics_source": f"{record['dynamics_source']} plus Gemini/GMOS-S GS-2007B-Q-36 raw extension attempt",
                "raw_dynamics_acquisition_gate_passed": bool(a383_acquisition and a383_acquisition["gates"]["raw_acquisition_gate_passed"]),
                "dynamics_reconstruction_attempted": True,
                "dynamics_reconstruction_gate_passed": False,
                "primary_obstruction": (
                    "the frozen A383 P2a calibration gate failed before science reduction: "
                    f"CuAr wavelength RMS values were {arc_rows[0]['wavelength_solution_rms_angstrom']:.3f} and "
                    f"{arc_rows[1]['wavelength_solution_rms_angstrom']:.3f} Angstrom versus the 0.200-Angstrom ceiling; "
                    "the 10.5-arcsec extension is unauthorized"
                ),
            })
        if system == "MS2137" and ms2137_geometry is not None:
            registration = ms2137_geometry["registration"]
            record.update({
                "candidate_class": "frozen_14arcsec_raw_dynamics_geometry_failed",
                "dynamics_source": f"{record['dynamics_source']} plus ESO MUSE {ms2137_geometry['cube_sha256'][:12]} raw-extension attempt",
                "raw_dynamics_acquisition_gate_passed": bool(ms2137_acquisition and ms2137_acquisition["gates"]["acquisition_header_gate_passed"]),
                "dynamics_reconstruction_attempted": True,
                "dynamics_reconstruction_gate_passed": False,
                "primary_obstruction": (
                    "the frozen MS2137 P2 geometry-and-signal gate failed before pPXF: "
                    f"the continuum registration offset was {registration['registration_offset_arcsec']:.3f} arcsec "
                    "versus the 1.0-arcsec ceiling, and the 0-0.5-arcsec annulus did not meet the frozen "
                    "20-spaxel-per-opposite-half rule; the 14.0-arcsec extension and all velocity fits are unauthorized"
                ),
            })
        if system == "A2537" and a2537_calibration is not None:
            arc_rows = a2537_calibration["arcs"]
            record.update({
                "candidate_class": "frozen_16arcsec_disturbed_control_calibration_failed",
                "dynamics_source": f"{record['dynamics_source']} plus Gemini/GMOS-S GS-2008B-Q-4 disturbed-control attempt",
                "raw_dynamics_acquisition_gate_passed": bool(a2537_acquisition and a2537_acquisition["gates"]["raw_acquisition_gate_passed"]),
                "dynamics_reconstruction_attempted": True,
                "dynamics_reconstruction_gate_passed": False,
                "primary_obstruction": (
                    "the frozen A2537 C2a disturbed-control calibration gate failed before any science frame was processed: "
                    f"CuAr wavelength RMS values were {arc_rows[0]['wavelength_solution_rms_angstrom']:.3f} and "
                    f"{arc_rows[1]['wavelength_solution_rms_angstrom']:.3f} Angstrom versus the 0.200-Angstrom ceiling; "
                    "the 16.0-arcsec extension is unauthorized, and A2537 cannot count as a non-disturbed pilot regardless"
                ),
            })
        records.append(record)
    return records


def promoted_records() -> list[dict]:
    cycle1 = pd.read_csv(ROOT / "data/derived/r1_replacement_cycle1_candidate_ledger.csv").set_index("system")
    cycle2 = pd.read_csv(ROOT / "data/derived/r1_replacement_cycle2_candidate_ledger.csv").set_index("system")
    m1206 = cycle1.loc["MACS J1206"]
    rxj2129 = cycle2.loc["RX J2129"]
    chandra = json.loads((ROOT / "results/r1_rxj2129_chandra_reduction/report.json").read_text())
    return [
        {
            "system": "MACS J1206",
            "candidate_class": "public_muse_reconstruction_structural_promotion",
            "dynamics_source": str(m1206["dynamics_source"]),
            "reported_or_numerical_dynamics_points": int(m1206["dynamics_bins"]),
            "numerical_dynamics_profile_local": True,
            "dynamics_covariance_local": True,
            "dynamics_internal_consistency_pass": False,
            "observable_lens_positions": int(m1206["published_image_rows"]),
            "source_redshift_likelihood_rows": int(m1206["strict_position_redshift_inputs"]),
            "lensing_points_on_dynamics_support": int(m1206["strict_inner_image_positions"]),
            "structural_radial_rank_upper_bound": int(m1206["structural_radial_rank_upper_bound"]),
            "three_plus_three_structural_pass": bool(m1206["structural_promotion_pass"]),
            "declared_lens_position_error": True,
            "coordinate_covariance_independent_of_fitted_gr_residuals": False,
            "bcg_starlight_model_local": True,
            "complete_baryonic_forward_inputs": False,
            "strict_r1_ready": False,
            "primary_obstruction": "the frozen homogeneous level-2 pPXF reconstruction fails internal-consistency gates; complete numerical baryons and theory-neutral lens covariance are also absent",
        },
        {
            "system": "RX J2129",
            "candidate_class": "public_muse_reconstruction_structural_promotion",
            "dynamics_source": str(rxj2129["dynamics_source"]),
            "reported_or_numerical_dynamics_points": int(rxj2129["resolved_bgg_dynamics_bins"]),
            "numerical_dynamics_profile_local": True,
            "dynamics_covariance_local": True,
            "dynamics_internal_consistency_pass": True,
            "observable_lens_positions": int(rxj2129["spectroscopic_multiple_image_positions"]),
            "source_redshift_likelihood_rows": int(rxj2129["spectroscopic_multiple_image_positions"]),
            "lensing_points_on_dynamics_support": int(rxj2129["strict_inner_image_positions"]),
            "structural_radial_rank_upper_bound": int(rxj2129["structural_radial_rank_upper_bound"]),
            "three_plus_three_structural_pass": bool(rxj2129["structural_promotion_pass"]),
            "declared_lens_position_error": True,
            "coordinate_covariance_independent_of_fitted_gr_residuals": False,
            "bcg_starlight_model_local": True,
            "complete_baryonic_forward_inputs": False,
            "strict_r1_ready": False,
            "dynamics_reconstruction_attempted": True,
            "dynamics_reconstruction_gate_passed": True,
            "primary_obstruction": (
                "the frozen Chandra calibrated-reduction gate failed: ObsID 552 retained 81.8% exposure, blank-sky BKGSCAL values fell outside 0.5-2.0, and event headers did not match the required CALDB; the gas likelihood remains blocked"
                if not chandra["calibrated_reduction_gate_pass"] else
                "the gas likelihood and a fresh thresholded lens-prediction gate remain incomplete"
            ),
        },
    ]


def a2261_record() -> dict:
    report = json.loads((ROOT / "results/r1_a2261_lens_observables/report.json").read_text())
    acquisition_path = ROOT / "results/r1_a2261_gemini_acquisition/report.json"
    cal2d_path = ROOT / "results/r1_a2261_gmos_science_cal2d/report.json"
    center_path = ROOT / "results/r1_a2261_gmos_continuum_center/report.json"
    acquisition = json.loads(acquisition_path.read_text()) if acquisition_path.exists() else None
    cal2d = json.loads(cal2d_path.read_text()) if cal2d_path.exists() else None
    center = json.loads(center_path.read_text()) if center_path.exists() else None
    raw_gate = bool(acquisition and acquisition["gates"]["raw_acquisition_gate_passed"])
    p2b_gate = bool(cal2d and cal2d["gates"]["P2b_individual_calibrated_2d_gate_passed"])
    p2c_pass = bool(center and center["gates"]["P2c_continuum_centroid_range_gate_passed"])
    radial = report["radial_overlap"]
    catalog = report["catalog"]
    return {
        "system": "Abell 2261",
        "candidate_class": (
            "frozen_36kpc_extended_raw_dynamics_failed" if center and not p2c_pass else
            "frozen_36kpc_extended_raw_dynamics_in_progress" if raw_gate else
            "figure_only_dynamics_near_miss"
        ),
        "dynamics_source": "Loubser et al. 2018 Gemini/GMOS",
        "reported_or_numerical_dynamics_points": 9,
        "numerical_dynamics_profile_local": False,
        "dynamics_covariance_local": False,
        "dynamics_internal_consistency_pass": False,
        "observable_lens_positions": int(catalog["images"]),
        "source_redshift_likelihood_rows": int(catalog["images_with_lens_independent_family_redshift"]),
        "lensing_points_on_dynamics_support": int(radial["images_inside_dynamics_support"]),
        "structural_radial_rank_upper_bound": int(radial["structural_radial_rank_upper_bound"]),
        "three_plus_three_structural_pass": False,
        "declared_lens_position_error": False,
        "coordinate_covariance_independent_of_fitted_gr_residuals": False,
        "bcg_starlight_model_local": True,
        "complete_baryonic_forward_inputs": False,
        "strict_r1_ready": False,
        "raw_dynamics_acquisition_gate_passed": raw_gate,
        "dynamics_reconstruction_attempted": center is not None,
        "dynamics_reconstruction_gate_passed": p2c_pass,
        "primary_obstruction": (
            f"the frozen continuum-centroid gate failed: the four independent centers span {center['individual_center_range_arcsec']:.3f} arcsec versus the 0.3-arcsec ceiling; sky modeling, pPXF, and the 36.0-kpc extension are unauthorized"
            if center and not p2c_pass else
            "the frozen raw acquisition and individual calibrated-2D gates pass, but no kinematic value has been fit; all nine signed bins, including both 7-10.5 arcsec outer bins, must pass and demonstrate at least 36.0-kpc realized support with covariance"
            if p2b_gate else
            "the exact raw set is acquired, but the frozen calibration/2-D gate is incomplete; the target remains 36.0 kpc because the third independent-family lens image is at 35.7082 kpc"
            if raw_gate else
            f"the nearest measured image is {radial['gap_beyond_dynamics_support_kpc']:.4f} kpc outside the published support; the velocity profile and both covariances are not machine-readable"
        ),
    }


def a1689_record() -> dict:
    lens_report = json.loads((ROOT / "results/r1_a1689_lens_prescreen/report.json").read_text())
    raw_report = json.loads((ROOT / "results/r1_a1689_gemini_acquisition/report.json").read_text())
    radial = lens_report["radial_overlap"]
    catalog = lens_report["catalog"]
    raw_gate = bool(raw_report["gates"]["raw_acquisition_gate_passed"])
    systematic_path = ROOT / "results/r1_a1689_gmos_systematics/report.json"
    systematic = json.loads(systematic_path.read_text()) if systematic_path.exists() else None
    p3_pass = bool(systematic and systematic["gates"]["P3e_systematic_shift_gate_passed"])
    return {
        "system": "Abell 1689",
        "candidate_class": "frozen_raw_dynamics_reconstruction_failed" if systematic else "raw_dynamics_acquired_geometry_pass",
        "dynamics_source": "Loubser et al. 2018 Gemini/GMOS GN-2008B-Q-5",
        "reported_or_numerical_dynamics_points": 9,
        "numerical_dynamics_profile_local": False,
        "dynamics_covariance_local": False,
        "dynamics_internal_consistency_pass": False,
        "observable_lens_positions": int(catalog["images"]),
        "source_redshift_likelihood_rows": int(catalog["images"]),
        "lensing_points_on_dynamics_support": int(radial["independently_redshift_anchored_images_inside_support"]),
        "structural_radial_rank_upper_bound": int(radial["distinct_image_radii_inside_support"]),
        "three_plus_three_structural_pass": False,
        "declared_lens_position_error": False,
        "coordinate_covariance_independent_of_fitted_gr_residuals": False,
        "bcg_starlight_model_local": True,
        "complete_baryonic_forward_inputs": False,
        "strict_r1_ready": False,
        "raw_dynamics_acquisition_gate_passed": raw_gate,
        "dynamics_reconstruction_attempted": systematic is not None,
        "dynamics_reconstruction_gate_passed": p3_pass,
        "primary_obstruction": (
            "the 200/200 covariance bootstrap passed, but the frozen 27-run pPXF systematic grid failed: signed-bin dispersion shifts reached 36.6% versus the 10% limit; A1689 remains geometry-only"
            if systematic else
            "the lens geometry and raw acquisition pass, but the frozen dynamics reconstruction is pending"
        ),
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_audit(
    output_path: Path = OUTPUT_PATH, report_path: Path = REPORT_PATH
) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = published_dynamics_records() + promoted_records() + [a2261_record(), a1689_record()]
    ledger = pd.DataFrame(records).sort_values("system").reset_index(drop=True)
    if ledger["system"].duplicated().any():
        raise RuntimeError("Duplicate system in same-system pilot gap ledger")
    ledger["raw_dynamics_acquisition_gate_passed"] = ledger[
        "raw_dynamics_acquisition_gate_passed"
    ].eq(True)
    ledger["geometry_prescreen_pass"] = (
        (ledger["lensing_points_on_dynamics_support"] >= 3)
        & (ledger["structural_radial_rank_upper_bound"] >= 3)
    )
    ledger["dynamics_reconstruction_attempted"] = ledger[
        "dynamics_reconstruction_attempted"
    ].eq(True)
    ledger["dynamics_reconstruction_gate_passed"] = ledger[
        "dynamics_reconstruction_gate_passed"
    ].eq(True)

    numerical = ledger.loc[ledger["numerical_dynamics_profile_local"]]
    numerical_with_lens = numerical.loc[numerical["observable_lens_positions"] >= 3]
    structural = ledger.loc[ledger["three_plus_three_structural_pass"]]
    geometry_pending = ledger.loc[
        ledger["geometry_prescreen_pass"]
        & ~ledger["numerical_dynamics_profile_local"]
        & ~ledger["dynamics_reconstruction_attempted"]
    ]
    failed_reconstruction = ledger.loc[
        ledger["dynamics_reconstruction_attempted"]
        & ~ledger["dynamics_reconstruction_gate_passed"]
    ]
    strict = ledger.loc[ledger["strict_r1_ready"]]
    baseline = config["current_verified_baseline"]
    measured = {
        "numerical_resolved_dynamics_systems": int(len(numerical)),
        "systems_with_local_observable_lens_positions": int(len(numerical_with_lens)),
        "systems_passing_three_plus_three_structural_overlap": int(len(structural)),
        "systems_with_complete_baryonic_forward_inputs": int(ledger["complete_baryonic_forward_inputs"].sum()),
        "systems_with_theory_neutral_joint_covariance": int(
            (ledger["dynamics_covariance_local"] & ledger["coordinate_covariance_independent_of_fitted_gr_residuals"]).sum()
        ),
        "strict_r1_ready_systems": int(len(strict)),
    }
    for key, expected in baseline.items():
        if key in measured and measured[key] != expected:
            raise RuntimeError(f"Frozen same-system baseline mismatch for {key}: {measured[key]} != {expected}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_path, index=False)
    report = {
        "report_version": "R1B-same-system-pilot-gap-0.2-after-control-cycle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "candidate_systems_evaluated": int(len(ledger)),
        **measured,
        "structural_pass_systems": structural["system"].tolist(),
        "raw_dynamics_geometry_qualified_pending_reconstruction": geometry_pending["system"].tolist(),
        "failed_frozen_raw_reconstruction_systems": failed_reconstruction["system"].tolist(),
        "target_strict_systems": int(baseline["target_strict_systems"]),
        "structural_system_gap": int(baseline["target_strict_systems"] - len(structural)),
        "strict_ready_system_gap": int(baseline["target_strict_systems"] - len(strict)),
        "a2261_near_miss_gap_kpc": 0.49010146333345617,
        "a2261_frozen_extended_support_target_kpc": 36.0,
        "cycle_1_checkpoint": {
            "required_additional_structural_passes": 3,
            "current_additional_structural_passes": 0,
            "passed": False,
            "rethink_if_completed_cycle_adds_fewer_than": 2,
        },
        "route_rethink_triggered": True,
        "route_rethink_reason": "The completed residual-blind raw-dynamics cycle produced zero additional structural promotions: A1689, A2261, A383, and MS2137 failed distinct frozen science gates, while the predeclared disturbed A2537 engineering control also failed its calibration gate. Continuing the same acquisition route would not address the two already-promoted systems' missing baryons and theory-neutral lens covariance.",
        "next_stage": {
            "name": "R1B3_RXJ2129_strict_observable_package",
            "primary_system": "RX J2129",
            "why": "RX J2129 already passes the three-dynamics-plus-three-lens structural gate and its frozen MUSE dynamics consistency gates; the shortest premise-level path is to replace the failed Chandra gas route and build measurement-level lens covariance.",
            "concrete_outcome": "One same-system package containing accepted dynamics covariance, stellar-light inputs, an independent X-ray gas density/temperature covariance, and theory-neutral image-position/redshift covariance, with no gravity-response fit.",
            "rethink_if": "No public XMM route passes predeclared exposure/background/calibration gates, or no measurement-level HST lens covariance can be built without borrowing fitted-GR residuals. If either occurs, retain RX J2129 as structurally promoted but not strict-ready and run the same availability prescreen on MACS J1206 before any new force-law work."
        },
        "decision": {
            "R1B1_gap_audit": "complete_for_current_numerical_dynamics_universe_plus_A2261_near_miss_and_A1689_raw_candidate",
            "R1B2_two_strict_systems": "not_authorized",
            "R1C_ten_system_freeze": "not_authorized",
            "R2_response_reconstruction": "not_authorized",
            "next_action": "Close the completed raw-dynamics acquisition cycle without threshold changes. Freeze and run the RX J2129 R1B3 strict-observable-package prescreen: public XMM gas feasibility plus theory-neutral HST image-position/redshift covariance feasibility. Keep all dynamical/Weyl responses separate and fit no gravity law.",
        },
        "authorization": config["authorization"],
        "output": display_path(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    arguments = parser.parse_args()
    print(json.dumps(build_audit(arguments.output, arguments.report), indent=2))
