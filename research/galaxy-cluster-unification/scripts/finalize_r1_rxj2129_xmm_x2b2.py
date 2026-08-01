#!/usr/bin/env python3
"""Close RX J2129 X2b2 and expose only the frozen X3 construction step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QPB_PATH = ROOT / "results/r1_rxj2129_xmm_event_processing/qpb_background_audit.json"
OUTER_PATH = (
    ROOT
    / "results/r1_rxj2129_xmm_event_processing/outer_annulus_transfer_audit.json"
)
PROTOCOL_PATH = ROOT / "configs/r1_rxj2129_xmm_background_mask_protocol.json"
MANIFEST_PATH = ROOT / "data/derived/r1_rxj2129_xmm_reduction_manifest.json"
REPORT_PATH = ROOT / "results/r1_rxj2129_xmm_event_processing/report.json"
NEXT_STAGE_PATH = ROOT / "configs/r1_rxj2129_strict_observable_next_stage.json"
EXECUTION_TARGETS_PATH = ROOT / "configs/r1_execution_targets.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def validate_inputs(qpb: dict[str, Any], outer: dict[str, Any]) -> list[str]:
    expected = ["MOS2", "pn"]
    assert qpb["stage"] == "X2b2_FWC_corner_subgate"
    assert qpb["minimum_instrument_gate_passed"] is True
    assert qpb["passing_instruments"] == expected
    assert qpb["excluded_at_FWC_corner_subgate"] == ["MOS1"]
    assert qpb["instrument_results"]["MOS1"]["sectors"]["5"]["passed"] is False
    assert (
        qpb["instrument_results"]["MOS1"]["sectors"]["5"]["FWC_corner_scale"]
        < 0.5
    )
    assert outer["stage"] == "X2b2_local_outer_annulus_transfer_subgate"
    assert outer["full_X2b2_background_gate_passed"] is True
    assert outer["passing_instruments"] == expected
    for instrument in expected:
        assert qpb["instrument_results"][instrument]["passed"] is True
        assert outer["instrument_results"][instrument]["passed"] is True
    return expected


def update_background_protocol(
    protocol: dict[str, Any], qpb: dict[str, Any], outer: dict[str, Any]
) -> None:
    protocol["protocol_version"] = "R1B3-RXJ2129-XMM-X2b-background-mask-1.3"
    protocol["status"] = "completed_X2b2_with_MOS2_and_pn_before_X3_annular_construction"
    protocol["frozen_utc"] = outer["generated_utc"]
    protocol["background"]["FWC_corner_subgate_result"]["full_X2_passed"] = False
    protocol["background"]["local_outer_annulus_subgate_result"] = {
        "artifact": relative(OUTER_PATH),
        "MOS2": {
            "observed_hard_band_counts": outer["instrument_results"]["MOS2"][
                "observed_counts"
            ],
            "ESAS_model_QPB_counts": outer["instrument_results"]["MOS2"][
                "ESAS_model_QPB_counts"
            ],
            "transfer_scale": outer["instrument_results"]["MOS2"][
                "outer_annulus_transfer_scale"
            ],
            "passed": True,
        },
        "pn": {
            "observed_hard_band_counts_after_OOT_subtraction": outer[
                "instrument_results"
            ]["pn"]["observed_counts"],
            "ESAS_model_QPB_counts": outer["instrument_results"]["pn"][
                "ESAS_model_QPB_counts"
            ],
            "transfer_scale": outer["instrument_results"]["pn"][
                "outer_annulus_transfer_scale"
            ],
            "passed": True,
        },
        "passing_instruments": outer["passing_instruments"],
        "full_X2b2_background_gate_passed": True,
    }
    protocol["execution_result"] = {
        "generated_utc": outer["generated_utc"],
        "FWC_corner_artifact": relative(QPB_PATH),
        "outer_annulus_artifact": relative(OUTER_PATH),
        "retained_instruments": outer["passing_instruments"],
        "excluded_instruments": {
            "MOS1": "Predeclared all-sector rule failed: CCD5 FWC/corner scale 0.3533430850668397 is below 0.5."
        },
        "full_X2_gate_passed": True,
        "next_gate": "X3 annular count/response construction and adequacy audit",
    }
    protocol["authorization"].update(
        {
            "construct_X3_annular_count_response_products": True,
            "fit_temperature_or_density_before_X3_adequacy_pass": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        }
    )


def update_manifest(
    manifest: dict[str, Any], qpb: dict[str, Any], outer: dict[str, Any]
) -> None:
    manifest["manifest_version"] = "R1B3-RXJ2129-XMM-X2b2-1.0"
    manifest["generated_utc"] = outer["generated_utc"]
    manifest["X2b2_background"] = {
        "protocol": relative(PROTOCOL_PATH),
        "FWC_corner_audit": relative(QPB_PATH),
        "outer_annulus_transfer_audit": relative(OUTER_PATH),
        "FWC_corner_result": {
            "MOS1": {
                "passed": False,
                "failed_sector": "CCD5",
                "failed_scale": qpb["instrument_results"]["MOS1"]["sectors"]["5"][
                    "FWC_corner_scale"
                ],
                "pooled_diagnostic_scale_not_used_for_override": qpb[
                    "instrument_results"
                ]["MOS1"]["pooled_diagnostic"]["FWC_corner_scale"],
            },
            "MOS2": {
                "passed": True,
                "pooled_diagnostic_scale": qpb["instrument_results"]["MOS2"][
                    "pooled_diagnostic"
                ]["FWC_corner_scale"],
            },
            "pn": {
                "passed": True,
                "pooled_diagnostic_scale": qpb["instrument_results"]["pn"][
                    "pooled_diagnostic"
                ]["FWC_corner_scale"],
                "OOT_scale": 0.0232,
            },
        },
        "outer_annulus_result": {
            instrument: {
                "observed_counts": outer["instrument_results"][instrument][
                    "observed_counts"
                ],
                "ESAS_model_QPB_counts": outer["instrument_results"][instrument][
                    "ESAS_model_QPB_counts"
                ],
                "transfer_scale": outer["instrument_results"][instrument][
                    "outer_annulus_transfer_scale"
                ],
                "central_95_interval": outer["instrument_results"][instrument][
                    "posterior"
                ]["central_95_interval"],
                "passed": True,
            }
            for instrument in ("MOS2", "pn")
        },
        "passing_instruments": ["MOS2", "pn"],
        "minimum_passing_instruments": 2,
        "invalid_outer_annulus_clobber_trial_quarantined_at": (
            "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/"
            "x2b/background/esas_outer_annulus_invalid_clobber_flag_yes"
        ),
        "invalid_partial_products_admitted": False,
        "gate_passed": True,
    }
    manifest["gates"].update(
        {
            "R1B3_XMM_X2b2_FWC_corner_gate_passed": True,
            "R1B3_XMM_X2b2_outer_annulus_gate_passed": True,
            "R1B3_XMM_X2_flare_background_gate_passed": True,
            "R1B3_XMM_X3_gas_likelihood_gate_passed": False,
        }
    )


def update_report(report: dict[str, Any], outer: dict[str, Any]) -> None:
    report.update(
        {
            "report_version": "R1B3-RXJ2129-XMM-event-processing-X2b2-1.0",
            "generated_utc": outer["generated_utc"],
            "stage": "X2b2_flare_mask_background",
            "status": "pass",
            "outcome": (
                "Full X2 passes with MOS2 and pn. MOS1 is excluded by the frozen "
                "sector-level FWC/corner rule; X3 product construction is authorized."
            ),
            "passing_instruments": ["MOS2", "pn"],
            "excluded_instruments": {
                "MOS1": "CCD5 FWC/corner scale 0.3533430850668397 failed the unchanged (0.5, 2.0) interval."
            },
            "evidence": [relative(QPB_PATH), relative(OUTER_PATH)],
        }
    )
    report["gates"].update(
        {
            "R1B3_XMM_X2b2_FWC_corner_gate_passed": True,
            "R1B3_XMM_X2b2_outer_annulus_gate_passed": True,
            "R1B3_XMM_X2_flare_background_gate_passed": True,
            "R1B3_XMM_X3_gas_likelihood_gate_passed": False,
        }
    )
    report["authorization"] = {
        "construct_X3_annular_count_response_products": True,
        "claim_X3_gas_likelihood_pass": False,
        "fit_temperature_or_density": False,
        "infer_dynamical_or_Weyl_response": False,
        "fit_new_force_or_action": False,
    }


def update_next_stage(config: dict[str, Any], outer: dict[str, Any]) -> None:
    config["protocol_version"] = "R1B3-RXJ2129-strict-observable-next-stage-0.2"
    config["status"] = "active_X3_annular_construction_and_HST_measurement_after_X2_pass"
    config["frozen_utc"] = outer["generated_utc"]
    config["disclosure"][1] = (
        "At the initial freeze, a published analysis implied substantially shorter "
        "flare-cleaned exposure than the gross XMM duration; exact cleaned values "
        "had not yet been measured locally."
    )
    update_note = (
        "Execution update: XMM X1, X2a, X2b1, and X2b2 now pass. MOS2 and pn "
        "are retained; MOS1 is excluded by its CCD5 FWC/corner sector scale."
    )
    if update_note not in config["disclosure"]:
        config["disclosure"].append(update_note)
    config["execution_status"] = {
        "XMM_X1_calibration": "pass",
        "XMM_X2a_flare_exposure": "pass: MOS1 42291.665 s; MOS2 43062.073 s; pn 32012.166 s",
        "XMM_X2b1_immutable_source_mask": "pass: 87 sources and 783 PSF evaluations",
        "XMM_X2b2_background": "pass: MOS2 and pn; MOS1 excluded",
        "XMM_X3_annular_count_response_adequacy": "in_progress",
        "HST_42x42_measurement_covariance": "pending",
        "gas_temperature_density_likelihood": "locked_until_X3_adequacy_pass",
        "evidence": [
            relative(MANIFEST_PATH),
            relative(QPB_PATH),
            relative(OUTER_PATH),
        ],
    }
    config["next_concrete_outcomes"] = {
        "X3_annular_products": {
            "radial_range_kpc": [10, 500],
            "minimum_accepted_annuli": 5,
            "minimum_net_counts_all_passing_instruments": 2000,
            "minimum_signal_to_noise_each_annulus": 5.0,
            "advance_if": (
                "MOS2+pn source, QPB, RMF, and ARF products are complete and the "
                "count/SNR thresholds pass under one frozen annular partition."
            ),
            "rethink_if": (
                "No frozen partition can support five annuli and 2000 net counts; "
                "stop the XMM branch without merging below five annuli or changing instruments."
            ),
        },
        "HST_measurement_covariance": {
            "required_images": 21,
            "required_inner_images": ["5.2", "6.3", "8.2"],
            "minimum_images_accepted": 18,
            "bootstrap_draws": 500,
            "required_covariance_shape": [42, 42],
            "advance_if": "All three inner images and at least 18 total images pass; covariance is PSD.",
            "rethink_if": "H1, H2, or H3 fails under the frozen pixel-measurement protocol.",
        },
    }
    config["authorization"].update(
        {
            "measure_HST_arc_pixels_after_separate_execution_freeze": True,
            "reduce_XMM_after_environment_freeze": True,
            "construct_X3_annular_count_response_products": True,
            "fit_XMM_temperature_or_density": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        }
    )


def update_execution_targets(targets: dict[str, Any]) -> None:
    targets["plan_version"] = "R1-execution-0.14-RXJ2129-XMM-X2-pass"
    targets["next_stage"] = (
        "R1B3 RX J2129 X3 MOS2+pn annular count/response adequacy plus independent "
        "HST 42x42 measurement covariance; no gas, gravity, dynamical-response, or "
        "Weyl-response fit until their gates pass"
    )
    targets["baseline"].update(
        {
            "rxj2129_xmm_X1_calibration_gate_pass": True,
            "rxj2129_xmm_X2a_flare_exposure_gate_pass": True,
            "rxj2129_xmm_X2b1_point_source_mask_gate_pass": True,
            "rxj2129_xmm_X2b2_background_gate_pass": True,
            "rxj2129_xmm_passing_instruments": ["MOS2", "pn"],
            "rxj2129_xmm_X3_gas_likelihood_gate_pass": False,
            "rxj2129_hst_measurement_covariance_complete": False,
        }
    )
    for milestone in targets["milestones"]:
        if milestone["id"] == "R1B":
            milestone["status"] = "active_RXJ2129_X3_annular_products_and_HST_covariance"
            milestone["outcome"] = (
                "XMM X1-X2 pass with MOS2 and pn. Construct and audit X3 annular "
                "count/response products while independently executing the frozen HST "
                "21-image, 42x42 measurement-covariance protocol."
            )
    branch = targets["active_branch_decisions"]["rxj2129_strict_readiness"]
    branch["status"] = "XMM_X2_pass_X3_annular_products_and_HST_covariance_active"
    for evidence in (relative(MANIFEST_PATH), relative(QPB_PATH), relative(OUTER_PATH)):
        if evidence not in branch["evidence"]:
            branch["evidence"].append(evidence)
    branch["baryonic_obstruction_result"] = (
        "The satellite stellar term is numeric and passes; the BCG/ICL split remains "
        "explicitly non-identifiable and must be marginalized. The independent XMM "
        "calibration, flare, mask, and quantitative background gates pass with MOS2 "
        "and pn; no gas temperature/density likelihood exists yet."
    )
    branch["remaining"] = (
        "construct and audit frozen MOS2+pn annular count/response products; only if "
        "their five-annulus, 2000-net-count, S/N gates pass, fit the frozen gas "
        "likelihood; independently measure the HST 21-image full 42x42 covariance; "
        "then re-audit strict readiness"
    )


def main() -> None:
    qpb = read_json(QPB_PATH)
    outer = read_json(OUTER_PATH)
    validate_inputs(qpb, outer)

    protocol = read_json(PROTOCOL_PATH)
    manifest = read_json(MANIFEST_PATH)
    report = read_json(REPORT_PATH)
    next_stage = read_json(NEXT_STAGE_PATH)
    execution_targets = read_json(EXECUTION_TARGETS_PATH)

    update_background_protocol(protocol, qpb, outer)
    update_manifest(manifest, qpb, outer)
    update_report(report, outer)
    update_next_stage(next_stage, outer)
    update_execution_targets(execution_targets)

    write_json(PROTOCOL_PATH, protocol)
    write_json(MANIFEST_PATH, manifest)
    write_json(REPORT_PATH, report)
    write_json(NEXT_STAGE_PATH, next_stage)
    write_json(EXECUTION_TARGETS_PATH, execution_targets)
    print(
        json.dumps(
            {
                "status": "X2b2_closed",
                "passing_instruments": ["MOS2", "pn"],
                "next_stage": "X3_annular_count_response_adequacy",
                "temperature_density_fit_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
