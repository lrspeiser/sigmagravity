#!/usr/bin/env python3
"""Finalize RX J2129 after the active H2 and X4 observable jobs finish."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "r1_rxj2129_terminal_observable_disposition_protocol.json"
H2_CONFIG = ROOT / "configs/r1_rxj2129_hst_h2_centroid_execution_protocol.json"
OUTPUT = ROOT / "results" / "r1_rxj2129_terminal_observable_disposition" / "report.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_artifact(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def verify_path_sha(
    path_value: str,
    expected_sha256: str,
    label: str,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    path = resolve_artifact(path_value)
    if not path.is_file():
        raise RuntimeError(f"{label} artifact is absent: {path}")
    observed_bytes = path.stat().st_size
    observed_sha256 = sha256(path)
    if expected_bytes is not None and observed_bytes != int(expected_bytes):
        raise RuntimeError(f"{label} artifact byte count changed: {path}")
    if observed_sha256.lower() != str(expected_sha256).lower():
        raise RuntimeError(f"{label} artifact checksum changed: {path}")
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "integrity_passed": True,
    }


def verify_artifact_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    required = {"path", "bytes", "sha256"}
    if not required.issubset(record):
        missing = sorted(required - set(record))
        raise RuntimeError(f"{label} artifact record lacks: {', '.join(missing)}")
    return verify_path_sha(
        str(record["path"]), record["sha256"], label, int(record["bytes"])
    )


def record_current_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "integrity_passed": True,
    }


def main() -> None:
    protocol = read(PROTOCOL)
    paths = {
        name: ROOT / relative_path
        for name, relative_path in protocol["inputs"].items()
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "terminal disposition is not ready; missing final inputs: "
            + ", ".join(missing)
        )
    ceiling, h2, x4 = (
        read(paths["public_data_ceiling"]),
        read(paths["H2_report"]),
        read(paths["X4_report"]),
    )
    if not ceiling["hard_public_data_shortfall_established"]:
        raise RuntimeError("the global public-data ceiling is not established")
    if h2["status"] not in {"pass", "fail"} or x4["status"] not in {"pass", "fail"}:
        raise RuntimeError("H2 or X4 lacks a terminal pass/fail status")
    h2_count_consistent = bool(
        h2["images_attempted"] == 21 and h2["band_fits_attempted"] == 42
    )
    h2_consistent = bool(
        h2_count_consistent
        and h2["status"] == ("pass" if all(h2["gates"].values()) else "fail")
    )
    x4_count_consistent = bool(
        x4["product_counts"]
        == x4["expected_product_counts"]
        == {
            "rmfs": 12,
            "direct_arfs": 12,
            "central_source_arfs": 12,
            "cross_region_arfs": 72,
        }
    )
    x4_consistent = bool(
        x4_count_consistent
        and x4["status"]
        == ("pass" if x4["gates"]["X4_response_products_passed"] else "fail")
    )
    if not h2_consistent or not x4_consistent:
        raise RuntimeError("a component status is inconsistent with its frozen gates")

    h2_artifacts = {
        name: verify_artifact_record(record, f"H2 {name}")
        for name, record in h2["outputs"].items()
    }
    if set(h2_artifacts) != {
        "H2_band_fit_ledger",
        "H2_image_ledger",
        "H2_centroid_draws",
        "H2_diagnostic",
    }:
        raise RuntimeError("H2 report does not enumerate the four frozen output artifacts")

    h2_config = read(H2_CONFIG)
    if h2["protocol_version"] != h2_config["protocol_version"]:
        raise RuntimeError("H2 report protocol version differs from its frozen execution config")
    h2_inputs: dict[str, dict[str, Any]] = {}
    for name, record in h2_config["inputs"].items():
        if name == "science_and_weight_files_from_parent":
            continue
        h2_inputs[name] = verify_artifact_record(record, f"H2 input {name}")
    parent_record = h2_config["parent_protocol"]
    h2_inputs["parent_protocol"] = verify_artifact_record(
        parent_record, "H2 parent protocol"
    )
    h1_record = h2_config["H1_gate"]
    h2_inputs["H1_report"] = verify_artifact_record(
        {
            "path": h1_record["report"],
            "bytes": h1_record["bytes"],
            "sha256": h1_record["sha256"],
        },
        "H2 H1 report",
    )
    h1 = read(resolve_artifact(h1_record["report"]))
    if h1["status"] != h1_record["required_status"]:
        raise RuntimeError("H2 predecessor H1 report does not have its required status")
    implementation = h2_config["implementation"]
    h2_inputs["runner"] = verify_path_sha(
        implementation["runner"],
        implementation["runner_sha256"],
        "H2 runner",
    )
    parent = read(resolve_artifact(parent_record["path"]))
    if parent["protocol_version"] != parent_record["required_version"]:
        raise RuntimeError("H2 parent protocol version differs from its frozen declaration")
    for band in h2_config["inputs"]["science_and_weight_files_from_parent"]:
        item = parent["inputs"][band]
        h2_inputs[f"{band}_science"] = verify_path_sha(
            item["path"], item["sha256"], f"H2 {band} science", item["bytes"]
        )
        h2_inputs[f"{band}_weight"] = verify_path_sha(
            item["weight_path"],
            item["weight_sha256"],
            f"H2 {band} weight",
            item["weight_bytes"],
        )
    if len(h2_inputs) != 11:
        raise RuntimeError("H2 execution config does not bind the expected 11 input artifacts")
    static_audit_path = resolve_artifact(implementation["static_audit_report"])
    static_audit = read(static_audit_path)
    if (
        static_audit["status"] != "pass"
        or static_audit["runner_sha256"].lower()
        != implementation["runner_sha256"].lower()
        or not all(static_audit["gates"].values())
        or static_audit["HST_arc_pixels_accessed_during_this_static_audit"] is not False
    ):
        raise RuntimeError("H2 static pre-pixel audit is not valid for the frozen runner")

    x4_inputs = {
        name: verify_artifact_record(record, f"X4 input {name}")
        for name, record in x4["inputs"].items()
    }
    if set(x4_inputs) != {
        "protocol",
        "map_convergence",
        "production_runner",
        "audit_implementation",
    }:
        raise RuntimeError("X4 report does not bind the four frozen implementation inputs")
    x4_manifest_path = resolve_artifact(str(x4["manifest"]))
    if not x4_manifest_path.is_file():
        raise RuntimeError(f"X4 manifest is absent: {x4_manifest_path}")
    x4_manifest = read(x4_manifest_path)
    if x4_manifest["product_counts"] != x4["product_counts"]:
        raise RuntimeError("X4 report and manifest product counts differ")
    x4_products = [
        verify_artifact_record(record, f"X4 product {index}")
        for index, record in enumerate(x4_manifest["products"])
    ]
    x4_response_product_count = sum(
        record.get("kind") != "detector_map" for record in x4_manifest["products"]
    )
    x4_detector_map_count = sum(
        record.get("kind") == "detector_map" for record in x4_manifest["products"]
    )
    if x4_response_product_count != 108:
        raise RuntimeError("X4 manifest does not enumerate 108 response products")
    if x4_detector_map_count != 8:
        raise RuntimeError("X4 manifest does not enumerate eight detector maps")
    if len({record["path"] for record in x4_products}) != len(x4_products):
        raise RuntimeError("X4 manifest contains duplicate product paths")

    h2_pass = h2["status"] == "pass"
    x4_pass = x4["status"] == "pass"
    if h2_pass and x4_pass:
        branch = "both_observable_production_gates_pass_global_ceiling_binding"
    elif not h2_pass and not x4_pass:
        branch = "both_observable_production_gates_fail_global_ceiling_binding"
    elif not h2_pass:
        branch = "H2_fails_X4_passes_global_ceiling_binding"
    else:
        branch = "H2_passes_X4_fails_global_ceiling_binding"

    report = {
        "report_version": "R1B3-RXJ2129-terminal-observable-disposition-0.2-integrity-bound",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "inputs": {
            "finalizer_implementation": record_current_artifact(
                Path(__file__).resolve()
            ),
            "protocol": {
                "path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(PROTOCOL),
            },
            **{
                name: {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(path),
                }
                for name, path in paths.items()
            },
        },
        "component_outcomes": {
            "H2": {
                "status": h2["status"],
                "images_attempted": h2["images_attempted"],
                "images_accepted": h2["images_accepted"],
                "gates": h2["gates"],
                "local_H3_authorization": h2["authorization"][
                    "assemble_H3_covariance"
                ],
            },
            "X4": {
                "status": x4["status"],
                "product_counts": x4["product_counts"],
                "expected_product_counts": x4["expected_product_counts"],
                "gates": x4["gates"],
                "local_X5_authorization": x4["authorization"][
                    "construct_X5_joint_likelihood_scaffold"
                ],
            },
        },
        "status_consistency_checks": {
            "H2_attempt_counts_are_21_images_and_42_band_fits": h2_count_consistent,
            "H2_status_derived_from_all_frozen_gates": h2_consistent,
            "X4_counts_match_12_RMF_12_direct_12_central_72_cross": x4_count_consistent,
            "X4_status_derived_from_frozen_product_gate": x4_consistent,
            "global_public_data_ceiling_passed": True,
        },
        "artifact_integrity": {
            "H2": {
                "execution_config": record_current_artifact(H2_CONFIG),
                "static_pre_pixel_audit": record_current_artifact(static_audit_path),
                "immutable_input_artifact_count": len(h2_inputs),
                "all_immutable_inputs_rehashed": all(
                    record["integrity_passed"] for record in h2_inputs.values()
                ),
                "immutable_inputs": h2_inputs,
                "artifact_count": len(h2_artifacts),
                "all_reported_artifacts_rehashed": all(
                    record["integrity_passed"] for record in h2_artifacts.values()
                ),
                "artifacts": h2_artifacts,
            },
            "X4": {
                "manifest_path": str(x4_manifest_path),
                "manifest_sha256": sha256(x4_manifest_path),
                "manifest_product_count": len(x4_products),
                "response_product_count": x4_response_product_count,
                "detector_map_count": x4_detector_map_count,
                "input_artifact_count": len(x4_inputs),
                "all_implementation_inputs_rehashed": all(
                    record["integrity_passed"] for record in x4_inputs.values()
                ),
                "implementation_inputs": x4_inputs,
                "all_manifest_products_rehashed": all(
                    record["integrity_passed"] for record in x4_products
                ),
            },
        },
        "branch": branch,
        "global_disposition": {
            "ten_system_hard_shortfall_changes": False,
            "minimum_strict_system_deficit_even_if_RXJ2129_passed": ceiling[
                "RXJ2129_outcome_independence"
            ]["minimum_remaining_strict_system_deficit_if_RXJ2129_passes"],
            "RXJ2129_counts_as_strict_ready_population_system": False,
            "population_R2_identifiable": False,
            "unification_claim": "withheld_due_public_data_identifiability_ceiling",
        },
        "authorization": {
            "retain_and_hash_H2_products": True,
            "retain_and_hash_X4_products": True,
            "assemble_H3_covariance": False,
            "construct_X5_joint_likelihood": False,
            "select_another_system": False,
            "reconstruct_dynamical_or_Weyl_response": False,
            "cross_validate_latent_response": False,
            "fit_new_force_or_action": False,
        },
        "decision": "close_RXJ2129_observable_branch_and_stop_population_unification_program",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
