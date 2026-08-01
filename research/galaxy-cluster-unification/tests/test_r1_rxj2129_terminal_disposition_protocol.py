import json
import hashlib
from pathlib import Path

import pytest

from scripts import finalize_r1_rxj2129_terminal_observable_disposition as finalizer


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs" / "r1_rxj2129_terminal_observable_disposition_protocol.json"
)


def test_terminal_protocol_is_outcome_blind_and_preserves_component_results() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == (
        "frozen_while_H2_and_X4_active_before_either_final_gate"
    )
    assert protocol["selection_blind"]
    assert not protocol["gravity_residuals_seen"]
    assert not protocol["final_H2_gate_seen"]
    assert not protocol["final_X4_gate_seen"]
    assert protocol["partial_execution_progress_seen"]
    assert protocol["authorization"]["allow_running_H2_and_X4_to_finish"]
    assert protocol["authorization"]["audit_and_hash_completed_H2_and_X4_products"]


def test_global_ceiling_supersedes_local_downstream_interfaces() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    ceiling = json.loads(
        (ROOT / protocol["inputs"]["public_data_ceiling"]).read_text(
            encoding="utf-8"
        )
    )
    assert ceiling["hard_public_data_shortfall_established"]
    assert ceiling["RXJ2129_outcome_independence"][
        "minimum_remaining_strict_system_deficit_if_RXJ2129_passes"
    ] == 9
    authorization = protocol["authorization"]
    assert not authorization["assemble_H3_covariance"]
    assert not authorization["construct_X5_joint_likelihood"]
    assert not authorization["select_another_system"]
    assert not authorization["reconstruct_dynamical_or_Weyl_response"]
    assert not authorization["cross_validate_latent_response"]
    assert not authorization["fit_new_force_or_action"]


def test_terminal_artifact_integrity_rehashes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"immutable observable product")
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    record = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    verified = finalizer.verify_artifact_record(record, "test")
    assert verified["integrity_passed"]
    assert verified["sha256"] == record["sha256"]


def test_terminal_artifact_integrity_rejects_post_report_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"first")
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    record = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    artifact.write_bytes(b"later")
    with pytest.raises(RuntimeError, match="checksum changed"):
        finalizer.verify_artifact_record(record, "test")


def test_terminal_finalizer_rehashes_complete_synthetic_component_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_json(relative: str, value: dict) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def artifact_record(relative: str, content: bytes) -> dict:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {
            "path": relative,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def existing_record(path: Path) -> dict:
        return {
            "path": str(path.relative_to(tmp_path)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    h2_outputs = {
        name: artifact_record(f"data/{name}.bin", name.encode())
        for name in (
            "H2_band_fit_ledger",
            "H2_image_ledger",
            "H2_centroid_draws",
            "H2_diagnostic",
        )
    }
    x4_products = [
        {
            "kind": "synthetic_response",
            **artifact_record(f"external/x4_{index:03d}.arf", bytes([index % 251])),
        }
        for index in range(108)
    ] + [
        {
            "kind": "detector_map",
            **artifact_record(f"external/map_{index:02d}.fits", bytes([index])),
        }
        for index in range(8)
    ]
    counts = {
        "rmfs": 12,
        "direct_arfs": 12,
        "central_source_arfs": 12,
        "cross_region_arfs": 72,
    }
    h2_direct_inputs = {
        name: artifact_record(f"h2_inputs/{name}.bin", name.encode())
        for name in (
            "coordinate_ledger",
            "registration_draws",
            "union_segmentation",
            "PSF_field",
        )
    }
    parent_bands = {}
    for band_name in ("F814W", "F125W"):
        science = artifact_record(
            f"h2_inputs/{band_name}_science.fits", f"{band_name} science".encode()
        )
        weight = artifact_record(
            f"h2_inputs/{band_name}_weight.fits", f"{band_name} weight".encode()
        )
        parent_bands[band_name] = {
            "path": science["path"],
            "bytes": science["bytes"],
            "sha256": science["sha256"],
            "weight_path": weight["path"],
            "weight_bytes": weight["bytes"],
            "weight_sha256": weight["sha256"],
        }
    parent_path = write_json(
        "configs/h2_parent.json",
        {"protocol_version": "synthetic-parent-0.1", "inputs": parent_bands},
    )
    h1_path = write_json("results/h1.json", {"status": "pass"})
    runner = artifact_record("scripts/h2_runner.py", b"frozen runner")
    static_path = write_json(
        "results/h2_static.json",
        {
            "status": "pass",
            "runner_sha256": runner["sha256"],
            "gates": {"runner": True, "pixels": True},
            "HST_arc_pixels_accessed_during_this_static_audit": False,
        },
    )
    h2_config_path = write_json(
        "configs/h2_config.json",
        {
            "protocol_version": "synthetic-H2-0.1",
            "inputs": h2_direct_inputs
            | {"science_and_weight_files_from_parent": ["F814W", "F125W"]},
            "parent_protocol": existing_record(parent_path)
            | {"required_version": "synthetic-parent-0.1"},
            "H1_gate": {
                "report": existing_record(h1_path)["path"],
                "bytes": existing_record(h1_path)["bytes"],
                "sha256": existing_record(h1_path)["sha256"],
                "required_status": "pass",
            },
            "implementation": {
                "runner": runner["path"],
                "runner_sha256": runner["sha256"],
                "static_audit_report": existing_record(static_path)["path"],
            },
        },
    )
    write_json(
        "results/ceiling.json",
        {
            "hard_public_data_shortfall_established": True,
            "RXJ2129_outcome_independence": {
                "minimum_remaining_strict_system_deficit_if_RXJ2129_passes": 9
            },
        },
    )
    write_json(
        "results/h2.json",
        {
            "protocol_version": "synthetic-H2-0.1",
            "status": "pass",
            "images_attempted": 21,
            "band_fits_attempted": 42,
            "images_accepted": 18,
            "gates": {"attempted": True, "accepted": True, "inner": True},
            "outputs": h2_outputs,
            "authorization": {"assemble_H3_covariance": True},
        },
    )
    write_json(
        "data/x4_manifest.json",
        {"product_counts": counts, "products": x4_products},
    )
    write_json(
        "results/x4.json",
        {
            "status": "pass",
            "product_counts": counts,
            "expected_product_counts": counts,
            "manifest": "data/x4_manifest.json",
            "inputs": {
                name: artifact_record(f"inputs/{name}.bin", name.encode())
                for name in (
                    "protocol",
                    "map_convergence",
                    "production_runner",
                    "audit_implementation",
                )
            },
            "gates": {"X4_response_products_passed": True},
            "authorization": {"construct_X5_joint_likelihood_scaffold": True},
        },
    )
    protocol = write_json(
        "configs/protocol.json",
        {
            "inputs": {
                "public_data_ceiling": "results/ceiling.json",
                "H2_report": "results/h2.json",
                "X4_report": "results/x4.json",
            }
        },
    )
    output = tmp_path / "results/terminal/report.json"
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    monkeypatch.setattr(finalizer, "PROTOCOL", protocol)
    monkeypatch.setattr(finalizer, "H2_CONFIG", h2_config_path)
    monkeypatch.setattr(finalizer, "OUTPUT", output)
    finalizer.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["inputs"]["finalizer_implementation"]["integrity_passed"]
    assert report["branch"] == "both_observable_production_gates_pass_global_ceiling_binding"
    assert report["artifact_integrity"]["H2"]["all_reported_artifacts_rehashed"]
    assert report["artifact_integrity"]["H2"]["immutable_input_artifact_count"] == 11
    assert report["artifact_integrity"]["H2"]["all_immutable_inputs_rehashed"]
    assert report["artifact_integrity"]["X4"]["all_manifest_products_rehashed"]
    assert report["artifact_integrity"]["X4"]["manifest_product_count"] == 116
    assert report["artifact_integrity"]["X4"]["response_product_count"] == 108
    assert report["artifact_integrity"]["X4"]["detector_map_count"] == 8
    assert report["artifact_integrity"]["X4"]["input_artifact_count"] == 4
    assert report["artifact_integrity"]["X4"]["all_implementation_inputs_rehashed"]
    assert report["authorization"]["assemble_H3_covariance"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
