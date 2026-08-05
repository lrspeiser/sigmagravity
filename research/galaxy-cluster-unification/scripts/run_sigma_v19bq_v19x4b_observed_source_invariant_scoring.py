#!/usr/bin/env python3
"""Run the frozen V19BP decision on terminal V19X4B/V19BMB products."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19bp_observed_source_invariant_scoring as inherited_v19bp

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STATE = "frozen_after_terminal_v19x4b_and_v19bmb_pass"
AUTHORIZED_X4B_STATUS = (
    "gas_state_posterior_and_common_grids_passed_source_invariant_scoring_authorized"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return inherited_v19bp.sha256(path)


def validate_static(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != FROZEN_STATE:
        raise RuntimeError("V19BQ is not frozen after terminal V19X4B and V19BMB")
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            path = ROOT / value
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"V19BQ parent changed: {value}")
    implementation = config["implementation"]
    runner = ROOT / implementation["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BQ configuration names another runner")
    for name in (
        "runner",
        "inherited_v19bp_runner",
        "gas_stream_module",
        "score_engine_module",
    ):
        path = ROOT / implementation[name]
        if not path.is_file() or sha256(path) != implementation[f"{name}_sha256"]:
            raise RuntimeError(f"V19BQ implementation changed: {name}")


def validate_terminal_reports(
    config: dict[str, Any], x4b_report_path: Path, bmb_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    x4b = load_json(x4b_report_path)
    bmb = load_json(bmb_report_path)
    runtime = config["terminal_authorization"]
    if (
        sha256(x4b_report_path) != config["parents"]["v19x4b_report_sha256"]
        or x4b.get("status") != runtime["required_v19x4b_status"]
        or x4b.get("config_sha256")
        != config["parents"]["v19x4b_config_sha256"]
        or x4b.get("runner_sha256")
        != config["parents"]["v19x4b_runner_sha256"]
        or not x4b.get("source_invariant_scoring_authorized")
        or not x4b.get("gates")
        or not all(x4b["gates"].values())
        or x4b.get("lensing_or_halo_payload_opened") is not False
    ):
        raise RuntimeError("V19BQ requires a passing target-sealed V19X4B report")
    if (
        sha256(bmb_report_path) != config["parents"]["v19bmb_report_sha256"]
        or bmb.get("status") != runtime["required_v19bmb_status"]
        or bmb.get("config_sha256")
        != config["parents"]["v19bmb_config_sha256"]
        or bmb.get("runner_sha256")
        != config["parents"]["v19bmb_runner_sha256"]
        or not bmb.get("invariant_scoring_ready")
        or not bmb.get("gates")
        or not all(bmb["gates"].values())
        or bmb.get("lensing_halo_action_or_gravity_payload_opened") is not False
    ):
        raise RuntimeError("V19BQ requires a passing target-sealed V19BMB report")
    if len(x4b.get("products", [])) != 12 or len(bmb.get("products", [])) != 2:
        raise RuntimeError("V19BQ terminal product inventory changed")
    for product in x4b["products"]:
        inherited_v19bp.validate_product(
            ROOT / product["relative_path"], product, "V19X4B"
        )
    for product in bmb["products"]:
        inherited_v19bp.validate_product(
            ROOT / product["relative_path"], product, "V19BMB"
        )
    return x4b, bmb


def execute(
    config: dict[str, Any], x4b_report_path: Path, bmb_report_path: Path
) -> dict[str, Any]:
    validate_static(config)
    x4b, bmb = validate_terminal_reports(config, x4b_report_path, bmb_report_path)
    output_root = ROOT / config["outputs"]["root"]
    branch_reports: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    for cluster in config["registered_inputs"]["clusters"]:
        stellar_product = inherited_v19bp.product_for(
            bmb["products"], cluster=cluster, role="stellar_morphology_control"
        )
        with np.load(ROOT / stellar_product["relative_path"]) as payload:
            stellar_ids = np.asarray(payload["bin_id"], dtype=np.int64)
            sample_id = np.asarray(payload["sample_id"], dtype=np.int64)
            stellar_ranks = {
                float(fwhm): np.asarray(
                    payload[f"light_percentile_rank_{float(fwhm):g}kpc"],
                    dtype=float,
                )
                for fwhm in config["variants"]["smoothing_fwhm_kpc"]
            }
        if not np.array_equal(sample_id, np.arange(sample_id.size)):
            raise RuntimeError(f"V19BQ {cluster} stellar draw sequence changed")
        for correlation in config["registered_inputs"]["rank_correlations"]:
            regional_product = inherited_v19bp.product_for(
                x4b["products"],
                cluster=cluster,
                role="regional_posterior",
                rank_correlation=float(correlation),
            )
            grid_product = inherited_v19bp.product_for(
                x4b["products"],
                cluster=cluster,
                role="common_grid_summary",
                rank_correlation=float(correlation),
            )
            with np.load(ROOT / regional_product["relative_path"]) as payload:
                region_ids = np.asarray(payload["bin_id"], dtype=np.int64)
                regional_fields = {
                    name: np.asarray(payload[name], dtype=float).T
                    for name in inherited_v19bp.REQUIRED_GAS_FIELDS
                }
                stored_correlation = float(payload["rank_correlation"])
            with np.load(ROOT / grid_product["relative_path"]) as payload:
                east_axis = np.asarray(payload["axis_east_kpc"], dtype=float)
                north_axis = np.asarray(payload["axis_north_kpc"], dtype=float)
                labels = np.asarray(payload["bin_id"], dtype=np.int64)
            if (
                not np.array_equal(region_ids, stellar_ids)
                or not math.isclose(
                    stored_correlation, float(correlation), abs_tol=1.0e-12
                )
                or any(
                    values.shape[0] != sample_id.size
                    for values in regional_fields.values()
                )
            ):
                raise RuntimeError(f"V19BQ {cluster} rho={correlation} pairing changed")
            candidates, arrays = inherited_v19bp.score_branch(
                regional_fields,
                region_ids=region_ids,
                label_grid=labels,
                east_axis_kpc=east_axis,
                north_axis_kpc=north_axis,
                stellar_ranks=stellar_ranks,
                config=config,
            )
            arrays["sample_id"] = sample_id
            arrays["rank_correlation"] = np.asarray(float(correlation))
            token = (
                f"rho_{float(correlation):+.1f}"
                .replace("+", "p")
                .replace("-", "m")
                .replace(".", "p")
            )
            output = output_root / cluster / f"source_invariant_scores_{token}.npz"
            inherited_v19bp.atomic_npz(output, arrays)
            products.append(
                {
                    "cluster": cluster,
                    "rank_correlation": float(correlation),
                    "role": "source_invariant_region_summary",
                    "relative_path": output.relative_to(ROOT).as_posix(),
                    "bytes": output.stat().st_size,
                    "sha256": sha256(output),
                }
            )
            branch_reports.append(
                {
                    "cluster": cluster,
                    "rank_correlation": float(correlation),
                    "regions": int(region_ids.size),
                    "draws": int(sample_id.size),
                    "candidates": candidates,
                }
            )
    expected_branches = len(config["registered_inputs"]["clusters"]) * len(
        config["registered_inputs"]["rank_correlations"]
    )
    decision = inherited_v19bp.aggregate_source_decision(
        branch_reports, expected_branches
    )
    gates = {
        "six_cluster_correlation_branches_scored": len(branch_reports)
        == expected_branches
        == 6,
        "six_hash_bound_source_only_products": len(products) == 6,
        "I4_direction_passes_both_clusters_all_branches": decision[
            "I4_direction_pass"
        ],
        "I4_amplitude_or_I5_scalar_passes_both_clusters_all_branches": decision[
            "I4_amplitude_pass"
        ]
        or decision["I5_scalar_pass"],
        "lensing_halo_action_and_gravity_payload_not_opened": True,
    }
    return {
        "status": (
            "observed_source_invariant_gates_passed_action_derivation_authorized"
            if decision["action_derivation_authorized"] and all(gates.values())
            else "observed_source_invariant_gates_failed_no_action_authorized"
        ),
        "x4b_report_sha256": sha256(x4b_report_path),
        "stellar_control_report_sha256": sha256(bmb_report_path),
        "branch_reports": branch_reports,
        "products": products,
        "aggregate_decision": decision,
        "gates": gates,
        "lensing_halo_action_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--x4b-report", type=Path, required=True)
    parser.add_argument("--stellar-report", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    try:
        result = execute(
            config, args.x4b_report.resolve(), args.stellar_report.resolve()
        )
    except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
        result = {
            "status": "v19bq_observed_source_invariant_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "branch_reports": [],
            "products": [],
            "aggregate_decision": {
                "I4_direction_pass": False,
                "I4_amplitude_pass": False,
                "I5_scalar_pass": False,
                "action_derivation_authorized": False,
            },
            "gates": {"execution_completed": False},
            "lensing_halo_action_or_gravity_payload_opened": False,
            "gravity_formula_or_parameter_changed": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["outputs"]["terminal_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report["status"])
    if report["status"] == "v19bq_observed_source_invariant_execution_failed_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
