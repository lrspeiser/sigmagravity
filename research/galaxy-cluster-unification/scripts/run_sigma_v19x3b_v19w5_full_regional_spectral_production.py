#!/usr/bin/env python3
"""Run the V19W5-authorized successor to frozen V19X3 regional production."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x3_full_regional_spectral_production as inherited_v19x3
import sigma_v19x2_unified_response_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_X2_STATUS = inherited_v19x3.AUTHORIZED_X2_STATUS
FROZEN_STATE = "frozen_after_terminal_v19w5_authorized_v19x2_pass"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_runner(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != FROZEN_STATE:
        raise RuntimeError(
            "V19X3B configuration is not frozen after a V19W5-authorized V19X2 pass"
        )
    runner = ROOT / config["implementation"]["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19X3B configuration names another runner")
    if adapter.sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19X3B runner changed after freeze")
    inherited = ROOT / config["implementation"]["inherited_v19x3_runner"]
    if adapter.sha256(inherited) != config["implementation"][
        "inherited_v19x3_runner_sha256"
    ]:
        raise RuntimeError("V19X3B inherited regional engine changed after freeze")


def validate_frozen_parents(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and adapter.sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19X3B parent changed after freeze: {value}")


def execute(config: dict[str, Any], output: Path, scratch: Path) -> dict[str, Any]:
    runtime = config["runtime_authorization"]
    x2_report_path = ROOT / runtime["required_v19x2_report"]
    x2_report, abundances = inherited_v19x3.validate_x2_authorization(
        config, x2_report_path
    )
    response_report_path = ROOT / runtime["required_response_report"]
    response_report, unified_index = adapter.authorize_unified_index(
        response_report_path,
        expected_config_sha256=config["parents"]["v19w5_config_sha256"],
        expected_runner_sha256=config["parents"]["v19w5_runner_sha256"],
        expected_cells=int(runtime["required_unified_cells"]),
        expected_products=int(runtime["required_unified_products"]),
        expected_status=runtime["required_response_status"],
        authority_label=runtime["response_authority"],
    )
    manifest = inherited_v19x3.inherited_v19x.load_manifest(config)
    plan = inherited_v19x3.build_full_region_plan(config, manifest)
    archives = {
        name: Path(path)
        for name, path in config["execution"]["response_archives"].items()
    }
    validated = adapter.validate_unified_archive(
        manifest,
        unified_index,
        archives,
        recovery_archive=runtime["recovery_archive"],
    )
    result = inherited_v19x3.run_full_regional_production(
        config, output, scratch, plan, validated, abundances
    )
    result.update(
        {
            "v19x2_report_sha256": adapter.sha256(x2_report_path),
            "response_report_sha256": adapter.sha256(response_report_path),
            "response_unified_index_sha256": response_report[
                "unified_product_index"
            ]["sha256"],
            "integrated_abundances_solar": abundances,
            "v19x2_status": x2_report["status"],
            "inherited_v19x3_engine_sha256": config["implementation"][
                "inherited_v19x3_runner_sha256"
            ],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_frozen_runner(config)
    validate_frozen_parents(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(config, output, args.scratch.resolve())
    except Exception as exc:  # noqa: BLE001 - retain terminal production failure
        result = {
            "status": "v19x3b_full_regional_spectral_production_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "source_map_construction_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": adapter.sha256(config_path),
        "runner_sha256": adapter.sha256(Path(__file__).resolve()),
        **result,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["source_map_construction_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
