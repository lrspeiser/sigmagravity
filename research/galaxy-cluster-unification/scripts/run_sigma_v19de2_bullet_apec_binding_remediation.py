#!/usr/bin/env python3
"""Bind exact AtomDB data before rerunning the unchanged V19DE profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19de_bullet_integrated_redshift_profile as v19de

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19de2_bullet_apec_binding_remediation.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19de2_bullet_apec_binding_remediation"
PREFLIGHT_STATUS = "v19de2_payload_blind_atomdb_binding_passed"
PASS_STATUS = "bullet_integrated_profile_apec_binding_remediation_passed"


def validate_file(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]) or v19de.sha256(path) != record["sha256"]:
        raise RuntimeError(f"V19DE2 frozen XSPEC model-data file changed: {path}")
    return path


def validate_frozen(config: dict[str, Any], *, hash_payload: bool) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    if config.get("freeze_state") != "frozen_after_v19de_invalid_apec_execution_before_source_rerun":
        raise RuntimeError("V19DE2 is not frozen before the source rerun")
    implementation = config["implementation"]
    if implementation["runner"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise RuntimeError("V19DE2 config names another runner")
    if implementation["runner_sha256"] != v19de.sha256(Path(__file__).resolve()):
        raise RuntimeError("V19DE2 runner changed after freeze")
    base_path = ROOT / config["parents"]["v19de_config"]["path"]
    if not base_path.is_file() or v19de.sha256(base_path) != config["parents"]["v19de_config"]["sha256"]:
        raise RuntimeError("V19DE2 scientific-method parent changed")
    invalid_path = ROOT / config["parents"]["v19de_invalid_report"]["path"]
    invalid_parent = config["parents"]["v19de_invalid_report"]
    if not invalid_path.is_file() or v19de.sha256(invalid_path) != invalid_parent["sha256"]:
        raise RuntimeError("V19DE2 invalid-execution parent changed")
    if v19de.load_json(invalid_path).get("status") != invalid_parent["required_status"]:
        raise RuntimeError("V19DE2 invalid-execution parent status changed")
    base_config = v19de.load_json(base_path)
    products = v19de.validate_frozen(base_config, hash_payload=hash_payload)
    auth = config["authorization"]
    if not (
        auth["bind_and_probe_xspec_model_data_without_source"]
        and auth["rerun_unchanged_v19de_integrated_profile_after_committed_preflight"]
        and not auth["change_v19de_scientific_method_or_gate"]
        and not auth["open_any_regional_source_line_or_velocity"]
        and not auth["open_obsid554_or_abell2146"]
        and not auth["open_lensing_halo_gravity_or_action"]
    ):
        raise RuntimeError("V19DE2 authorization boundary is open")
    return base_config, base_path, products


def probe_component(component: Any, bins: list[list[float]]) -> dict[str, Any]:
    low = [float(row[0]) for row in bins]
    high = [float(row[1]) for row in bins]
    values = np.asarray(component(low, high), dtype=float)
    total = float(values.sum())
    if values.size != len(bins) or not np.isfinite(values).all() or not math.isfinite(total) or total <= 0:
        raise RuntimeError(f"V19DE2 {component.name} model-data probe is not finite and positive")
    return {"model": component.type, "values": values.tolist(), "integrated_flux": total}


def bind_and_probe_model_data(config: dict[str, Any]) -> dict[str, Any]:
    from sherpa.astro import ui

    model_data = config["xspec_model_data"]
    headas = Path(os.environ.get("HEADAS", "")).resolve()
    expected_headas = Path(model_data["headas"]).resolve()
    if headas != expected_headas:
        raise RuntimeError(f"V19DE2 HEADAS changed: {headas} != {expected_headas}")
    init_path = validate_file(model_data["xspec_init"])
    continuum = validate_file(model_data["apec_continuum"])
    lines = validate_file(model_data["apec_lines"])
    match = re.search(
        r"^\s*ATOMDB_VERSION\s*:\s*([^\s#]+)",
        init_path.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE,
    )
    if match is None or match.group(1) != model_data["atomdb_version"]:
        raise RuntimeError("V19DE2 Xspec.init AtomDB version changed")
    root = Path(model_data["apec_root"])
    if continuum != Path(f"{root}_coco.fits") or lines != Path(f"{root}_line.fits"):
        raise RuntimeError("V19DE2 APECROOT no longer names the frozen model-data pair")
    ui.clean()
    ui.set_xsxset("APECROOT", str(root))
    probe_parameters = model_data["probe_parameters"]
    probes: list[dict[str, Any]] = []
    for model_name in model_data["required_models"]:
        component = ui.create_model_component(model_name, f"v19de2_probe_{model_name}")
        component.kT = float(probe_parameters["kT_keV"])
        component.Abundanc = float(probe_parameters["abundance_solar"])
        component.Redshift = float(probe_parameters["redshift"])
        component.norm = float(probe_parameters["normalization"])
        probes.append(probe_component(component, model_data["probe_bins_keV"]))
    return {
        "headas": str(headas),
        "xspec_init": {"path": str(init_path), "bytes": init_path.stat().st_size, "sha256": v19de.sha256(init_path)},
        "atomdb_version": model_data["atomdb_version"],
        "apec_root": str(root),
        "apec_continuum": {"bytes": continuum.stat().st_size, "sha256": v19de.sha256(continuum)},
        "apec_lines": {"bytes": lines.stat().st_size, "sha256": v19de.sha256(lines)},
        "positive_model_probes": probes,
    }


def execute(config: dict[str, Any], output: Path) -> dict[str, Any]:
    base_config, base_path, _ = validate_frozen(config, hash_payload=True)
    model_data = bind_and_probe_model_data(config)
    configured_probes: dict[str, Any] = {}
    original_configure_session = v19de.configure_session

    def configure_session_with_probe(
        method: dict[str, Any], products: dict[str, Path], branch: str
    ) -> dict[str, Any]:
        session = original_configure_session(method, products, branch)
        expected_root = str(Path(config["xspec_model_data"]["apec_root"]))
        actual_root = str(session["ui"].get_xsxset("APECROOT"))
        if actual_root != expected_root:
            raise RuntimeError(f"V19DE2 APECROOT was lost before {branch}: {actual_root}")
        configured_probes[branch] = {
            "component_1": probe_component(session["first"], config["xspec_model_data"]["probe_bins_keV"]),
            "component_2": probe_component(session["second"], config["xspec_model_data"]["probe_bins_keV"]),
        }
        return session

    v19de.configure_session = configure_session_with_probe
    try:
        result = v19de.execute(base_config, base_path, output)
    finally:
        v19de.configure_session = original_configure_session
    base_passed = result["status"] == v19de.PASS_STATUS
    result["base_v19de_status"] = result["status"]
    result["status"] = PASS_STATUS if base_passed else "bullet_integrated_profile_remediation_scientific_gate_failed"
    result["model_data_binding"] = model_data
    result["configured_component_probes"] = configured_probes
    result["integrated_systematic_and_goodness_stage_authorized"] = bool(
        base_passed and result["integrated_systematic_and_goodness_stage_authorized"]
    )
    return result


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    base_config, _, products = validate_frozen(config, hash_payload=False)
    model_data = bind_and_probe_model_data(config)
    coarse = v19de.inclusive_grid(
        float(base_config["profile"]["optical_redshift_center"]),
        float(base_config["profile"]["half_range"]),
        float(base_config["profile"]["coarse_step"]),
    )
    fine = v19de.inclusive_grid(
        float(base_config["profile"]["optical_redshift_center"]),
        float(base_config["profile"]["fine_half_width"]),
        float(base_config["profile"]["fine_step"]),
    )
    return {
        "status": PREFLIGHT_STATUS,
        "unchanged_v19de_config_sha256": v19de.sha256(ROOT / config["parents"]["v19de_config"]["path"]),
        "branches": list(base_config["model"]["branches"]),
        "coarse_points_per_branch": len(coarse),
        "fine_points_per_branch": len(fine),
        "multistarts_per_point": 2,
        "integrated_product_sizes_verified": {
            key: products[key].stat().st_size for key, value in base_config["data"].items() if isinstance(value, dict)
        },
        "model_data_binding": model_data,
        "source_pha_response_scientific_arrays_opened": False,
        "source_line_temperature_abundance_redshift_or_velocity_fitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = v19de.load_json(config_path)
    try:
        result = preflight(config) if args.preflight_only else execute(config, output)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "v19de2_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "integrated_systematic_and_goodness_stage_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19de.sha256(config_path),
        "runner_sha256": v19de.sha256(Path(__file__).resolve()),
        **result,
        "posterior_predictive_or_thermal_sobol_run": False,
        "regional_source_line_or_velocity_opened": False,
        "obsid554_or_abell2146_opened": False,
        "lensing_halo_gravity_or_action_opened": False,
    }
    name = "preflight_report.json" if args.preflight_only else "report.json"
    v19de.atomic_json(output / name, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_exception")}, indent=2, sort_keys=True))
    required = PREFLIGHT_STATUS if args.preflight_only else PASS_STATUS
    if report["status"] != required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
