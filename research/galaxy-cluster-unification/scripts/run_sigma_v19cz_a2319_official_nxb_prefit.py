#!/usr/bin/env python3
"""Run the target-sealed, official-state Resolve NXB prefit on spent A2319 data."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fit_sigma_v19cy_a2319_spectra as fitter
import prepare_sigma_v19cy_a2319_response_inputs as preparation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sigma_v19cz_a2319_official_nxb_prefit.json"
REPORT = ROOT / "results/sigma_v19cz_a2319_official_nxb_prefit/report.json"
ARTIFACT_ROOT = ROOT / "data/processed/sigma_v19cz_a2319_official_nxb_prefit"
EXPECTED_PROTOCOL = "SIGMA-V19CZ-A2319-OFFICIAL-NXB-PREFIT-1.0.0"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(record: dict[str, Any]) -> Path:
    path = ROOT / record["path"]
    if not path.is_file():
        raise RuntimeError(f"registered input is missing: {path}")
    if path.stat().st_size != record["bytes"]:
        raise RuntimeError(f"registered input size changed: {path}")
    if preparation.sha256(path) != record["sha256"]:
        raise RuntimeError(f"registered input hash changed: {path}")
    return path


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG)
    if config.get("protocol_version") != EXPECTED_PROTOCOL:
        raise RuntimeError("unexpected V19CZ protocol version")
    parent = config["parent"]
    parent_report = ROOT / parent["v19cy_report"]
    parent_index = ROOT / parent["v19cy_artifact_index"]
    if preparation.sha256(parent_report) != parent["v19cy_report_sha256"]:
        raise RuntimeError("V19CY terminal report changed")
    if preparation.sha256(parent_index) != parent["v19cy_artifact_index_sha256"]:
        raise RuntimeError("V19CY artifact index changed")
    old_result = load_json(parent_report)
    if old_result.get("terminal_gate_passed") is not False:
        raise RuntimeError("V19CZ requires the frozen V19CY development failure")
    verify_file(config["nxb_model"])
    verify_file(config["diagonal_response"])
    identities: set[tuple[str, str]] = set()
    for row in config["inputs"]:
        verify_file(row)
        identity = (row["branch"], row["region"])
        if identity in identities:
            raise RuntimeError(f"duplicate V19CZ input: {identity}")
        identities.add(identity)
    regions = {row["region"] for row in config["inputs"]}
    if len(config["inputs"]) != 10 or len(regions) != 7:
        raise RuntimeError("V19CZ requires ten spectra across seven A2319 regions")
    base_path = ROOT / config["base_config"]
    base = load_json(base_path)
    base["nxb_protocol"]["second_stage_free_policy"] = config["fit_policy"][
        "second_stage_free_policy"
    ]
    if base["fit_protocol"]["nxb_constraint_band_keV"] != config["fit_policy"][
        "band_keV"
    ]:
        raise RuntimeError("V19CZ changed the registered NXB fit band")
    return config, base


def build_bundles(config: dict[str, Any]) -> dict[str, list[dict[str, Path]]]:
    bundles: dict[str, list[dict[str, Path]]] = defaultdict(list)
    for row in config["inputs"]:
        bundles[row["region"]].append({"nxb_pha": ROOT / row["path"]})
    return dict(bundles)


def summarize_prefit(prefit: dict[str, Any]) -> dict[str, Any]:
    free = set(prefit["metadata"]["nxb_free_parameter_indices"])
    all_hits = set(prefit["hard_bound_hits"])
    statistic = float(prefit["statistic"])
    dof = int(prefit["dof"])
    return {
        "region": prefit["region"],
        "statistic": statistic,
        "dof": dof,
        "reduced_chi_square": statistic / dof,
        "converged": bool(prefit["converged"]),
        "free_parameter_indices": sorted(free),
        "free_parameter_hard_bound_hits": sorted(free & all_hits),
        "all_numeric_hard_bound_hits": sorted(all_hits),
        "source_spectra_loaded": bool(prefit["source_spectra_loaded"]),
        "source_energy_distribution_used": bool(
            prefit["source_energy_distribution_used"]
        ),
        "statistic_by_spectrum": prefit["statistic_by_spectrum"],
    }


def artifact_index(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": preparation.sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def generate() -> dict[str, Any]:
    config, base = validate()
    if REPORT.exists() or ARTIFACT_ROOT.exists():
        raise RuntimeError("refusing to overwrite an existing V19CZ result")
    model_path = ROOT / config["nxb_model"]["path"]
    expression, specs = fitter.parse_nxb_model(model_path.read_text(encoding="utf-8"))
    official_local_free = fitter.nxb_free_parameter_indices(
        specs, 1, config["fit_policy"]["second_stage_free_policy"]
    )
    bundles = build_bundles(config)
    distribution = base["runtime"]["wsl_distribution"]
    native_temp = Path(f"//wsl.localhost/{distribution}/tmp")
    staging = Path(tempfile.mkdtemp(prefix="sigma_v19cz_nxb_", dir=native_temp))
    prefits: list[dict[str, Any]] = []
    runtime_error: str | None = None
    try:
        for region, bundle in bundles.items():
            prefits.append(
                fitter.run_nxb_prefit(
                    base, region, bundle, expression, specs, staging
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        # Retain exact partial evidence and fail closed.
        runtime_error = str(exc)
    ARTIFACT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, ARTIFACT_ROOT)
    shutil.rmtree(staging)
    summaries = [summarize_prefit(row) for row in prefits]
    gates = config["gates"]
    gate_results = {
        "all_regions_completed": len(summaries) == gates["required_regions"],
        "all_prefits_converged": bool(summaries)
        and all(row["converged"] for row in summaries),
        "all_reduced_chi_square_within_limit": bool(summaries)
        and all(
            row["reduced_chi_square"]
            <= gates["maximum_reduced_chi_square_each_region"]
            for row in summaries
        ),
        "no_free_parameter_at_hard_bound": bool(summaries)
        and all(not row["free_parameter_hard_bound_hits"] for row in summaries),
        "no_source_spectrum_or_energy_used": bool(summaries)
        and all(
            not row["source_spectra_loaded"]
            and not row["source_energy_distribution_used"]
            for row in summaries
        ),
        "runtime_completed_without_error": runtime_error is None,
    }
    terminal_pass = all(gate_results.values())
    artifacts = artifact_index(ARTIFACT_ROOT)
    report = {
        "protocol_version": "SIGMA-V19CZ-A2319-OFFICIAL-NXB-PREFIT-RESULT-1.0.0",
        "status": (
            "passed_official_nxb_prefit_gate"
            if terminal_pass
            else "failed_official_nxb_prefit_gate"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": preparation.sha256(CONFIG),
        "base_config_sha256": preparation.sha256(ROOT / config["base_config"]),
        "parent_report_sha256": config["parent"]["v19cy_report_sha256"],
        "official_second_stage_local_free_indices": official_local_free,
        "prefits": summaries,
        "gate_results": gate_results,
        "runtime_error": runtime_error,
        "terminal_gate_passed": terminal_pass,
        "a2319_source_fit_authorized": terminal_pass,
        "signed_gas_current_constructed": False,
        "validation_or_holdout_accessed": False,
        "lensing_halo_or_gravity_target_accessed": False,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(row["bytes"] for row in artifacts),
        "artifacts": artifacts,
        "decision": config["decision"]["pass" if terminal_pass else "fail"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_name(REPORT.name + ".writing")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, REPORT)
    return report


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
