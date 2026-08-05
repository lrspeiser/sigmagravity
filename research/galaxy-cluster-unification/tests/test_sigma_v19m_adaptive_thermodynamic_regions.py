from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19m_adaptive_thermodynamic_regions.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19m_adaptive_thermodynamic_regions.py"
REPORT = ROOT / "results" / "sigma_v19m_adaptive_thermodynamic_regions" / "report.json"
TOPOLOGY = (
    ROOT
    / "results"
    / "sigma_v19m_adaptive_thermodynamic_regions"
    / "topology_diagnostics.json"
)
VISUAL = (
    ROOT / "results" / "sigma_v19m_adaptive_thermodynamic_regions" / "visual_audit.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19m_was_frozen_before_binning_or_spectra() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    integrity = config["integrity"]
    assert integrity["v19m_region_outcome_known_at_freeze"] is False
    assert integrity["v19m_contbin_run_at_freeze"] is False
    assert integrity["spectrum_or_response_constructed_at_freeze"] is False
    assert config["sample"]["same_binning_executable_thresholds_and_gates"] is True
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected


def test_v19m_region_gate_passed_without_downstream_access() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "both_adaptive_thermodynamic_region_gates_passed"
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    clusters = {row["cluster"]: row for row in report["clusters"]}
    assert clusters["BULLET"]["region_count"] == 392
    assert clusters["BULLET"]["valid_region_count"] == 366
    assert clusters["ABELL2146"]["region_count"] == 138
    assert clusters["ABELL2146"]["valid_region_count"] == 128
    assert all(all(row["gates"].values()) for row in clusters.values())
    assert report["regional_spectral_extraction_authorized"] is True
    assert report["spectrum_or_response_constructed"] is False
    assert report["temperature_density_mach_or_speed_fitted"] is False
    assert report["lensing_target_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False


def test_v19m_post_hash_topology_and_visual_audits_passed() -> None:
    topology = json.loads(TOPOLOGY.read_text(encoding="utf-8"))
    visual = json.loads(VISUAL.read_text(encoding="utf-8"))
    assert topology["v19m_report_sha256"] == sha256(REPORT)
    assert visual["v19m_report_sha256"] == sha256(REPORT)
    assert visual["topology_diagnostics_sha256"] == sha256(TOPOLOGY)
    assert all(row["all_admitted_regions_one_connected_component"] for row in topology["clusters"])
    assert all(row["passed"] for row in visual["clusters"])
    assert visual["scientific_threshold_changed"] is False
    assert visual["region_selected_or_rejected_by_visual_appearance"] is False
