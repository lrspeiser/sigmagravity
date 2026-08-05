from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v18b_replacement_pair_readiness.json"
REPORT = ROOT / "results" / "sigma_v18b_replacement_pair_readiness" / "report.json"
SCRIPT = ROOT / "scripts" / "audit_sigma_v18b_replacement_pair_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v18b_readiness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inherits_member_gate_and_forbids_formula_selection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / config["parents"]["dynamical_stress_gate"]).read_text(
            encoding="utf-8"
        )
    )
    required = parent["stage_B_collisionless_member_stress"][
        "required_for_each_cluster"
    ]
    assert required["minimum_secure_members_inside_1_8_Mpc"] == 50
    assert config["universal_selection"]["minimum_secure_members_inside_aperture"] == 50
    assert config["universal_selection"]["projected_aperture_kpc"] == 1800.0
    assert config["authorization"]["formula_or_spatial_kernel_selection_authorized"] is False
    assert config["authorization"]["lensing_target_access_authorized"] is False


def test_photometry_representations_give_same_ab_magnitude() -> None:
    module = load_module()
    magnitude = 20.0
    flux = 10.0 ** (-0.4 * (magnitude + 48.6))
    from_magnitude = module.ab_magnitude(
        {"m": str(magnitude)}, {"photometry_column": "m", "photometry_kind": "ab_magnitude"}
    )
    from_flux = module.ab_magnitude(
        {"f": str(flux)}, {"photometry_column": "f", "photometry_kind": "cgs_fnu"}
    )
    assert math.isclose(from_magnitude, from_flux, rel_tol=0.0, abs_tol=1e-12)


def test_report_authorizes_only_source_construction() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    by_cluster = {item["cluster"]: item for item in report["clusters"]}
    assert by_cluster["MACS0416"]["selected_secure_members"] == 231
    assert by_cluster["PLCKG287"]["selected_secure_members"] == 129
    assert all(item["member_gate_passed"] for item in by_cluster.values())
    assert all(item["common_f160w_weight_rule"] for item in by_cluster.values())
    assert report["source_construction_authorized"] is True
    assert report["formula_or_spatial_kernel_selection_authorized"] is False
    assert report["lensing_target_opened"] is False
    assert report["holdout_opened"] is False
    assert report["gravity_parameters_fit"] == 0


def test_report_hashes_every_frozen_input() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == digest(CONFIG)
    for cluster_name, cluster in config["clusters"].items():
        for key in ("member_catalog", "baryon_report"):
            assert report["input_hashes"][f"{cluster_name}_{key}"] == digest(
                ROOT / cluster[key]
            )
        baryon_key = "baryon_sources" if "baryon_sources" in cluster else "baryon_map"
        assert report["input_hashes"][f"{cluster_name}_{baryon_key}"] == digest(
            ROOT / cluster[baryon_key]
        )
