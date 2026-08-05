from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19d_member_catalog_extraction.json"
REPORT = ROOT / "results" / "sigma_v19d_member_catalog_extraction" / "report.json"
SCRIPT = ROOT / "scripts" / "build_sigma_v19d_member_catalogs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19d_members", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_sexagesimal_coordinate_conversion() -> None:
    module = load_module()
    assert math.isclose(module.ra_hms_to_deg("06:58:00"), 104.5, rel_tol=1e-12)
    assert math.isclose(
        module.dec_dms_to_deg("-55:54:24"), -55.906666666666666, rel_tol=1e-12
    )


def test_catalogs_retain_every_published_row_and_uncertainty() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for cluster, definition in config["catalogs"].items():
        catalog_rows = rows(ROOT / definition["output"])
        assert len(catalog_rows) == definition["expected_rows"]
        assert len({row["object_id"] for row in catalog_rows}) == len(catalog_rows)
        assert all(float(row["cz_uncertainty_km_s"]) > 0 for row in catalog_rows)
        assert all(row["cluster"] == cluster for row in catalog_rows)
        assert all(row["source_arxiv_id"] == definition["source_arxiv_id"] for row in catalog_rows)


def test_known_rows_are_parsed_without_unit_or_coordinate_loss() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bullet = rows(ROOT / config["catalogs"]["BULLET"]["output"])
    abell = rows(ROOT / config["catalogs"]["ABELL2146"]["output"])
    assert bullet[0]["object_id"] == "01"
    assert bullet[0]["heliocentric_cz_km_s"] == "87495"
    assert bullet[0]["cz_uncertainty_km_s"] == "38"
    assert math.isclose(float(bullet[0]["ra_deg"]), 104.48333333333333)
    assert abell[0]["object_id"] == "5"
    assert abell[0]["ra_deg"] == "238.9647"
    assert abell[0]["heliocentric_cz_km_s"] == "69484"
    assert abell[0]["cz_uncertainty_km_s"] == "24"
    assert abell[0]["subcluster_label"] == "B"


def test_abell_subcluster_and_bcg_labels_are_retained() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    abell = rows(ROOT / config["catalogs"]["ABELL2146"]["output"])
    assert {row["subcluster_label"] for row in abell} == {"A", "B"}
    assert sum(row["is_bcg"] == "True" for row in abell) == 2


def test_report_hashes_inputs_and_outputs_and_keeps_targets_sealed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["total_rows"] == 141
    assert report["input_hashes"]["config"] == digest(CONFIG)
    for key in ("source_acquisition_config", "source_acquisition_manifest"):
        assert report["input_hashes"][key] == digest(ROOT / config["parents"][key])
    for cluster, definition in config["catalogs"].items():
        output = ROOT / definition["output"]
        assert report["catalogs"][cluster]["output_sha256"] == digest(output)
    assert report["all_rows_retain_quoted_cz_uncertainty"] is True
    assert report["missing_uncertainties_inferred"] is False
    assert report["lensing_or_halo_payload_used"] is False
    assert report["gravity_parameters_fit"] == 0
