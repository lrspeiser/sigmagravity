from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19z_nsc_member_photometry.json"
RUNNER = ROOT / "scripts" / "download_sigma_v19z_nsc_member_photometry.py"
REPORT = ROOT / "results" / "sigma_v19z_nsc_member_photometry" / "provenance.json"
COVERAGE = ROOT / "results" / "sigma_v19z_nsc_member_photometry" / "coverage_analysis.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19z_download", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v19z_is_frozen_before_member_level_nsc_queries() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["implementation"]["runner_sha256"] == sha256(RUNNER)
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert config["gates"]["exact_member_query_count"] == 141
    assert config["source"]["pre_freeze_center_only_coverage"]["BULLET"][
        "nsc_dr2_objects"
    ] == 222
    assert config["source"]["pre_freeze_center_only_coverage"]["BULLET"][
        "des_dr2_objects"
    ] == 0
    assert config["source"]["pre_freeze_center_only_coverage"]["ABELL2146"][
        "nsc_dr2_objects"
    ] == 104
    assert config["integrity"]["member_level_nsc_query_executed_at_freeze"] is False
    assert (
        config["integrity"]["member_level_nsc_candidate_coverage_known_at_freeze"]
        is False
    )


def test_v19z_forbids_matching_mass_inference_and_target_opening() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    assert authorization["query_only_frozen_member_cones"] is True
    assert authorization["extract_published_bullet_bri_losslessly"] is True
    for key in (
        "select_nsc_counterpart",
        "apply_photometric_quality_cut",
        "derive_stellar_mass",
        "construct_mass_current",
        "read_lensing_or_halo_payload",
        "fit_gravity_parameters",
        "open_holdout",
    ):
        assert authorization[key] is False


def test_v19z_extracts_all_published_bullet_photometry_losslessly(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = tmp_path / "bullet_published_bri.csv"
    summary = module.extract_bullet_photometry(config, output)
    rows = list(csv.DictReader(output.open("r", encoding="utf-8", newline="")))
    assert summary["rows"] == 78
    assert summary["complete_bri_rows"] == 72
    assert summary["missing_bri_rows"] == 6
    assert len(rows) == 78
    complete = [row for row in rows if row["published_bri_available"] == "True"]
    assert len(complete) == 72
    for row in complete:
        assert float(row["r_bessel_mag"]) == pytest.approx(
            float(row["b_bessel_mag"]) - float(row["b_minus_r_bessel_mag"])
        )
        assert float(row["i_bessel_mag"]) == pytest.approx(
            float(row["b_bessel_mag"]) - float(row["b_minus_i_bessel_mag"])
        )
    assert summary["counterpart_selection_performed"] is False


def test_v19z_query_uses_frozen_q3c_cone_and_ordering() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    adql = module.build_adql(
        config,
        {
            "ra_deg": 104.5,
            "dec_deg": -55.9,
            "query_radius_arcsec": 6.0,
        },
    )
    assert "FROM nsc_dr2.object" in adql
    assert "'t'=q3c_radial_query(ra,dec,104.5,-55.9," in adql
    assert "ORDER BY id" in adql
    assert "0.00166666666666667" in adql
    url = module.build_query_url(config, adql)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["REQUEST"] == ["doQuery"]
    assert query["LANG"] == ["ADQL"]
    assert query["FORMAT"] == ["csv"]
    assert query["QUERY"] == [adql]


def response_payload(columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    row = {column: "" for column in columns}
    row.update({"id": "nsc-1", "ra": "239.0", "dec": "66.3"})
    writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def test_v19z_full_mock_acquisition_retains_all_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload = response_payload(config["source"]["query_columns"])
    requested_urls: list[str] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return payload

    def fake_urlopen(request, timeout):
        assert timeout == config["source"]["timeout_seconds"]
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    report = module.acquire(
        CONFIG,
        raw_override=tmp_path / "raw",
        output_override=tmp_path / "results",
        derived_override=tmp_path / "derived" / "bullet_bri.csv",
    )
    assert report["status"] == (
        "all_frozen_nsc_member_cones_and_published_bullet_bri_acquired"
    )
    assert report["query_count"] == 141
    assert report["http_200_count"] == 141
    assert report["total_candidate_rows"] == 141
    assert len(requested_urls) == 141
    assert report["published_bullet_photometry"]["complete_bri_rows"] == 72
    assert report["by_cluster"]["BULLET"]["member_queries"] == 78
    assert report["by_cluster"]["ABELL2146"]["member_queries"] == 63
    assert report["counterpart_selection_performed"] is False
    assert report["filter_transformation_performed"] is False
    assert report["stellar_mass_inference_performed"] is False
    for row in report["records"]:
        assert Path(row["csv_path"]).exists()
        assert Path(row["query_url_path"]).exists()
        assert Path(row["adql_path"]).exists()
        assert row["counterpart_selected"] is False


def test_v19z_real_report_if_acquisition_has_run() -> None:
    if not REPORT.exists():
        pytest.skip("V19Z member-level acquisition has not run after freeze")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == sha256(CONFIG)
    assert report["input_hashes"]["runner"] == sha256(RUNNER)
    assert report["query_count"] == 141
    assert report["http_200_count"] == 141
    assert report["published_bullet_photometry"]["complete_bri_rows"] == 72
    assert report["counterpart_selection_performed"] is False
    assert report["filter_transformation_performed"] is False
    assert report["stellar_mass_inference_performed"] is False
    assert report["lensing_or_halo_payload_opened"] is False
    for row in report["records"]:
        assert sha256(ROOT / row["csv_path"]) == row["csv_sha256"]
        assert sha256(ROOT / row["query_url_path"]) == row["query_url_sha256"]
        assert sha256(ROOT / row["adql_path"]) == row["adql_sha256"]


def test_v19z_coverage_analysis_if_it_has_run() -> None:
    if not COVERAGE.exists():
        pytest.skip("V19Z descriptive candidate-coverage analysis has not run")
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    assert coverage["acquisition_report_sha256"] == sha256(REPORT)
    assert coverage["candidate_rows"] == 244
    assert coverage["published_bullet_complete_bri_rows"] == 72
    assert coverage["clusters"]["BULLET"]["member_cones"] == 78
    assert coverage["clusters"]["ABELL2146"]["member_cones"] == 63
    assert coverage["counterpart_selection_performed"] is False
    assert coverage["photometric_quality_cut_performed"] is False
    assert coverage["filter_transformation_performed"] is False
    assert coverage["stellar_mass_inference_performed"] is False
    assert coverage["mass_current_constructed"] is False
