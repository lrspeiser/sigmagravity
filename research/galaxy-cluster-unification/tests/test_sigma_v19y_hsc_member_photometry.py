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
CONFIG = ROOT / "configs" / "sigma_v19y_hsc_member_photometry.json"
RUNNER = ROOT / "scripts" / "download_sigma_v19y_hsc_member_photometry.py"
REPORT = ROOT / "results" / "sigma_v19y_hsc_member_photometry" / "provenance.json"
COVERAGE = ROOT / "results" / "sigma_v19y_hsc_member_photometry" / "coverage_analysis.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19y_download", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v19y_is_frozen_before_member_level_hsc_queries() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["implementation"]["runner_sha256"] == sha256(RUNNER)
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert config["gates"]["exact_member_query_count"] == 141
    assert config["member_catalogs"]["BULLET"]["expected_rows"] == 78
    assert config["member_catalogs"]["ABELL2146"]["expected_rows"] == 63
    assert config["member_catalogs"]["BULLET"]["query_radius_arcsec"] == 6.0
    assert config["member_catalogs"]["ABELL2146"]["query_radius_arcsec"] == 1.0
    assert config["integrity"]["member_level_query_executed_at_original_freeze"] is False
    assert config["integrity"]["candidate_coverage_known_at_original_freeze"] is False
    correction = config["pre_execution_schema_gate_correction"]
    assert correction["opened_member_cones"] == 30
    assert correction["candidate_rows_seen"] == 247
    assert correction["pre_correction_file_count"] == 60
    assert correction["counterpart_selected_before_correction"] is False
    assert correction["gravity_formula_or_parameter_changed_before_correction"] is False


def test_v19y_forbids_selection_mass_inference_and_target_opening() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    assert authorization["query_only_frozen_member_cones"] is True
    assert authorization["retain_every_candidate_row"] is True
    for key in (
        "select_hsc_counterpart",
        "apply_photometric_quality_cut",
        "derive_stellar_mass",
        "construct_mass_current",
        "read_lensing_or_halo_payload",
        "fit_gravity_parameters",
        "open_holdout",
    ):
        assert authorization[key] is False


def test_v19y_query_url_uses_frozen_radius_and_columns() -> None:
    module = load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    url = module.build_query_url(
        config,
        {
            "ra_deg": 104.5,
            "dec_deg": -55.9,
            "query_radius_arcsec": 6.0,
        },
    )
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "catalogs.mast.stsci.edu"
    assert float(query["radius"][0]) == pytest.approx(6.0 / 3600.0)
    assert query["columns"][0] == "[" + ",".join(
        config["source"]["query_columns"]
    ) + "]"


def response_payload(columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    row = {column: "" for column in columns}
    row.update({"MatchID": "12345", "MatchRA": "239.0", "MatchDec": "66.3"})
    writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def test_v19y_parser_accepts_documented_empty_cone_and_exact_schema() -> None:
    module = load_module()
    columns = json.loads(CONFIG.read_text(encoding="utf-8"))["source"][
        "query_columns"
    ]
    assert module.parse_response(b"", columns) == (0, [])
    assert module.parse_response(response_payload(columns), columns) == (1, columns)
    f606w_only = columns[:12] + columns[15:18]
    assert module.parse_response(response_payload(f606w_only), columns) == (
        1,
        f606w_only,
    )
    partial_triplet = columns[:12] + columns[15:17]
    with pytest.raises(RuntimeError, match="complete ordered"):
        module.parse_response(response_payload(partial_triplet), columns)
    with pytest.raises(RuntimeError, match="complete ordered"):
        module.parse_response(b"MatchID,MatchRA,MatchDec\n1,2,3\n", columns)


def test_v19y_full_acquisition_retains_all_141_mock_cones(
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
    )
    assert report["status"] == "all_frozen_member_cones_acquired_unmatched_and_hashed"
    assert report["query_count"] == 141
    assert report["http_200_count"] == 141
    assert report["total_candidate_rows"] == 141
    assert len(requested_urls) == 141
    assert report["by_cluster"]["BULLET"]["member_queries"] == 78
    assert report["by_cluster"]["ABELL2146"]["member_queries"] == 63
    assert report["counterpart_selection_performed"] is False
    assert report["stellar_mass_inference_performed"] is False
    assert all(row["counterpart_selected"] is False for row in report["records"])
    for row in report["records"]:
        csv_path = Path(row["csv_path"])
        url_path = Path(row["query_url_path"])
        assert csv_path.exists()
        assert url_path.exists()
        assert sha256(csv_path) == row["csv_sha256"]
        assert sha256(url_path) == row["query_url_sha256"]


def test_v19y_real_report_if_acquisition_has_run() -> None:
    if not REPORT.exists():
        pytest.skip("V19Y member-level acquisition has not run after the freeze commit")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == sha256(CONFIG)
    assert report["input_hashes"]["runner"] == sha256(RUNNER)
    assert report["query_count"] == 141
    assert report["http_200_count"] == 141
    assert len(report["records"]) == 141
    assert report["all_raw_candidate_rows_retained"] is True
    assert report["counterpart_selection_performed"] is False
    assert report["stellar_mass_inference_performed"] is False
    assert report["lensing_or_halo_payload_opened"] is False
    for row in report["records"]:
        csv_path = ROOT / row["csv_path"]
        url_path = ROOT / row["query_url_path"]
        assert sha256(csv_path) == row["csv_sha256"]
        assert sha256(url_path) == row["query_url_sha256"]


def test_v19y_coverage_analysis_if_it_has_run() -> None:
    if not COVERAGE.exists():
        pytest.skip("V19Y descriptive candidate-coverage analysis has not run")
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    assert coverage["acquisition_report_sha256"] == sha256(REPORT)
    assert coverage["candidate_rows"] == 855
    assert coverage["clusters"]["BULLET"]["member_cones"] == 78
    assert coverage["clusters"]["ABELL2146"]["member_cones"] == 63
    assert coverage["counterpart_selection_performed"] is False
    assert coverage["photometric_quality_cut_performed"] is False
    assert coverage["stellar_mass_inference_performed"] is False
    assert coverage["mass_current_constructed"] is False
