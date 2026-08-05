import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19ax_delve_dr3_coadd_acquisition" / "report.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v19ax_failed_the_weight_support_gate_without_photometry():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "failed_closed"
    assert report["products"]["count"] == 12
    assert report["products"]["shapes"] == [[2331, 2333]]
    assert report["gate_results"] == {
        "all_candidates_inside_every_product": True,
        "common_shape": True,
        "exact_products": True,
        "exact_sia_rows": True,
        "minimum_positive_weight_fraction_every_band": False,
        "no_source_photometry_or_association": True,
    }
    assert not report["source_photometry_or_candidate_association_computed"]


def test_v19ax_weight_support_cannot_reach_the_prior_completeness_gate():
    report = json.loads(REPORT.read_text())
    support = report["weight_support"]
    assert support["candidate_centers_with_positive_weight_by_band"] == {
        "g": 441,
        "i": 568,
        "r": 568,
        "z": 566,
    }
    assert support["candidate_centers_with_positive_weight_all_griz"] == 439
    assert support["maximum_possible_complete_candidate_fraction"] == 439 / 568
    assert support["members_with_at_least_one_all_griz_supported_candidate"] == 52
    union = report["post_failure_optimistic_union"]
    assert union["prior_signed_stack_complete_candidates"] == 200
    assert union["coadd_and_prior_stack_intersection"] == 164
    assert union["union_candidates"] == 475
    assert union["union_fraction"] == 475 / 568
    assert union["members_with_at_least_one_union_candidate"] == 57
    assert union["union_fraction"] < 0.9


def test_v19ax_manifest_and_products_are_hash_bound():
    report = json.loads(REPORT.read_text())
    manifest = ROOT / report["products"]["manifest"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report["products"][
        "manifest_sha256"
    ]
    manifest_rows = rows(manifest)
    assert len(manifest_rows) == 12
    assert {(row["band"], row["product"]) for row in manifest_rows} == {
        (band, product) for band in "griz" for product in ("image", "mask", "weight")
    }
    for row in manifest_rows:
        path = ROOT / row["output_path"]
        assert path.stat().st_size == int(row["bytes"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_v19ax_sia_inputs_are_hash_bound():
    report = json.loads(REPORT.read_text())
    raw = ROOT / "data" / "raw" / "sigma_v19ax_delve_dr3_coadd_acquisition"
    assert hashlib.sha256((raw / "sia_response.xml").read_bytes()).hexdigest() == report[
        "sia"
    ]["response_sha256"]
    assert hashlib.sha256((raw / "sia_query_url.txt").read_bytes()).hexdigest() == report[
        "sia"
    ]["query_url_sha256"]
    summarizer = ROOT / report["post_failure_summarizer"]
    assert hashlib.sha256(summarizer.read_bytes()).hexdigest() == report[
        "post_failure_summarizer_sha256"
    ]
