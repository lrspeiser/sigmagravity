import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19aw_delve_candidate_coverage" / "report.json"
HYPOTHESES = (
    ROOT
    / "data"
    / "derived"
    / "sigma_v19au_ambiguous_candidate_image_measurement"
    / "candidate_hypotheses.csv"
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v19aw_failed_closed_without_selecting_a_candidate():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "failed_closed"
    assert report["field"]["rows"] == 2351
    assert report["candidate_coverage"] == {
        "candidates": 568,
        "complete_signed_flux_griz_candidates": 56,
        "complete_signed_flux_griz_fraction": 56 / 568,
        "matches": 108,
        "matching_states": {"multiple": 0, "no_match": 460, "unique": 108},
        "members_with_at_least_one_complete_candidate": 38,
    }
    assert report["gate_results"] == {
        "all_candidates_evaluated": True,
        "complete_candidate_fraction": False,
        "every_member_has_complete_candidate": False,
        "exact_field_rows": True,
        "no_counterpart_selected": True,
    }
    assert not report["candidate_selected_or_ranked"]


def test_v19aw_outputs_are_hash_bound():
    report = json.loads(REPORT.read_text())
    raw = ROOT / "data" / "raw" / "sigma_v19aw_delve_dr3_candidate_coverage"
    expected = {
        raw / "bullet_field.csv": report["field"]["raw_csv_sha256"],
        raw / "bullet_field.adql": report["field"]["raw_adql_sha256"],
        raw / "bullet_field.query_url.txt": report["field"]["raw_query_url_sha256"],
    }
    for name in ("candidate_matches", "candidate_coverage"):
        path = ROOT / report["outputs"][name]
        expected[path] = report["outputs"][f"{name}_sha256"]
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_nearest_object_diagnostic_does_not_rescue_catalog_matching():
    field = rows(
        ROOT
        / "data"
        / "raw"
        / "sigma_v19aw_delve_dr3_candidate_coverage"
        / "bullet_field.csv"
    )
    unique = list({row["candidate_id"]: row for row in rows(HYPOTHESES)}.values())
    field_sky = SkyCoord(
        [float(row["alphawin_j2000"]) for row in field],
        [float(row["deltawin_j2000"]) for row in field],
        unit="deg",
    )
    candidate_sky = SkyCoord(
        [float(row["candidate_ra_deg"]) for row in unique],
        [float(row["candidate_dec_deg"]) for row in unique],
        unit="deg",
    )
    _, separation, _ = candidate_sky.match_to_catalog_sky(field_sky)
    arcsec = separation.arcsec
    assert np.sum(arcsec <= 0.5) == 108
    assert np.sum(arcsec <= 1.0) == 135
    assert np.isclose(np.median(arcsec), 2.5236710637652733)


def test_catalog_provenance_explains_most_of_the_coverage_loss():
    coverage = {
        row["candidate_id"]: row
        for row in rows(
            ROOT
            / "data"
            / "derived"
            / "sigma_v19aw_delve_candidate_coverage"
            / "candidate_coverage.csv"
        )
    }
    unique = {row["candidate_id"]: row for row in rows(HYPOTHESES)}
    counts = {}
    for prefix in ("HSC", "NSC"):
        selected = [key for key in unique if key.startswith(f"{prefix}:")]
        counts[prefix] = {
            "candidates": len(selected),
            "matched": sum(int(coverage[key]["delve_matches_within_radius"]) > 0 for key in selected),
            "complete": sum(
                coverage[key]["has_complete_signed_flux_griz_match"] == "True"
                for key in selected
            ),
        }
    assert counts == {
        "HSC": {"candidates": 529, "matched": 85, "complete": 36},
        "NSC": {"candidates": 39, "matched": 23, "complete": 20},
    }
