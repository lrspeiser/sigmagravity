import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19ay_signed_flux_likelihood_validation" / "report.json"


def test_v19ay_failed_without_opening_ambiguous_candidates():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "failed_closed"
    assert report["validation"] == {
        "mean_reciprocal_rank": 0.5,
        "top1_retrievals": 1,
        "true_pair_ranks": {"21": 2, "26": 2, "57": 1, "66": 4, "71": 4},
    }
    assert report["gate_results"] == {
        "all_scores_finite": True,
        "exact_score_rows": True,
        "minimum_mrr": False,
        "minimum_top1": False,
        "no_ambiguous_candidate_scoring": True,
    }
    assert not report["ambiguous_candidate_scoring_performed"]


def test_v19ay_output_is_complete_and_hash_bound():
    report = json.loads(REPORT.read_text())
    path = ROOT / report["outputs"]["validation_scores"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["outputs"][
        "validation_scores_sha256"
    ]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 25
    assert sum(row["is_true_pair"] == "True" for row in rows) == 5
    assert all(row["log_photometric_score"] for row in rows)
