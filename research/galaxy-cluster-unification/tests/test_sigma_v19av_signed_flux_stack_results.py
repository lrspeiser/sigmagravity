import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19av_signed_flux_stack" / "report.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v19av_failed_only_candidate_completeness():
    report = json.loads(REPORT.read_text())
    assert report["decision"] == "failed_closed"
    assert report["counts"] == {
        "anchor_stack_rows": 75,
        "candidate_complete_griz": 200,
        "candidate_stack_rows": 2840,
        "members_with_at_least_one_complete_candidate": 57,
    }
    assert report["gate_results"] == {
        "all_stack_rows_present": True,
        "candidate_complete_griz_fraction": False,
        "every_member_has_complete_candidate": True,
        "no_candidate_association_scored": True,
        "validation_color_error": True,
        "validation_mrr": True,
        "validation_top1": True,
    }
    assert report["candidate_complete_griz_fraction"] == 200 / 568


def test_v19av_preserves_anchor_validation_behavior():
    report = json.loads(REPORT.read_text())
    metrics = report["validation_metrics"]
    assert metrics["top1_retrievals"] == 3
    assert metrics["mean_reciprocal_rank"] >= 0.65
    assert all(value <= 0.25 for value in metrics["median_absolute_error_mag"].values())
    assert metrics["true_pair_ranks"] == {"21": 3, "26": 1, "57": 1, "66": 3, "71": 1}


def test_v19av_outputs_are_hash_bound_and_have_exact_rows():
    report = json.loads(REPORT.read_text())
    expected_rows = {"anchor_stacks": 75, "candidate_stacks": 2840}
    for name in (
        "anchor_stacks",
        "candidate_stacks",
        "development_color_fit",
        "validation_predictions",
        "validation_retrieval",
    ):
        path = ROOT / report["outputs"][name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report["outputs"][f"{name}_sha256"]
        if name in expected_rows:
            assert len(rows(path)) == expected_rows[name]
