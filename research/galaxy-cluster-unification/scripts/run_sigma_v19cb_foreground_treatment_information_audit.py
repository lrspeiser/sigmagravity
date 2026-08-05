from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cb_foreground_treatment_information_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def candidate_weight(diagnostic: dict[str, str], treatment: dict[str, Any]) -> float:
    if diagnostic["quality_controlled_foreground_contamination"] == "true":
        return float(treatment["quality_controlled_weight"])
    if diagnostic["foreground_astrometric_evidence"] == "true":
        return float(treatment["other_foreground_evidence_weight"])
    return 1.0


def rank(values: list[tuple[float, str]]) -> list[tuple[float, str]]:
    return sorted(values, key=lambda item: (-item[0], item[1]))


def top_margin(order: list[tuple[float, str]]) -> float:
    if not order or order[0][0] <= 0:
        return 0.0
    if len(order) == 1 or order[1][0] <= 0:
        return float("inf")
    return order[0][0] / order[1][0]


def fmt(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.12g}"


def finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else float("inf")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    robust = sum(row["robust_margin_ge_3"] == "true" for row in rows)
    return {
        "release_rows": len(rows),
        "zero_positive_candidate_rows": sum(
            int(row["minimum_positive_score_candidates"]) == 0 for row in rows
        ),
        "median_positive_candidates": finite_median(
            [float(row["minimum_positive_score_candidates"]) for row in rows]
        ),
        "same_top_all_kernels": sum(
            row["same_top_all_kernel_branches"] == "true" for row in rows
        ),
        "robust_margin_ge_3": robust,
        "robust_fraction": robust / len(rows),
        "median_minimum_margin": finite_median(
            [float(row["minimum_top_to_second_margin"]) for row in rows]
        ),
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    honesty = config["honesty_boundary"]
    if not honesty["complete_v19ca_source_result_inspected_before_freeze"]:
        raise RuntimeError("V19CB must disclose post-source design")
    if honesty["gravity_kinematic_or_lensing_target_inspected"]:
        raise RuntimeError("V19CB claims forbidden target access")
    for item in config["parents"].values():
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19CB parent hash mismatch: {item['path']}")

    score_rows = load_csv(ROOT / config["parents"]["v19bz_candidate_scores"]["path"])
    release_rows = load_csv(ROOT / config["parents"]["v19bz_release_information"]["path"])
    diagnostic_rows = load_csv(ROOT / config["parents"]["v19ca_diagnostics"]["path"])
    scores_by_release: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        scores_by_release[row["source_row_id"]].append(row)
    release_by_id = {row["source_row_id"]: row for row in release_rows}
    diagnostics = {row["object_id"]: row for row in diagnostic_rows}
    if len(release_by_id) != 711 or set(scores_by_release) != set(release_by_id):
        raise RuntimeError("V19CB release inventory mismatch")
    if any(row["object_id"] not in diagnostics for row in score_rows):
        raise RuntimeError("V19CB diagnostic coverage mismatch")

    kernels = list(config["kernel_labels"])
    primary = config["primary_kernel_label"]
    threshold = float(config["descriptive_robust_margin"])
    output: list[dict[str, Any]] = []
    for treatment in config["treatment_branches"]:
        for source_row_id in sorted(release_by_id, key=int):
            candidates = scores_by_release[source_row_id]
            top_ids: dict[str, str] = {}
            margins: dict[str, float] = {}
            positive_counts: dict[str, int] = {}
            for kernel in kernels:
                values = [
                    (
                        float(candidate[f"spatial_lr_{kernel}"])
                        * candidate_weight(diagnostics[candidate["object_id"]], treatment),
                        candidate["object_id"],
                    )
                    for candidate in candidates
                ]
                order = rank(values)
                top_ids[kernel] = order[0][1] if order and order[0][0] > 0 else ""
                margins[kernel] = top_margin(order)
                positive_counts[kernel] = sum(value > 0 for value, _ in values)
            nonempty = [value for value in top_ids.values() if value]
            same_top = len(nonempty) == len(kernels) and len(set(nonempty)) == 1
            minimum_margin = min(margins.values())
            release = release_by_id[source_row_id]
            row: dict[str, Any] = {
                "treatment": treatment["id"],
                "source_row_id": source_row_id,
                "wallaby_name": release["wallaby_name"],
                "team_release": release["team_release"],
                "field": release["field"],
                "candidate_count": len(candidates),
                "minimum_positive_score_candidates": min(positive_counts.values()),
                "same_top_all_kernel_branches": str(same_top).lower(),
                "minimum_top_to_second_margin": fmt(minimum_margin),
                "robust_margin_ge_3": str(same_top and minimum_margin >= threshold).lower(),
                "primary_top_object_id": top_ids[primary],
            }
            for kernel in kernels:
                row[f"top_object_id_{kernel}"] = top_ids[kernel]
                row[f"top_margin_{kernel}"] = fmt(margins[kernel])
            output.append(row)

    output_path = ROOT / config["outputs"]["release_branch_information"]
    report_path = ROOT / config["outputs"]["report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(csv_bytes(output))

    branch_summary: dict[str, Any] = {}
    for treatment in config["treatment_branches"]:
        rows = [row for row in output if row["treatment"] == treatment["id"]]
        summary = summarize(rows)
        summary["field_summary"] = {
            field: summarize([row for row in rows if row["field"] == field])
            for field in sorted({row["field"] for row in rows})
        }
        branch_summary[treatment["id"]] = summary
    best_fraction = max(row["robust_fraction"] for row in branch_summary.values())

    boundary = dict(config["access_boundary"])
    gates = {
        "parent_hashes_exact": True,
        "all_711_release_rows_and_four_branches_reported": len(output) == 711 * 4,
        "all_four_spatial_kernel_branches_used": len(kernels) == 4,
        "no_mask_treatment_counterpart_or_sample_selected": not boundary[
            "hard_star_mask_authorized"
        ]
        and not boundary["treatment_branch_selected"]
        and not boundary["candidate_or_galaxy_removed"]
        and not boundary["optical_counterpart_selected"],
        "kinematic_gravity_lensing_and_solar_targets_remain_sealed": not any(
            boundary[key]
            for key in (
                "wallaby_kinematic_table_row_read",
                "rotation_speed_or_velocity_field_read",
                "gravity_formula_residual_or_halo_result_read",
                "development_validation_holdout_split_selected",
                "gravity_action_or_constant_changed",
                "lensing_payload_opened",
                "solar_system_optimization_performed",
            )
        ),
        "post_source_exploratory_status_reported_honestly": honesty[
            "complete_v19ca_source_result_inspected_before_freeze"
        ]
        and not honesty["this_is_a_preregistered_theory_or_holdout_gate"],
    }
    resolution_threshold = float(config["exploratory_resolution_threshold"])
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_exploratory_foreground_treatment_information_audit",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "honesty_boundary": honesty,
        "branch_summary": branch_summary,
        "best_robust_fraction": best_fraction,
        "exploratory_resolution_threshold": resolution_threshold,
        "association_resolved_by_foreground_astrometry": best_fraction >= resolution_threshold,
        "decision": (
            "foreground_astrometry_resolves_spatial_association"
            if best_fraction >= resolution_threshold
            else "foreground_astrometry_reduces_crowding_but_does_not_resolve_association"
        ),
        "next_evidence": [
            "optical image pixels and survey bitmasks",
            "source deblending and foreground-overlap uncertainty",
            "independent source-only counterpart validation",
            "probabilistic mixture propagation rather than a hard match",
        ],
        "output": {
            "path": config["outputs"]["release_branch_information"],
            "rows": len(output),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "claim_boundary": config["claim_boundary"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all(gates.values()):
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps({key: report[key] for key in ("decision", "branch_summary", "output", "gate_results")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
