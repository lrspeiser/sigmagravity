from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cl_registered_cluster_morphology_acquisition.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def finite(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def read_vizier_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    if "<html" in text.lower() or "<!doctype" in text.lower():
        raise ValueError(f"HTML response is not a catalog: {path}")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) < 4:
        raise ValueError(f"Missing VizieR table payload: {path}")
    header = lines[0].split("\t")
    rows = list(csv.DictReader(io.StringIO("\n".join([lines[0], *lines[3:]])), delimiter="\t"))
    return header, rows


def index_unique_aliases(
    rows: list[dict[str, str]], name_fields: tuple[str, ...]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    grouped: dict[str, set[int]] = {}
    for row_index, row in enumerate(rows):
        for field in name_fields:
            value = row.get(field, "").strip()
            if value:
                grouped.setdefault(normalize(value), set()).add(row_index)
    ambiguous = sorted(key for key, indices in grouped.items() if len(indices) != 1)
    return {
        key: rows[next(iter(indices))]
        for key, indices in grouped.items()
        if len(indices) == 1
    }, ambiguous


def _primary_record(candidate: str, row: dict[str, str], medians: dict[str, float]) -> dict[str, Any]:
    values = {
        "z": finite(row.get("z")),
        "logc": finite(row.get("logc")),
        "e_logc": finite(row.get("e_logc")),
        "logw": finite(row.get("logw")),
        "e_logw": finite(row.get("e_logw")),
        "logP3_P0": finite(row.get("logP3/P0")),
        "e_logP3_P0": finite(row.get("e_logP3/P0")),
        "logalpha": finite(row.get("logalpha")),
        "e_logalpha": finite(row.get("e_logalpha")),
        "kappa": finite(row.get("kappa")),
        "delta": finite(row.get("delta")),
        "e_delta": finite(row.get("e_delta")),
    }
    delta, error = values["delta"], values["e_delta"]
    multimetric = all(values[key] is not None for key in ("logc", "logw", "logP3_P0"))
    if delta is None or error is None:
        primary_state = "missing_primary_state"
    elif delta + error < 0:
        primary_state = "secure_relaxed"
    elif delta - error > 0:
        primary_state = "secure_disturbed"
    else:
        primary_state = "boundary_intermediate"

    relaxed_votes: int | None = None
    disturbed_votes: int | None = None
    if multimetric:
        relaxed_votes = sum(
            (
                values["logc"] >= medians["logc"],
                values["logw"] <= medians["logw"],
                values["logP3_P0"] <= medians["logP3_P0"],
            )
        )
        disturbed_votes = 3 - relaxed_votes

    if primary_state == "boundary_intermediate":
        morphology_class = "boundary_intermediate"
    elif primary_state == "secure_relaxed" and relaxed_votes is not None:
        morphology_class = "confirmed_relaxed" if relaxed_votes >= 2 else "discordant_extreme"
    elif primary_state == "secure_disturbed" and disturbed_votes is not None:
        morphology_class = "confirmed_disturbed" if disturbed_votes >= 2 else "discordant_extreme"
    elif primary_state in {"secure_relaxed", "secure_disturbed"}:
        morphology_class = f"{primary_state}_missing_multimetric"
    else:
        morphology_class = "missing_primary_state"

    return {
        "candidate_id": candidate,
        "catalog_name": row["Name"].strip(),
        "catalog_normalized_id": normalize(row["Name"]),
        **values,
        "primary_state": primary_state,
        "multimetric_finite": multimetric,
        "relaxed_direction_votes": relaxed_votes,
        "disturbed_direction_votes": disturbed_votes,
        "morphology_class": morphology_class,
        "eligible_for_balanced_metadata_pool": morphology_class
        in {"confirmed_relaxed", "confirmed_disturbed", "boundary_intermediate", "discordant_extreme"},
    }


def _secondary_record(candidate: str, row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": candidate,
        "catalog_name": row["Name"].strip(),
        "subcluster": row["Subcluster"].strip(),
        **{field: finite(row.get(field)) for field in ("cC", "e_cC", "wC", "e_wC", "cX", "e_cX", "wX", "e_wX")},
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    clean_path = ROOT / config["clean_frame"]["path"]
    parent = load_json(parent_path)
    clean = load_json(clean_path)

    source_paths = {key: ROOT / value["path"] for key, value in config["sources"].items()}
    primary_header, primary_rows = read_vizier_tsv(source_paths["primary_table"])
    secondary_header, secondary_rows = read_vizier_tsv(source_paths["secondary_table"])
    # SimbadName is a source-table catalog alias, not a coordinate match.  It
    # resolves known formatting/rounding aliases such as ACTCLJ0206-0114 vs
    # ACT-CL J0206.2-0114 and SPTCLJ0534-5005 vs SPT-CL J0533-5005.
    primary_index, primary_ambiguous = index_unique_aliases(
        primary_rows, ("Name", "SimbadName")
    )
    secondary_index, secondary_ambiguous = index_unique_aliases(
        secondary_rows, ("Name",)
    )
    candidates = clean["clean_source_frame_candidate_ids"]

    primary_pairs = [(candidate, primary_index[normalize(candidate)]) for candidate in candidates if normalize(candidate) in primary_index]
    finite_metric_rows = [
        row
        for _, row in primary_pairs
        if all(finite(row.get(key)) is not None for key in ("logc", "logw", "logP3/P0"))
        and finite(row.get("delta")) is not None
        and finite(row.get("e_delta")) is not None
    ]
    medians = {
        "logc": median(finite(row["logc"]) for row in finite_metric_rows),
        "logw": median(finite(row["logw"]) for row in finite_metric_rows),
        "logP3_P0": median(finite(row["logP3/P0"]) for row in finite_metric_rows),
    }
    primary_ledger = [_primary_record(candidate, row, medians) for candidate, row in primary_pairs]
    secondary_ledger = [
        _secondary_record(candidate, secondary_index[normalize(candidate)])
        for candidate in candidates
        if normalize(candidate) in secondary_index
    ]
    primary_by_candidate = {row["candidate_id"]: row for row in primary_ledger}
    secondary_primary_overlap = sorted(set(primary_by_candidate) & {row["candidate_id"] for row in secondary_ledger})

    class_counts = {name: sum(row["morphology_class"] == name for row in primary_ledger) for name in sorted({row["morphology_class"] for row in primary_ledger})}
    eligible = [row for row in primary_ledger if row["eligible_for_balanced_metadata_pool"]]
    requirements = parent["diversity_and_admission_requirements"]
    diversity = {
        "eligible_pool_size": len(eligible),
        "confirmed_relaxed": class_counts.get("confirmed_relaxed", 0),
        "confirmed_disturbed": class_counts.get("confirmed_disturbed", 0),
        "boundary_or_discordant": class_counts.get("boundary_intermediate", 0) + class_counts.get("discordant_extreme", 0),
    }

    hashes_exact = {key: sha256(path) == config["sources"][key]["sha256"] for key, path in source_paths.items()}
    required_primary = {"Name", "z", "logc", "e_logc", "logw", "e_logw", "logP3/P0", "e_logP3/P0", "logalpha", "e_logalpha", "kappa", "delta", "e_delta"}
    required_secondary = {"Name", "Subcluster", "cC", "e_cC", "wC", "e_wC", "cX", "e_cX", "wX", "e_wX"}
    gates = {
        "parent_and_clean_frame_hashes_and_decisions_exact": (
            sha256(parent_path) == config["parent"]["sha256"]
            and parent["decision"] == config["parent"]["required_decision"]
            and sha256(clean_path) == config["clean_frame"]["sha256"]
            and len(candidates) == config["clean_frame"]["required_candidates"]
        ),
        "all_four_source_files_hash_exact_and_tables_not_html": all(hashes_exact.values()),
        "registered_row_counts_and_required_columns_exact": (
            len(primary_rows) == config["sources"]["primary_table"]["rows"]
            and len(secondary_rows) == config["sources"]["secondary_table"]["rows"]
            and required_primary <= set(primary_header)
            and required_secondary <= set(secondary_header)
        ),
        "crossmatch_is_one_to_one_and_coordinate_free": (
            not any(normalize(candidate) in primary_ambiguous for candidate in candidates)
            and not any(normalize(candidate) in secondary_ambiguous for candidate in candidates)
            and "coordinate" in config["frozen_rule_reference"]["crossmatch"]
            and "no coordinates" in config["frozen_rule_reference"]["crossmatch"]
        ),
        "frozen_primary_state_and_multimetric_rules_reproduced": (
            len(finite_metric_rows) > 0
            and config["frozen_rule_reference"]["secure_relaxed"] == "delta+e_delta < 0"
            and config["frozen_rule_reference"]["secure_disturbed"] == "delta-e_delta > 0"
            and config["frozen_rule_reference"]["confirmation"].startswith("at least two of three")
        ),
        "minimum_balanced_morphology_pool_passes": (
            diversity["eligible_pool_size"] >= requirements["minimum_metadata_shortlist"]
            and diversity["confirmed_relaxed"] >= requirements["minimum_secure_relaxed"]
            and diversity["confirmed_disturbed"] >= requirements["minimum_secure_disturbed"]
            and diversity["boundary_or_discordant"] >= requirements["minimum_boundary_or_discordant"]
        ),
        "no_cluster_lensing_target_formula_constant_or_solar_setting_selected": (
            not config["authorization"]["select_or_admit_final_cluster"]
            and not config["authorization"]["open_raw_lensing_coordinates_or_halo_map"]
            and not config["authorization"]["score_or_fit_sigma_gravity"]
            and not config["authorization"]["select_or_modify_action_or_constants"]
            and not config["authorization"]["perform_detailed_solar_optimization"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared V19CL gate names differ")

    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_registered_cluster_morphology_acquisition",
        "decision": "balanced_source_morphology_pool_established_source_completeness_required" if all(gates.values()) else "registered_cluster_morphology_acquisition_failed_closed",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_integrity": {"hashes_exact": hashes_exact, "primary_rows": len(primary_rows), "secondary_rows": len(secondary_rows)},
        "crossmatch": {
            "clean_candidates": len(candidates),
            "primary_exact_one_to_one": len(primary_ledger),
            "primary_all_required_metrics_finite": len(finite_metric_rows),
            "secondary_exact_one_to_one": len(secondary_ledger),
            "primary_secondary_same_candidate_overlap": len(secondary_primary_overlap),
            "overlap_candidate_ids": secondary_primary_overlap,
            "coordinate_matching_used": False,
        },
        "clean_finite_primary_medians": medians,
        "morphology_class_counts": class_counts,
        "diversity_gate_inputs": diversity,
        "eligible_balanced_metadata_pool_ids": [row["candidate_id"] for row in eligible],
        "missing_or_unconfirmed_primary_ids": [row["candidate_id"] for row in primary_ledger if not row["eligible_for_balanced_metadata_pool"]],
        "secondary_crosscheck_disposition": "No same-candidate primary/secondary overlap; preserve secondary rows for later independent descriptive checks without inventing coordinate or lensing-based aliases." if not secondary_primary_overlap else "Report same-candidate concentration and centroid-shift values descriptively; do not replace primary state.",
        "gate_results": gates,
        "access_boundary_audit": {
            "clusters_selected_or_admitted": 0,
            "raw_lensing_coordinates_or_halo_maps_opened": False,
            "sigma_gravity_scored_or_fit": False,
            "action_or_constants_changed": False,
            "detailed_solar_optimization_performed": False,
        },
        "required_next_work": [
            "Apply the already frozen source-completeness, redshift, source-side mass, gas/BCG concentration, and single-core/multipeak metadata gates without opening lensing outcomes.",
            "Seal raw image positions and comparator predictions separately before admitting the balanced final six.",
            "Complete the independent V19W5 response recovery and unchanged source-only chain before V19BS can authorize any new action.",
        ],
        "claim_boundary": config["claim_boundary"],
        "outputs": config["outputs"],
    }
    return report, primary_ledger, secondary_ledger


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("candidate_id\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    report, primary, secondary = build_report()
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(ROOT / config["outputs"]["primary_ledger"], primary)
    write_csv(ROOT / config["outputs"]["secondary_ledger"], secondary)
    print(json.dumps({"decision": report["decision"], "crossmatch": report["crossmatch"], "classes": report["morphology_class_counts"], "gates": report["gate_results"]}, indent=2, sort_keys=True))
    if not all(report["gate_results"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
