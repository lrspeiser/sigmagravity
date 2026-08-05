from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19ch_jwst_slice_clean_source_frame.json"
)
MAX_TEXT_AUDIT_BYTES = 20_000_000
SELF_AUDIT_PATHS = {
    "research/galaxy-cluster-unification/configs/sigma_v19ch_jwst_slice_clean_source_frame.json",
    "research/galaxy-cluster-unification/docs/SIGMA_V19CH_JWST_SLICE_CLEAN_SOURCE_FRAME.md",
    "research/galaxy-cluster-unification/results/sigma_v19ch_jwst_slice_clean_source_frame/report.json",
    "research/galaxy-cluster-unification/scripts/audit_sigma_v19ch_jwst_slice_clean_source_frame.py",
    "research/galaxy-cluster-unification/tests/test_sigma_v19ch_jwst_slice_clean_source_frame.py",
}
CATALOG_MARKERS = (
    b"SPT",
    b"PSZ",
    b"ACT-CL",
    b"ACT_CL",
    b"ACTCL",
    b"ACT-C_",
    b"MACS",
    b"RXJ",
    b"RXC",
    b"RBS",
    b"RCS",
    b"ABELL",
    b"CARLA",
    b"A611",
    b"Z2089",
    b"Z7160",
    b"Z2661",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def aliases(target: str, config: dict[str, Any]) -> set[str]:
    canonical = normalize(target)
    values = {canonical}
    if canonical.startswith("PSZ2G"):
        values.add("PSZ1G" + canonical[len("PSZ2G") :])
    if canonical.startswith("SPTCLJ"):
        values.add("SPTCL" + canonical[len("SPTCLJ") :])
    if canonical.startswith("ACTCLJ"):
        values.add("ACTCL" + canonical[len("ACTCLJ") :])
    if canonical.startswith("ACTCJ"):
        values.add("ACTCLJ" + canonical[len("ACTCJ") :])
    if canonical.startswith("MACSMACSJ"):
        values.add(canonical[len("MACS") :])
    if canonical.startswith("RBSMACSJ"):
        values.add(canonical[len("RBS") :])
    macs = re.search(
        r"(?:(?:MACS|RBS)_)?MACSJ(\d{2})(\d{2})(\d{2})(?:\.\d)?([+-])(\d{2})(\d{2})",
        target.upper(),
    )
    if macs:
        hours, minutes, seconds, sign, degrees, arcminutes = macs.groups()
        decimal_minute_tenths = int(
            (int(minutes) + int(seconds) / 60.0) * 10.0
        )
        values.add(
            normalize(
                f"MACS J{hours}{decimal_minute_tenths:03d}"
                f"{sign}{degrees}{arcminutes}"
            )
        )
    if re.fullmatch(r"A\d+", target.upper()):
        values.add("ABELL" + target[1:])
    for value in config["alias_rules"]["known_current_source_aliases"].get(
        target, []
    ):
        values.add(normalize(value))
    # Very short aliases such as A68 are unsafe substring searches. Their
    # exposure disposition is applied explicitly rather than by text scan.
    return {value for value in values if len(value) >= 6}


def tracked_paths() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in completed.stdout.split(b"\0")
        if value
    ]


@lru_cache(maxsize=1)
def repository_identity_audit(
    config_path_text: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    config = load_json(Path(config_path_text))
    targets = config["source_frame"]["target_ids"]
    alias_to_targets: dict[str, set[str]] = {}
    for target in targets:
        for alias in aliases(target, config):
            alias_to_targets.setdefault(alias, set()).add(target)
    pattern = re.compile(
        "|".join(
            re.escape(value)
            for value in sorted(alias_to_targets, key=len, reverse=True)
        )
    )

    hits: dict[str, set[str]] = {}
    paths = tracked_paths()
    content_scanned = 0
    large_or_binary_skipped: list[str] = []
    for relative in paths:
        if relative in SELF_AUDIT_PATHS:
            continue
        normalized_path = normalize(relative)
        for match in pattern.finditer(normalized_path):
            for target in alias_to_targets[match.group(0)]:
                hits.setdefault(target, set()).add(relative)

        path = REPO / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_TEXT_AUDIT_BYTES:
            large_or_binary_skipped.append(relative)
            continue
        payload = path.read_bytes()
        if payload.count(b"\0") > 10:
            large_or_binary_skipped.append(relative)
            continue
        upper = payload.upper()
        if not any(marker in upper for marker in CATALOG_MARKERS):
            continue
        content_scanned += 1
        text = normalize(payload.decode("utf-8", errors="ignore"))
        matched_aliases = {match.group(0) for match in pattern.finditer(text)}
        for alias in matched_aliases:
            for target in alias_to_targets[alias]:
                hits.setdefault(target, set()).add(relative)

    return (
        {target: sorted(paths) for target, paths in sorted(hits.items())},
        {
            "tracked_paths": len(paths),
            "content_scanned_candidate_marker_files": content_scanned,
            "large_or_binary_content_skipped": len(large_or_binary_skipped),
            "large_or_binary_boundary_examples": sorted(large_or_binary_skipped)[:20],
            "all_tracked_path_names_audited": True,
            "max_decoded_content_bytes": MAX_TEXT_AUDIT_BYTES,
            "self_audit_paths_excluded": sorted(SELF_AUDIT_PATHS),
        },
    )


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    parent = load_json(parent_path)
    parent_actual_sha = sha256(parent_path)
    targets = config["source_frame"]["target_ids"]
    target_set = set(targets)
    hits, audit_coverage = repository_identity_audit(str(config_path))
    hit_targets = set(hits)
    expected_hits = set(config["repository_audit"]["expected_identity_hit_targets"])
    incident = config["coordinate_exposure_incident"]
    exposed_current = set(incident["current_source_frame_ids_failed_closed"])
    clean = sorted(target_set - hit_targets - exposed_current)
    quarantined_or_spent = sorted(hit_targets | exposed_current)
    authorizations = config["authorization"]

    gates = {
        "v19cg_parent_hash_exact_and_decision_passes": (
            parent_actual_sha == config["parent"]["sha256"]
            and parent["decision"]
            == "original_cluster_shortlist_retired_from_prospective_holdout_role"
            and all(parent["gate_results"].values())
        ),
        "source_frame_has_182_unique_targets_and_pinned_provenance": (
            len(targets) == len(target_set) == 182
            and len(config["source_frame"]["apt_package_sha256_at_freeze"]) == 64
            and config["source_frame"]["program_status_at_freeze"] == "completed"
        ),
        "whole_repository_identity_audit_matches_expected_quarantine_set": (
            hit_targets == expected_hits
            and audit_coverage["all_tracked_path_names_audited"]
        ),
        "all_14_paper_systems_fail_closed_and_12_current_ids_are_removed": (
            len(incident["paper_sample_aliases"]) == 14
            and len(set(incident["paper_sample_aliases"])) == 14
            and len(exposed_current) == 12
            and exposed_current <= target_set
            and incident["raw_coordinate_values_entered_ephemeral_filter_process"]
            and incident[
                "raw_coordinate_values_returned_visibly_for_at_least_one_system"
            ]
            and not incident["coordinate_values_copied_into_repository"]
            and not incident["coordinate_values_used_for_score_selection_or_physics"]
            and not (exposed_current & set(clean))
        ),
        "at_least_160_zero_hit_unexposed_candidates_remain": len(clean) >= 160,
        "no_replacement_cluster_raw_target_formula_constant_or_solar_setting_selected": (
            not authorizations["select_replacement_cluster"]
            and not authorizations["open_raw_lensing_coordinate_or_map"]
            and not authorizations["select_or_change_action_or_gravity_formula"]
            and not authorizations["fit_universal_constants"]
            and not authorizations["perform_detailed_solar_optimization"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared V19CH gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every V19CH gate must be mandatory")
    passed = all(gates.values())

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_clean_cluster_source_frame_audit",
        "decision": (
            "clean_cluster_source_frame_established_metadata_stratification_required"
            if passed
            else "clean_cluster_source_frame_failed_closed"
        ),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            "path": config["parent"]["path"],
            "expected_sha256": config["parent"]["sha256"],
            "actual_sha256": parent_actual_sha,
            "exact": parent_actual_sha == config["parent"]["sha256"],
            "decision": parent["decision"],
        },
        "source_frame": {
            "program": config["source_frame"]["program"],
            "program_page": config["source_frame"]["program_page"],
            "apt_package": config["source_frame"]["apt_package"],
            "apt_package_sha256_at_freeze": config["source_frame"][
                "apt_package_sha256_at_freeze"
            ],
            "target_count": len(targets),
            "target_ids": targets,
        },
        "repository_identity_hits": hits,
        "repository_audit_coverage": audit_coverage,
        "coordinate_exposure_incident": incident,
        "clean_source_frame_candidate_ids": clean,
        "quarantined_or_raw_exposed_ids": quarantined_or_spent,
        "summary": {
            "external_source_frame_targets": len(targets),
            "repository_identity_hit_targets": len(hit_targets),
            "raw_exposed_current_source_targets": len(exposed_current),
            "hit_and_exposure_overlap": len(hit_targets & exposed_current),
            "quarantined_or_spent_unique_targets": len(quarantined_or_spent),
            "zero_hit_unexposed_candidates": len(clean),
            "replacement_clusters_selected": 0,
            "clusters_admitted": 0,
        },
        "required_next_cluster_work": config["required_next_cluster_work"],
        "access_boundary_audit": {
            "used_only_external_program_identity_for_source_frame": True,
            "recorded_pdf_exposure_fail_closed": True,
            "opened_raw_target_payload_after_freeze": False,
            "selected_replacement_cluster": False,
            "changed_action_formula_or_constants": False,
            "performed_detailed_solar_optimization": False,
        },
        "gate_results": gates,
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }


def main() -> None:
    report = build_report()
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "summary": report["summary"],
                "gate_results": report["gate_results"],
                "output": output.relative_to(ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != (
        "clean_cluster_source_frame_established_metadata_stratification_required"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
