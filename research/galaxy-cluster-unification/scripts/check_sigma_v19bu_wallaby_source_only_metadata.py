from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bu_wallaby_source_only_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("WALLABY source-only CSV has no header")
        return list(reader.fieldnames), list(reader)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    payload_path = ROOT / config["source_only_payload"]["path"]
    parent_actual = sha256(parent_path)
    payload_actual = sha256(payload_path)
    columns, rows = load_csv(payload_path)

    names = [row["name"] for row in rows]
    name_counts = Counter(names)
    unique_names = set(names)
    kflags = sorted({row["kflag"] for row in rows})
    summary = {
        "row_count": len(rows),
        "unique_source_names": len(unique_names),
        "duplicate_name_rows": len(rows) - len(unique_names),
        "maximum_rows_per_name": max(name_counts.values()),
        "team_release_row_counts": dict(
            sorted(Counter(row["team_release"] for row in rows).items())
        ),
        "kflag_row_counts": dict(sorted(Counter(row["kflag"] for row in rows).items())),
        "kflag_unique_name_counts": {
            flag: len({row["name"] for row in rows if row["kflag"] == flag})
            for flag in kflags
        },
        "qflag_row_counts": dict(sorted(Counter(row["qflag"] for row in rows).items())),
        "reliability_at_least_0_9_rows": sum(
            bool(row["rel"]) and float(row["rel"]) >= 0.9 for row in rows
        ),
        "rows_with_hi_mass": sum(bool(row["log_m_hi"]) for row in rows),
    }

    declared_payload = config["source_only_payload"]
    expected_summary = config["observed_source_summary"]
    boundary = config["access_boundary"]
    authorization = config["authorization"]
    allowed = declared_payload["allowed_columns"]
    forbidden = set(config["sealed_target_columns"])

    payload_exact = (
        payload_actual == declared_payload["sha256"]
        and payload_path.stat().st_size == declared_payload["bytes"]
        and text_sha256(config["primary_source"]["adql_query"])
        == config["primary_source"]["adql_query_sha256"]
    )
    source_summary_exact = all(
        summary[key] == expected_summary[key]
        for key in (
            "team_release_row_counts",
            "kflag_row_counts",
            "kflag_unique_name_counts",
            "qflag_row_counts",
            "reliability_at_least_0_9_rows",
            "rows_with_hi_mass",
        )
    )
    row_shape_exact = all(
        summary[key] == declared_payload[key]
        for key in (
            "row_count",
            "unique_source_names",
            "duplicate_name_rows",
            "maximum_rows_per_name",
        )
    )
    target_sealed = (
        boundary["source_finding_table_read"]
        and boundary["kinematic_model_table_schema_names_read"]
        and not boundary["kinematic_model_table_rows_read"]
        and not boundary["rotation_speed_values_read"]
        and not boundary["velocity_field_or_cube_opened"]
        and not boundary["systemic_velocity_values_read"]
        and not boundary["inclination_or_kinematic_position_angle_values_read"]
        and not boundary["model_residual_or_halo_result_read"]
        and not authorization["read_kinematic_values"]
        and not authorization["download_velocity_cubes_or_fields"]
    )
    no_selection = (
        not boundary["final_holdout_sample_selected"]
        and not boundary["gravity_formula_or_constant_changed"]
        and not boundary["solar_system_optimization_performed"]
        and not authorization["select_final_holdout_galaxies"]
        and not authorization["fit_or_change_gravity"]
        and not authorization["perform_detailed_solar_optimization"]
    )

    gates = {
        "parent_hash_exact": parent_actual == config["parent"]["sha256"],
        "source_payload_hash_and_size_exact": payload_exact,
        "source_columns_equal_whitelist": columns == allowed,
        "all_target_columns_absent": forbidden.isdisjoint(columns),
        "published_592_unique_sources_recovered": (
            summary["unique_source_names"] == 592 and row_shape_exact
        ),
        "duplicate_release_rows_are_explicit": (
            summary["duplicate_name_rows"] == 119
            and summary["maximum_rows_per_name"] == 2
        ),
        "source_summary_reproduces_exactly": source_summary_exact,
        "kinematic_rows_and_values_remain_sealed": target_sealed,
        "final_sample_theory_constants_and_solar_remain_unselected": no_selection,
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_wallaby_source_only_metadata_checkpoint",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            "path": config["parent"]["path"],
            "expected_sha256": config["parent"]["sha256"],
            "actual_sha256": parent_actual,
            "exact": parent_actual == config["parent"]["sha256"],
        },
        "source_payload_audit": {
            "path": declared_payload["path"],
            "expected_sha256": declared_payload["sha256"],
            "actual_sha256": payload_actual,
            "bytes": payload_path.stat().st_size,
            "columns": columns,
            **summary,
        },
        "source_target_boundary": {
            "source_table": config["primary_source"]["source_table"],
            "sealed_target_table": config["primary_source"]["sealed_target_table"],
            "allowed_source_columns": allowed,
            "sealed_target_columns": config["sealed_target_columns"],
            "kinematic_value_rows_read": False,
            "final_sample_selected": False,
        },
        "access_boundary_audit": boundary,
        "authorization_audit": authorization,
        "gate_results": gates,
        "decision": (
            "passed_source_only_candidate_universe_not_holdout_selection"
            if all(gates.values())
            else "failed_source_only_candidate_universe"
        ),
        "next_work_after_action_freeze": config["next_work_after_action_freeze"],
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    report = build_report()
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "output": output.relative_to(ROOT).as_posix(),
                "source_payload_audit": report["source_payload_audit"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_source_only_candidate_universe_not_holdout_selection":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
