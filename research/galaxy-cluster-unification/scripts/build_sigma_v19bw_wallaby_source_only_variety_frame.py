from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bw_wallaby_source_only_variety_frame.json"
V19BV_SCRIPT = ROOT / "scripts" / "build_sigma_v19bv_wallaby_canonical_source_rows.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bv_for_v19bw", V19BV_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V19BV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V19BV)

AXES = (
    "hi_mass",
    "hi_compactness_proxy",
    "axis_ratio",
    "distance",
    "source_extent_pixels",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def field_name(team_release: str) -> str:
    if team_release.startswith("Hydra"):
        return "Hydra"
    if team_release.startswith("Norma"):
        return "Norma"
    if team_release.startswith("NGC 4636"):
        return "NGC 4636"
    raise ValueError(f"unsupported team release: {team_release}")


def row_metrics(row: dict[str, str]) -> dict[str, float]:
    major = max(float(row["ell3s_maj"]), float(row["ell3s_min"]))
    minor = min(float(row["ell3s_maj"]), float(row["ell3s_min"]))
    distance = float(row["dist_h"])
    hi_mass = float(row["log_m_hi"])
    source_snr = float(row["f_sum"]) / float(row["err_f_sum"])
    if min(major, minor, distance, source_snr) <= 0:
        raise ValueError(f"non-positive source-only metric for {row['name']}")
    metrics = {
        "hi_mass": hi_mass,
        "hi_compactness_proxy": hi_mass
        - math.log10(major * minor * distance * distance),
        "axis_ratio": minor / major,
        "distance": math.log10(distance),
        "source_extent_pixels": math.log10(major),
        "source_snr_log10": math.log10(source_snr),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"non-finite source-only metric for {row['name']}")
    return metrics


def type7_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty list")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def quartile(value: float, edges: list[float]) -> str:
    if value <= edges[0]:
        return "Q1"
    if value <= edges[1]:
        return "Q2"
    if value <= edges[2]:
        return "Q3"
    return "Q4"


def availability_label(rows: list[dict[str, str]]) -> str:
    count = sum(row["kflag"] == "2" for row in rows)
    if count == len(rows):
        return "all_policies"
    if count:
        return "some_policies"
    return "no_policy"


def build_artifacts(config_path: Path = DEFAULT_CONFIG) -> tuple[bytes, dict[str, Any]]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    canonical_path = ROOT / config["canonical_input"]["path"]
    raw_path = ROOT / config["raw_alternative_input"]["path"]
    parent_actual = sha256(parent_path)
    canonical_actual = sha256(canonical_path)
    raw_actual = sha256(raw_path)
    canonical_columns, canonical_rows = load_csv(canonical_path)
    raw_columns, raw_rows = load_csv(raw_path)
    canonical_by_name = {row["name"]: row for row in canonical_rows}
    if len(canonical_by_name) != len(canonical_rows):
        raise ValueError("canonical input is not one row per name")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[row["name"]].append(row)
    policies = V19BV.policy_keys()
    policy_rows: dict[str, dict[str, dict[str, str]]] = {}
    for name, candidates in grouped.items():
        policy_rows[name] = {
            policy: min(candidates, key=key) for policy, key in policies.items()
        }
    if set(policy_rows) != set(canonical_by_name):
        raise ValueError("canonical and raw candidate universes differ")

    metrics_by_name = {
        name: row_metrics(row)
        for name, row in canonical_by_name.items()
    }
    canonical_available_names = sorted(
        name for name, row in canonical_by_name.items() if row["kflag"] == "2"
    )
    edges = {
        axis: [
            type7_quantile(
                [metrics_by_name[name][axis] for name in canonical_available_names], q
            )
            for q in (0.25, 0.50, 0.75)
        ]
        for axis in AXES
    }

    output_rows: list[dict[str, str]] = []
    ambiguity_count = 0
    policy_bin_change_count = 0
    for name in sorted(canonical_by_name):
        row = canonical_by_name[name]
        metrics = metrics_by_name[name]
        selected_rows = list(policy_rows[name].values())
        selected_ids = {selected["id"] for selected in selected_rows}
        source_sensitive = len(selected_ids) > 1
        ambiguity_count += source_sensitive
        policy_metric_bins = {
            axis: sorted(
                {
                    quartile(row_metrics(selected)[axis], edges[axis])
                    for selected in selected_rows
                }
            )
            for axis in AXES
        }
        policy_bin_changed = any(len(labels) > 1 for labels in policy_metric_bins.values())
        policy_bin_change_count += policy_bin_changed
        bins = {axis: quartile(metrics[axis], edges[axis]) for axis in AXES}
        output_rows.append(
            {
                "name": name,
                "canonical_id": row["id"],
                "team_release": row["team_release"],
                "release_field": field_name(row["team_release"]),
                "canonical_kflag": row["kflag"],
                "canonical_qflag": row["qflag"],
                "canonical_reliability": row["rel"],
                "canonical_source_snr": row["source_snr"],
                "kinematic_availability_across_policies": availability_label(selected_rows),
                "source_row_policy_sensitive": str(source_sensitive).lower(),
                "source_metric_bin_policy_sensitive": str(policy_bin_changed).lower(),
                **{axis: f"{metrics[axis]:.12g}" for axis in AXES},
                **{f"{axis}_quartile": bins[axis] for axis in AXES},
                **{
                    f"{axis}_policy_quartiles": "|".join(policy_metric_bins[axis])
                    for axis in AXES
                },
                "variety_cell": ":".join(bins[axis] for axis in AXES),
            }
        )

    output_columns = list(output_rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=output_columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    payload = stream.getvalue().encode("utf-8")

    available_rows = [row for row in output_rows if row["canonical_kflag"] == "2"]
    axis_counts = {
        axis: dict(
            sorted(Counter(row[f"{axis}_quartile"] for row in available_rows).items())
        )
        for axis in AXES
    }
    field_counts = dict(
        sorted(Counter(row["release_field"] for row in available_rows).items())
    )
    availability_counts = dict(
        sorted(
            Counter(
                row["kinematic_availability_across_policies"] for row in output_rows
            ).items()
        )
    )
    variety_cells = Counter(row["variety_cell"] for row in available_rows)
    min_quartile = min(min(counts.values()) for counts in axis_counts.values())
    min_field = min(field_counts.values())
    boundary = config["access_boundary"]
    sealed_columns_absent = set(config["sealed_target_columns"]).isdisjoint(
        canonical_columns
    ) and set(config["sealed_target_columns"]).isdisjoint(raw_columns)
    no_split_columns = not any(
        any(token in column.lower() for token in ("holdout", "validation", "development", "fold"))
        for column in output_columns
    )
    gates = {
        "parent_and_input_hashes_exact": (
            parent_actual == config["parent"]["sha256"]
            and canonical_actual == config["canonical_input"]["sha256"]
            and raw_actual == config["raw_alternative_input"]["sha256"]
        ),
        "complete_candidate_universe_preserved": (
            len(output_rows) == len({row["name"] for row in output_rows}) == 592
            and len(raw_rows) == 711
        ),
        "published_kinematic_availability_lane_reproduced": (
            len(available_rows) == config["availability_lane"]["expected_names"] == 109
        ),
        "all_five_variety_axes_have_quartile_coverage": (
            set(axis_counts) == set(AXES)
            and all(set(counts) == {"Q1", "Q2", "Q3", "Q4"} for counts in axis_counts.values())
            and min_quartile >= config["binning"]["minimum_names_per_axis_quartile"]
        ),
        "all_three_release_fields_are_represented": (
            set(field_counts) == {"Hydra", "Norma", "NGC 4636"}
            and min_field >= config["binning"]["minimum_names_per_release_field"]
        ),
        "release_row_systematic_is_propagated": (
            ambiguity_count == config["raw_alternative_input"]["policy_sensitive_names"] == 92
            and boundary["raw_alternative_rows_retained"]
            and policy_bin_change_count > 0
        ),
        "kinematic_gravity_and_solar_targets_remain_sealed": (
            sealed_columns_absent
            and boundary["source_columns_only"]
            and not boundary["kinematic_table_rows_read"]
            and not boundary["rotation_speed_or_velocity_field_read"]
            and not boundary["inclination_or_kinematic_angle_read"]
            and not boundary["gravity_formula_residual_or_halo_result_read"]
            and not boundary["gravity_action_or_constant_changed"]
            and not boundary["solar_system_optimization_performed"]
        ),
        "no_final_sample_or_evidence_split_claimed": (
            no_split_columns
            and not boundary["development_validation_holdout_split_selected"]
            and not boundary["final_galaxy_sample_selected"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    output_path = ROOT / config["outputs"]["variety_frame_csv"]
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_source_only_variety_frame",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "input_audit": {
            "parent": {"path": config["parent"]["path"], "sha256": parent_actual},
            "canonical": {
                "path": config["canonical_input"]["path"],
                "sha256": canonical_actual,
                "rows": len(canonical_rows),
            },
            "raw_alternatives": {
                "path": config["raw_alternative_input"]["path"],
                "sha256": raw_actual,
                "rows": len(raw_rows),
            },
        },
        "variety_frame": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": bytes_sha256(payload),
            "bytes": len(payload),
            "rows": len(output_rows),
            "canonical_kinematic_availability_names": len(available_rows),
            "availability_across_policy_counts_all_names": availability_counts,
            "axis_quartile_edges": edges,
            "axis_quartile_counts_in_availability_lane": axis_counts,
            "release_field_counts_in_availability_lane": field_counts,
            "unique_multiaxis_variety_cells_in_availability_lane": len(variety_cells),
            "largest_multiaxis_variety_cell": max(variety_cells.values()),
            "source_row_policy_sensitive_names": ambiguity_count,
            "source_metric_bin_policy_sensitive_names": policy_bin_change_count,
            "minimum_axis_quartile_count": min_quartile,
            "minimum_release_field_count": min_field,
            "output_columns": output_columns,
        },
        "metric_definitions": config["variety_axes"],
        "binning": config["binning"],
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "decision": (
            "passed_source_only_variety_frame_not_final_split"
            if all(gates.values())
            else "failed_source_only_variety_frame"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }
    return payload, report


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    payload, report = build_artifacts()
    output_path = ROOT / config["outputs"]["variety_frame_csv"]
    report_path = ROOT / config["outputs"]["report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "variety_frame": report["variety_frame"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_source_only_variety_frame_not_final_split":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
