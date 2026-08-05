from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bv_wallaby_canonical_source_rows.json"
CANONICAL_POLICY = "QFLAG_KFLAG_RELIABILITY_SNR_PIXELS_RELEASE_CATALOGUE"


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
            raise ValueError("source CSV has no header")
        return list(reader.fieldnames), list(reader)


def release_number(row: dict[str, str]) -> int:
    match = re.search(r"\bTR(\d+)\b", row["team_release"])
    if match is None:
        raise ValueError(f"team release has no numeric TR suffix: {row['team_release']}")
    return int(match.group(1))


def values(row: dict[str, str]) -> dict[str, float | int]:
    error = float(row["err_f_sum"])
    if error <= 0:
        raise ValueError("integrated-flux uncertainty must be positive")
    kflag_rank = {"2": 0, "1": 1, "0": 2}
    if row["kflag"] not in kflag_rank:
        raise ValueError(f"unsupported kflag: {row['kflag']}")
    return {
        "qflag": float(row["qflag"]),
        "kflag_rank": kflag_rank[row["kflag"]],
        "reliability": float(row["rel"]),
        "snr": float(row["f_sum"]) / error,
        "pixels": float(row["n_pix"]),
        "release": release_number(row),
        "catalogue_id": int(row["catalogue_id"]),
        "id": int(row["id"]),
    }


def policy_keys() -> dict[str, Callable[[dict[str, str]], tuple[float | int, ...]]]:
    def v(row: dict[str, str]) -> dict[str, float | int]:
        return values(row)

    return {
        CANONICAL_POLICY: lambda row: (
            v(row)["qflag"],
            v(row)["kflag_rank"],
            -v(row)["reliability"],
            -v(row)["snr"],
            -v(row)["pixels"],
            -v(row)["release"],
            v(row)["catalogue_id"],
            v(row)["id"],
        ),
        "QFLAG_RELEASE_KFLAG_RELIABILITY_SNR_PIXELS": lambda row: (
            v(row)["qflag"],
            -v(row)["release"],
            v(row)["kflag_rank"],
            -v(row)["reliability"],
            -v(row)["snr"],
            -v(row)["pixels"],
            v(row)["catalogue_id"],
            v(row)["id"],
        ),
        "KFLAG_QFLAG_RELIABILITY_SNR_PIXELS_RELEASE": lambda row: (
            v(row)["kflag_rank"],
            v(row)["qflag"],
            -v(row)["reliability"],
            -v(row)["snr"],
            -v(row)["pixels"],
            -v(row)["release"],
            v(row)["catalogue_id"],
            v(row)["id"],
        ),
        "RELIABILITY_QFLAG_KFLAG_SNR_PIXELS_RELEASE": lambda row: (
            -v(row)["reliability"],
            v(row)["qflag"],
            v(row)["kflag_rank"],
            -v(row)["snr"],
            -v(row)["pixels"],
            -v(row)["release"],
            v(row)["catalogue_id"],
            v(row)["id"],
        ),
        "SNR_QFLAG_KFLAG_RELIABILITY_PIXELS_RELEASE": lambda row: (
            -v(row)["snr"],
            v(row)["qflag"],
            v(row)["kflag_rank"],
            -v(row)["reliability"],
            -v(row)["pixels"],
            -v(row)["release"],
            v(row)["catalogue_id"],
            v(row)["id"],
        ),
    }


def choice_reason(rows: list[dict[str, str]]) -> str:
    if len(rows) == 1:
        return "unique_source_name"
    ranked = sorted(rows, key=policy_keys()[CANONICAL_POLICY])
    chosen = values(ranked[0])
    runner_up = values(ranked[1])
    criteria = (
        ("qflag", "lower_qflag"),
        ("kflag_rank", "higher_kinematic_availability_flag"),
        ("reliability", "higher_source_reliability"),
        ("snr", "higher_integrated_source_snr"),
        ("pixels", "larger_source_mask"),
        ("release", "newer_team_release"),
        ("catalogue_id", "catalogue_id_tiebreak"),
        ("id", "archive_primary_id_tiebreak"),
    )
    for field, label in criteria:
        if chosen[field] != runner_up[field]:
            return label
    raise ValueError("duplicate source rows are identical through the total ordering")


def render_canonical_csv(
    input_columns: list[str], rows: list[dict[str, str]]
) -> tuple[bytes, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)
    policies = policy_keys()
    canonical_rows: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    distinct_choice_counts: Counter[int] = Counter()
    per_policy_choices: dict[str, list[str]] = {policy: [] for policy in policies}

    for name in sorted(grouped):
        candidates = grouped[name]
        choices = {
            policy: min(candidates, key=key) for policy, key in policies.items()
        }
        distinct = len({row["id"] for row in choices.values()})
        distinct_choice_counts[distinct] += 1
        chosen = choices[CANONICAL_POLICY]
        reason = choice_reason(candidates)
        reasons[reason] += 1
        enriched = dict(chosen)
        enriched.update(
            {
                "source_snr": f"{values(chosen)['snr']:.12g}",
                "duplicate_group_size": str(len(candidates)),
                "canonical_choice_reason": reason,
                "all_prespecified_policies_agree": str(distinct == 1).lower(),
                "prespecified_distinct_choice_count": str(distinct),
                "canonical_policy_id": CANONICAL_POLICY,
            }
        )
        canonical_rows.append(enriched)
        for policy, row in choices.items():
            per_policy_choices[policy].append(f"{name}:{row['id']}")

    derived_columns = [
        "source_snr",
        "duplicate_group_size",
        "canonical_choice_reason",
        "all_prespecified_policies_agree",
        "prespecified_distinct_choice_count",
        "canonical_policy_id",
    ]
    output_columns = input_columns + derived_columns
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=output_columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(canonical_rows)
    payload = stream.getvalue().encode("utf-8")

    duplicate_names = [name for name, group in grouped.items() if len(group) > 1]
    duplicate_agree = sum(
        len(
            {
                min(grouped[name], key=key)["id"]
                for key in policies.values()
            }
        )
        == 1
        for name in duplicate_names
    )
    summary = {
        "input_rows": len(rows),
        "unique_source_names": len(grouped),
        "duplicate_names_resolved": len(duplicate_names),
        "maximum_input_rows_per_name": max(len(group) for group in grouped.values()),
        "canonical_rows": len(canonical_rows),
        "duplicate_names_with_all_five_policies_agreeing": duplicate_agree,
        "duplicate_names_policy_ambiguous": len(duplicate_names) - duplicate_agree,
        "all_name_distinct_choice_counts": {
            str(key): count for key, count in sorted(distinct_choice_counts.items())
        },
        "canonical_choice_reason_counts": dict(sorted(reasons.items())),
        "canonical_team_release_counts": dict(
            sorted(Counter(row["team_release"] for row in canonical_rows).items())
        ),
        "canonical_qflag_counts": dict(
            sorted(Counter(row["qflag"] for row in canonical_rows).items())
        ),
        "canonical_kflag_counts": dict(
            sorted(Counter(row["kflag"] for row in canonical_rows).items())
        ),
        "policy_choice_sha256": {
            policy: hashlib.sha256("\n".join(choices).encode("utf-8")).hexdigest()
            for policy, choices in per_policy_choices.items()
        },
        "output_columns": output_columns,
    }
    return payload, summary


def build_artifacts(config_path: Path = DEFAULT_CONFIG) -> tuple[bytes, dict[str, Any]]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    input_path = ROOT / config["input"]["path"]
    parent_actual = sha256(parent_path)
    input_actual = sha256(input_path)
    input_columns, rows = load_csv(input_path)
    payload, summary = render_canonical_csv(input_columns, rows)
    expected = config["expected_source_only_result"]
    boundary = config["access_boundary"]
    policy_ids = set(policy_keys())
    declared_policy_ids = {config["canonical_policy"]["id"]} | set(
        config["robustness_policies"]
    )

    expected_summary_exact = all(
        summary[key] == expected[key]
        for key in (
            "canonical_rows",
            "duplicate_names_resolved",
            "maximum_input_rows_per_name",
            "duplicate_names_with_all_five_policies_agreeing",
            "duplicate_names_policy_ambiguous",
        )
    )
    sealed = (
        boundary["raw_alternative_rows_retained"]
        and boundary["source_columns_only"]
        and not boundary["kinematic_table_rows_read"]
        and not boundary["rotation_speed_or_velocity_field_read"]
        and not boundary["inclination_or_kinematic_angle_read"]
        and not boundary["gravity_formula_residual_or_halo_result_read"]
        and not boundary["gravity_formula_or_constant_changed"]
        and not boundary["solar_system_optimization_performed"]
    )
    gates = {
        "parent_and_input_hashes_exact": (
            parent_actual == config["parent"]["sha256"]
            and input_actual == config["input"]["sha256"]
        ),
        "canonical_policy_is_source_only_and_total": (
            policy_ids == declared_policy_ids
            and summary["canonical_rows"] == len(rows) - config["input"]["duplicate_names"]
        ),
        "one_canonical_row_per_unique_name": (
            summary["canonical_rows"] == summary["unique_source_names"] == 592
        ),
        "all_raw_alternatives_remain_retained": (
            input_path.is_file()
            and input_actual == config["input"]["sha256"]
            and summary["input_rows"] == 711
        ),
        "prespecified_policy_ambiguity_is_quantified_and_retained": (
            summary["duplicate_names_with_all_five_policies_agreeing"] == 27
            and summary["duplicate_names_policy_ambiguous"] == 92
            and summary["all_name_distinct_choice_counts"] == {"1": 500, "2": 92}
            and len(set(summary["policy_choice_sha256"].values())) == 4
            and boundary["raw_alternative_rows_retained"]
        ),
        "canonical_summary_reproduces_exactly": expected_summary_exact,
        "kinematic_gravity_and_solar_targets_remain_sealed": sealed,
        "no_final_holdout_selection_claimed": (
            not boundary["final_holdout_sample_selected"]
            and expected["final_holdout_galaxies_selected"] == 0
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    output_path = ROOT / config["outputs"]["canonical_csv"]
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_wallaby_canonical_source_row_checkpoint",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            "path": config["parent"]["path"],
            "expected_sha256": config["parent"]["sha256"],
            "actual_sha256": parent_actual,
            "exact": parent_actual == config["parent"]["sha256"],
        },
        "input_audit": {
            "path": config["input"]["path"],
            "expected_sha256": config["input"]["sha256"],
            "actual_sha256": input_actual,
            "rows": len(rows),
            "unique_names": len({row["name"] for row in rows}),
        },
        "canonical_output": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": bytes_sha256(payload),
            "bytes": len(payload),
            **summary,
        },
        "policy": {
            "canonical": config["canonical_policy"],
            "robustness_policies": config["robustness_policies"],
            "published_flag_semantics": config["published_flag_semantics"],
        },
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "decision": (
            "passed_canonical_source_rows_not_holdout_selection"
            if all(gates.values())
            else "failed_canonical_source_rows"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }
    return payload, report


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    payload, report = build_artifacts()
    output_path = ROOT / config["outputs"]["canonical_csv"]
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
                "output": report["canonical_output"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_canonical_source_rows_not_holdout_selection":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
