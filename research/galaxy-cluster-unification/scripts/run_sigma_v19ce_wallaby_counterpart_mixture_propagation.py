from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19ce_wallaby_counterpart_mixture_propagation.json"
)


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


def foreground_weight(diagnostic: dict[str, str], treatment: dict[str, Any]) -> float:
    if diagnostic["quality_controlled_foreground_contamination"] == "true":
        return float(treatment["quality_controlled_weight"])
    if diagnostic["foreground_astrometric_evidence"] == "true":
        return float(treatment["other_foreground_evidence_weight"])
    return 1.0


def normalized_weights(
    candidates: list[dict[str, str]],
    diagnostics: dict[str, dict[str, str]],
    treatment: dict[str, Any],
    kernel: str,
) -> tuple[list[float | None], list[float], float]:
    raw = [
        max(0.0, float(row[f"spatial_lr_{kernel}"]))
        * foreground_weight(diagnostics[row["object_id"]], treatment)
        for row in candidates
    ]
    total = math.fsum(raw)
    if total <= 0:
        return [None] * len(candidates), raw, total
    weights = [value / total for value in raw]
    correction = 1.0 - math.fsum(weights)
    largest = max(range(len(weights)), key=weights.__getitem__)
    weights[largest] += correction
    return weights, raw, total


def effective_candidates(weights: list[float]) -> float:
    return 1.0 / math.fsum(value * value for value in weights)


def entropy_nats(weights: list[float]) -> float:
    return -math.fsum(value * math.log(value) for value in weights if value > 0)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.16g}"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    for spec in config["parents"].values():
        path = ROOT / spec["path"]
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise RuntimeError(f"V19CE parent changed: {spec['path']}")
    v19cb_config = load_json(ROOT / config["parents"]["v19cb_config"]["path"])
    v19cb_report = load_json(ROOT / config["parents"]["v19cb_report"]["path"])
    if v19cb_report["association_resolved_by_foreground_astrometry"]:
        raise RuntimeError("V19CE is unnecessary because V19CB reports resolution")

    scores = load_csv(ROOT / config["parents"]["candidate_spatial_scores"]["path"])
    releases = load_csv(ROOT / config["parents"]["release_information"]["path"])
    diagnostic_rows = load_csv(ROOT / config["parents"]["foreground_diagnostics"]["path"])
    diagnostics = {row["object_id"]: row for row in diagnostic_rows}
    release_by_id = {row["source_row_id"]: row for row in releases}
    candidates_by_release: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scores:
        candidates_by_release[row["source_row_id"]].append(row)
    if set(candidates_by_release) != set(release_by_id):
        raise RuntimeError("V19CE candidate/release inventory differs")
    if any(row["object_id"] not in diagnostics for row in scores):
        raise RuntimeError("V19CE lacks foreground coverage for a candidate")

    treatments = v19cb_config["treatment_branches"]
    kernels = v19cb_config["kernel_labels"]
    scenario_labels = [
        (treatment["id"], kernel)
        for treatment in treatments
        for kernel in kernels
    ]
    candidate_output: list[dict[str, Any]] = []
    scenario_output: list[dict[str, Any]] = []
    defined_normalization_errors: list[float] = []
    undefined_scenarios = 0

    for source_row_id in sorted(release_by_id, key=int):
        release = release_by_id[source_row_id]
        candidates = sorted(
            candidates_by_release[source_row_id], key=lambda row: row["object_id"]
        )
        weight_columns: dict[tuple[str, str], list[float | None]] = {}
        for treatment in treatments:
            for kernel in kernels:
                weights, raw, total = normalized_weights(
                    candidates, diagnostics, treatment, kernel
                )
                label = (treatment["id"], kernel)
                weight_columns[label] = weights
                defined = weights[0] is not None
                if defined:
                    finite = [float(value) for value in weights if value is not None]
                    normalization_error = abs(math.fsum(finite) - 1.0)
                    defined_normalization_errors.append(normalization_error)
                    maximum = max(finite)
                    top_index = min(
                        index for index, value in enumerate(finite) if value == maximum
                    )
                    effective = effective_candidates(finite)
                    entropy = entropy_nats(finite)
                    top_object = candidates[top_index]["object_id"]
                else:
                    undefined_scenarios += 1
                    normalization_error = ""
                    maximum = ""
                    effective = ""
                    entropy = ""
                    top_object = ""
                scenario_output.append(
                    {
                        "source_row_id": source_row_id,
                        "wallaby_name": release["wallaby_name"],
                        "team_release": release["team_release"],
                        "field": release["field"],
                        "treatment": treatment["id"],
                        "kernel": kernel,
                        "candidate_count": len(candidates),
                        "positive_raw_weight_candidates": sum(value > 0 for value in raw),
                        "raw_weight_sum": f"{total:.16g}",
                        "scenario_defined": str(defined).lower(),
                        "normalization_error": fmt(
                            float(normalization_error)
                            if normalization_error != ""
                            else None
                        ),
                        "effective_candidate_count": fmt(
                            float(effective) if effective != "" else None
                        ),
                        "entropy_nats": fmt(float(entropy) if entropy != "" else None),
                        "maximum_candidate_weight": fmt(
                            float(maximum) if maximum != "" else None
                        ),
                        "diagnostic_top_object_id": top_object,
                        "counterpart_selected": "false",
                    }
                )

        for index, candidate in enumerate(candidates):
            diagnostic = diagnostics[candidate["object_id"]]
            row: dict[str, Any] = {
                "source_row_id": source_row_id,
                "wallaby_name": release["wallaby_name"],
                "team_release": release["team_release"],
                "field": release["field"],
                "object_id": candidate["object_id"],
                "separation_arcsec": candidate["separation_arcsec"],
                "inside_map": candidate["inside_map"],
                "distance_to_nonzero_support_beams": candidate[
                    "distance_to_nonzero_support_beams"
                ],
                "extended_candidate_diagnostic": candidate[
                    "extended_candidate_diagnostic"
                ],
                "foreground_astrometric_evidence": diagnostic[
                    "foreground_astrometric_evidence"
                ],
                "quality_controlled_foreground_contamination": diagnostic[
                    "quality_controlled_foreground_contamination"
                ],
            }
            for kernel in kernels:
                row[f"spatial_lr_{kernel}"] = candidate[f"spatial_lr_{kernel}"]
            for treatment, kernel in scenario_labels:
                row[f"p_{treatment}_{kernel}"] = fmt(
                    weight_columns[(treatment, kernel)][index]
                )
            row["counterpart_selected"] = "false"
            candidate_output.append(row)

    inventory = config["scenario_inventory"]
    defined_scenarios = len(scenario_output) - undefined_scenarios
    max_norm_error = max(defined_normalization_errors, default=0.0)
    undefined_keys = {
        (row["source_row_id"], row["treatment"], row["kernel"])
        for row in scenario_output
        if row["scenario_defined"] == "false"
    }
    undefined_explicit = len(undefined_keys) == undefined_scenarios and all(
        all(
            row[f"p_{treatment}_{kernel}"] == ""
            for row in candidate_output
            if row["source_row_id"] == source_row_id
        )
        for source_row_id, treatment, kernel in undefined_keys
    )
    access = config["access_boundary"]
    gates = {
        "all_parent_hashes_exact_and_v19cb_unresolved": not v19cb_report[
            "association_resolved_by_foreground_astrometry"
        ],
        "all_711_releases_18550_candidates_and_16_scenarios_carried": len(
            release_by_id
        )
        == inventory["expected_release_rows"]
        and len(candidate_output) == inventory["expected_candidate_rows"]
        and len(treatments) == inventory["expected_treatments"]
        and len(kernels) == inventory["expected_kernels"]
        and len(scenario_output) == inventory["expected_release_scenarios"],
        "every_defined_scenario_normalizes_to_one": max_norm_error <= 1e-12,
        "every_undefined_scenario_is_explicit_and_has_blank_weights": undefined_explicit,
        "no_candidate_release_treatment_or_kernel_selected_or_removed": len(
            candidate_output
        )
        == len(scores)
        and all(row["counterpart_selected"] == "false" for row in candidate_output)
        and not config["mixture_definition"]["hard_counterpart_selected"]
        and not config["mixture_definition"]["treatment_selected"]
        and not config["mixture_definition"]["kernel_selected"],
        "kinematic_gravity_halo_lensing_holdout_and_solar_targets_sealed": not any(
            access[key]
            for key in (
                "wallaby_kinematic_table_row_read",
                "rotation_curve_or_velocity_field_read",
                "gravity_formula_or_residual_read",
                "halo_or_lensing_result_read",
                "sample_split_or_holdout_label_selected",
                "gravity_action_or_constant_changed",
                "solar_system_optimization_performed",
            )
        ),
        "future_likelihood_marginalizes_identity_without_target_feedback": config[
            "future_use_contract"
        ]["primary_result_must_marginalize_counterpart_identity"]
        and config["future_use_contract"][
            "candidate_identity_may_not_be_selected_by_gravity_fit"
        ],
        "no_new_tunable_parameter_or_gravity_formula": config[
            "mixture_definition"
        ]["new_tunable_hyperparameters"]
        == 0
        and not access["gravity_action_or_constant_changed"],
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("V19CE implemented and declared gate sets differ")

    candidate_path = ROOT / config["outputs"]["candidate_mixture_weights"]
    scenario_path = ROOT / config["outputs"]["release_scenario_summary"]
    report_path = ROOT / config["outputs"]["report"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(csv_bytes(candidate_output))
    scenario_path.write_bytes(csv_bytes(scenario_output))

    treatment_summary: dict[str, Any] = {}
    for treatment in treatments:
        rows = [
            row
            for row in scenario_output
            if row["treatment"] == treatment["id"]
        ]
        finite_effective = [
            float(row["effective_candidate_count"])
            for row in rows
            if row["scenario_defined"] == "true"
        ]
        treatment_summary[treatment["id"]] = {
            "scenarios": len(rows),
            "defined_scenarios": sum(
                row["scenario_defined"] == "true" for row in rows
            ),
            "undefined_scenarios": sum(
                row["scenario_defined"] == "false" for row in rows
            ),
            "median_effective_candidate_count": median(finite_effective),
        }
    report = {
        "protocol_version": config["protocol_version"],
        "status": "wallaby_counterpart_mixture_payload_completed",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "release_rows": len(release_by_id),
        "candidate_rows": len(candidate_output),
        "treatments": len(treatments),
        "kernels": len(kernels),
        "release_scenarios": len(scenario_output),
        "defined_scenarios": defined_scenarios,
        "undefined_scenarios": undefined_scenarios,
        "maximum_defined_normalization_error": max_norm_error,
        "treatment_summary": treatment_summary,
        "outputs": {
            "candidate_mixture_weights": {
                "path": config["outputs"]["candidate_mixture_weights"],
                "rows": len(candidate_output),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256(candidate_path),
            },
            "release_scenario_summary": {
                "path": config["outputs"]["release_scenario_summary"],
                "rows": len(scenario_output),
                "bytes": scenario_path.stat().st_size,
                "sha256": sha256(scenario_path),
            },
        },
        "gate_results": gates,
        "decision": (
            "counterpart_uncertainty_ready_for_target_blind_marginalization"
            if all(gates.values())
            else "counterpart_mixture_failed_closed"
        ),
        "access_boundary_audit": access,
        "claim_boundary": config["claim_boundary"],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not all(gates.values()):
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "decision",
                    "release_rows",
                    "candidate_rows",
                    "release_scenarios",
                    "defined_scenarios",
                    "undefined_scenarios",
                    "treatment_summary",
                    "outputs",
                    "gate_results",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
