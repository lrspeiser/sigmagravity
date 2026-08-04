"""Mechanism-level synthesis for the three Sigma v7 carrier formulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _nested_gate(report: Mapping[str, object], gate: str) -> bool:
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or gate not in gates:
        raise ValueError(f"report is missing required gate {gate!r}")
    value = gates[gate]
    if not isinstance(value, bool):
        raise TypeError(f"gate {gate!r} must be boolean")
    return value


def audit_positive_spin2_sequence(
    reports: Sequence[Mapping[str, object]],
    *,
    formulation_names: Sequence[str],
    failure_gate_names: Sequence[str],
    minimum_distinct_failures: int = 3,
) -> dict[str, object]:
    """Verify that distinct formulations fail their frozen useful-lensing gate."""

    minimum = int(minimum_distinct_failures)
    if minimum < 1:
        raise ValueError("minimum_distinct_failures must be positive")
    if not (
        len(reports) == len(formulation_names) == len(failure_gate_names)
    ):
        raise ValueError("reports, formulation_names, and failure_gate_names must align")

    formulations: list[dict[str, object]] = []
    candidates: list[str] = []
    for report, formulation, gate in zip(
        reports, formulation_names, failure_gate_names, strict=True
    ):
        candidate = report.get("candidate")
        decision = report.get("decision")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("every report must declare a candidate")
        if not isinstance(decision, str) or not decision:
            raise ValueError("every report must declare a decision")
        gate_value = _nested_gate(report, gate)
        observational_data = bool(report.get("observational_data_accessed", False))
        raw_holdout = bool(report.get("raw_holdout_opened", False))
        candidates.append(candidate)
        formulations.append(
            {
                "formulation": str(formulation),
                "candidate": candidate,
                "failed_gate": str(gate),
                "gate_value": gate_value,
                "decision": decision,
                "observational_data_accessed": observational_data,
                "raw_holdout_opened": raw_holdout,
            }
        )

    distinct_candidates = len(set(candidates))
    failed_count = sum(not bool(item["gate_value"]) for item in formulations)
    no_raw_holdout = not any(bool(item["raw_holdout_opened"]) for item in formulations)
    gates = {
        "minimum_distinct_formulations": distinct_candidates >= minimum,
        "minimum_failed_useful_lensing_gates": failed_count >= minimum,
        "no_raw_holdout_consumed": no_raw_holdout,
        "mechanism_reset_required": distinct_candidates >= minimum
        and failed_count >= minimum,
    }
    return {
        "formulations": formulations,
        "distinct_candidate_count": distinct_candidates,
        "failed_gate_count": failed_count,
        "gates": {name: bool(value) for name, value in gates.items()},
    }
