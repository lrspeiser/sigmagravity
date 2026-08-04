from __future__ import annotations

import pytest

from voidscreen.sigma_v7_falsification import audit_positive_spin2_sequence


def _report(candidate: str, gate: str, value: bool) -> dict[str, object]:
    return {
        "candidate": candidate,
        "decision": f"retire_{candidate}",
        "gates": {gate: value},
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
    }


def test_three_distinct_failed_carriers_trigger_mechanism_reset() -> None:
    gates = ["useful_amplitude", "geometry_discrimination", "metric_projection"]
    reports = [_report(f"candidate_{index}", gate, False) for index, gate in enumerate(gates)]
    audit = audit_positive_spin2_sequence(
        reports,
        formulation_names=["v7A", "v7B", "v7C"],
        failure_gate_names=gates,
    )
    assert audit["distinct_candidate_count"] == 3
    assert audit["failed_gate_count"] == 3
    assert all(audit["gates"].values())


def test_passing_or_duplicate_formulation_does_not_trigger_reset() -> None:
    gates = ["a", "b", "c"]
    reports = [
        _report("same", "a", False),
        _report("same", "b", False),
        _report("different", "c", True),
    ]
    audit = audit_positive_spin2_sequence(
        reports,
        formulation_names=["v7A", "v7B", "v7C"],
        failure_gate_names=gates,
    )
    assert not audit["gates"]["minimum_distinct_formulations"]
    assert not audit["gates"]["minimum_failed_useful_lensing_gates"]
    assert not audit["gates"]["mechanism_reset_required"]


@pytest.mark.parametrize(
    ("reports", "names", "gates"),
    [
        ([], [], []),
        ([_report("a", "gate", False)], [], ["gate"]),
        ([_report("a", "gate", False)], ["v7A"], ["missing"]),
    ],
)
def test_invalid_falsification_inputs_are_rejected(reports, names, gates) -> None:
    with pytest.raises(ValueError):
        audit_positive_spin2_sequence(
            reports,
            formulation_names=names,
            failure_gate_names=gates,
            minimum_distinct_failures=0 if not reports else 3,
        )
