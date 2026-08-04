from __future__ import annotations

import json
from pathlib import Path

import pytest

from voidscreen.sigma_v14_mechanism_reset import (
    audit_reset_protocol,
    massive_p_form_degrees_of_freedom,
    massless_p_form_degrees_of_freedom,
    p_form_screen,
    point_source_scaling,
)

ROOT = Path(__file__).resolve().parents[1]


def test_four_dimensional_p_forms_add_no_new_orientation_class() -> None:
    assert [massless_p_form_degrees_of_freedom(rank) for rank in range(4)] == [
        1,
        2,
        1,
        0,
    ]
    assert [massive_p_form_degrees_of_freedom(rank) for rank in range(4)] == [
        1,
        3,
        3,
        1,
    ]
    rows = p_form_screen()
    assert len(rows) == 8
    assert next(
        row
        for row in rows
        if row["mass_class"] == "massless" and row["form_rank"] == 2
    )["four_dimensional_dual_class"] == "scalar"
    assert next(
        row
        for row in rows
        if row["mass_class"] == "massive" and row["form_rank"] == 2
    )["four_dimensional_dual_class"] == "vector"


def test_direct_scalar_charge_rank2_field_misses_newtonian_scaling() -> None:
    newton = point_source_scaling(spatial_dimension=3, operator_order=2)
    rank_two = point_source_scaling(spatial_dimension=3, operator_order=4)
    assert (newton.potential_power, newton.force_power) == (-1, -2)
    assert (rank_two.potential_power, rank_two.force_power) == (1, 0)
    assert rank_two.force_power != newton.force_power


def test_invalid_scaling_and_dimension_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        point_source_scaling(spatial_dimension=0, operator_order=2)
    with pytest.raises(ValueError):
        point_source_scaling(spatial_dimension=3, operator_order=3)
    with pytest.raises(ValueError):
        point_source_scaling(spatial_dimension=2, operator_order=2)
    with pytest.raises(ValueError):
        massless_p_form_degrees_of_freedom(1, spacetime_dimension=2)


def test_frozen_reset_protocol_is_complete_and_keeps_data_closed() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "sigma_v14_mechanism_reset.json").read_text(
            encoding="utf-8"
        )
    )
    report = audit_reset_protocol(protocol, project_root=ROOT)
    assert report["retired_mechanism_count"] == 9
    assert len(report["mechanism_reset_rows"]) == 4
    assert report["all_verification_gates_pass"]
    assert not report["missing_evidence"]
    assert not report["action_written"]
    assert not report["observational_data_accessed"]
    assert not report["theory_viable"]
    assert all(report["verification_gates"].values())


def test_successor_explicitly_excludes_every_recent_failed_placement() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "sigma_v14_mechanism_reset.json").read_text(
            encoding="utf-8"
        )
    )
    forbidden = " ".join(
        protocol["v14a_postulates"]["forbidden_placements"]
    ).lower()
    for required in (
        "adm trace",
        "ordinary covariant component kinetic",
        "material-coordinate",
        "localized retarded multiplier",
        "direct rank-two gauge charge",
    ):
        assert required in forbidden
    assert protocol["v14a_postulates"]["physical_constant_budget"] <= 5
    assert not protocol["observational_data_authorized"]
