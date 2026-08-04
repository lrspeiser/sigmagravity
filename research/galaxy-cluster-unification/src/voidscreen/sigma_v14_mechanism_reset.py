"""Theory-only mechanism-reset checks for Sigma v14A.

This module does not implement a gravity theory.  It makes the reset auditable:
retired mechanisms must have evidence, every three-formulation reset must have
three distinct members, and a proposed successor must state how it differs
from every retired carrier class before an action is attempted.

It also records two elementary screens used during the reset.  In four
dimensions ordinary p-forms add no new healthy orientation representation:
a massless two-form is dual to a scalar and a massive two-form is dual to a
vector.  A scalar-charge rank-two gauge electrostatic equation is fourth order;
in three spatial dimensions its point-source potential scales as ``r`` rather
than ``1/r``.  Such a gauge field may therefore be considered only as a
zero-monopole tidal channel, not as the Newtonian mass monopole.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PointSourceScaling:
    """Power-law exponents for a polyharmonic point-source equation."""

    spatial_dimension: int
    operator_order: int
    potential_power: int
    force_power: int


def massless_p_form_degrees_of_freedom(
    form_rank: int,
    *,
    spacetime_dimension: int = 4,
) -> int:
    """Return the local polarizations of a massless Abelian p-form."""

    if spacetime_dimension < 3:
        raise ValueError("spacetime_dimension must be at least three")
    if not 0 <= form_rank <= spacetime_dimension - 2:
        return 0
    return comb(spacetime_dimension - 2, form_rank)


def massive_p_form_degrees_of_freedom(
    form_rank: int,
    *,
    spacetime_dimension: int = 4,
) -> int:
    """Return the local polarizations of a massive Abelian p-form."""

    if spacetime_dimension < 2:
        raise ValueError("spacetime_dimension must be at least two")
    if not 0 <= form_rank <= spacetime_dimension - 1:
        return 0
    return comb(spacetime_dimension - 1, form_rank)


def point_source_scaling(
    *,
    spatial_dimension: int,
    operator_order: int,
) -> PointSourceScaling:
    """Return generic point-source powers for ``nabla^order phi = delta``.

    Away from logarithmic exceptional dimensions, the Green function scales
    as ``r**(operator_order-spatial_dimension)`` and its force as one radial
    derivative less.  The reset uses only three dimensions at orders two and
    four, which are not logarithmic cases.
    """

    if spatial_dimension < 1:
        raise ValueError("spatial_dimension must be positive")
    if operator_order < 2 or operator_order % 2:
        raise ValueError("operator_order must be a positive even order")
    potential_power = operator_order - spatial_dimension
    if potential_power == 0:
        raise ValueError("the logarithmic Green-function case needs separate handling")
    return PointSourceScaling(
        spatial_dimension=spatial_dimension,
        operator_order=operator_order,
        potential_power=potential_power,
        force_power=potential_power - 1,
    )


def p_form_screen() -> list[dict[str, int | str]]:
    """Return the four-dimensional p-form representation screen."""

    rows: list[dict[str, int | str]] = []
    duals = {
        ("massless", 0): "scalar",
        ("massless", 1): "vector",
        ("massless", 2): "scalar",
        ("massless", 3): "no local polarization",
        ("massive", 0): "scalar",
        ("massive", 1): "vector",
        ("massive", 2): "vector",
        ("massive", 3): "scalar",
    }
    for mass_class in ("massless", "massive"):
        for rank in range(4):
            if mass_class == "massless":
                count = massless_p_form_degrees_of_freedom(rank)
            else:
                count = massive_p_form_degrees_of_freedom(rank)
            rows.append(
                {
                    "mass_class": mass_class,
                    "form_rank": rank,
                    "local_degrees_of_freedom": count,
                    "four_dimensional_dual_class": duals[(mass_class, rank)],
                }
            )
    return rows


def audit_reset_protocol(
    protocol: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Validate evidence coverage and successor distinctness."""

    retired = protocol["retired_mechanisms"]
    resets = protocol["mechanism_resets"]
    postulates = protocol["v14a_postulates"]
    retired_ids = [str(row["id"]) for row in retired]
    evidence_rows = []
    missing_evidence: list[str] = []
    for row in retired:
        paths = [project_root / str(value) for value in row["evidence"]]
        missing = [
            str(path.relative_to(project_root)).replace("\\", "/")
            for path in paths
            if not path.is_file()
        ]
        missing_evidence.extend(missing)
        evidence_rows.append(
            {
                "mechanism_id": row["id"],
                "evidence_file_count": len(paths),
                "all_evidence_files_exist": not missing,
                "missing_evidence": missing,
            }
        )

    reset_rows = []
    for reset in resets:
        members = [str(value) for value in reset["member_versions"]]
        reset_rows.append(
            {
                "reset_id": reset["id"],
                "member_count": len(members),
                "members_are_distinct": len(set(members)) == len(members),
                "three_failure_rule_satisfied": (
                    len(members) >= 3 and len(set(members)) == len(members)
                ),
            }
        )

    forbidden = [str(value).lower() for value in postulates["forbidden_placements"]]
    constants = [str(value) for value in postulates["constants_reserved"]]
    newton = point_source_scaling(spatial_dimension=3, operator_order=2)
    scalar_charge_rank_two = point_source_scaling(
        spatial_dimension=3,
        operator_order=4,
    )
    gates = {
        "all_retired_mechanisms_have_unique_ids": len(set(retired_ids))
        == len(retired_ids),
        "all_retired_evidence_files_exist": not missing_evidence,
        "every_recorded_reset_has_three_distinct_formulations": all(
            bool(row["three_failure_rule_satisfied"]) for row in reset_rows
        ),
        "one_physical_metric_frozen": bool(postulates["one_physical_metric"]),
        "direct_rank2_mass_charge_forbidden": any(
            "direct rank-two gauge charge" in value for value in forbidden
        ),
        "clock_or_adm_trace_reuse_forbidden": any(
            "adm trace" in value for value in forbidden
        ),
        "ordinary_spacetime_tensor_kinetic_reuse_forbidden": any(
            "ordinary covariant component kinetic" in value for value in forbidden
        ),
        "material_triad_reuse_forbidden": any(
            "material-coordinate" in value for value in forbidden
        ),
        "localized_retarded_multiplier_reuse_forbidden": any(
            "localized retarded multiplier" in value for value in forbidden
        ),
        "constant_budget_is_respected": len(constants)
        == int(postulates["physical_constant_budget"])
        <= int(protocol["maximum_physical_constants"]),
        "observational_data_remains_closed": not bool(
            protocol["observational_data_authorized"]
        ),
        "direct_rank2_mass_charge_does_not_recover_newton": (
            scalar_charge_rank_two.potential_power != newton.potential_power
            and scalar_charge_rank_two.force_power != newton.force_power
        ),
    }
    return {
        "status": "Sigma v14A mechanism reset and postulate audit",
        "candidate": protocol["candidate"],
        "retired_mechanism_count": len(retired),
        "retired_mechanism_ids": retired_ids,
        "evidence_rows": evidence_rows,
        "missing_evidence": missing_evidence,
        "mechanism_reset_rows": reset_rows,
        "screened_new_languages": protocol["screened_new_languages"],
        "p_form_representation_screen": p_form_screen(),
        "point_source_scaling_screen": {
            "newtonian_scalar": newton.__dict__,
            "direct_scalar_charge_rank2": scalar_charge_rank_two.__dict__,
        },
        "v14a_postulates": postulates,
        "v14a_action_kill_gates": protocol["v14a_action_kill_gates"],
        "verification_gates": gates,
        "all_verification_gates_pass": all(gates.values()),
        "action_written": False,
        "constraint_and_hamiltonian_gate_passed": False,
        "weak_metric_gate_passed": False,
        "observational_data_accessed": False,
        "theory_viable": False,
        "decision": (
            "Advance the gauge-reduced, zero-monopole tidal carrier only to a "
            "covariant source/action derivation. Do not treat a higher-rank "
            "gauge mass charge as gravity, do not reuse an ordinary spacetime "
            "tensor kinetic term, material triad, retarded multiplier pair, or "
            "clock/ADM trace, and do not open observational data."
        ),
        "scope_limit": protocol["scope_limit"],
        "data_policy": protocol["data_policy"],
    }
