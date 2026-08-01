from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from voidscreen.basin_survivors import (
    algebraic_inverse_field_scaling,
    canonical_scalar_exterior_energy_fraction,
    direct_force_amplitude_for_field_energy_fraction,
    nonlinear_flux_law_scaling,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbm0_survivor_derivation_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nbm0_survivor_derivation",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    flux = nonlinear_flux_law_scaling(2.0)
    algebraic_flat = algebraic_inverse_field_scaling(0.0)
    algebraic_btfr = algebraic_inverse_field_scaling(0.5)
    benchmark = protocol["benchmark_compactness"]
    energy = {}
    for domain in ["galaxy", "cluster"]:
        compactness = benchmark[domain]
        energy[domain] = {
            "compactness": compactness,
            "field_energy_fraction_at_source_d_one": float(
                canonical_scalar_exterior_energy_fraction(1.0, compactness)
            ),
            "direct_force_amplitude_for_target_field_energy": float(
                direct_force_amplitude_for_field_energy_fraction(
                    benchmark["target_field_energy_fraction"], compactness
                )
            ),
        }

    decisions = [
        {
            "id": "A8.1",
            "decision": "excluded_as_retired_AQUAL_limit",
            "reason": "The unique isotropic flux power satisfying flat speed and v_flat^4 proportional to M_b is m=2: div(|grad Phi| grad Phi) proportional rho, the deep AQUAL/MOND equation.",
        },
        {
            "id": "A8.2",
            "decision": "reject",
            "reason": "For X proportional M/r and Phi_extra proportional X^n, flat speed requires n=0 while the square-root mass amplitude requires n=1/2; no single algebraic power satisfies both.",
        },
        {
            "id": "A8.3",
            "decision": "reject",
            "reason": "Assigning a square-root charge to an identified galaxy is nonadditive under subdivision and requires a noncovariant object-segmentation rule.",
        },
        {
            "id": "A8.4",
            "decision": "reject_within_one_state_variable_scope",
            "reason": "A history-dependent amplitude is a second state variable or per-object initial condition; with a unique static state it returns to A8.1-A8.3. No four-global one-field closure remains.",
        },
        {
            "id": "A9.1",
            "decision": "reject",
            "reason": "Canonical exterior field energy is compactness-suppressed. Supplying five times the baryonic mass requires direct-force amplitudes 1e7 for a galaxy and 1e6 for a cluster.",
        },
        {
            "id": "A9.2",
            "decision": "reject_for_galaxy_support",
            "reason": "Uniform positive vacuum energy has negative pressure but gives outward acceleration proportional to r; an exterior spherical shell supplies no broad internal 1/r force in the GR/Newtonian limit.",
        },
        {
            "id": "A9.3",
            "decision": "reject_as_nonindependent_branch",
            "reason": "Linearized nonminimal curvature coupling is a scalar fifth-force/metric-slip model already rejected by A2-A4; making it nonlinear returns to A8.",
        },
        {
            "id": "A9.4",
            "decision": "reclassify_outside_fixed_premises",
            "reason": "A stable condensate with an independent cosmological energy reservoir can lens and cluster, but it is a new gravitating dark component rather than pure void-pressure modified gravity.",
        },
        {
            "id": "A9.5",
            "decision": "reject",
            "reason": "Negative kinetic energy can make negative gravitational charge but fails the positive-Hamiltonian action-health gate.",
        },
        {
            "id": "A10",
            "decision": "advance_as_declared_loophole",
            "reason": "A constitutive void medium changes gravitational flux geometry rather than adding a force. It must be benchmarked against existing Refracted Gravity and then tested with a reciprocal nonlocal basin field.",
        },
    ]
    frame = pd.DataFrame(decisions)
    report = {
        "report_version": "NBM0-A2-survivor-derivation-0.1",
        "status": "A8 and A9 exhausted under fixed premises; one constitutive-boundary loophole admitted",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "astronomical_fit_performed": False,
        "nonlinear_flux_uniqueness": {
            "response_power": flux.response_power,
            "acceleration_mass_exponent": flux.acceleration_mass_exponent,
            "acceleration_radial_exponent": flux.acceleration_radial_exponent,
            "circular_speed_radial_exponent": flux.circular_speed_radial_exponent,
            "circular_speed_fourth_power_mass_exponent": flux.circular_speed_fourth_power_mass_exponent,
            "decision": "unique desired scaling is the retired deep-AQUAL/MOND equation",
        },
        "single_inverse_field_algebraic_gate": {
            "flat_candidate": algebraic_flat.__dict__,
            "btfr_candidate": algebraic_btfr.__dict__,
            "simultaneous_solution_exists": False,
        },
        "canonical_field_energy_budget": energy,
        "subfamily_decisions": decisions,
        "decision_counts": frame["decision"].value_counts().sort_index().to_dict(),
        "fixed_premise_result": "No A8/A9 subfamily survives all fixed premises.",
        "new_active_family": "A10 constitutive void-basin flux refraction",
        "A10_next_gates": [
            "Reproduce the analytic slab/disk flux-confinement limit and the spherical constant-permittivity limit.",
            "Separate the existing density-permittivity Refracted Gravity benchmark from a project-specific nonlocal basin field X; make no originality claim for permittivity gravity.",
            "Derive reciprocal X backreaction from an action before fitting epsilon(X).",
            "Require one universal epsilon(X), no class switch, same-metric lensing, Solar-System epsilon->1, and whole-system validation.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "subfamily_decisions.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
