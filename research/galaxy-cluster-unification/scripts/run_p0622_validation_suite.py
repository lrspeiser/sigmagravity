#!/usr/bin/env python3
"""One-command runner for the P0622 comprehensive scientific test suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = [
    "tests/test_arc_invariants.py",
    "tests/test_route_template.py",
    "tests/test_baryon_morphology.py",
    "tests/test_solar_system_tail.py",
    "tests/test_p0554_local_cross_domain_sensitivity.py",
    "tests/test_p0554_multiscale_elasticity.py",
    "tests/test_p0612_cross_stage_parameter_impact_results.py",
    "tests/test_p0613_bounded_endpoint_cross_domain_results.py",
    "tests/test_p0614_composite_formula_audit_results.py",
    "tests/test_p0615_self_coupled_quadrupole_route_results.py",
    "tests/test_p0616_frozen_self_coupled_transfer_results.py",
    "tests/test_p0617_self_coupled_support_phase_atlas_results.py",
    "tests/test_p0618_universal_route_phase_results.py",
    "tests/test_p0619_frozen_tangential_transfer_results.py",
    "tests/test_p0620_parameter_impact_synthesis_results.py",
    "tests/test_p0622_comprehensive_regime_diagnostics.py",
]


def run(command: list[str]) -> None:
    shown = " ".join(command)
    print(f"\n> {shown}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild P0622 outputs and run the frozen 77-check scientific suite."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Validate existing result artifacts without rebuilding them.",
    )
    args = parser.parse_args()
    if not args.skip_build:
        run([sys.executable, "scripts/run_p0622_comprehensive_regime_diagnostics.py"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            *TEST_MODULES,
            "-o",
            "addopts=",
            "-q",
        ]
    )
    print("\nP0622 comprehensive suite complete: all scientific checks passed.")


if __name__ == "__main__":
    main()
