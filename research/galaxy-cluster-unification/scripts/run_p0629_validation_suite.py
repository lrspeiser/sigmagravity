#!/usr/bin/env python3
"""Rebuild and validate the P0623-P0629 density/path-survival investigation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPTS = [
    "scripts/run_p0623_density_path_survival.py",
    "scripts/run_p0624_deep_porous_cross_domain.py",
    "scripts/run_p0625_bounded_porosity_survival.py",
    "scripts/run_p0626_compact_scalar_angular_route.py",
    "scripts/run_p0627_or_strength_phase_atlas.py",
    "scripts/run_p0628_selected_density_route_synthesis.py",
    "scripts/run_p0629_hierarchical_density_survival.py",
]


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        for script in BUILD_SCRIPTS:
            run([sys.executable, script])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_p0623_p0629_density_path_survival.py",
            "-o",
            "addopts=",
            "-q",
        ]
    )
    print("\nP0623-P0629 density/path-survival validation complete.")


if __name__ == "__main__":
    main()
