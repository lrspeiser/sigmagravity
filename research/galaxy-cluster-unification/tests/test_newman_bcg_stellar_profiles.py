from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reconstruct_newman_bcg_stellar_profiles import (  # noqa: E402
    annular_surface_density,
    projected_enclosed_fraction,
)


def test_dpie_projected_mass_normalizes_to_total() -> None:
    radius = np.asarray([0.0, 1.0, 100.0, 1e8])
    fraction = projected_enclosed_fraction(radius, core_kpc=0.75, cut_kpc=52.7)
    assert fraction[0] == 0.0
    assert np.all(np.diff(fraction) > 0)
    assert np.isclose(fraction[-1], 1.0, rtol=1e-6)
    density = annular_surface_density(
        np.asarray([0.0]),
        np.asarray([1e8]),
        total_mass_msun=1.0e12,
        core_kpc=0.75,
        cut_kpc=52.7,
    )
    recovered = density[0] * np.pi * 1e16
    assert np.isclose(recovered, 1.0e12, rtol=1e-6)


def test_reconstructs_seven_stellar_components_without_passing_r1(tmp_path: Path) -> None:
    profile = tmp_path / "profiles.csv"
    covariance = tmp_path / "covariance.csv"
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reconstruct_newman_bcg_stellar_profiles.py"),
            "--profile-output",
            str(profile),
            "--covariance-output",
            str(covariance),
            "--report-output",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    profiles = pd.read_csv(profile)
    assert report["summary"]["newman_systems_reconstructed"] == 7
    assert report["summary"]["a2537_reference_annuli_inside_dynamics_support"] == 3
    assert report["summary"]["systems_with_complete_baryonic_profile"] == 0
    assert report["summary"]["systems_passing_complete_R1_gate"] == 0
    assert set(profiles["sps_imf"]) == {"Chabrier"}
    assert (profiles["stellar_surface_density_msun_kpc2"] > 0).all()
