from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_relics_ensemble_audit_and_radial_overlap(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    audit_report = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_relics_lens_ensembles.py"),
            "--manifest-output",
            str(manifest),
            "--report-output",
            str(audit_report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    files = pd.read_csv(manifest)
    assert audit["totals"]["systems"] == 3
    assert audit["totals"]["mcmc_range_maps"] == 300
    assert len(files) == 303
    assert audit["classification"]["observable_level_likelihood"] is False
    assert audit["classification"]["rerunnable_lenstool_inputs_local"] is False

    profiles = tmp_path / "profiles.csv"
    covariance = tmp_path / "covariance.csv"
    centers = tmp_path / "centers.csv"
    radial_report = tmp_path / "radial.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reconstruct_relics_radial_kappa.py"),
            "--profile-output",
            str(profiles),
            "--covariance-output",
            str(covariance),
            "--center-audit-output",
            str(centers),
            "--report-output",
            str(radial_report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    radial = json.loads(radial_report.read_text(encoding="utf-8"))
    assert radial["summary"]["systems_reconstructed"] == 3
    assert radial["summary"]["systems_with_three_full_lensing_annuli_inside_dynamics_support"] == 1
    assert radial["summary"]["systems_with_verified_bcg_centered_three_plus_three_overlap"] == 1
    assert radial["systems"]["A2537"]["passes_three_overlapping_lensing_annuli"]
    assert not radial["systems"]["MACS J0417"]["passes_three_overlapping_lensing_annuli"]
    assert not radial["systems"]["MACS J0949"]["passes_three_overlapping_lensing_annuli"]
    center_audit = pd.read_csv(centers)
    assert center_audit["centering_verified"].sum() == 3
    assert center_audit.loc[
        center_audit["system"] == "A2537", "centering_verified"
    ].iloc[0]
    assert "later frozen raw-dynamics calibration gate failed" in radial[
        "classification"
    ]["remaining_requirements"]
    assert radial["summary"]["systems_passing_complete_R1_gate"] == 0
