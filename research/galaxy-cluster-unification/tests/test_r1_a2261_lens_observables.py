from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_a2261_protocol_is_observable_only_and_frozen() -> None:
    protocol = json.loads((ROOT / "configs/r1_a2261_lens_observable_protocol.json").read_text())
    assert protocol["frozen_before_local_catalog_ingest"] is True
    assert "LensPerfect zLens values" in protocol["lens_sources"]["forbidden_as_input"]
    assert protocol["authorization"]["fit_lens_mass_model"] is False
    assert protocol["authorization"]["fit_gravity_response"] is False


def test_a2261_catalog_and_same_radius_gate() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_a2261_lens_observables.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_a2261_lens_observables/report.json").read_text())
    images = pd.read_csv(ROOT / "data/derived/r1_a2261_lens_observables.csv")
    families = pd.read_csv(ROOT / "data/derived/r1_a2261_lens_families.csv")
    assert len(images) == 30
    assert len(families) == 12
    assert report["gates"]["catalog_integrity_passed"] is True
    assert report["catalog"]["family_wide_position_dof_after_source_positions"] == 36
    assert report["catalog"]["spectroscopic_redshift_families"] == 1
    assert images.loc[images["family_id"] == 4, "family_redshift"].eq(3.377).all()
    assert images["gravity_target_used"].eq(False).all()
    assert report["radial_overlap"]["images_inside_dynamics_support"] == 0
    assert report["radial_overlap"]["nearest_image_radius_kpc"] > 15.0
    assert report["gates"]["observable_coordinate_likelihood_passed"] is False
    assert report["gates"]["same_radius_bridge_passed"] is False
    assert report["gates"]["strict_r1_readiness_passed"] is False
    assert report["authorization"]["promote_to_same_system_response_sample"] is False
    assert report["authorization"]["fit_weyl_response"] is False
