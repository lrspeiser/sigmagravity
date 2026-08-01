from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_lens_prescreen_is_frozen_and_geometry_only() -> None:
    config = json.loads((ROOT / "configs/r1_a1689_lens_prescreen_protocol.json").read_text())
    assert config["frozen_before_local_catalog_ingest"] is True
    assert config["dynamics_support"]["one_sided_radius_kpc"] == 15.0
    bridge = config["pre_registered_gates"]["same_radius_bridge"]
    assert bridge["minimum_independently_redshift_anchored_images_inside_dynamics_support"] == 3
    assert bridge["minimum_independently_redshift_anchored_families_inside_dynamics_support"] == 2
    assert bridge["minimum_distinct_image_radii_inside_dynamics_support"] == 3
    assert config["authorization"]["fit_gravity_response"] is False


def test_a1689_lens_prescreen_passes_without_claiming_a_likelihood() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_a1689_lens_prescreen.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_a1689_lens_prescreen/report.json").read_text())
    assert report["catalog"]["images"] == 135
    assert report["catalog"]["families"] == 42
    assert report["catalog"]["independently_redshift_anchored_families"] == 42
    assert report["radial_overlap"]["independently_redshift_anchored_images_inside_support"] == 6
    assert report["radial_overlap"]["independently_redshift_anchored_families_inside_support"] == 6
    assert report["radial_overlap"]["distinct_image_radii_inside_support"] == 6
    assert report["gates"]["same_radius_bridge_passed"] is True
    assert report["gates"]["raw_gemini_reconstruction_authorized"] is True
    assert report["gates"]["observable_coordinate_likelihood_ready"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
