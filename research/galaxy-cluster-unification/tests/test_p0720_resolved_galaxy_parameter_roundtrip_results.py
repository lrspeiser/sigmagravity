from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from voidscreen.resolved_galaxy_generator import package_content_hash

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "p0720_resolved_galaxy_parameter_roundtrip"


def test_p0720_real_map_roundtrip_passes_commissioning_gates() -> None:
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["galaxies"] == 13
    assert report["components_scored"] == 39
    assert all(report["checks"].values())
    assert report["gravity_parameters"] == 0
    assert report["velocity_targets_used"] is False
    assert report["aggregate"]["total"]["median_normalized_l2"] <= 0.20
    assert report["aggregate"]["total"]["maximum_normalized_l2"] <= 0.35
    assert report["aggregate"]["total"]["median_pixel_correlation"] >= 0.97
    assert report["maximum_3d_projection_relative_error"] <= 1e-12


def test_p0720_preserves_parameter_identity_and_declares_3d_priors() -> None:
    catalog = pd.read_csv(RESULT / "parameter_catalog.csv")
    vertical = pd.read_csv(RESULT / "vertical_prior_ensemble.csv")
    assert len(catalog) == 13
    assert catalog.parameter_content_sha256.nunique() == 13
    assert (catalog.gravity_parameter_count == 0).all()
    assert not catalog.velocity_targets_used.any()
    assert len(vertical) == 78
    assert (vertical.status == "assumed_prior_not_measured").all()
    assert vertical.projectionRelativeError.max() <= 1e-12
    assert (
        vertical.groupby(["galaxy", "component"]).massWeightedZ2Kpc2.nunique() > 1
    ).all()


def test_p0720_all_saved_parameter_packages_are_formula_independent() -> None:
    paths = sorted((RESULT / "parameters").glob("*.json"))
    assert len(paths) == 13
    for path in paths:
        package = json.loads(path.read_text(encoding="utf-8"))
        assert package["contentSha256"] == package_content_hash(package)
        assert package["gravityParameters"] == {}
        assert package["velocityTargetsUsed"] is False
        assert package["verticalStructure"]["status"] == "assumed_prior_not_measured"
        assert package["sourceObservables"]["stellarMassToLightAssumption"]["solarUnits"] == 0.5
