from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_extracts_observable_level_image_positions_and_audits_support(tmp_path: Path) -> None:
    images_path = tmp_path / "images.csv"
    support_path = tmp_path / "support.csv"
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "extract_r1_strong_lens_observables.py"),
            "--image-output",
            str(images_path),
            "--support-output",
            str(support_path),
            "--report-output",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    images = pd.read_csv(images_path)
    support = pd.read_csv(support_path)
    assert report["systems"]["A2390"]["published_image_positions"] == 13
    assert report["systems"]["A2537"]["published_image_positions"] == 16
    assert report["systems"]["A383"]["observable_level_image_positions"] == 14
    assert report["systems"]["A611"]["published_image_positions"] == 12
    assert report["systems"]["MACS J0326"]["observable_level_image_positions"] == 8
    assert report["systems"]["MACS J0417"]["published_image_positions"] == 57
    assert report["systems"]["MACS J0949"]["published_image_positions"] == 20
    assert report["systems"]["MACS J1427"]["observable_level_image_positions"] == 2
    assert report["systems"]["MS2137"]["published_image_positions"] == 11
    assert report["systems"]["A2537"]["fiducial_source_families"] == 4
    assert report["systems"]["MACS J0417"]["fiducial_source_families"] == 15
    assert report["systems"]["MACS J0417"]["fiducial_images_with_spectroscopic_source_redshift"] == 21
    assert report["systems"]["MACS J0949"]["fiducial_source_families"] == 6
    assert len(images) == 175
    assert images["observable_level_image_position"].sum() == 166
    assert report["summary"]["systems"] == 10
    assert report["summary"]["systems_with_published_position_uncertainty"] == 10
    assert (
        report["summary"]["images_with_strict_position_and_spectroscopic_redshift_likelihood_inputs"]
        == 106
    )
    strict = images.loc[images["alternative_metric_likelihood_ready"]]
    assert strict["likelihood_source_archive"].notna().all()
    assert strict["likelihood_source_equation"].notna().all()
    assert report["summary"]["systems_with_three_verified_lensing_families_inside_dynamics_support"] == 0
    assert support["inside_dynamics_support"].sum() == 4
    assert report["systems"]["A2537"]["strict_likelihood_source_families_inside_dynamics_support"] == 1
    assert report["systems"]["A383"]["strict_likelihood_source_families_inside_dynamics_support"] == 1
    assert report["systems"]["MS2137"]["strict_likelihood_source_families_inside_dynamics_support"] == 1
    assert report["systems"]["MACS J1427"]["radial_support_auditable"] is False
    a2537_family2 = images.loc[
        (images["system"] == "A2537") & (images["source_family"] == "2")
    ]
    assert (a2537_family2["source_redshift"] == 2.786).all()
    assert a2537_family2["alternative_metric_likelihood_ready"].all()
    assert images.loc[
        images["system"] == "MACS J0417", "alternative_metric_likelihood_ready"
    ].sum() == 21
    j0326_predictions = images.loc[
        (images["system"] == "MACS J0326") & images["image_id"].isin(["1.1", "2.4", "3.4", "3.5"])
    ]
    assert not j0326_predictions["observable_level_image_position"].any()
    j1427_bad = images.loc[
        (images["system"] == "MACS J1427") & images["image_id"].isin(["1.1", "1.2", "1.3", "2.3"])
    ]
    assert not j1427_bad["observable_level_image_position"].any()
