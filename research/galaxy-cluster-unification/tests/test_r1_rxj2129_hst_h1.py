from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rxj2129_hst_h1_passes_the_frozen_measurement_only_gates() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_hst_centroid_covariance_protocol.json").read_text()
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_hst_h1/report.json").read_text()
    )
    registration = report["registration"]
    psf_freeze = protocol["H1_execution_freeze"]["spatial_PSF"]

    assert report["protocol_version"] == protocol["protocol_version"]
    assert report["status"] == "pass"
    assert all(report["gates"].values())
    assert registration["mutual_match_count"] >= protocol["registration"][
        "minimum_matches"
    ]
    assert registration["cross_validated_RMS_arcsec"] <= protocol["registration"][
        "maximum_cross_validated_registration_RMS_arcsec"
    ]
    assert registration["bootstrap_draws_complete"] == protocol["registration"][
        "bootstrap_draws"
    ]
    assert registration["bootstrap_full_rank_fraction"] >= protocol[
        "H1_execution_freeze"
    ]["matching_and_affine_fit"]["minimum_full_rank_draw_fraction"]
    assert report["segmentation"]["retained_component_count"] > 0
    assert report["segmentation"]["masked_pixels_after_dilation"] > 0

    for band in ("F814W", "F125W"):
        result = report["PSF"][band]
        assert result["candidates_after_arc_exclusion"] >= protocol["psf"][
            "minimum_stars"
        ]
        assert (
            result["successful_fit_fraction"]
            >= psf_freeze["minimum_successful_fit_fraction"]
        )
        assert result["bootstrap_draws_complete"] == psf_freeze["bootstrap_draws"]

    assert report["authorization"] == {
        "execute_H2_arc_centroids": True,
        "assemble_H3_covariance": False,
        "use_lens_or_gravity_model": False,
        "infer_dynamical_or_Weyl_response": False,
        "fit_new_force_or_action": False,
    }


def test_rxj2129_hst_h1_products_are_complete_and_hash_locked() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_hst_centroid_covariance_protocol.json").read_text()
    )
    report = json.loads(
        (ROOT / "results/r1_rxj2129_hst_h1/report.json").read_text()
    )

    for name, record in report["outputs"].items():
        path = ROOT / record["path"]
        assert record["path"] == protocol["outputs"][name]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]

    registration = np.load(
        ROOT / protocol["outputs"]["H1_registration_draws"], allow_pickle=False
    )
    assert registration["bootstrap_coefficients_normalized"].shape == (500, 3, 2)

    psf = np.load(ROOT / protocol["outputs"]["H1_PSF_field"], allow_pickle=False)
    assert psf["f814w_field_bootstrap"].shape == (500, 3, 4)
    assert psf["f125w_field_bootstrap"].shape == (500, 3, 4)

    with fits.open(
        ROOT / protocol["outputs"]["H1_union_segmentation"], memmap=True
    ) as hdus:
        mask = hdus[0].data
        assert mask.shape == tuple(protocol["inputs"]["F814W"]["shape"])
        assert mask.dtype == np.dtype("uint8")
        assert np.count_nonzero(mask) == report["segmentation"][
            "masked_pixels_after_dilation"
        ]

    invalid = json.loads(
        (ROOT / "results/r1_rxj2129_hst_h1/invalid_v0_3/report.json").read_text()
    )
    assert invalid["status"] == "fail"
    assert invalid["authorization"]["execute_H2_arc_centroids"] is False
