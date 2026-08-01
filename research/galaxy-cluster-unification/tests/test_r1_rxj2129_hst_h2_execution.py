from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_hst_h2_centroid_execution_protocol.json"
REPORT = ROOT / "results/r1_rxj2129_hst_h2/report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def test_h2_execution_products_reproduce_every_frozen_acceptance_gate() -> None:
    if not REPORT.is_file():
        pytest.skip("immutable H2 execution is still running")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for record in report["outputs"].values():
        path = _path(record["path"])
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]

    band = pd.read_csv(_path(config["outputs"]["H2_band_fit_ledger"]), dtype={"image_id": str})
    image = pd.read_csv(_path(config["outputs"]["H2_image_ledger"]), dtype={"image_id": str})
    assert len(band) == report["band_fits_attempted"] == 42
    assert len(image) == report["images_attempted"] == 21
    assert band[["image_id", "band"]].duplicated().sum() == 0
    assert image["image_id"].is_unique
    assert set(band["band"]) == {"F814W", "F125W"}
    assert band.groupby("image_id")["band"].nunique().eq(2).all()

    fit_success = band["fit_success"].astype(bool)
    successful_columns = [
        "centroid_ra_deg",
        "centroid_dec_deg",
        "source_flux_SNR",
        "bootstrap_successful_fraction",
        "standard_error_east_arcsec_floored",
        "standard_error_north_arcsec_floored",
        "centroid_inside_central_half",
    ]
    assert np.isfinite(band.loc[fit_success, successful_columns[:-1]].to_numpy(dtype=float)).all()
    assert band.loc[fit_success, "centroid_inside_central_half"].notna().all()
    assert band.loc[~fit_success, "failure_reason"].astype(str).str.strip().ne("").all()

    thresholds = config["image_acceptance"]
    expected_acceptance: dict[str, bool] = {}
    for image_id, rows in band.groupby("image_id", sort=False):
        image_row = image.loc[image["image_id"] == image_id].iloc[0]
        complete = bool(rows["fit_success"].astype(bool).all())
        assert bool(image_row["both_band_fits_complete"]) is complete
        accepted = False
        if complete:
            gates = (
                (rows["source_flux_SNR"] >= thresholds["minimum_source_flux_SNR_each_band"])
                & (
                    rows["bootstrap_successful_fraction"]
                    >= thresholds["minimum_successful_bootstrap_fraction_each_band"]
                )
                & rows["centroid_inside_central_half"].astype(bool)
                & (
                    rows["standard_error_east_arcsec_floored"]
                    <= thresholds["maximum_per_coordinate_standard_error_arcsec"]
                )
                & (
                    rows["standard_error_north_arcsec_floored"]
                    <= thresholds["maximum_per_coordinate_standard_error_arcsec"]
                )
            )
            cross = bool(
                image_row["cross_band_separation_arcsec"]
                <= thresholds["maximum_cross_band_centroid_difference_arcsec"]
            )
            assert bool(image_row["cross_band_gate"]) is cross
            accepted = bool(gates.all() and cross)
        else:
            assert not bool(image_row["cross_band_gate"])
        expected_acceptance[image_id] = accepted
        assert bool(image_row["accepted"]) is accepted

    accepted_ids = {name for name, accepted in expected_acceptance.items() if accepted}
    required = set(thresholds["required_inner_images"])
    expected_gates = {
        "all_immutable_images_attempted": len(image) == 21 and len(band) == 42,
        "minimum_images_accepted": len(accepted_ids)
        >= thresholds["minimum_total_images_accepted"],
        "all_required_inner_images_accepted": required.issubset(accepted_ids),
    }
    assert report["images_accepted"] == len(accepted_ids)
    assert report["accepted_inner_images"] == sorted(required & accepted_ids)
    assert report["gates"] == expected_gates
    assert report["status"] == ("pass" if all(expected_gates.values()) else "fail")
    assert report["authorization"]["assemble_H3_covariance"] is all(
        expected_gates.values()
    )
    assert report["authorization"]["infer_dynamical_or_Weyl_response"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False

    with np.load(
        _path(config["outputs"]["H2_centroid_draws"]), allow_pickle=False
    ) as draws:
        assert draws["image_ids"].shape == (21,)
        assert draws["bands"].tolist() == ["F814W", "F125W"]
        assert draws["east_north_arcsec"].shape == (21, 2, 500, 2)
        assert draws["reference_pixels"].shape == (21, 2, 500, 2)
        assert draws["successful"].shape == (21, 2, 500)
        assert np.array_equal(
            draws["successful"], np.isfinite(draws["east_north_arcsec"]).all(axis=-1)
        )
        assert draws["image_ids"].astype(str).tolist() == image["image_id"].tolist()
        for image_index, image_id in enumerate(draws["image_ids"].astype(str)):
            for band_index, band_name in enumerate(draws["bands"].astype(str)):
                row = band.loc[
                    (band["image_id"] == image_id) & (band["band"] == band_name)
                ].iloc[0]
                expected_successes = int(draws["successful"][image_index, band_index].sum())
                if bool(row["fit_success"]):
                    assert int(row["bootstrap_draws_successful"]) == expected_successes
                    assert np.isclose(
                        row["bootstrap_successful_fraction"], expected_successes / 500
                    )
                else:
                    assert expected_successes == 0
