import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_sigma_v19cy_a2319_chandra_image as builder


def test_inputs_are_frozen_and_preparation_passed():
    config, report = builder.validate_inputs()
    assert report["terminal_gate_passed"] is True
    assert config["authorization"]["access_A3667_validation"] is False
    assert config["authorization"]["access_A754_holdout"] is False


def test_crop_is_positive_square_and_preserves_celestial_center(tmp_path: Path):
    shape = (200, 200)
    data = np.ones(shape, dtype=np.float32)
    data[0, 0] = np.nan
    data[1, 1] = -1.0
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100.5, 100.5]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.crval = [290.299, 43.9345]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    source = tmp_path / "source.img"
    output = tmp_path / "output.img"
    fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(source)
    result = builder.crop_positive_image(
        source,
        output,
        {"crop_center_ra_deg": 290.299, "crop_center_dec_deg": 43.9345, "crop_width_arcmin": 2.0},
    )
    assert result["shape"] == [120, 120]
    assert result["positive_pixels"] > 0
    assert abs(result["center_ra_deg"] - 290.299) < 1e-5
    assert abs(result["center_dec_deg"] - 43.9345) < 1e-5


def test_merge_command_uses_both_frozen_obsids_and_is_deterministic(tmp_path: Path):
    config, _ = builder.validate_inputs()
    command = builder.merge_obs_command(config, tmp_path)
    assert "acisf03231N004_evt2" in command
    assert "acisf15187N003_evt2" in command
    assert "bands=0.5:7.0:2.3" in command
    assert "binsize=1" in command
    assert "parallel=no" in command
    assert "random=7" in command
