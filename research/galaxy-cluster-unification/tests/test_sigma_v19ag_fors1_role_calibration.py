import importlib.util
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19ag_fors1_role_calibration.py"
CONFIG = ROOT / "configs" / "sigma_v19ag_fors1_role_calibration.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ag", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 2080
    header["NAXIS2"] = 2048
    header["ESO DET OUTPUTS"] = 4
    header["ESO DET CHIP1 ID"] = "TK2048EB4-1 1604"
    header["ESO DET READ CLOCK"] = "Readout ABCD (low)"
    for port, (x, y) in enumerate(((1, 1), (2048, 1), (1, 2049), (2048, 2049)), start=1):
        header[f"ESO DET OUT{port} X"] = x
        header[f"ESO DET OUT{port} Y"] = y
        header[f"ESO DET OUT{port} NX"] = 1008
        header[f"ESO DET OUT{port} NY"] = 1024
        header[f"ESO DET OUT{port} PRSCX"] = 16
        header[f"ESO DET OUT{port} OVSCX"] = 16
    return header


def test_frozen_runner_hash_and_parent_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]


def test_detector_regions_exactly_cover_2016_by_2048():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    regions = MODULE.detector_regions(synthetic_header(), config["detector_model"])
    assert len(regions) == 4
    assert [region["label"] for region in regions] == [
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
    ]


def test_portwise_row_prescan_subtraction_and_valid_concatenation():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    header = synthetic_header()
    regions = MODULE.detector_regions(header, config["detector_model"])
    raw = np.zeros((2048, 2080), dtype=np.float32)
    expected = np.zeros((2048, 2016), dtype=np.float32)
    for index, region in enumerate(regions, start=1):
        row_level = np.arange(1024, dtype=np.float32)[:, None] + 100 * index
        raw[region["prescan"]] = row_level
        raw[region["valid"]] = row_level + index
        raw[region["overscan"]] = 60000 + index
        expected[region["mosaic_y"], region["mosaic_x"]] = index
    corrected, diagnostics = MODULE.correct_ports(raw, regions, (2048, 2016))
    np.testing.assert_allclose(corrected, expected)
    assert len(diagnostics) == 4


def test_boundary_fraction_detects_vertical_step():
    image = np.ones((2048, 2016), dtype=np.float32)
    image[:, 1008:] *= 1.1
    metric = MODULE.boundary_fraction(image, 24)
    assert metric["vertical_fraction"] > 0.09
    assert metric["horizontal_fraction"] == 0.0


def test_bias_block_shape_exactly_tiles_active_mosaic():
    image = np.arange(2048 * 2016, dtype=np.float32).reshape(2048, 2016)
    blocks = MODULE.block_medians(image, (64, 63))
    assert blocks.shape == (32, 32)
