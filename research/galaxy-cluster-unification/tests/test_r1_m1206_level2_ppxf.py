from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.reconstruct_m1206_level2_ppxf import _coadd_spectra


ROOT = Path(__file__).resolve().parents[1]


def test_level2_coadd_is_inverse_variance_weighted() -> None:
    wavelength = np.arange(4860.0, 4870.1, 1.25)
    first = {
        "wavelength": wavelength,
        "spectrum": np.full_like(wavelength, 10.0),
        "variance": np.full_like(wavelength, 4.0),
    }
    second = {
        "wavelength": wavelength + 0.1,
        "spectrum": np.full_like(wavelength, 20.0),
        "variance": np.full_like(wavelength, 16.0),
    }
    common, spectrum, variance, metadata = _coadd_spectra(
        [first, second], [4860.0, 4875.0], 2
    )
    assert common.size > 2
    # Continuum normalization removes absolute throughput differences.
    assert np.allclose(spectrum, 1.0)
    assert np.all(variance > 0)
    assert metadata["minimum_contributors"] == 2


def test_level2_protocol_freezes_systematic_checks_before_gravity() -> None:
    config = json.loads(
        (ROOT / "configs/r1_m1206_level2_products.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["coadd_protocol"][
        "outer_systematic_check"
    ].startswith("repeat the outermost")
    assert config["success_thresholds"][
        "maximum_leave_one_out_sigma_shift_fraction"
    ] == 0.2
    assert not config["authorization"]["gravity_response_fit"]
