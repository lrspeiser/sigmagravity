from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.reconstruct_rxj2129_covariance import (
    _anchored_block_summaries,
    _construct_covariances,
    _protocols,
)


ROOT = Path(__file__).resolve().parents[1]


def test_anchored_blocks_partition_annular_spaxels_once() -> None:
    data = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4) + 1
    variance = np.ones_like(data)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    summaries = _anchored_block_summaries(
        data, variance, mask, block_shape=(2, 2), origin=(0, 0)
    )

    assert len(summaries["keys"]) == 4
    assert summaries["member_count"].sum() == mask.sum()
    np.testing.assert_allclose(
        summaries["spectrum"].sum(axis=0), data[:, mask].sum(axis=1)
    )
    np.testing.assert_allclose(
        summaries["variance"].sum(axis=0), variance[:, mask].sum(axis=1)
    )


def test_covariance_sum_preserves_coherent_systematic_shifts_and_is_psd() -> None:
    bootstrap = np.array(
        [
            [100.0, 110.0, 120.0, 130.0],
            [101.0, 111.0, 121.0, 131.0],
            [99.0, 109.0, 119.0, 129.0],
        ]
    )
    baseline = np.array([100.0, 110.0, 120.0, 130.0])
    sensitivity = np.array(
        [
            [102.0, 112.0, 122.0, 132.0],
            [98.0, 108.0, 118.0, 128.0],
        ]
    )
    bootstrap_covariance, systematic_covariance, total_covariance = (
        _construct_covariances(bootstrap, baseline, sensitivity)
    )

    assert np.all(systematic_covariance > 0)
    np.testing.assert_allclose(
        total_covariance, bootstrap_covariance + systematic_covariance
    )
    assert np.linalg.eigvalsh(total_covariance).min() >= -1e-10


def test_frozen_covariance_protocol_has_all_required_runs_and_templates() -> None:
    config = json.loads(
        (ROOT / "configs/r1_rxj2129_covariance_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(
        (ROOT / "data/raw/r1_ppxf_templates/provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["spatial_block_bootstrap"]["replicates"] == 100
    assert config["spatial_block_bootstrap"]["block_shape_spaxels"] == [2, 2]
    assert len(_protocols(config)) == 10
    assert sum(protocol["baseline"] for protocol in _protocols(config)) == 1
    configured_hashes = {
        item["family"]: item["sha256"] for item in config["inputs"]["template_products"]
    }
    provenance_hashes = {
        item["family"]: item["sha256"] for item in provenance["products"]
    }
    assert configured_hashes == provenance_hashes
    assert not config["authorization"]["gravity_response_fit"]


def test_resolution_corrected_protocol_uses_xsl_and_five_one_factor_shifts() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/r1_rxj2129_covariance_resolution_corrected_protocol.json"
        ).read_text(encoding="utf-8")
    )
    protocols = _protocols(config)
    baseline = [item for item in protocols if item["baseline"]]
    included = [
        item for item in protocols if item["include_in_systematic_covariance"]
    ]

    assert len(protocols) == 10
    assert len(baseline) == 1
    assert baseline[0]["template_family"] == "XSL"
    assert baseline[0]["protocol_id"] == "xsl_fwhm2.6_mask6"
    assert len(included) == 5
    assert config["spatial_block_bootstrap"]["spectral_configuration"][
        "template_family"
    ] == "XSL"

    parent = json.loads(
        (
            ROOT
            / "configs/r1_rxj2129_ppxf_resolution_corrected_protocol.json"
        ).read_text(encoding="utf-8")
    )
    template_max = max(
        parent["spectral_fit"]["template_fwhm_rest_angstrom_over_fit_range"]
    )
    galaxy_min = min(
        parent["spectral_fit"]["galaxy_fwhm_rest_angstrom_over_sensitivity_range"]
    )
    assert template_max < galaxy_min
    assert not config["authorization"]["gravity_response_fit"]
