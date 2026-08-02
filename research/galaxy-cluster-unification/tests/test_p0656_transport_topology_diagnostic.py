from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0656_transport_topology_diagnostic"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_has_no_fit_or_selection_authority():
    result = report()
    assert result["status"] == "descriptive_only"
    assert result["coverage"]["fields"] == 4
    assert result["coverage"]["spent_image_positions"] == 22
    assert result["coverage"]["field_image_rows"] == 88
    assert result["coverage"]["pairwise_field_comparisons"] == 6
    assert result["coverage"]["fitted_parameters"] == 0
    assert result["candidate_selected_or_advanced"] is False


def test_gather_fields_have_similar_local_gradient_scales():
    summaries = pd.read_csv(RESULTS / "field_summaries.csv").set_index("field")
    gather = summaries.loc[
        ["P0652_finite_gather", "P0653_compact_gather", "P0654_padded_gather"]
    ]
    assert gather.correction_gradient_spectral_norm_max.max() < 0.20
    assert gather.correction_gradient_spectral_norm_max.max() / gather.correction_gradient_spectral_norm_max.min() < 1.04
    assert gather.correction_convergence_RMS.max() < 0.022
    assert gather.correction_shear_RMS.max() < 0.041


def test_deposition_focuses_much_more_strongly_than_gather():
    summaries = pd.read_csv(RESULTS / "field_summaries.csv").set_index("field")
    gather = summaries.loc["P0654_padded_gather"]
    deposit = summaries.loc["P0655_padded_deposit"]
    assert deposit.image_correction_RMS_arcsec > 1.9 * gather.image_correction_RMS_arcsec
    assert deposit.correction_convergence_RMS > 6.0 * gather.correction_convergence_RMS
    assert deposit.correction_shear_RMS > 4.0 * gather.correction_shear_RMS
    assert deposit.correction_gradient_spectral_norm_max > 4.5 * gather.correction_gradient_spectral_norm_max
    assert deposit.negative_fixed_mapping_determinants == 10
    assert gather.negative_fixed_mapping_determinants == 8


def test_gather_fields_are_pairwise_aligned_but_not_identical():
    pairs = pd.read_csv(RESULTS / "pairwise_field_comparisons.csv")
    gather = pairs[
        ~pairs.left.str.contains("deposit") & ~pairs.right.str.contains("deposit")
    ]
    assert gather.vector_cosine_correlation.min() > 0.87
    assert gather.difference_RMS_over_left.min() > 0.38


def test_failed_padded_gather_image_is_near_fixed_critical_boundary():
    images = pd.read_csv(RESULTS / "image_local_topology.csv")
    selected = images[
        (images.field == "P0654_padded_gather") & (images.image_id == "6b")
    ].iloc[0]
    assert abs(selected.fixed_full_lens_mapping_determinant) < 1e-3
    assert selected.correction_tangential_arcsec > 0.33


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0656_transport_topology_diagnostic.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0656_transport_topology_diagnostic.py"
    )
    assert (RESULTS / "transport_topology_diagnostic.png").stat().st_size > 20000
