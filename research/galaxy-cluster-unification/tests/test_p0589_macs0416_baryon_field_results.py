import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0589_residual_blindness_and_mass_conservation():
    report = json.loads((ROOT / "results/p0589_macs0416_baryon_field/report.json").read_text())
    assert report["blindness"] == {
        "kappa_pixels_read": 0,
        "dark_halo_coordinates_read": 0,
        "image_residuals_calculated": 0,
        "formula_parameters_changed": 0,
    }
    sources = pd.read_csv(ROOT / report["outputs"]["nominal_sources"])
    assert np.isclose(sources.mass_msun.sum(), report["nominal"]["total_mass_msun"], rtol=1e-12)
    assert {"member_star", "bcg_stars", "icl_nuisance", "hot_gas"}.issubset(set(sources.component))


def test_p0589_variants_are_finite_and_curl_free():
    table = pd.read_csv(ROOT / "results/p0589_macs0416_baryon_field/variant_metrics.csv")
    assert len(table) == 13
    assert np.all(np.isfinite(table.select_dtypes(include=["number"])))
    assert table.normalized_curl_rms.max() < 1e-10
    assert table.metric_minimum_eigenvalue.min() > 0.0
    nominal = table[table.variant_id == "nominal"].iloc[0]
    assert nominal.field_difference_rms_arcsec < 1e-12
    assert nominal.field_vector_cosine_similarity > 0.999999
