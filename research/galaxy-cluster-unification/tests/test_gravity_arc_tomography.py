from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.gravity_arc_tomography import (
    combine_f160_photometry,
    photometric_membership_weights,
    read_relics_catalog,
    sinkhorn_transport,
)


ROOT = Path(__file__).resolve().parents[1]


def test_relics_parser_preserves_repeated_f160_columns():
    path = ROOT / "data/raw/relics_gravity_arc_catalogs/hlsp_relics_hst_acs-wfc3ir_rxc0949p17_multi_v1_cat.txt"
    catalog = read_relics_catalog(path)
    assert len(catalog.columns) == 94
    assert sum(
        name == "f160w_fluxnJy" or name.startswith("f160w_fluxnJy__")
        for name in catalog
    ) == 3


def test_f160_combination_takes_highest_significance_valid_measurement():
    catalog = pd.DataFrame(
        {
            "f160w_fluxnJy": [10.0, -1.0],
            "f160w_fluxnJy__2": [20.0, 5.0],
            "f160w_sig": [3.0, -2.0],
            "f160w_sig__2": [8.0, 6.0],
        }
    )
    flux, significance = combine_f160_photometry(catalog)
    assert np.allclose(flux, [20.0, 5.0])
    assert np.allclose(significance, [8.0, 6.0])


def test_membership_hard_interval_and_soft_weight_are_bounded():
    catalog = pd.DataFrame(
        {"zb": [0.3, 1.2], "zbmin": [0.2, 1.0], "zbmax": [0.4, 1.4], "odds": [0.8, 0.9]}
    )
    hard, soft = photometric_membership_weights(catalog, 0.3)
    assert hard.tolist() == [True, False]
    assert np.all((soft >= 0.0) & (soft <= 1.0))
    assert soft[0] > soft[1]


def test_sinkhorn_matches_both_marginals():
    source = np.array([0.3, 0.7])
    target = np.array([0.2, 0.5, 0.3])
    cost = np.array([[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]])
    plan = sinkhorn_transport(source, target, cost, entropy=0.5)
    assert np.allclose(plan.sum(axis=1), source, atol=1.0e-7)
    assert np.allclose(plan.sum(axis=0), target, atol=1.0e-7)
