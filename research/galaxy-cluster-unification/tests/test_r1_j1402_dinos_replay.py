import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "r1_j1402_dinos_replay" / "report.json"
PRODUCTS = ROOT / "data" / "derived" / "r1_j1402_dinos_replay_products.npz"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_replay_decision_is_derived_from_every_exact_gate() -> None:
    item = report()
    assert item["exact_replay_gate_pass"] == all(item["checks"].values())
    assert item["authorization"]["run_heldout_sector_and_coordinate_controls"] == item[
        "exact_replay_gate_pass"
    ]
    assert not item["authorization"]["compute_lens_response"]
    assert not item["authorization"]["optimize_nonlinear_model"]
    assert not item["nonlinear_fit_performed"]


def test_reconstructed_parameter_contract_is_exact() -> None:
    item = report()
    chain = item["chain"]
    assert chain["parameter_count"] == 23
    assert chain["stored_parameter_names"] == chain["reconstructed_parameter_names"]
    assert item["checks"]["reconstructed_parameter_order_matches_chain_exactly"]
    assert item["checks"]["stored_best_index_and_likelihood_match_frozen_contract"]


def test_replay_products_have_three_complete_predictive_images() -> None:
    item = report()
    assert [row["band"] for row in item["bands"]] == ["F435W", "F555W", "F814W"]
    assert sum(row["retained_pixels"] for row in item["bands"]) == item[
        "likelihood_replay"
    ]["used_pixels"]
    with np.load(PRODUCTS, allow_pickle=False) as products:
        for row in item["bands"]:
            band = row["band"]
            assert products[f"{band}_model"].shape == tuple(row["image_shape"])
            assert products[f"{band}_mask"].shape == tuple(row["image_shape"])
            assert np.isfinite(products[f"{band}_model"]).all()


def test_replay_never_authorizes_physics_or_r2() -> None:
    authorization = report()["authorization"]
    assert not authorization["infer_gravity_response"]
    assert not authorization["fit_new_force_or_action"]
    assert not authorization["authorize_R2"]
