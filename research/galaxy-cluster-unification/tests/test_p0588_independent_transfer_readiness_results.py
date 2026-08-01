import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report():
    return json.loads(
        (ROOT / "results/p0588_independent_transfer_readiness/report.json").read_text(
            encoding="utf-8"
        )
    )


def test_p0588_protocol_is_frozen_and_hashed():
    report = load_report()
    protocol = ROOT / report["protocol"]["path"]
    assert report["protocol"]["sha256"] == sha256(protocol)
    assert report["status"] == "complete"


def test_p0588_does_not_misclassify_derived_lens_products_as_raw():
    evidence = pd.read_csv(
        ROOT / "results/p0588_independent_transfer_readiness/evidence_inventory.csv"
    ).set_index("product")
    assert evidence.loc[
        "CCCP/MENeaCS weak-lensing masses", "classification"
    ] == "model_dependent_lens_summary"
    assert evidence.loc[
        "RELICS radial kappa profiles and covariance", "classification"
    ] == "model_dependent_lens_reconstruction"
    assert load_report()["local_inventory"][
        "local_raw_weak_shear_or_magnification_likelihoods"
    ] == 0


def test_p0588_macs0416_fresh_inputs_are_real_and_counted():
    prep = load_report()["macs0416_preparation"]
    assert prep["selection_used_formula_residual"] is False
    assert prep["buffalo_catalog_rows"] == 18801
    assert prep["valid_spectroscopic_redshifts"] == 660
    assert prep["spectroscopic_member_candidates"] == 247
    assert prep["multiple_image_positions"] == 237
    assert prep["source_families_published"] == 88
    assert prep["chandra_derived_gas_components"] == 4


def test_p0588_member_catalog_obeys_frozen_selection():
    members = pd.read_csv(
        ROOT / "data/derived/p0588_macs0416_spectroscopic_member_candidates.csv"
    )
    assert len(members) == 247
    assert members.zspec.between(0.376, 0.416, inclusive="both").all()
    assert (members.zspec_quality >= 3).all()
    assert (members.f160w_flux_catalog_units > 0.0).all()


def test_p0588_keeps_strict_claim_closed_but_selects_next_target():
    report = load_report()
    assert report["local_inventory"]["strict_fresh_strong_lens_ready_systems"] == 0
    assert report["decision"]["next_target"] == "MACS0416"
    assert report["decision"]["strict_validation_authorized"] is False
    assert report["decision"]["descriptive_fresh_transfer_after_baryon_map"] is True
