from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_caminha_protocol_separates_observables_and_model_products() -> None:
    protocol = json.loads((ROOT / "configs/r1_clash_caminha2019_ingest_protocol.json").read_text())
    assert protocol["frozen_before_local_product_ingest"] is True
    assert len(protocol["systems"]) == 8
    assert sum(row["tian2020_target"] for row in protocol["systems"]) == 6
    assert "convergence maps" in protocol["product_separation"]["intentionally_not_downloaded"]
    assert protocol["authorization"]["infer_weyl_response"] is False


def test_caminha_public_packages_and_observable_catalogs() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_clash_caminha2019.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_clash_caminha2019/report.json").read_text())
    images = pd.read_csv(ROOT / "data/derived/r1_clash_caminha2019_image_observables.csv")
    systems = pd.read_csv(ROOT / "data/derived/r1_clash_caminha2019_system_summary.csv")
    covariances = np.load(ROOT / "data/derived/r1_clash_caminha2019_coordinate_covariances.npz")
    assert len(systems) == 8
    assert systems["tian2020_target"].sum() == 6
    assert report["published_tablea2_rows"] == 150
    assert report["systems_with_observable_catalog"] == 8
    assert report["systems_with_complete_rerunnable_lenstool_package"] == 8
    assert report["systems_with_local_model_chain"] == 8
    assert report["spectroscopic_metric_neutral_likelihood_rows"] > 0
    assert report["gates"]["package_integrity_passed"] is True
    assert report["gates"]["observable_coordinate_likelihood_acquired_for_all_eight"] is True
    assert report["tian2020_intersection"]["confirmed_local_catalog_count_after_ingest"] == 13
    assert images.loc[images["metric_neutral_likelihood_row"], "spectroscopic_redshift"].notna().all()
    assert images["model_catalog_redshift_used_as_metric_neutral_input"].eq(False).all()
    assert images["gravity_target_used"].eq(False).all()
    assert systems["chain_schema_consistent"].all()
    assert systems["model_chain_metric_dependent"].all()
    assert systems["metric_neutral_weyl_posterior_acquired"].eq(False).all()
    assert len(covariances.files) == 16
    assert report["authorization"]["reuse_lenstool_chain_as_alternative_metric_posterior"] is False
    assert report["authorization"]["fit_gravity_response"] is False
