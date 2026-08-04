import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17o_halo_scale_driver_audit.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17o_halo_scale_drivers.py"
REPORT = ROOT / "results" / "sigma_v17o_halo_scale_driver_audit" / "report.json"
PREDICTIONS = ROOT / "results" / "sigma_v17o_halo_scale_driver_audit" / "predictions.csv"
CLUSTER_FITS = ROOT / "results" / "sigma_v17o_halo_scale_driver_audit" / "cluster_nfw_fits.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17o_halo_scale", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17o_is_hash_locked_and_cannot_open_holdouts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["authorization"]["untouched_holdout_opened"] is False
    assert config["authorization"]["empirical_theory_fit_authorized"] is False
    assert config["parent"]["sha256"] == _sha256(ROOT / config["parent"]["protocol"])
    assert config["parent"]["report_sha256"] == _sha256(ROOT / config["parent"]["report"])
    for entry in config["inputs"].values():
        assert entry["sha256"] == _sha256(ROOT / entry["path"])


def test_nfw_cluster_fitter_recovers_a_manufactured_profile() -> None:
    runner = _load_runner()
    radius = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 700.0])
    acceleration, _, expected_scale = runner.nfw_profile(radius, 15.1, 4.3, 0.35)
    block = pd.DataFrame(
        {
            "radius_kpc": radius,
            "log_g_total": np.log10(acceleration),
            "err_log_g_total": np.full(len(radius), 0.02),
        }
    )

    fitted = runner.fit_cluster_nfw(block, 0.35)

    assert fitted["fit_success"] is True
    assert fitted["fit_at_boundary"] is False
    assert fitted["m200_msun"] == pytest.approx(10**15.1, rel=1e-5)
    assert fitted["concentration"] == pytest.approx(4.3, rel=1e-5)
    assert fitted["halo_scale_kpc"] == pytest.approx(expected_scale, rel=1e-5)


def test_galaxy_half_mass_proxy_is_monotonic_and_finite() -> None:
    runner = _load_runner()
    row = pd.Series(
        {
            "catalog__disk_mass_solar": 5e10,
            "catalog__bulge_mass_solar": 1e10,
            "catalog__gas_mass_solar": 1e10,
            "catalog__disk_scale_kpc": 3.0,
            "catalog__effective_radius_kpc": 5.0,
            "catalog__HI_radius_kpc": 20.0,
            "catalog__bulge_scale_fit_kpc": 0.7,
        }
    )

    radius = runner.galaxy_half_mass_radius(row)

    assert 0.5 < radius < 20.0


def test_mass_extent_bridge_implements_the_preregistered_equation() -> None:
    runner = _load_runner()
    frame = pd.DataFrame(
        {
            "domain": ["galaxy", "galaxy", "cluster", "cluster"],
            "log_mond_radius": [0.8, 1.2, 2.4, 2.8],
            "log_baryonic_radius": [0.2, 0.5, 1.6, 1.9],
        }
    )
    expected_intercept = 0.31
    expected_mass_weight = 0.67
    frame["log_halo_scale"] = (
        expected_intercept
        + expected_mass_weight * frame.log_mond_radius
        + (1.0 - expected_mass_weight) * frame.log_baryonic_radius
    )

    fitted = runner.fit_model(frame, "mass_extent_bridge_diagnostic")
    predicted = runner.predict(frame, "mass_extent_bridge_diagnostic", fitted)

    assert fitted["beta"][0] == pytest.approx(expected_intercept, abs=1e-12)
    assert fitted["beta"][1] == pytest.approx(expected_mass_weight, abs=1e-12)
    assert predicted == pytest.approx(frame.log_halo_scale.to_numpy(), abs=1e-12)


def test_v17o_report_has_cross_domain_coverage_and_no_type_switch() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTIONS)
    cluster_fits = pd.read_csv(CLUSTER_FITS)

    assert report["status"] == "completed_halo_scale_driver_audit"
    assert report["holdout_opened"] is False
    assert report["coverage"]["galaxies"] >= 100
    assert report["coverage"]["clusters"] >= 15
    assert report["coverage"]["cluster_nfw_fit_max_rmse_dex"] <= 0.05
    assert set(predictions.domain) == {"galaxy", "cluster"}
    assert predictions.system.nunique() == len(predictions)
    assert len(cluster_fits) >= 15
    assert {
        "m200_msun",
        "concentration",
        "r200_kpc",
        "halo_scale_kpc",
        "fit_rmse_dex",
    }.issubset(cluster_fits.columns)
    for model in report["models"].values():
        assert set(model["out_of_fold"]["domains"]) == {"galaxy", "cluster"}
        assert set(model["within_domain_parameters"]) == {"galaxy", "cluster"}
        assert set(model["domain_transfer"]) == {
            "galaxy_to_cluster",
            "cluster_to_galaxy",
        }
        assert all(
            "source_parameters" in transfer for transfer in model["domain_transfer"].values()
        )
    sensitivity = report["target_cut_sensitivity"]
    assert sensitivity["strict_systems"] == len(predictions)
    assert sensitivity["relaxed_systems"] >= sensitivity["strict_systems"]
    assert set(sensitivity["models"]) == set(report["models"])


def test_v17o_outcome_does_not_claim_a_physical_halo() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["selection"]["outcome"] in {
        "select_MOND_acceleration_radius_scale_mechanism",
        "select_CRG_density_radius_scale_mechanism",
        "select_AeST_cutoff_radius_scale_mechanism",
        "select_baryonic_extent_scale_mechanism",
        "select_fixed_geometric_bridge_scale_mechanism",
        "select_continuous_mass_extent_field_invariant",
        "halo_scale_not_identifiable_from_current_diagnostic_products",
    }
    assert any("not raw observations" in statement for statement in report["claim_boundary"])
