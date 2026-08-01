from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0641_registered_cluster_baryon_maps"


def test_four_physical_maps_close_exactly_to_external_baryon_masses():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    systems = pd.read_csv(RESULTS / "systems.csv")
    assert report["status"] == "ready"
    assert len(systems) == 4
    assert np.max(np.abs(systems["stellar_mass_recovery_fraction"] - 1.0)) <= 1.0e-10
    assert np.max(np.abs(systems["gas_mass_recovery_fraction"] - 1.0)) <= 1.0e-10


def test_maps_contain_measured_stellar_and_gas_morphology_and_uncertainties():
    systems = pd.read_csv(RESULTS / "systems.csv")
    assert (systems["selected_members"] >= 90).all()
    assert (systems["gas_stellar_centroid_offset_kpc"] > 0.0).all()
    assert systems["central_gas_stellar_cosine_overlap"].between(0.0, 1.0).all()
    for path_text in systems["map_path"]:
        with np.load(ROOT / path_text) as data:
            required = {
                "stellar_surface_density_msun_kpc2",
                "gas_surface_density_msun_kpc2",
                "baryon_surface_density_msun_kpc2",
                "baryon_surface_density_low_msun_kpc2",
                "baryon_surface_density_high_msun_kpc2",
                "gas_shape_exponent_0p4_msun_kpc2",
                "gas_shape_exponent_0p6_msun_kpc2",
            }
            assert required <= set(data.files)
            assert all(np.isfinite(data[key]).all() for key in required)
            assert np.sum(data["baryon_surface_density_msun_kpc2"]) > 0.0


def test_every_chandra_observation_contributes_to_registered_maps():
    observations = pd.read_csv(RESULTS / "chandra_reprojection.csv")
    assert len(observations) == 19
    assert (observations["covered_grid_fraction"] > 0.0).all()
    assert (observations["sampled_counts"] > 0.0).all()


def test_lensing_remains_sealed_and_gravity_has_no_object_parameters():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert not any(report["blind_state"].values())
    assert report["gates"]["blind_state_untouched"] is True
    assert report["gates"]["zero_per_cluster_gravity_parameters"] is True
