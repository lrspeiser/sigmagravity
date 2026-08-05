from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19x4_gas_state_posterior.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19x4_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_config() -> dict:
    return {
        "posterior": {
            "draws": 8,
            "seed": 19,
            "line_of_sight_depth_factor": {"minimum": 0.5, "maximum": 2.0},
            "temperature_bounds_keV": [1.0, 30.0],
            "normalization_bounds": [1e-10, 1.0],
        },
        "physical_constants_and_composition": {
            "electron_to_hydrogen_ratio": 1.2,
            "mean_mass_per_electron_proton_masses": 1.17,
            "mean_particle_mass_proton_masses": 0.61,
            "adiabatic_index": 5.0 / 3.0,
        },
        "geometry": {
            "output_pixel_arcsec": 1.0,
            "clusters": {
                "TEST": {
                    "redshift": 0.2,
                    "kpc_per_arcsec": 4.0,
                    "seed_offset": 0,
                }
            },
        },
        "common_grid": {
            "half_width_kpc": 8.0,
            "spacing_kpc": 4.0,
            "smoothing_fwhm_kpc": [4.0, 8.0],
        },
    }


def passing_region(bin_id: int, ordered: bool = True) -> dict:
    temperature = (
        {"lower_keV": 6.0, "upper_keV": 10.0}
        if ordered
        else {"lower_keV": None, "upper_keV": None}
    )
    normalization = (
        {"lower": 8e-4, "upper": 1.4e-3}
        if ordered
        else {"lower": None, "upper": None}
    )
    return {
        "cluster": "TEST",
        "bin_id": bin_id,
        "fit": {
            "parameters": {
                "temperature_keV": 8.0,
                "normalization": 1e-3,
            },
            "temperature_confidence_68_percent": temperature,
            "normalization_confidence_68_percent": normalization,
            "gates": {"all_passed": ordered},
        },
    }


def test_cluster_branch_retains_failed_profiles_with_full_bounds() -> None:
    runner = load_runner()
    arrays, summary = runner.cluster_branch(
        fixture_config(),
        "TEST",
        [passing_region(0), passing_region(2, ordered=False)],
        {0: {"pixels": 10.0}, 2: {"pixels": 20.0}},
        0.0,
    )
    assert arrays["temperature_keV"].shape == (2, 8)
    assert arrays["gas_surface_density_msun_kpc2"].shape == (2, 8)
    assert np.all(arrays["gas_surface_density_msun_kpc2"] > 0.0)
    assert summary["temperature_sampling_modes"] == {
        "asymmetric_log_profile": 1,
        "full_frozen_log_bound_fallback": 1,
    }
    assert summary["normalization_sampling_modes"] == {
        "asymmetric_log_profile": 1,
        "full_frozen_log_bound_fallback": 1,
    }
    assert summary["individual_quality_passes"] == 1


def test_common_grid_products_share_axes_and_conserve_surface_mass() -> None:
    runner = load_runner()
    config = fixture_config()
    arrays, _ = runner.cluster_branch(
        config,
        "TEST",
        [passing_region(0), passing_region(1)],
        {0: {"pixels": 10.0}, 1: {"pixels": 10.0}},
        0.0,
    )
    binmap = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 0, 0, 1, -1],
            [-1, 0, 0, 1, -1],
            [-1, 0, 0, 1, -1],
            [-1, -1, -1, -1, -1],
        ]
    )
    maps, summary = runner.build_common_maps(
        config,
        "TEST",
        arrays,
        binmap,
        {"logicalx": 3.0, "logicaly": 3.0},
    )
    assert maps["bin_id"].shape == (5, 5)
    assert summary["represented_region_ids"] == 2
    assert summary["maximum_surface_density_smoothing_mass_relative_error"] < 1e-12
    assert "gas_surface_density_msun_kpc2_median_8kpc" in maps
