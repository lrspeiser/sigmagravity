#!/usr/bin/env python3
"""Audit the pinned historical Dolphin environment without evaluating a lens model."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import h5py
import lenstronomy
import numpy as np
import scipy
import yaml
from dolphin.processor.config import ModelConfig
from dolphin.processor.data import ImageData as DolphinImageData
from dolphin.processor.data import PSFData as DolphinPSFData
from lenstronomy.Data.imaging_data import ImageData as LenstronomyImageData
from lenstronomy.Data.psf import PSF


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r1_j1402_dinos_coordinate_replay_protocol.json"
CORRECTION_PATH = ROOT / "configs" / "r1_j1402_dinos_interface_correction_protocol.json"
SOURCE_MANIFEST_PATH = ROOT / "data" / "raw" / "r1_j1402" / "software" / "source_manifest.json"
PIP_FREEZE_PATH = ROOT / "data" / "derived" / "r1_j1402_dinos_environment_pip_freeze.txt"
INITIAL_REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_environment" / "report.json"
REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_environment_corrected" / "report.json"


def dist_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    io_directory = ROOT / "data/raw/r1_j1402/dinos_repo/2_dolphin_modelling"
    config = ModelConfig(str(io_directory / "settings/SDSSJ1402+6321_config.yml"))
    settings = yaml.safe_load(
        (io_directory / "settings/SDSSJ1402+6321_config.yml").read_text(
            encoding="utf-8"
        )
    )
    operative_masks = config.get_masks()
    kwargs_numerics = config.get_kwargs_numerics()
    psf_supersampling_factor = config.get_psf_supersampled_factor()
    band_rows: list[dict] = []

    for index, band in enumerate(settings["band"]):
        image_path = io_directory / f"data/SDSSJ1402+6321/image_SDSSJ1402+6321_{band}.h5"
        psf_path = io_directory / f"data/SDSSJ1402+6321/psf_SDSSJ1402+6321_{band}.h5"
        kwargs_data = DolphinImageData(str(image_path)).kwargs_data
        kwargs_psf = DolphinPSFData(str(psf_path)).kwargs_psf
        kwargs_psf["point_source_supersampling_factor"] = psf_supersampling_factor
        with h5py.File(image_path, "r") as handle:
            expected_keys = sorted(handle.keys())
            exact_dictionary = all(
                np.array_equal(kwargs_data[key], handle[key][()]) for key in handle.keys()
            )
        image_instance = LenstronomyImageData(**kwargs_data)
        raw_psf = np.asarray(kwargs_psf["kernel_point_source"], dtype=float)
        psf_instance = PSF(**kwargs_psf)
        operative_psf = np.asarray(psf_instance.kernel_point_source, dtype=float)
        mask = np.asarray(operative_masks[index])
        band_rows.append(
            {
                "band": band,
                "kwargs_data_keys": sorted(kwargs_data),
                "expected_HDF5_keys": expected_keys,
                "Dolphin_passes_HDF5_dictionary_exactly": bool(exact_dictionary),
                "lenstronomy_image_instantiated": image_instance is not None,
                "image_shape": list(np.asarray(kwargs_data["image_data"]).shape),
                "raw_PSF_sum": float(np.sum(raw_psf)),
                "operative_PSF_sum": float(np.sum(operative_psf)),
                "operative_PSF_shape": list(operative_psf.shape),
                "operative_PSF_normalized_to_1e_12": bool(
                    abs(float(np.sum(operative_psf)) - 1.0) <= 1e-12
                ),
                "operative_PSF_normalized_to_float32_tolerance": bool(
                    abs(float(np.sum(operative_psf)) - 1.0)
                    <= correction["PSF_float32_normalization_correction"][
                        "corrected_maximum_absolute_sum_error"
                    ]
                ),
                "operative_mask_shape": list(mask.shape),
                "operative_mask_finite_binary": bool(
                    np.isfinite(mask).all() and set(np.unique(mask)).issubset({0, 1})
                ),
                "operative_mask_retained_pixels": int(np.sum(mask == 1)),
                "kwargs_numerics": kwargs_numerics[index],
            }
        )

    observed_versions = {
        "python": platform.python_version(),
        "dolphin_distribution": dist_version("dolphin"),
        "lenstronomy": lenstronomy.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": dist_version("h5py"),
        "PyYAML": dist_version("PyYAML"),
        "emcee": dist_version("emcee"),
        "schwimmbad": dist_version("schwimmbad"),
        "corner": dist_version("corner"),
        "matplotlib": dist_version("matplotlib"),
        "numba": dist_version("numba"),
        "llvmlite": dist_version("llvmlite"),
        "astropy": dist_version("astropy"),
    }
    critical = protocol["software_lock"]["critical_packages"]
    checks = {
        "python_3_10_environment": observed_versions["python"].startswith("3.10."),
        "Dolphin_source_commit_and_requirements_blob_locked": bool(
            source["commit"] == protocol["software_lock"]["Dolphin"]["commit"]
            and source["requirements_git_blob_sha1"]
            == protocol["software_lock"]["Dolphin"]["expected_requirements_blob_sha1"]
        ),
        "critical_package_versions_match": all(
            observed_versions[name] == version for name, version in critical.items()
        ),
        "historical_numba_compatibility_pair_matches": bool(
            observed_versions["numba"] == "0.58.1"
            and observed_versions["llvmlite"] == "0.41.1"
        ),
        "historical_astropy_compatibility_pin_matches": observed_versions[
            "astropy"
        ]
        == "5.3.4",
        "Dolphin_passes_all_three_HDF5_dictionaries_exactly": all(
            item["Dolphin_passes_HDF5_dictionary_exactly"] for item in band_rows
        ),
        "locked_lenstronomy_instantiates_all_three_images": all(
            item["lenstronomy_image_instantiated"] for item in band_rows
        ),
        "locked_lenstronomy_normalizes_all_three_PSFs_within_frozen_float32_tolerance": all(
            item["operative_PSF_normalized_to_float32_tolerance"]
            for item in band_rows
        ),
        "locked_Dolphin_constructs_three_binary_likelihood_masks": all(
            item["operative_mask_finite_binary"]
            and item["operative_mask_retained_pixels"] >= 100
            for item in band_rows
        ),
        "external_numpy_pickle_not_loaded": True,
        "forward_model_not_evaluated": True,
    }
    gate_pass = all(checks.values())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol_version"],
        "correction_protocol": correction["protocol_version"],
        "initial_failed_environment_report": str(
            INITIAL_REPORT_PATH.relative_to(ROOT)
        ).replace("\\", "/"),
        "source_manifest": str(SOURCE_MANIFEST_PATH.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "pip_freeze": str(PIP_FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "versions": observed_versions,
        "bands": band_rows,
        "checks": checks,
        "gate_pass": gate_pass,
        "decision": "environment_interface_gate_pass_authorize_stored_chain_replay"
        if gate_pass
        else "stop_J1402_environment_interface_failure",
        "authorization": {
            "evaluate_only_stored_chain_coordinates": gate_pass,
            "optimize_nonlinear_model": False,
            "compute_lens_response": False,
            "reduce_KCWI": False,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "authorize_R2": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(report["decision"])


if __name__ == "__main__":
    main()
