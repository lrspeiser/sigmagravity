#!/usr/bin/env python3
"""No-fit structural audit of the released J1402 Dinos coordinate contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r1_j1402_dinos_coordinate_replay_protocol.json"
CORRECTION_PATH = ROOT / "configs" / "r1_j1402_dinos_interface_correction_protocol.json"
REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_coordinate_audit_corrected" / "report.json"


def coordinate_roundtrip(matrix: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> float:
    ny, nx = shape
    pixels = np.asarray(
        [
            [0.0, 0.0],
            [nx - 1.0, 0.0],
            [0.0, ny - 1.0],
            [nx - 1.0, ny - 1.0],
            [(nx - 1.0) / 2.0, (ny - 1.0) / 2.0],
            [0.371 * (nx - 1.0), 0.619 * (ny - 1.0)],
        ]
    )
    angles = origin[:, None] + matrix @ pixels.T
    reconstructed = np.linalg.solve(matrix, angles - origin[:, None]).T
    return float(np.max(np.abs(reconstructed - pixels)))


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    settings_path = ROOT / protocol["released_inputs"]["settings"]
    data_directory = ROOT / protocol["released_inputs"]["data_directory"]
    chain_path = ROOT / protocol["released_inputs"]["chain"]
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    bands = settings["band"]
    mask_settings = settings["mask"]
    band_rows: list[dict] = []
    psf_rows: list[dict] = []

    for index, band in enumerate(bands):
        image_path = data_directory / f"image_SDSSJ1402+6321_{band}.h5"
        psf_path = data_directory / f"psf_SDSSJ1402+6321_{band}.h5"
        with h5py.File(image_path, "r") as handle:
            image = np.asarray(handle["image_data"][()])
            exposure = np.asarray(handle["exposure_time"][()])
            matrix = np.asarray(handle["transform_pix2angle"][()], dtype=float)
            origin = np.asarray(
                [handle["ra_at_xy_0"][()], handle["dec_at_xy_0"][()]], dtype=float
            )
            ra_shift = float(handle["ra_shift"][()])
            dec_shift = float(handle["dec_shift"][()])
            background_rms = float(handle["background_rms"][()])
        settings_matrix = np.asarray(mask_settings["transform_matrix"][index], dtype=float)
        settings_ra = float(mask_settings["ra_at_xy_0"][index])
        determinant = float(np.linalg.det(matrix))
        pixel_scale = float(np.sqrt(abs(determinant)))
        ny, nx = image.shape
        center_pixel = np.asarray([(nx - 1.0) / 2.0, (ny - 1.0) / 2.0])
        center_angle = origin + matrix @ center_pixel
        band_rows.append(
            {
                "band": band,
                "image_shape": [int(value) for value in image.shape],
                "declared_size": int(mask_settings["size"][index]),
                "image_shape_matches_settings": list(image.shape)
                == [int(mask_settings["size"][index])] * 2,
                "transform_matches_settings_bitwise": bool(
                    np.array_equal(matrix, settings_matrix)
                ),
                "ra_origin_matches_settings_bitwise": bool(
                    float(origin[0]) == settings_ra
                ),
                "origin_arcsec": origin.tolist(),
                "transform_pix2angle": matrix.tolist(),
                "determinant": determinant,
                "condition_number": float(np.linalg.cond(matrix)),
                "pixel_scale_arcsec": pixel_scale,
                "roundtrip_maximum_pixel_error": coordinate_roundtrip(
                    matrix, origin, image.shape
                ),
                "raw_array_center_angle_arcsec": center_angle.tolist(),
                "ra_shift": ra_shift,
                "dec_shift": dec_shift,
                "settings_centroid_offset": mask_settings["centroid_offset"][index],
                "stored_shift_is_negative_settings_offset": bool(
                    np.array_equal(
                        np.asarray([ra_shift, dec_shift]),
                        -np.asarray(mask_settings["centroid_offset"][index], dtype=float),
                    )
                ),
                "background_rms": background_rms,
                "image_all_finite": bool(np.isfinite(image).all()),
                "exposure_all_finite": bool(np.isfinite(exposure).all()),
                "exposure_positive_fraction": float(np.mean(exposure > 0)),
                "background_rms_positive_finite": bool(
                    np.isfinite(background_rms) and background_rms > 0
                ),
                "image_minimum": float(np.min(image)),
                "image_maximum": float(np.max(image)),
            }
        )
        with h5py.File(psf_path, "r") as handle:
            psf = np.asarray(handle["kernel_point_source"][()], dtype=float)
        psf_rows.append(
            {
                "band": band,
                "shape": [int(value) for value in psf.shape],
                "finite": bool(np.isfinite(psf).all()),
                "nonnegative": bool(np.all(psf >= 0)),
                "sum": float(np.sum(psf)),
                "normalized_to_one": bool(abs(float(np.sum(psf)) - 1.0) <= 1e-6),
                "odd_square_61": list(psf.shape) == [61, 61],
            }
        )

    released_mask_path = data_directory / "mask_SDSSJ1402+6321_F435W.h5"
    with h5py.File(released_mask_path, "r") as handle:
        released_mask = np.asarray(handle["mask"][()])
    configured_masks = [np.asarray(item) for item in mask_settings["custom_mask"]]
    configured_mask = configured_masks[0]
    mask_report = {
        "released_shape": [int(value) for value in released_mask.shape],
        "configured_shape": [int(value) for value in configured_mask.shape],
        "all_configured_shapes": [
            [int(value) for value in item.shape] for item in configured_masks
        ],
        "bitwise_equal": bool(np.array_equal(released_mask, configured_mask)),
        "bitwise_equal_to_complement": bool(
            np.array_equal(
                released_mask,
                1 - configured_mask.reshape(released_mask.shape),
            )
        ),
        "complement_disagreeing_pixels": int(
            np.count_nonzero(
                released_mask
                != 1 - configured_mask.reshape(released_mask.shape)
            )
        ),
        "released_unique_values": sorted(float(value) for value in np.unique(released_mask)),
        "retained_fraction_if_one_is_retained": float(np.mean(released_mask == 1)),
    }

    with h5py.File(chain_path, "r") as handle:
        group = handle[protocol["chain_contract"]["group"]]
        samples = group["samples"]
        likelihood = group["log_likelihood"]
        parameter_names = [value.decode("utf-8") for value in group["param_list"][()]]
        chunk_rows = 50_000
        samples_finite = True
        likelihood_finite = True
        maximum_log_likelihood = -np.inf
        best_index = -1
        for start in range(0, samples.shape[0], chunk_rows):
            stop = min(start + chunk_rows, samples.shape[0])
            sample_chunk = samples[start:stop]
            likelihood_chunk = likelihood[start:stop]
            samples_finite &= bool(np.isfinite(sample_chunk).all())
            likelihood_finite &= bool(np.isfinite(likelihood_chunk).all())
            local = int(np.argmax(likelihood_chunk))
            if float(likelihood_chunk[local]) > maximum_log_likelihood:
                maximum_log_likelihood = float(likelihood_chunk[local])
                best_index = start + local
        best_sample = np.asarray(samples[best_index], dtype=float)
        chain_shape = [int(value) for value in samples.shape]
    chain_report = {
        "samples_shape": chain_shape,
        "parameter_count": len(parameter_names),
        "parameter_names": parameter_names,
        "all_samples_finite": samples_finite,
        "all_log_likelihoods_finite": likelihood_finite,
        "maximum_log_likelihood": maximum_log_likelihood,
        "best_sample_index": best_index,
        "best_sample": best_sample.tolist(),
        "best_sample_finite": bool(np.isfinite(best_sample).all()),
        "walker_step_identity": bool(
            chain_shape[0]
            == protocol["chain_contract"]["walkers"]
            * protocol["chain_contract"]["steps"]
        ),
    }

    checks = {
        "three_frozen_bands_present": bands == protocol["coordinate_contract"]["band_order"],
        "all_image_shapes_match_settings": all(
            item["image_shape_matches_settings"] for item in band_rows
        ),
        "all_transforms_match_settings_bitwise": all(
            item["transform_matches_settings_bitwise"] for item in band_rows
        ),
        "all_ra_origins_match_settings_bitwise": all(
            item["ra_origin_matches_settings_bitwise"] for item in band_rows
        ),
        "all_transforms_finite_invertible_and_roundtrip": all(
            np.isfinite(item["determinant"])
            and abs(item["determinant"]) > 1e-8
            and item["condition_number"] < 2.0
            and item["roundtrip_maximum_pixel_error"] <= 1e-10
            for item in band_rows
        ),
        "all_stored_shifts_are_negative_declared_offsets": all(
            item["stored_shift_is_negative_settings_offset"] for item in band_rows
        ),
        "standalone_mask_matches_documented_complement_semantics": bool(
            mask_report["bitwise_equal_to_complement"]
            and mask_report["complement_disagreeing_pixels"] == 0
        ),
        "all_images_exposures_and_noise_are_usable": all(
            item["image_all_finite"]
            and item["exposure_all_finite"]
            and item["exposure_positive_fraction"] == 1.0
            and item["background_rms_positive_finite"]
            for item in band_rows
        ),
        "all_raw_PSFs_are_valid_for_locked_automatic_normalization": all(
            item["finite"]
            and item["nonnegative"]
            and item["odd_square_61"]
            and np.isfinite(item["sum"])
            and item["sum"] > 0
            for item in psf_rows
        ),
        "chain_shape_matches_frozen_contract": chain_report["samples_shape"]
        == protocol["chain_contract"]["samples_shape"],
        "chain_is_finite_and_walker_shape_closes": bool(
            chain_report["all_samples_finite"]
            and chain_report["all_log_likelihoods_finite"]
            and chain_report["best_sample_finite"]
            and chain_report["walker_step_identity"]
        ),
        "chain_maximum_matches_frozen_contract": bool(
            chain_report["best_sample_index"]
            == protocol["chain_contract"]["best_sample_index"]
            and chain_report["maximum_log_likelihood"]
            == protocol["chain_contract"]["stored_max_log_likelihood"]
        ),
        "external_numpy_pickle_was_not_loaded": True,
    }
    gate_pass = all(checks.values())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol_version"],
        "correction_protocol": correction["protocol_version"],
        "initial_failed_report": correction["initial_failed_report"],
        "corrections": {
            "mask": correction["mask_correction"],
            "PSF": correction["PSF_correction"],
            "unchanged_scientific_gates": correction["unchanged_scientific_gates"],
        },
        "fit_performed": False,
        "forward_model_evaluated": False,
        "band_coordinates": band_rows,
        "PSFs": psf_rows,
        "mask": mask_report,
        "chain": chain_report,
        "scalar_pixel_size_discrepancy": {
            "settings_scalar_arcsec": float(settings["pixel_size"]),
            "operative_matrix_scales_arcsec": [
                item["pixel_scale_arcsec"] for item in band_rows
            ],
            "resolution": "The released Dolphin v0.0.1 data interface passes the complete HDF5 kwargs_data dictionary to lenstronomy. The explicit per-band transform matrices are therefore operative; the scalar setting is retained as metadata and is not substituted.",
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "decision": "corrected_structural_coordinate_gate_pass_authorize_locked_environment_install"
        if gate_pass
        else "stop_J1402_structural_coordinate_failure",
        "authorization": {
            "install_locked_Dolphin_environment": gate_pass,
            "evaluate_forward_model": False,
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
