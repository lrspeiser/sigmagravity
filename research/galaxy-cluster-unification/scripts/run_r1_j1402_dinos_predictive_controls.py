#!/usr/bin/env python3
"""Run frozen J1402 sector holdouts and coordinate-corruption controls."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from dolphin.processor.config import ModelConfig
from dolphin.processor.data import ImageData as DolphinImageData
from dolphin.processor.data import PSFData as DolphinPSFData
from lenstronomy.Data.imaging_data import ImageData as LenstronomyImageData
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from scipy.signal import fftconvolve


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r1_j1402_dinos_predictive_controls_protocol.json"
CORRECTION_PATH = (
    ROOT / "configs" / "r1_j1402_dinos_predictive_controls_implementation_correction.json"
)
REPLAY_REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_replay" / "report.json"
COORDINATE_PROTOCOL_PATH = ROOT / "configs" / "r1_j1402_dinos_coordinate_replay_protocol.json"
REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_predictive_controls" / "report.json"
PRODUCT_PATH = ROOT / "data" / "derived" / "r1_j1402_dinos_predictive_controls.npz"
COORDINATE_FIELDS = [
    "ra_at_xy_0",
    "dec_at_xy_0",
    "transform_pix2angle",
    "ra_shift",
    "dec_shift",
]


def load_best_sample(coordinate_protocol: dict) -> tuple[np.ndarray, list[str]]:
    chain_path = ROOT / coordinate_protocol["released_inputs"]["chain"]
    group_name = coordinate_protocol["chain_contract"]["group"]
    best_index = coordinate_protocol["chain_contract"]["best_sample_index"]
    with h5py.File(chain_path, "r") as handle:
        group = handle[group_name]
        sample = np.asarray(group["samples"][best_index], dtype=float)
        names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in group["param_list"][()]
        ]
    return sample, names


def released_data_rows(config: ModelConfig, data_directory: Path) -> list[list[dict]]:
    rows = []
    psf_factor = config.get_psf_supersampled_factor()
    for band, numerics in zip(config.settings["band"], config.get_kwargs_numerics()):
        kwargs_data = DolphinImageData(
            str(data_directory / f"image_SDSSJ1402+6321_{band}.h5")
        ).kwargs_data
        kwargs_psf = DolphinPSFData(
            str(data_directory / f"psf_SDSSJ1402+6321_{band}.h5")
        ).kwargs_psf
        kwargs_psf["point_source_supersampling_factor"] = psf_factor
        rows.append([kwargs_data, kwargs_psf, numerics])
    return rows


def corrupt_data_rows(rows: list[list[dict]], variant: str) -> list[list[dict]]:
    result = copy.deepcopy(rows)
    if variant == "baseline":
        return result
    if variant == "scalar_0p04":
        for kwargs_data, _, _ in result:
            matrix = np.asarray(kwargs_data["transform_pix2angle"], dtype=float)
            current_scale = np.sqrt(abs(np.linalg.det(matrix)))
            kwargs_data["transform_pix2angle"] = matrix * (0.04 / current_scale)
        return result
    if variant == "zero_shifts":
        for kwargs_data, _, _ in result:
            kwargs_data["ra_shift"] = 0.0
            kwargs_data["dec_shift"] = 0.0
        return result
    if variant == "swap_F555W_F814W_coordinate_maps":
        left = {field: copy.deepcopy(rows[1][0][field]) for field in COORDINATE_FIELDS}
        right = {field: copy.deepcopy(rows[2][0][field]) for field in COORDINATE_FIELDS}
        for field in COORDINATE_FIELDS:
            result[1][0][field] = right[field]
            result[2][0][field] = left[field]
        return result
    raise ValueError(f"unknown coordinate variant {variant}")


def make_sector_masks(rows: list[list[dict]], operative_masks, sector_count: int):
    sector_masks: list[list[np.ndarray]] = [[] for _ in range(sector_count)]
    sector_indices: list[np.ndarray] = []
    for (kwargs_data, _, _), operative in zip(rows, operative_masks):
        data = LenstronomyImageData(**kwargs_data)
        x, y = data.pixel_coordinates
        angle = np.mod(np.arctan2(y, x), 2 * np.pi)
        index = np.floor(angle / (2 * np.pi / sector_count)).astype(int)
        index = np.minimum(index, sector_count - 1)
        operative = np.asarray(operative, dtype=bool)
        sector_indices.append(index)
        for sector in range(sector_count):
            sector_masks[sector].append(operative & (index == sector))
    return sector_masks, sector_indices


def pixel_coordinate_maps(rows: list[list[dict]]) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for kwargs_data, _, _ in rows:
        x, y = LenstronomyImageData(**kwargs_data).pixel_coordinates
        result.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    return result


def maximum_coordinate_difference(
    baseline: list[tuple[np.ndarray, np.ndarray]],
    candidate: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    return float(
        max(
            np.max(np.abs(candidate_axis - baseline_axis))
            for baseline_band, candidate_band in zip(baseline, candidate)
            for baseline_axis, candidate_axis in zip(baseline_band, candidate_band)
        )
    )


def build_sequence(config: ModelConfig, rows, training_masks) -> FittingSequence:
    kwargs_likelihood = config.get_kwargs_likelihood()
    kwargs_likelihood["image_likelihood_mask_list"] = training_masks
    return FittingSequence(
        {"multi_band_list": rows, "multi_band_type": "multi-linear"},
        config.get_kwargs_model(),
        config.get_kwargs_constraints(),
        kwargs_likelihood,
        config.get_kwargs_params(),
        mpi=False,
        verbose=False,
    )


def coherent_psf_significance(standardized, heldout_mask, kernel) -> float:
    kernel = np.asarray(kernel, dtype=float)
    kernel = kernel / np.sum(kernel)
    reverse = kernel[::-1, ::-1]
    numerator = fftconvolve(standardized * heldout_mask, reverse, mode="same")
    denominator = np.sqrt(
        np.maximum(
            fftconvolve(heldout_mask.astype(float), reverse**2, mode="same"), 0.0
        )
    )
    full_support = float(np.sqrt(np.sum(kernel**2)))
    valid = heldout_mask & (denominator >= 0.8 * full_support)
    if not np.any(valid):
        raise RuntimeError("no heldout pixel has the frozen 80-percent PSF support")
    return float(np.max(np.abs(numerator[valid] / denominator[valid])))


def score_sector(config, rows, operative_masks, heldout_masks, best_sample, names):
    training_masks = [
        np.asarray(operative, dtype=bool) & ~np.asarray(heldout, dtype=bool)
        for operative, heldout in zip(operative_masks, heldout_masks)
    ]
    sequence = build_sequence(config, rows, training_masks)
    count, reconstructed_names = sequence.param_class.num_param()
    if count != len(best_sample) or reconstructed_names != names:
        raise RuntimeError("parameter contract changed in predictive control")
    kwargs_result = sequence.param_class.args2kwargs(best_sample)
    likelihood = sequence.likelihoodModule
    im_sim = likelihood.image_likelihood.imSim
    masked_solver_models, model_errors, _, _ = im_sim.image_linear_solve(
        **kwargs_result, inv_bool=False
    )
    models = [
        image_model.image(**kwargs_result)
        for image_model in im_sim._imageModel_list
    ]
    band_rows = []
    total_chi_square = 0.0
    total_pixels = 0
    for index, (band, model, masked_solver_model, model_error, heldout) in enumerate(
        zip(
            config.settings["band"],
            models,
            masked_solver_models,
            model_errors,
            heldout_masks,
        )
    ):
        image_model = im_sim._imageModel_list[index]
        model = np.asarray(model, dtype=float)
        masked_solver_model = np.asarray(masked_solver_model, dtype=float)
        model_error = np.asarray(model_error, dtype=float)
        heldout = np.asarray(heldout, dtype=bool)
        data = np.asarray(image_model.Data.data, dtype=float)
        variance = np.asarray(image_model.Data.C_D_model(model), dtype=float) + np.abs(
            model_error
        )
        standardized = (model - data) / np.sqrt(variance)
        residual = standardized[heldout]
        chi_square = float(np.sum(residual**2))
        pixels = int(np.count_nonzero(heldout))
        coherent = coherent_psf_significance(
            standardized, heldout, image_model.PSF.kernel_point_source
        )
        total_chi_square += chi_square
        total_pixels += pixels
        band_rows.append(
            {
                "band": band,
                "heldout_pixels": pixels,
                "training_pixels": int(np.count_nonzero(training_masks[index])),
                "chi_square": chi_square,
                "reduced_chi_square": float(chi_square / pixels),
                "standardized_residual_mean": float(np.mean(residual)),
                "standardized_residual_rms": float(np.sqrt(np.mean(residual**2))),
                "maximum_PSF_matched_coherent_residual_sigma": coherent,
                "masked_linear_solver_return_heldout_absolute_max": float(
                    np.max(np.abs(masked_solver_model[heldout]))
                ),
                "complete_forward_model_heldout_absolute_max": float(
                    np.max(np.abs(model[heldout]))
                ),
                "model_and_variance_finite": bool(
                    np.isfinite(model).all()
                    and np.isfinite(variance[heldout]).all()
                    and np.all(variance[heldout] > 0)
                ),
            }
        )
    return {
        "bands": band_rows,
        "heldout_pixels": total_pixels,
        "chi_square": total_chi_square,
        "reduced_chi_square": float(total_chi_square / total_pixels),
        "maximum_PSF_matched_coherent_residual_sigma": max(
            item["maximum_PSF_matched_coherent_residual_sigma"] for item in band_rows
        ),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY_REPORT_PATH.read_text(encoding="utf-8"))
    coordinate_protocol = json.loads(
        COORDINATE_PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    if not replay["exact_replay_gate_pass"] or not replay["authorization"][
        "run_heldout_sector_and_coordinate_controls"
    ]:
        raise RuntimeError("exact replay did not authorize predictive controls")
    settings_path = ROOT / coordinate_protocol["released_inputs"]["settings"]
    data_directory = ROOT / coordinate_protocol["released_inputs"]["data_directory"]
    config = ModelConfig(str(settings_path))
    best_sample, names = load_best_sample(coordinate_protocol)
    released_rows = released_data_rows(config, data_directory)
    operative_masks = [np.asarray(item, dtype=bool) for item in config.get_masks()]
    sector_count = protocol["sector_geometry"]["count"]
    sector_masks, sector_indices = make_sector_masks(
        released_rows, operative_masks, sector_count
    )

    variants = [
        "baseline",
        "scalar_0p04",
        "zero_shifts",
        "swap_F555W_F814W_coordinate_maps",
    ]
    results = {}
    coordinate_maps = {}
    for variant in variants:
        print(f"VARIANT {variant}", flush=True)
        rows = corrupt_data_rows(released_rows, variant)
        coordinate_maps[variant] = pixel_coordinate_maps(rows)
        sectors = []
        for sector, heldout_masks in enumerate(sector_masks):
            score = score_sector(
                config,
                rows,
                operative_masks,
                heldout_masks,
                best_sample,
                names,
            )
            score["sector"] = sector
            sectors.append(score)
            print(
                f"  sector={sector} reduced_chi2={score['reduced_chi_square']:.6f} "
                f"coherent_sigma={score['maximum_PSF_matched_coherent_residual_sigma']:.6f}",
                flush=True,
            )
        results[variant] = {
            "sectors": sectors,
            "aggregate_heldout_pixels": int(
                sum(item["heldout_pixels"] for item in sectors)
            ),
            "aggregate_chi_square": float(sum(item["chi_square"] for item in sectors)),
            "aggregate_reduced_chi_square": float(
                sum(item["chi_square"] for item in sectors)
                / sum(item["heldout_pixels"] for item in sectors)
            ),
            "maximum_sector_reduced_chi_square": float(
                max(item["reduced_chi_square"] for item in sectors)
            ),
            "maximum_PSF_matched_coherent_residual_sigma": float(
                max(
                    item["maximum_PSF_matched_coherent_residual_sigma"]
                    for item in sectors
                )
            ),
        }

    baseline_coordinate_maps = coordinate_maps["baseline"]
    for variant in variants:
        results[variant]["maximum_instantiated_coordinate_difference_vs_baseline_arcsec"] = (
            maximum_coordinate_difference(
                baseline_coordinate_maps, coordinate_maps[variant]
            )
        )

    baseline_chi = results["baseline"]["aggregate_chi_square"]
    floor = protocol["coordinate_controls"][
        "minimum_worsening_total_chi_square"
    ]
    negative_controls = {}
    for variant in variants[1:]:
        delta = results[variant]["aggregate_chi_square"] - baseline_chi
        negative_controls[variant] = {
            "aggregate_chi_square": results[variant]["aggregate_chi_square"],
            "delta_vs_released_mapping": float(delta),
            "fractional_change_vs_released_mapping": float(delta / baseline_chi),
            "worsens_by_frozen_numerical_floor": bool(delta >= floor),
            "changes_instantiated_pixel_coordinates": bool(
                results[variant][
                    "maximum_instantiated_coordinate_difference_vs_baseline_arcsec"
                ]
                > 0
            ),
        }

    thresholds = protocol["predictive_metrics"]
    minimum_pixels = protocol["sector_geometry"][
        "minimum_heldout_pixels_per_band_sector"
    ]
    checks = {
        "upstream_exact_replay_gate_passed": True,
        "all_full_masked_band_reduced_chi_squares_pass": all(
            item["reduced_chi_square_per_retained_pixel"]
            <= thresholds["full_masked_reduced_chi_square_per_band_maximum"]
            for item in replay["bands"]
        ),
        "six_half_open_sky_sectors_partition_every_operative_mask": all(
            np.array_equal(
                sum(
                    np.asarray(sector_masks[s][b], dtype=int)
                    for s in range(sector_count)
                ),
                np.asarray(operative_masks[b], dtype=int),
            )
            for b in range(len(operative_masks))
        ),
        "every_band_sector_has_minimum_heldout_pixels": all(
            band["heldout_pixels"] >= minimum_pixels
            for sector in results["baseline"]["sectors"]
            for band in sector["bands"]
        ),
        "all_baseline_sector_models_and_variances_are_finite": all(
            band["model_and_variance_finite"]
            for sector in results["baseline"]["sectors"]
            for band in sector["bands"]
        ),
        "linear_solver_masked_return_is_zero_filled_on_every_heldout_sector": all(
            band["masked_linear_solver_return_heldout_absolute_max"] == 0
            for sector in results["baseline"]["sectors"]
            for band in sector["bands"]
        ),
        "complete_forward_model_is_nonzero_on_every_heldout_sector": all(
            band["complete_forward_model_heldout_absolute_max"] > 0
            for sector in results["baseline"]["sectors"]
            for band in sector["bands"]
        ),
        "every_coordinate_corruption_changes_instantiated_coordinates": all(
            item["changes_instantiated_pixel_coordinates"]
            for item in negative_controls.values()
        ),
        "maximum_six_sector_heldout_reduced_chi_square_passes": bool(
            results["baseline"]["maximum_sector_reduced_chi_square"]
            <= thresholds["maximum_six_sector_heldout_reduced_chi_square"]
        ),
        "maximum_coherent_heldout_residual_passes": bool(
            results["baseline"]["maximum_PSF_matched_coherent_residual_sigma"]
            <= thresholds["maximum_coherent_heldout_residual_sigma"]
        ),
        "every_coordinate_corruption_worsens_heldout_likelihood": all(
            item["worsens_by_frozen_numerical_floor"]
            for item in negative_controls.values()
        ),
        "nonlinear_parameters_never_optimized": True,
        "lens_response_not_computed": True,
        "external_numpy_pickle_not_loaded": True,
    }
    gate_pass = all(checks.values())
    PRODUCT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRODUCT_PATH,
        **{
            f"{config.settings['band'][band]}_sector_index": sector_indices[band]
            for band in range(len(sector_indices))
        },
        **{
            f"{config.settings['band'][band]}_operative_mask": operative_masks[band].astype(
                np.uint8
            )
            for band in range(len(operative_masks))
        },
    )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol_version"],
        "implementation_correction": {
            "path": str(CORRECTION_PATH.relative_to(ROOT)).replace("\\", "/"),
            "id": correction["correction_id"],
            "scientific_protocol_changed": correction["scientific_protocol_changed"],
        },
        "upstream_replay_report": str(REPLAY_REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "nonlinear_fit_performed": False,
        "lens_response_computed": False,
        "sector_geometry": protocol["sector_geometry"],
        "predictive_thresholds": thresholds,
        "released_and_corrupted_results": results,
        "negative_controls": negative_controls,
        "checks": checks,
        "predictive_coordinate_gate_pass": gate_pass,
        "decision": (
            "predictive_and_coordinate_controls_pass_authorize_frozen_lens_response_Jacobian"
            if gate_pass
            else "stop_J1402_lens_promotion_after_predictive_or_coordinate_control_failure"
        ),
        "outputs": {
            "sector_products": str(PRODUCT_PATH.relative_to(ROOT)).replace("\\", "/")
        },
        "authorization": {
            "compute_frozen_lens_response_Jacobian": gate_pass,
            "optimize_nonlinear_lens_model": False,
            "reduce_KCWI": False,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
            "authorize_R2": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)
    print(report["decision"], flush=True)


if __name__ == "__main__":
    main()
