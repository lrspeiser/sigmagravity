#!/usr/bin/env python3
"""Replay the released J1402 Dinos maximum-likelihood sample without refitting."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from dolphin.processor.config import ModelConfig
from dolphin.processor.data import ImageData as DolphinImageData
from dolphin.processor.data import PSFData as DolphinPSFData
from lenstronomy.Workflow.fitting_sequence import FittingSequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r1_j1402_dinos_coordinate_replay_protocol.json"
ENVIRONMENT_REPORT_PATH = (
    ROOT / "results" / "r1_j1402_dinos_fastell_environment" / "report.json"
)
REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_replay" / "report.json"
PRODUCT_PATH = ROOT / "data" / "derived" / "r1_j1402_dinos_replay_products.npz"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def build_fitting_sequence(config: ModelConfig, data_directory: Path) -> FittingSequence:
    """Reproduce Processor.get_kwargs_data_joint without its absent lens_list file."""
    kwargs_numerics = config.get_kwargs_numerics()
    psf_supersampling_factor = config.get_psf_supersampled_factor()
    multi_band_list = []
    for band, kwargs_num in zip(config.settings["band"], kwargs_numerics):
        image_path = data_directory / f"image_SDSSJ1402+6321_{band}.h5"
        psf_path = data_directory / f"psf_SDSSJ1402+6321_{band}.h5"
        kwargs_data = DolphinImageData(str(image_path)).kwargs_data
        kwargs_psf = DolphinPSFData(str(psf_path)).kwargs_psf
        kwargs_psf["point_source_supersampling_factor"] = psf_supersampling_factor
        multi_band_list.append([kwargs_data, kwargs_psf, kwargs_num])
    kwargs_data_joint = {
        "multi_band_list": multi_band_list,
        "multi_band_type": "multi-linear",
    }
    return FittingSequence(
        kwargs_data_joint,
        config.get_kwargs_model(),
        config.get_kwargs_constraints(),
        config.get_kwargs_likelihood(),
        config.get_kwargs_params(),
        mpi=False,
        verbose=False,
    )


def load_best_sample(protocol: dict, chain_path: Path) -> tuple[np.ndarray, float, int, list[str]]:
    group_name = protocol["chain_contract"]["group"]
    expected_index = int(protocol["chain_contract"]["best_sample_index"])
    with h5py.File(chain_path, "r") as handle:
        group = handle[group_name]
        sample = np.asarray(group["samples"][expected_index], dtype=float)
        stored_log_likelihood = float(group["log_likelihood"][expected_index])
        parameter_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in group["param_list"][()]
        ]
    return sample, stored_log_likelihood, expected_index, parameter_names


def standardized_residual_products(im_sim, models, model_error_maps, bands):
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for index, (band, model, model_error) in enumerate(
        zip(bands, models, model_error_maps)
    ):
        image_model = im_sim._imageModel_list[index]
        data = np.asarray(image_model.Data.data, dtype=float)
        model = np.asarray(model, dtype=float)
        model_error = np.asarray(model_error, dtype=float)
        mask = np.asarray(image_model.likelihood_mask, dtype=bool)
        variance = np.asarray(image_model.Data.C_D_model(model), dtype=float) + np.abs(
            model_error
        )
        standardized = (model - data) / np.sqrt(variance)
        retained = standardized[mask]
        chi_square = float(np.sum(retained**2))
        pixel_log_likelihood = float(
            image_model.Data.log_likelihood(model, mask, model_error)
        )
        rows.append(
            {
                "band": band,
                "image_shape": [int(value) for value in model.shape],
                "retained_pixels": int(np.count_nonzero(mask)),
                "model_all_finite": bool(np.isfinite(model).all()),
                "variance_positive_finite_on_mask": bool(
                    np.isfinite(variance[mask]).all() and np.all(variance[mask] > 0)
                ),
                "pixel_log_likelihood_without_positive_flux_penalty": pixel_log_likelihood,
                "chi_square": chi_square,
                "reduced_chi_square_per_retained_pixel": float(
                    chi_square / np.count_nonzero(mask)
                ),
                "standardized_residual_mean": float(np.mean(retained)),
                "standardized_residual_rms": float(np.sqrt(np.mean(retained**2))),
                "standardized_residual_maximum_absolute": float(
                    np.max(np.abs(retained))
                ),
            }
        )
        arrays[f"{band}_data"] = data
        arrays[f"{band}_model"] = model
        arrays[f"{band}_model_error"] = model_error
        arrays[f"{band}_variance"] = variance
        arrays[f"{band}_mask"] = mask.astype(np.uint8)
        arrays[f"{band}_standardized_residual"] = standardized
    return rows, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--products", type=Path, default=PRODUCT_PATH)
    parser.add_argument(
        "--environment-report", type=Path, default=ENVIRONMENT_REPORT_PATH
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    environment_report_path = args.environment_report.resolve()
    environment = json.loads(environment_report_path.read_text(encoding="utf-8"))
    if not environment.get("gate_pass", False):
        raise RuntimeError("The corrected historical-environment interface gate did not pass")
    if not environment["authorization"]["evaluate_only_stored_chain_coordinates"]:
        raise RuntimeError("The upstream report does not authorize stored-chain replay")

    settings_path = ROOT / protocol["released_inputs"]["settings"]
    data_directory = ROOT / protocol["released_inputs"]["data_directory"]
    chain_path = ROOT / protocol["released_inputs"]["chain"]
    config = ModelConfig(str(settings_path))
    best_sample, stored_log_likelihood, best_index, stored_names = load_best_sample(
        protocol, chain_path
    )

    fitting_sequence = build_fitting_sequence(config, data_directory)
    parameter_count, reconstructed_names = fitting_sequence.param_class.num_param()
    names_match = reconstructed_names == stored_names
    count_matches = parameter_count == len(best_sample) == len(stored_names)

    likelihood = fitting_sequence.likelihoodModule
    replayed_log_likelihood = float(likelihood.logL(best_sample, verbose=True))
    kwargs_result = fitting_sequence.param_class.args2kwargs(best_sample)
    im_sim = likelihood.image_likelihood.imSim
    models, model_error_maps, covariance_matrices, linear_parameters = (
        im_sim.image_linear_solve(**kwargs_result, inv_bool=False)
    )
    band_rows, product_arrays = standardized_residual_products(
        im_sim, models, model_error_maps, config.settings["band"]
    )

    used_pixels = int(sum(item["retained_pixels"] for item in band_rows))
    absolute_delta = float(abs(replayed_log_likelihood - stored_log_likelihood))
    absolute_delta_per_used_pixel = float(absolute_delta / used_pixels)
    fractional_total_delta = float(
        absolute_delta / max(abs(stored_log_likelihood), np.finfo(float).tiny)
    )
    tolerances = protocol["forward_replay_gate"]["likelihood_tolerance"]
    checks = {
        "upstream_environment_gate_passed": True,
        "stored_best_index_and_likelihood_match_frozen_contract": bool(
            best_index == protocol["chain_contract"]["best_sample_index"]
            and stored_log_likelihood
            == protocol["chain_contract"]["stored_max_log_likelihood"]
        ),
        "reconstructed_parameter_count_matches_chain": bool(count_matches),
        "reconstructed_parameter_order_matches_chain_exactly": bool(names_match),
        "stored_sample_is_finite": bool(np.isfinite(best_sample).all()),
        "forward_solution_completed_in_all_three_bands": bool(
            len(models) == len(config.settings["band"]) == 3
            and all(item["model_all_finite"] for item in band_rows)
        ),
        "model_variance_is_positive_finite_on_every_retained_pixel": all(
            item["variance_positive_finite_on_mask"] for item in band_rows
        ),
        "stored_total_likelihood_reproduced_per_pixel": bool(
            absolute_delta_per_used_pixel
            <= tolerances["maximum_absolute_delta_per_used_pixel"]
        ),
        "stored_total_likelihood_reproduced_fractionally": bool(
            fractional_total_delta <= tolerances["maximum_fractional_total_delta"]
        ),
        "no_nonlinear_optimization_performed": True,
        "external_numpy_pickle_not_loaded": True,
    }
    exact_replay_gate_pass = all(checks.values())

    product_arrays.update(
        {
            "best_sample": best_sample,
            "stored_log_likelihood": np.asarray(stored_log_likelihood),
            "replayed_log_likelihood": np.asarray(replayed_log_likelihood),
            "linear_parameters": np.asarray(linear_parameters, dtype=object),
            "covariance_matrices": np.asarray(covariance_matrices, dtype=object),
        }
    )
    args.products.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.products, **product_arrays)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol_version"],
        "stage": "exact_stored_chain_replay_before_heldout_controls",
        "upstream_environment_report": relative(environment_report_path),
        "software_versions": environment["versions"],
        "nonlinear_fit_performed": False,
        "lens_response_computed": False,
        "chain": {
            "best_sample_index": best_index,
            "parameter_count": parameter_count,
            "stored_parameter_names": stored_names,
            "reconstructed_parameter_names": reconstructed_names,
            "best_sample": best_sample.tolist(),
        },
        "likelihood_replay": {
            "stored_log_likelihood": stored_log_likelihood,
            "replayed_log_likelihood": replayed_log_likelihood,
            "used_pixels": used_pixels,
            "absolute_delta": absolute_delta,
            "absolute_delta_per_used_pixel": absolute_delta_per_used_pixel,
            "fractional_total_delta": fractional_total_delta,
            "frozen_tolerances": tolerances,
            "positive_flux_penalty_note": (
                "The released check_positive_flux=True convention subtracts 1e8 "
                "within each band whose multi-linear amplitudes are not all positive. "
                "The exact historical total retains those penalties; per-band pixel "
                "likelihoods below exclude them and are reported separately."
            ),
        },
        "bands": band_rows,
        "checks": checks,
        "exact_replay_gate_pass": exact_replay_gate_pass,
        "decision": (
            "exact_stored_chain_replay_pass_authorize_heldout_and_coordinate_controls"
            if exact_replay_gate_pass
            else "stop_J1402_before_heldout_controls"
        ),
        "outputs": {"replay_products": relative(args.products)},
        "authorization": {
            "run_heldout_sector_and_coordinate_controls": exact_replay_gate_pass,
            "optimize_nonlinear_model": False,
            "compute_lens_response": False,
            "reduce_KCWI": False,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
            "authorize_R2": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["likelihood_replay"], indent=2))
    print(json.dumps({item["band"]: item["reduced_chi_square_per_retained_pixel"] for item in band_rows}, indent=2))
    print(report["decision"])


if __name__ == "__main__":
    main()
