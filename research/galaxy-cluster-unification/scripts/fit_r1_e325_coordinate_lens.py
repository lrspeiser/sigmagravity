#!/usr/bin/env python3
"""Fit the frozen E325 coordinate lens map with semilinear source inversion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lenstronomy
import numpy as np
import scipy
from astropy.io import fits
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.param_util import phi_q2_ellipticity
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import lsmr


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_e325_image_jacobian_execution_protocol.json"
UPSTREAM_PATH = ROOT / "results/r1_e325_arc_mask/report.json"
SOURCE_OUTPUT = ROOT / "data/derived/r1_e325_coordinate_lens_source.npz"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def gradient_regularizer(grid_size: int) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    row = 0
    for iy in range(grid_size):
        for ix in range(grid_size - 1):
            left = iy * grid_size + ix
            rows.extend([row, row])
            columns.extend([left, left + 1])
            values.extend([-1.0, 1.0])
            row += 1
    for iy in range(grid_size - 1):
        for ix in range(grid_size):
            lower = iy * grid_size + ix
            rows.extend([row, row])
            columns.extend([lower, lower + grid_size])
            values.extend([-1.0, 1.0])
            row += 1
    return coo_matrix((values, (rows, columns)), shape=(row, grid_size**2)).tocsr()


def source_mapping(
    beta_x: np.ndarray, beta_y: np.ndarray, grid_size: int, bounds: list[float]
) -> csr_matrix:
    xmin, xmax, ymin, ymax = bounds
    ux = (beta_x - xmin) / (xmax - xmin) * (grid_size - 1)
    uy = (beta_y - ymin) / (ymax - ymin) * (grid_size - 1)
    inside = (ux >= 0) & (ux < grid_size - 1) & (uy >= 0) & (uy < grid_size - 1)
    row_index = np.nonzero(inside)[0]
    x0 = np.floor(ux[inside]).astype(int)
    y0 = np.floor(uy[inside]).astype(int)
    dx = ux[inside] - x0
    dy = uy[inside] - y0
    columns = np.column_stack(
        [
            y0 * grid_size + x0,
            y0 * grid_size + x0 + 1,
            (y0 + 1) * grid_size + x0,
            (y0 + 1) * grid_size + x0 + 1,
        ]
    ).ravel()
    rows = np.repeat(row_index, 4)
    values = np.column_stack(
        [
            (1 - dx) * (1 - dy),
            dx * (1 - dy),
            (1 - dx) * dy,
            dx * dy,
        ]
    ).ravel()
    return coo_matrix(
        (values, (rows, columns)), shape=(len(beta_x), grid_size**2)
    ).tocsr()


def psf_projection(mask: np.ndarray, pad_mask: np.ndarray, kernel: np.ndarray) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    pad_y, pad_x = np.nonzero(pad_mask)
    pad_lookup = np.full(mask.shape, -1, dtype=int)
    pad_lookup[pad_y, pad_x] = np.arange(len(pad_y))
    mask_y, mask_x = np.nonzero(mask)
    half_y, half_x = np.array(kernel.shape) // 2
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for output_row, (iy, ix) in enumerate(zip(mask_y, mask_x, strict=True)):
        for ky in range(kernel.shape[0]):
            py = iy + ky - half_y
            if py < 0 or py >= mask.shape[0]:
                continue
            for kx in range(kernel.shape[1]):
                px = ix + kx - half_x
                if px < 0 or px >= mask.shape[1]:
                    continue
                source_row = pad_lookup[py, px]
                if source_row >= 0 and kernel[ky, kx] != 0:
                    rows.append(output_row)
                    columns.append(int(source_row))
                    values.append(float(kernel[ky, kx]))
    projection = coo_matrix(
        (values, (rows, columns)), shape=(len(mask_y), len(pad_y))
    ).tocsr()
    return projection, pad_x, pad_y


def lens_kwargs(parameters: np.ndarray) -> list[dict[str, float]]:
    theta_e, gamma, e1, e2, center_x, center_y, gamma1, gamma2 = parameters
    return [
        {
            "theta_E": float(theta_e),
            "gamma": float(gamma),
            "e1": float(e1),
            "e2": float(e2),
            "center_x": float(center_x),
            "center_y": float(center_y),
        },
        {"gamma1": float(gamma1), "gamma2": float(gamma2), "ra_0": 0.0, "dec_0": 0.0},
    ]


def solve_source(
    operator: csr_matrix,
    data: np.ndarray,
    variance: np.ndarray,
    regularizer: csr_matrix,
    amplitude: float,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sqrt_weight = 1.0 / np.sqrt(variance)
    weighted_operator = operator.multiply(sqrt_weight[:, None]).tocsr()
    weighted_data = data * sqrt_weight
    column_norms = np.sqrt(np.asarray(weighted_operator.power(2).sum(axis=0)).ravel())
    nonzero_norms = column_norms[column_norms > 0]
    scale = float(np.median(nonzero_norms)) if len(nonzero_norms) else 1.0
    augmented = vstack(
        [weighted_operator, regularizer * (np.sqrt(amplitude) * scale)], format="csr"
    )
    right_hand_side = np.r_[weighted_data, np.zeros(regularizer.shape[0])]
    result = lsmr(
        augmented,
        right_hand_side,
        atol=tolerance,
        btol=tolerance,
        maxiter=maximum_iterations,
    )
    source = result[0]
    prediction = operator @ source
    residual = data - prediction
    chi_square = float(np.sum(residual**2 / variance))
    regularization_penalty = float(amplitude * scale**2 * np.sum((regularizer @ source) ** 2))
    return source, prediction, {
        "chi_square": chi_square,
        "regularization_penalty": regularization_penalty,
        "objective": chi_square + regularization_penalty,
        "linear_iterations": int(result[2]),
        "linear_stop_code": int(result[1]),
        "regularization_scale": scale,
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    if not upstream["authorization"]["implement_frozen_image_level_jacobian"]:
        raise RuntimeError("Arc-mask gate did not authorize coordinate fitting")
    software = config["software_lock"]
    if lenstronomy.__version__ != software["lenstronomy_version"] or scipy.__version__ != software["scipy_version"]:
        raise RuntimeError("Frozen E325 coordinate-fit software versions changed")

    mask_path = ROOT / config["inputs"]["masks_and_residuals"]
    colour_path = ROOT / config["inputs"]["colour_model"]
    with fits.open(mask_path, memmap=False) as hdul:
        data = np.asarray(hdul["RESIDUAL"].data, dtype=float)
        variance = np.asarray(hdul["VARIANCE"].data, dtype=float)
        mask = np.asarray(hdul["ARCMASK"].data, dtype=bool)
        header = hdul["ARCMASK"].header
    colour = np.load(colour_path)
    noise_inflation = float(colour["noise_inflation"])
    variance *= noise_inflation**2
    common_psf = np.asarray(colour["common_psf"], dtype=float)
    center = np.array(common_psf.shape) // 2
    psf_half = 5
    kernel = common_psf[
        center[0] - psf_half : center[0] + psf_half + 1,
        center[1] - psf_half : center[1] + psf_half + 1,
    ].copy()
    kernel /= kernel.sum()
    pad_mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=psf_half)
    projection, pad_x, pad_y = psf_projection(mask, pad_mask, kernel)

    pixel_scale = float(abs(header["CDELT1"]) * 3600.0)
    cx = float(header["CRPIX1"] - 1.0)
    cy = float(header["CRPIX2"] - 1.0)
    x_pad = (pad_x - cx) * pixel_scale
    y_pad = (pad_y - cy) * pixel_scale
    masked_data = data[mask]
    masked_variance = variance[mask]
    lens_model = LensModel(lens_model_list=config["coordinate_lens_fit"]["model"])
    source_config = config["semilinear_source"]
    grid_size = int(config["coordinate_lens_fit"]["engineering_source_grid"])
    regularizer = gradient_regularizer(grid_size)
    amplitude = float(config["coordinate_lens_fit"]["engineering_regularization_amplitude"])
    evaluations: list[dict[str, object]] = []

    def evaluate(parameters: np.ndarray, keep: bool = False):
        beta_x, beta_y = lens_model.ray_shooting(x_pad, y_pad, lens_kwargs(parameters))
        source_map = source_mapping(
            np.asarray(beta_x), np.asarray(beta_y), grid_size, source_config["source_plane_bounds_arcsec"]
        )
        operator = (projection @ source_map).tocsr()
        source, prediction, metrics = solve_source(
            operator,
            masked_data,
            masked_variance,
            regularizer,
            amplitude,
            int(source_config["maximum_linear_iterations"]),
            float(source_config["linear_tolerance"]),
        )
        ellipticity_penalty = 0.0
        if np.hypot(parameters[2], parameters[3]) >= 0.3:
            ellipticity_penalty = 1e6 * (np.hypot(parameters[2], parameters[3]) - 0.3) ** 2
        objective = metrics["objective"] / len(masked_data) + ellipticity_penalty
        evaluations.append(
            {
                "parameters": parameters.tolist(),
                "objective_per_pixel": float(objective),
                "chi_square_per_pixel": metrics["chi_square"] / len(masked_data),
                "linear_iterations": metrics["linear_iterations"],
            }
        )
        if keep:
            return objective, source, prediction, operator, metrics
        return objective

    initial_config = config["coordinate_lens_fit"]["initial"]
    phi = np.deg2rad(180.0 - initial_config["position_angle_east_of_north_deg"])
    e1, e2 = phi_q2_ellipticity(phi, initial_config["axis_ratio"])
    initial = np.asarray(
        [
            initial_config["theta_E"],
            initial_config["gamma"],
            e1,
            e2,
            initial_config["center_x_arcsec"],
            initial_config["center_y_arcsec"],
            initial_config["gamma1"],
            initial_config["gamma2"],
        ],
        dtype=float,
    )
    bounds_config = config["coordinate_lens_fit"]["bounds"]
    parameter_names = config["coordinate_lens_fit"]["free_parameters"]
    bounds = [tuple(bounds_config[name]) for name in parameter_names]
    initial_objective = float(evaluate(initial))
    result = minimize(
        evaluate,
        initial,
        method="Powell",
        bounds=bounds,
        options={
            "maxfev": int(config["coordinate_lens_fit"]["maximum_function_evaluations"]),
            "xtol": 1e-3,
            "ftol": 1e-3,
        },
    )
    final_objective, source, prediction, operator, metrics = evaluate(result.x, keep=True)
    source_image = source.reshape(grid_size, grid_size)
    source_abs = np.abs(source_image)
    threshold = 0.01 * source_abs.max() if source_abs.size else np.inf
    occupied_fraction = float((source_abs >= threshold).mean()) if source_abs.max() > 0 else 0.0
    edge = np.zeros_like(source_image, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge_flux_fraction = float(source_abs[edge].sum() / source_abs.sum()) if source_abs.sum() else 1.0
    chi_square_per_pixel = metrics["chi_square"] / len(masked_data)
    fit_gate = config["coordinate_lens_fit"]["fit_gate"]
    coordinate_gate = bool(
        chi_square_per_pixel <= fit_gate["maximum_masked_reduced_chi_square"]
        and occupied_fraction >= fit_gate["minimum_source_grid_occupied_fraction"]
        and edge_flux_fraction <= fit_gate["maximum_source_grid_edge_flux_fraction"]
    )
    SOURCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SOURCE_OUTPUT,
        source=source_image,
        masked_prediction=prediction,
        masked_data=masked_data,
        masked_variance=masked_variance,
        mask=mask,
        pad_mask=pad_mask,
        best_parameters=result.x,
        parameter_names=np.asarray(parameter_names),
        source_plane_bounds_arcsec=np.asarray(source_config["source_plane_bounds_arcsec"]),
        psf_kernel=kernel,
    )
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "published_E325_lens_or_gamma_posterior_used": False,
        "inputs": {
            "protocol": {"path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(CONFIG_PATH)},
            "arc_mask_report": {"path": str(UPSTREAM_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(UPSTREAM_PATH)},
            "masks_and_residuals": {"path": str(mask_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(mask_path)},
            "colour_model": {"path": str(colour_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(colour_path)},
        },
        "model": {
            "lens_models": config["coordinate_lens_fit"]["model"],
            "source_grid": grid_size,
            "source_plane_bounds_arcsec": source_config["source_plane_bounds_arcsec"],
            "masked_pixels": int(len(masked_data)),
            "padded_mapping_pixels": int(len(pad_x)),
            "source_operator_nonzero_entries": int(operator.nnz),
            "initial_parameters": dict(zip(parameter_names, initial.tolist(), strict=True)),
            "best_parameters": dict(zip(parameter_names, result.x.tolist(), strict=True)),
        },
        "optimizer": {
            "method": "Powell",
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "initial_objective_per_pixel": initial_objective,
            "final_objective_per_pixel": float(final_objective),
            "logged_evaluations": len(evaluations),
        },
        "fit": {
            "chi_square": metrics["chi_square"],
            "chi_square_per_masked_pixel": chi_square_per_pixel,
            "regularization_penalty": metrics["regularization_penalty"],
            "linear_iterations": metrics["linear_iterations"],
            "source_occupied_fraction": occupied_fraction,
            "source_edge_absolute_flux_fraction": edge_flux_fraction,
        },
        "gates": {
            "masked_fit_quality_passed": chi_square_per_pixel <= fit_gate["maximum_masked_reduced_chi_square"],
            "source_occupied_fraction_passed": occupied_fraction >= fit_gate["minimum_source_grid_occupied_fraction"],
            "source_edge_flux_passed": edge_flux_fraction <= fit_gate["maximum_source_grid_edge_flux_fraction"],
            "coordinate_map_engineering_gate_passed": coordinate_gate,
            "heldout_visit_gate_passed": False,
            "rank_three_candidate_admission_passed": False,
        },
        "outputs": {
            "coordinate_source": str(SOURCE_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "coordinate_source_sha256": sha256(SOURCE_OUTPUT),
        },
        "decision": "continue_to_heldout_visit_and_full_jacobian" if coordinate_gate else "stop_E325_coordinate_map_fit",
        "authorization": {
            "run_heldout_visit_and_full_jacobian": coordinate_gate,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    output_path = ROOT / config["outputs"]["coordinate_fit"]
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
