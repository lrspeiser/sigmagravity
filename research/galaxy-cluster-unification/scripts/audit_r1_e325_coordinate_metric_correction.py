#!/usr/bin/env python3
"""Apply the frozen metric-only correction to the preserved E325 J2 fit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from lenstronomy.LensModel.lens_model import LensModel

from fit_r1_e325_coordinate_lens import lens_kwargs, psf_projection, source_mapping


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_PATH = ROOT / "configs/r1_e325_coordinate_fit_metric_correction.json"
EXECUTION_PATH = ROOT / "configs/r1_e325_image_jacobian_execution_protocol.json"
OUTPUT_PATH = ROOT / "data/derived/r1_e325_coordinate_lens_fit.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
    invalid_report_path = ROOT / correction["invalid_report"]
    invalid_source_path = ROOT / correction["invalid_source"]
    invalid = json.loads(invalid_report_path.read_text(encoding="utf-8"))
    saved = np.load(invalid_source_path)
    source_image = np.asarray(saved["source"], dtype=float)
    source = source_image.ravel()
    mask = np.asarray(saved["mask"], dtype=bool)
    pad_mask = np.asarray(saved["pad_mask"], dtype=bool)
    parameters = np.asarray(saved["best_parameters"], dtype=float)
    kernel = np.asarray(saved["psf_kernel"], dtype=float)
    bounds = np.asarray(saved["source_plane_bounds_arcsec"], dtype=float).tolist()
    mask_path = ROOT / execution["inputs"]["masks_and_residuals"]
    with fits.open(mask_path, memmap=False) as hdul:
        header = hdul["ARCMASK"].header
        variance = np.asarray(hdul["VARIANCE"].data, dtype=float)[mask]
    colour = np.load(ROOT / execution["inputs"]["colour_model"])
    variance *= float(colour["noise_inflation"]) ** 2

    projection, pad_x, pad_y = psf_projection(mask, pad_mask, kernel)
    pixel_scale = float(abs(header["CDELT1"]) * 3600.0)
    cx = float(header["CRPIX1"] - 1.0)
    cy = float(header["CRPIX2"] - 1.0)
    x_pad = (pad_x - cx) * pixel_scale
    y_pad = (pad_y - cy) * pixel_scale
    lens_model = LensModel(lens_model_list=execution["coordinate_lens_fit"]["model"])
    beta_x, beta_y = lens_model.ray_shooting(x_pad, y_pad, lens_kwargs(parameters))
    grid_size = source_image.shape[0]
    mapping = source_mapping(np.asarray(beta_x), np.asarray(beta_y), grid_size, bounds)
    operator = (projection @ mapping).tocsr()
    whitened = operator.multiply((1.0 / np.sqrt(variance))[:, None]).tocsr()
    column_norms = np.sqrt(np.asarray(whitened.power(2).sum(axis=0)).ravel())
    constrained = column_norms > 0
    absolute_source = np.abs(source)
    constrained_values = absolute_source[constrained]
    maximum = float(constrained_values.max()) if len(constrained_values) else 0.0
    occupied_fraction = float(
        (constrained_values >= 0.01 * maximum).mean()
    ) if maximum > 0 else 0.0
    edge = np.zeros_like(source_image, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    constrained_grid = constrained.reshape(source_image.shape)
    denominator = float(absolute_source[constrained].sum())
    edge_flux_fraction = float(
        np.abs(source_image)[edge & constrained_grid].sum() / denominator
    ) if denominator > 0 else 1.0
    xmin, xmax, ymin, ymax = bounds
    boundary_reached = bool(
        np.any(np.asarray(beta_x) <= xmin)
        or np.any(np.asarray(beta_x) >= xmax)
        or np.any(np.asarray(beta_y) <= ymin)
        or np.any(np.asarray(beta_y) >= ymax)
    )
    threshold = float(correction["correction"]["threshold"])
    occupied_pass = occupied_fraction >= execution["coordinate_lens_fit"]["fit_gate"]["minimum_source_grid_occupied_fraction"]
    edge_pass = edge_flux_fraction <= threshold
    boundary_pass = not boundary_reached
    coordinate_gate = bool(
        invalid["gates"]["masked_fit_quality_passed"]
        and occupied_pass
        and edge_pass
        and boundary_pass
    )
    report = dict(invalid)
    report["report_version"] = correction["protocol_version"]
    report["generated_utc"] = datetime.now(timezone.utc).isoformat()
    report["invalid_v0_1_preserved"] = {
        "report": correction["invalid_report"],
        "report_sha256": sha256(invalid_report_path),
        "source": correction["invalid_source"],
        "source_sha256": sha256(invalid_source_path),
        "reason": correction["failure_observed"],
    }
    report["metric_correction"] = {
        "protocol": str(CORRECTION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(CORRECTION_PATH),
        "optimizer_rerun": False,
        "threshold_changed": False,
        "source_cells_total": int(len(source)),
        "source_cells_data_constrained": int(constrained.sum()),
        "source_cells_data_constrained_fraction": float(constrained.mean()),
        "corrected_source_occupied_fraction": occupied_fraction,
        "corrected_source_edge_absolute_flux_fraction": edge_flux_fraction,
        "image_mapped_source_boundary_reached": boundary_reached,
    }
    report["fit"]["invalid_unconstrained_source_occupied_fraction"] = report["fit"].pop("source_occupied_fraction")
    report["fit"]["invalid_unconstrained_source_edge_absolute_flux_fraction"] = report["fit"].pop("source_edge_absolute_flux_fraction")
    report["fit"]["source_occupied_fraction_on_data_constrained_cells"] = occupied_fraction
    report["fit"]["source_edge_absolute_flux_fraction_on_data_constrained_cells"] = edge_flux_fraction
    report["gates"]["source_occupied_fraction_passed"] = occupied_pass
    report["gates"]["source_edge_flux_passed"] = edge_pass
    report["gates"]["source_mapping_stays_inside_fixed_bounds_passed"] = boundary_pass
    report["gates"]["coordinate_map_engineering_gate_passed"] = coordinate_gate
    report["decision"] = (
        "continue_to_heldout_visit_and_full_jacobian"
        if coordinate_gate
        else "stop_E325_coordinate_map_fit_after_metric_correction"
    )
    report["authorization"]["run_heldout_visit_and_full_jacobian"] = coordinate_gate
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
