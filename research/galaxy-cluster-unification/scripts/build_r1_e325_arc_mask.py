#!/usr/bin/env python3
"""Freeze E325 arc and negative-control masks with the preregistered algorithm."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.signal import fftconvolve


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_e325_arc_mask_protocol.json"
UPSTREAM_PATH = ROOT / "results/r1_e325_hst_preprocessing/report.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def hat_basis(radius: np.ndarray, knots: np.ndarray) -> np.ndarray:
    identity = np.eye(len(knots))
    return np.column_stack(
        [np.interp(radius, knots, identity[:, index], left=0.0, right=0.0) for index in range(len(knots))]
    )


def weighted_linear_fit(matrix: np.ndarray, values: np.ndarray, variance: np.ndarray) -> np.ndarray:
    weight_sqrt = 1.0 / np.sqrt(variance)
    weighted_matrix = matrix * weight_sqrt[:, None]
    weighted_values = values * weight_sqrt
    return np.linalg.lstsq(weighted_matrix, weighted_values, rcond=1e-10)[0]


def azimuthal_span(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    angles = np.sort((np.degrees(np.arctan2(y[mask], x[mask])) + 360.0) % 360.0)
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(np.r_[angles, angles[0] + 360.0])
    return float(360.0 - gaps.max())


def equal_area_rotated_control(
    arc_mask: np.ndarray, annulus: np.ndarray, radius: np.ndarray, angle: np.ndarray
) -> np.ndarray:
    rotated = np.rot90(arc_mask, k=1)
    control = rotated & annulus & ~arc_mask
    target = int(arc_mask.sum())
    missing_locations = np.argwhere(rotated & arc_mask)
    available = annulus & ~arc_mask & ~control
    for location in missing_locations:
        if int(control.sum()) >= target:
            break
        iy, ix = location
        candidate_y, candidate_x = np.nonzero(available)
        if len(candidate_y) == 0:
            break
        radial_cost = np.abs(radius[candidate_y, candidate_x] - radius[iy, ix])
        angular_delta = np.abs(
            np.angle(np.exp(1j * (angle[candidate_y, candidate_x] - angle[iy, ix])))
        )
        order = np.lexsort((candidate_x, candidate_y, angular_delta, radial_cost))
        chosen = order[0]
        cy, cx = candidate_y[chosen], candidate_x[chosen]
        control[cy, cx] = True
        available[cy, cx] = False
    if int(control.sum()) < target:
        candidate_y, candidate_x = np.nonzero(available)
        needed = target - int(control.sum())
        order = np.lexsort((candidate_x, candidate_y, radius[candidate_y, candidate_x]))
        chosen = order[:needed]
        control[candidate_y[chosen], candidate_x[chosen]] = True
    if int(control.sum()) > target:
        candidate_y, candidate_x = np.nonzero(control)
        order = np.lexsort((candidate_x, candidate_y, radius[candidate_y, candidate_x]))
        drop = order[target:]
        control[candidate_y[drop], candidate_x[drop]] = False
    return control


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    if not upstream["authorization"]["freeze_arc_and_negative_control_masks"]:
        raise RuntimeError("Blind preprocessing gate did not authorize morphology inspection")
    if config["arc_morphology_seen_at_freeze"]:
        raise RuntimeError("Arc-mask protocol was not frozen blind")
    input_path = ROOT / config["inputs"]["registered_cutouts"]
    psf_path = ROOT / config["inputs"]["psf_family"]
    with fits.open(input_path, memmap=False) as hdul:
        header = hdul["F475COA"].header.copy()
        f475 = np.asarray(hdul["F475COA"].data, dtype=float)
        w475 = np.asarray(hdul["F475WHT"].data, dtype=float)
        f814 = np.asarray(hdul["F814COA"].data, dtype=float)
        w814 = np.asarray(hdul["F814WHT"].data, dtype=float)
        f475_visits = [np.asarray(hdul[f"F475V{number}S"].data, dtype=float) for number in (1, 2)]
        w475_visits = [np.asarray(hdul[f"F475V{number}W"].data, dtype=float) for number in (1, 2)]
    psfs = np.load(psf_path)
    psf475 = np.asarray(psfs[config["common_psf"]["f475_psf_key"]], dtype=float)
    psf814 = np.asarray(psfs[config["common_psf"]["f814_psf_key"]], dtype=float)
    common_psf = fftconvolve(psf475, psf814, mode="full")
    common_psf /= common_psf.sum()

    f475_matched = fftconvolve(f475, psf814, mode="same")
    f814_matched = fftconvolve(f814, psf475, mode="same")
    variance475 = fftconvolve(np.where(w475 > 0, 1.0 / w475, 0.0), psf814**2, mode="same")
    variance814 = fftconvolve(np.where(w814 > 0, 1.0 / w814, 0.0), psf475**2, mode="same")

    shape = f475.shape
    pixel_scale = float(abs(header["CDELT1"]) * 3600.0)
    yy, xx = np.indices(shape, dtype=float)
    cx = float(header["CRPIX1"] - 1.0)
    cy = float(header["CRPIX2"] - 1.0)
    x_arcsec = (xx - cx) * pixel_scale
    y_arcsec = (yy - cy) * pixel_scale
    radius = np.hypot(x_arcsec, y_arcsec)
    angle = np.arctan2(y_arcsec, x_arcsec)

    colour = config["lens_colour_model"]
    knots = np.asarray(colour["radial_knots_arcsec"], dtype=float)
    hats = hat_basis(radius.ravel(), knots).reshape(*shape, len(knots))
    design = np.dstack(
        [
            np.ones(shape),
            x_arcsec,
            y_arcsec,
            *[f814_matched * hats[:, :, index] for index in range(len(knots))],
        ]
    )
    fit_domain = (
        (radius >= colour["fit_radial_domain_arcsec"][0])
        & (radius <= colour["fit_radial_domain_arcsec"][1])
        & np.isfinite(f475_matched)
        & np.isfinite(f814_matched)
        & (variance475 > 0)
        & (variance814 > 0)
    )
    retained = fit_domain.copy()
    coefficients = np.zeros(design.shape[-1])
    variance = variance475.copy()
    clip_history: list[int] = []
    for _ in range(int(colour["iterations"])):
        coefficients = weighted_linear_fit(
            design[retained], f475_matched[retained], variance[retained]
        )
        colour_scale = np.tensordot(hats, coefficients[3:], axes=([2], [0]))
        model = np.tensordot(design, coefficients, axes=([2], [0]))
        variance = variance475 + colour_scale**2 * variance814
        residual = f475_matched - model
        sigma = np.sqrt(variance)
        newly_retained = retained & (
            (residual <= colour["positive_clip_sigma"] * sigma)
            & (np.abs(residual) <= colour["absolute_clip_sigma"] * sigma)
        )
        clip_history.append(int(retained.sum() - newly_retained.sum()))
        retained = newly_retained

    colour_scale = np.tensordot(hats, coefficients[3:], axes=([2], [0]))
    model = np.tensordot(design, coefficients, axes=([2], [0]))
    residual = f475_matched - model
    variance = variance475 + colour_scale**2 * variance814
    valid = np.isfinite(residual) & np.isfinite(variance) & (variance > 0)
    inverse_variance = np.where(valid, 1.0 / variance, 0.0)
    numerator = fftconvolve(residual * inverse_variance, common_psf[::-1, ::-1], mode="same")
    denominator2 = fftconvolve(inverse_variance, common_psf[::-1, ::-1] ** 2, mode="same")
    matched_snr = np.zeros(shape, dtype=float)
    matched_snr[denominator2 > 0] = numerator[denominator2 > 0] / np.sqrt(denominator2[denominator2 > 0])

    detection = config["detection"]
    annulus = (
        (radius >= detection["radial_domain_arcsec"][0])
        & (radius <= detection["radial_domain_arcsec"][1])
        & valid
    )
    noise_sample = matched_snr[annulus & (np.abs(matched_snr) < 4.0)]
    median = float(np.median(noise_sample))
    noise_inflation = max(1.0, float(1.4826 * np.median(np.abs(noise_sample - median))))
    corrected_snr = (matched_snr - median) / noise_inflation
    raw_detection = annulus & (corrected_snr >= detection["corrected_signal_to_noise_threshold"])
    labels, component_count_raw = label(raw_detection, structure=np.ones((3, 3), dtype=int))
    retained_components: list[int] = []
    cleaned = np.zeros(shape, dtype=bool)
    for component in range(1, component_count_raw + 1):
        component_mask = labels == component
        if int(component_mask.sum()) >= detection["minimum_connected_pixels_before_dilation"]:
            cleaned |= component_mask
            retained_components.append(component)
    arc_mask = binary_dilation(cleaned, structure=np.ones((3, 3), dtype=bool), iterations=detection["dilation_pixels"])
    arc_mask &= annulus
    eroded = binary_erosion(arc_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    extra_dilated = binary_dilation(arc_mask, structure=np.ones((3, 3), dtype=bool), iterations=1) & annulus
    negative_control = equal_area_rotated_control(arc_mask, annulus, radius, angle)

    support = config["pre_rank_radial_support"]
    knot_intersections = {
        str(knot): bool(
            np.any(arc_mask & (np.abs(radius - knot) <= support["knot_neighborhood_half_width_arcsec"]))
        )
        for knot in support["response_knots_arcsec"]
    }
    knot_count = sum(knot_intersections.values())
    arc_span = azimuthal_span(arc_mask, x_arcsec, y_arcsec)

    visit_metrics: list[dict[str, float]] = []
    visit_residuals: list[np.ndarray] = []
    visit_variances: list[np.ndarray] = []
    for science, weight in zip(f475_visits, w475_visits, strict=True):
        matched = fftconvolve(science, psf814, mode="same")
        visit_variance = fftconvolve(np.where(weight > 0, 1.0 / weight, 0.0), psf814**2, mode="same")
        visit_residual = matched - model
        integrated_snr = float(
            np.sum(visit_residual[arc_mask] / visit_variance[arc_mask])
            / np.sqrt(np.sum(1.0 / visit_variance[arc_mask]))
        )
        visit_metrics.append({"integrated_arc_signal_to_noise": integrated_snr})
        visit_residuals.append(visit_residual)
        visit_variances.append(visit_variance)
    values1 = visit_residuals[0][arc_mask]
    values2 = visit_residuals[1][arc_mask]
    difference_variance = visit_variances[0][arc_mask] + visit_variances[1][arc_mask]
    visit_design = np.column_stack([values1, np.ones_like(values1)])
    visit_coefficients = weighted_linear_fit(visit_design, values2, difference_variance)
    difference = values2 - visit_design @ visit_coefficients
    visit_difference_dof = max(1, len(difference) - 2)
    visit_difference_reduced_chi_square = float(
        np.sum(difference**2 / difference_variance) / visit_difference_dof
    )

    mask_size_pass = int(arc_mask.sum()) >= detection["minimum_final_pixels"]
    component_pass = len(retained_components) >= detection["minimum_connected_components"]
    span_pass = arc_span >= detection["minimum_azimuthal_span_deg"]
    negative_pass = bool(
        int(negative_control.sum()) == int(arc_mask.sum())
        and not np.any(negative_control & arc_mask)
    )
    support_pass = knot_count >= support["minimum_knot_neighborhoods_intersected"]
    visits_pass = bool(
        all(
            metric["integrated_arc_signal_to_noise"]
            >= config["visit_controls"]["minimum_integrated_arc_signal_to_noise_each_F475W_visit"]
            for metric in visit_metrics
        )
        and visit_difference_reduced_chi_square
        <= config["visit_controls"]["maximum_visit_difference_reduced_chi_square"]
    )
    complete = mask_size_pass and component_pass and span_pass and negative_pass and support_pass and visits_pass

    output_path = ROOT / config["outputs"]["masks_and_residuals"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primary_header = fits.Header()
    primary_header["PROTOVER"] = config["protocol_version"]
    primary_header["NOISEINF"] = noise_inflation
    hdus = [
        fits.PrimaryHDU(header=primary_header),
        fits.ImageHDU(residual.astype(np.float32), header=header, name="RESIDUAL"),
        fits.ImageHDU(variance.astype(np.float32), header=header, name="VARIANCE"),
        fits.ImageHDU(corrected_snr.astype(np.float32), header=header, name="SNR"),
        fits.ImageHDU(model.astype(np.float32), header=header, name="LENSMODEL"),
        fits.ImageHDU(arc_mask.astype(np.uint8), header=header, name="ARCMASK"),
        fits.ImageHDU(eroded.astype(np.uint8), header=header, name="ERODED"),
        fits.ImageHDU(extra_dilated.astype(np.uint8), header=header, name="DILATED"),
        fits.ImageHDU(negative_control.astype(np.uint8), header=header, name="NEGCTRL"),
    ]
    fits.HDUList(hdus).writeto(output_path, overwrite=True, checksum=True)
    model_path = ROOT / config["outputs"]["colour_model"]
    np.savez_compressed(
        model_path,
        coefficients=coefficients,
        radial_knots_arcsec=knots,
        retained_fit_mask=retained,
        colour_scale=colour_scale.astype(np.float32),
        common_psf=common_psf,
        noise_inflation=np.asarray(noise_inflation),
        visit_scale_offset=visit_coefficients,
    )

    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "gravity_residuals_inspected": False,
        "inputs": {
            "protocol": {"path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(CONFIG_PATH)},
            "preprocessing_report": {"path": str(UPSTREAM_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(UPSTREAM_PATH)},
            "registered_cutouts": {"path": str(input_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(input_path)},
            "psf_family": {"path": str(psf_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(psf_path)},
        },
        "colour_model": {
            "fit_pixels_initial": int(fit_domain.sum()),
            "fit_pixels_retained": int(retained.sum()),
            "newly_clipped_pixels_by_iteration": clip_history,
            "coefficients": coefficients.tolist(),
        },
        "detection": {
            "noise_sample_pixels": int(len(noise_sample)),
            "matched_filter_noise_median": median,
            "drizzle_noise_inflation": noise_inflation,
            "raw_detected_pixels": int(raw_detection.sum()),
            "raw_connected_components": int(component_count_raw),
            "retained_connected_components": int(len(retained_components)),
            "final_arc_pixels": int(arc_mask.sum()),
            "eroded_arc_pixels": int(eroded.sum()),
            "extra_dilated_arc_pixels": int(extra_dilated.sum()),
            "negative_control_pixels": int(negative_control.sum()),
            "azimuthal_span_deg": arc_span,
            "knot_neighborhood_intersections": knot_intersections,
            "knot_neighborhoods_intersected": int(knot_count),
        },
        "visit_controls": {
            "visits": visit_metrics,
            "relative_scale": float(visit_coefficients[0]),
            "relative_offset": float(visit_coefficients[1]),
            "difference_degrees_of_freedom": int(visit_difference_dof),
            "difference_reduced_chi_square": visit_difference_reduced_chi_square,
        },
        "gates": {
            "minimum_mask_size_passed": mask_size_pass,
            "minimum_component_count_passed": component_pass,
            "minimum_azimuthal_span_passed": span_pass,
            "equal_area_nonoverlapping_negative_control_passed": negative_pass,
            "minimum_three_knot_neighborhoods_passed": support_pass,
            "both_visit_consistency_passed": visits_pass,
            "complete_arc_mask_gate_passed": complete,
            "rank_three_candidate_admission_passed": False,
        },
        "outputs": {
            "masks_and_residuals": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "masks_and_residuals_sha256": sha256(output_path),
            "colour_model": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "colour_model_sha256": sha256(model_path),
        },
        "decision": "authorize_frozen_image_level_jacobian_implementation" if complete else "retain_as_acquired_control_and_replace_candidate",
        "authorization": {
            "implement_frozen_image_level_jacobian": complete,
            "reconstruct_MUSE_numerical_kinematics": complete,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "ten_system_effect": {
            "previous_structural_ceiling": 3,
            "updated_structural_ceiling": 3,
            "minimum_new_rank_three_systems_still_required": 7,
        },
    }
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
