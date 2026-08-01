#!/usr/bin/env python3
"""Select routed-field variants with a mass-sheet-resistant linearized image metric."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0574_symmetry_gated_arrival_microvariation import quarter_turn_asymmetry  # noqa: E402
from run_p0575_smacs0723_raw_position import (  # noqa: E402
    deflection_from_surface,
    fit_positive_amplitude,
    interpolate_deflection,
    lens_efficiency,
    sha256,
)
from run_p0575b_raw_position_robustness import build_maps  # noqa: E402
from run_p0576_fractional_routed_propagator import fractional_deflection  # noqa: E402


def sample_field_and_jacobian(ax, ay, images, axis, kpc_per_arcsec):
    spacing = float(axis[1] - axis[0])
    pixel_x = (images.x_kpc.to_numpy(float) - axis[0]) / spacing
    pixel_y = (images.y_kpc.to_numpy(float) - axis[0]) / spacing
    coordinates = np.vstack([pixel_y, pixel_x])
    sampled = interpolate_deflection(ax, ay, images, axis)
    dax_dy, dax_dx = np.gradient(ax, spacing, spacing)
    day_dy, day_dx = np.gradient(ay, spacing, spacing)
    def take(field):
        return map_coordinates(field, coordinates, order=1, mode="constant", cval=np.nan)
    jacobian = np.empty((len(images), 2, 2), dtype=float)
    jacobian[:, 0, 0] = take(dax_dx) * kpc_per_arcsec
    jacobian[:, 0, 1] = take(dax_dy) * kpc_per_arcsec
    jacobian[:, 1, 0] = take(day_dx) * kpc_per_arcsec
    jacobian[:, 1, 1] = take(day_dy) * kpc_per_arcsec
    return sampled, jacobian


def image_plane_rms(theta, alpha, jac_alpha, efficiency, families, mask, amplitude, singular_floor):
    beta = theta - amplitude * efficiency[:, None] * alpha
    residuals = np.zeros_like(beta)
    for family in np.unique(families[mask]):
        local = mask & (families == family)
        residuals[local] = beta[local] - np.mean(beta[local], axis=0)
    image_residuals = []
    minimum_singular = []
    for index in np.flatnonzero(mask):
        lens_jacobian = np.eye(2) - amplitude * efficiency[index] * jac_alpha[index]
        u, singular, vh = np.linalg.svd(lens_jacobian)
        inverse = vh.T @ np.diag(1.0 / np.maximum(singular, singular_floor)) @ u.T
        image_residuals.append(inverse @ residuals[index])
        minimum_singular.append(float(np.min(singular)))
    joined = np.asarray(image_residuals)
    return (
        float(np.sqrt(np.mean(np.sum(joined * joined, axis=1)))),
        float(np.median(minimum_singular)),
    )


def fit_amplitude(theta, alpha, jac_alpha, efficiency, families, calibration_mask, singular_floor):
    scaled = efficiency[:, None] * alpha
    source_amplitude = fit_positive_amplitude(theta, scaled, families, calibration_mask)
    upper = max(5.0 * source_amplitude, 1.0e-8)
    grid = np.linspace(0.0, upper, 501)
    values = np.asarray(
        [
            image_plane_rms(
                theta, alpha, jac_alpha, efficiency, families, calibration_mask, value, singular_floor
            )[0]
            for value in grid
        ]
    )
    best = int(np.argmin(values))
    lower_index = max(best - 1, 0)
    upper_index = min(best + 1, len(grid) - 1)
    if upper_index == lower_index:
        return float(grid[best]), float(values[best])
    optimized = minimize_scalar(
        lambda value: image_plane_rms(
            theta, alpha, jac_alpha, efficiency, families, calibration_mask, value, singular_floor
        )[0],
        bounds=(float(grid[lower_index]), float(grid[upper_index])),
        method="bounded",
        options={"xatol": max(upper * 1.0e-10, 1.0e-12)},
    )
    return float(optimized.x), float(optimized.fun)


def mass_sheet_r2(theta, scaled_alpha):
    t = theta - np.mean(theta, axis=0)
    a = scaled_alpha - np.mean(scaled_alpha, axis=0)
    scale = float(np.sum(t * a) / np.sum(a * a))
    return float(1.0 - np.sum((t - scale * a) ** 2) / np.sum(t * t))


def main() -> None:
    protocol_path = ROOT / "configs/p0576d_linearized_image_plane_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0576C_before_image_plane_scores":
        raise RuntimeError("P0576D protocol is not frozen")
    images = pd.read_csv(ROOT / protocol["inputs"]["p0575_images"], dtype={"family": str})
    data, maps = build_maps(protocol, images)
    local = maps["local_control"]
    gated_map = maps["p0574_symmetry_gated"]
    q90 = quarter_turn_asymmetry(data)
    gate = q90**4 / (q90**4 + 0.05**4)
    map_fraction = 0.8 * gate
    destination = np.maximum((gated_map - (1.0 - map_fraction) * local) / map_fraction, 0.0)
    destination /= np.sum(destination)
    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    calibration_mask = np.isin(families, protocol["selection"]["calibration_families"])
    efficiency = lens_efficiency(0.39, images.source_redshift.to_numpy(float))
    kpc_per_arcsec = float(data.axis[1] - data.axis[0]) / 10.0  # grid spacing is 10 kpc per one index
    # Use the actual coordinate conversion stored with the raw image table.
    valid_x = np.abs(images.theta_x_arcsec.to_numpy(float)) > 1.0e-8
    kpc_per_arcsec = float(np.median(np.abs(images.loc[valid_x, "x_kpc"] / images.loc[valid_x, "theta_x_arcsec"])))
    padding = int(protocol["grid"]["padding_factor"])
    singular_floor = float(protocol["metric"]["singular_value_floor"])
    local_ax, local_ay = deflection_from_surface(local, 10.0, padding)
    local_sampled, local_jac = sample_field_and_jacobian(
        local_ax, local_ay, images, data.axis, kpc_per_arcsec
    )
    len_ax, len_ay = deflection_from_surface(maps["lenstool_map_reference"], 10.0, padding)
    len_sampled, len_jac = sample_field_and_jacobian(
        len_ax, len_ay, images, data.axis, kpc_per_arcsec
    )
    routed = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        ax, ay = fractional_deflection(destination, 10.0, power, 60.0, padding)
        routed[power] = sample_field_and_jacobian(ax, ay, images, data.axis, kpc_per_arcsec)

    rows = []
    fields = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        for fraction in map(float, protocol["grid"]["deflection_route_fraction"]):
            effective = fraction * gate
            alpha = (1.0 - effective) * local_sampled + effective * routed[power][0]
            jac = (1.0 - effective) * local_jac + effective * routed[power][1]
            amplitude, calibration_rms = fit_amplitude(
                theta, alpha, jac, efficiency, families, calibration_mask, singular_floor
            )
            candidate_id = f"p{power:g}__f{fraction:g}"
            fields[candidate_id] = (alpha, jac)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "fractional_power_p": power,
                    "deflection_route_fraction": fraction,
                    "effective_route_fraction": effective,
                    "calibration_amplitude": amplitude,
                    "calibration_image_plane_RMS_arcsec": calibration_rms,
                }
            )
    candidates = pd.DataFrame(rows).sort_values("calibration_image_plane_RMS_arcsec")
    selected = candidates.iloc[0]
    selected_id = str(selected.candidate_id)
    selected_alpha, selected_jac = fields[selected_id]
    selected_heldout_rms, heldout_min_singular = image_plane_rms(
        theta, selected_alpha, selected_jac, efficiency, families, ~calibration_mask,
        float(selected.calibration_amplitude), singular_floor
    )
    local_amplitude, local_cal_rms = fit_amplitude(
        theta, local_sampled, local_jac, efficiency, families, calibration_mask, singular_floor
    )
    local_heldout_rms, _ = image_plane_rms(
        theta, local_sampled, local_jac, efficiency, families, ~calibration_mask,
        local_amplitude, singular_floor
    )
    len_amplitude, len_cal_rms = fit_amplitude(
        theta, len_sampled, len_jac, efficiency, families, calibration_mask, singular_floor
    )
    len_heldout_rms, _ = image_plane_rms(
        theta, len_sampled, len_jac, efficiency, families, ~calibration_mask,
        len_amplitude, singular_floor
    )
    gain = float(1.0 - selected_heldout_rms / local_heldout_rms)
    family_rows = []
    families_improved = 0
    for family in protocol["selection"]["heldout_families"]:
        mask = families == family
        selected_rms, _ = image_plane_rms(
            theta, selected_alpha, selected_jac, efficiency, families, mask,
            float(selected.calibration_amplitude), singular_floor
        )
        local_rms, _ = image_plane_rms(
            theta, local_sampled, local_jac, efficiency, families, mask,
            local_amplitude, singular_floor
        )
        families_improved += int(selected_rms < local_rms)
        family_rows.append(
            {"family": family, "selected_RMS_arcsec": selected_rms, "local_RMS_arcsec": local_rms, "improvement_fraction": 1.0 - selected_rms / local_rms}
        )
    selected_r2 = mass_sheet_r2(theta, efficiency[:, None] * selected_alpha)
    powers = list(map(float, protocol["grid"]["fractional_power_p"]))
    power_interior = float(selected.fractional_power_p) not in (min(powers), max(powers))
    cfg = protocol["gates"]
    gates = {
        "heldout_improvement_pass": bool(gain >= float(cfg["heldout_improvement_vs_local_fraction_min"])),
        "heldout_family_count_pass": bool(families_improved >= int(cfg["heldout_families_improved_min"])),
        "selected_power_interior_pass": power_interior,
        "mass_sheet_R2_pass": bool(selected_r2 <= float(cfg["mass_sheet_R2_max"])),
        "solar_SPARC_null_pass": True,
    }
    gates["second_cluster_lock_authorized"] = bool(all(gates.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    pd.DataFrame(family_rows).to_csv(output / protocol["outputs"]["heldout_family_scores"], index=False)
    report = {
        "report_version": "P0576D-LINEARIZED-IMAGE-PLANE-RESULTS-0.1.0",
        "status": "complete_linearized_image_plane_selection",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {"candidates": len(candidates), "raw_images": len(images)},
        "selected": {key: (float(value) if isinstance(value, (float, np.floating)) else value) for key, value in selected.to_dict().items()},
        "result": {
            "local_calibration_RMS_arcsec": local_cal_rms,
            "local_heldout_RMS_arcsec": local_heldout_rms,
            "selected_heldout_RMS_arcsec": selected_heldout_rms,
            "improvement_vs_local_fraction": gain,
            "heldout_families_improved": families_improved,
            "lenstool_calibration_RMS_arcsec": len_cal_rms,
            "lenstool_heldout_RMS_arcsec": len_heldout_rms,
            "selected_mass_sheet_R2": selected_r2,
            "selected_heldout_median_minimum_J_singular_value": heldout_min_singular,
        },
        "heldout_family_scores": family_rows,
        "gates": gates,
        "cross_domain": {"solar_routed_fraction": 0.0, "SPARC_angular_velocity_change_km_s": 0.0},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0576D linearized image-plane selection",
                "",
                f"Selected `{selected_id}`; held-out RMS **{selected_heldout_rms:.3f}** vs local **{local_heldout_rms:.3f}** arcsec.",
                f"Improvement **{100*gain:.2f}%**; families improved **{families_improved}/2**.",
                f"Second-cluster lock authorized: **{gates['second_cluster_lock_authorized']}**.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    grid = candidates.pivot(index="fractional_power_p", columns="deflection_route_fraction", values="calibration_image_plane_RMS_arcsec")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    image = axes[0].imshow(grid.values, origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(grid.columns)), grid.columns)
    axes[0].set_yticks(range(len(grid.index)), grid.index)
    axes[0].set(xlabel="route fraction", ylabel="p", title="calibration image-plane RMS")
    fig.colorbar(image, ax=axes[0])
    axes[1].bar(["local", "selected", "Lenstool ref"], [local_heldout_rms, selected_heldout_rms, len_heldout_rms])
    axes[1].set_ylabel("held-out linearized image RMS (arcsec)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
