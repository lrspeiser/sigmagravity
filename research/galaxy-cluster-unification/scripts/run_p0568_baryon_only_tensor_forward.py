#!/usr/bin/env python3
"""Compress the P0567 inverse geometry into baryon-only forward tensors."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import (  # noqa: E402
    deposit_baryons,
    fresh_systems,
    lens_source_map,
)


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
AU_M = 149_597_870_700.0
KPC_M = 3.085677581491367e19
JULIAN_YEAR_DAYS = 365.25
RAD_TO_MAS = 206_264_806.24709636


@dataclass
class Context:
    data: object
    target: np.ndarray
    glafic_target: np.ndarray
    aperture: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def padded_fourier_geometry(shape: tuple[int, int], spacing: float):
    ny, nx = shape
    padded_shape = (2 * ny, 2 * nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(padded_shape[0], d=spacing)
    kx = 2.0 * np.pi * np.fft.fftfreq(padded_shape[1], d=spacing)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    return padded_shape, kx_grid, ky_grid


def pad_center(image: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice]]:
    ny, nx = image.shape
    padded = np.zeros((2 * ny, 2 * nx), dtype=float)
    y0, x0 = ny // 2, nx // 2
    crop = (slice(y0, y0 + ny), slice(x0, x0 + nx))
    padded[crop] = image
    return padded, crop


def baryon_derivatives(source: np.ndarray, spacing: float) -> dict[str, np.ndarray]:
    padded, crop = pad_center(source)
    _, kx, ky = padded_fourier_geometry(source.shape, spacing)
    k2 = kx * kx + ky * ky
    source_hat = np.fft.fft2(padded)
    potential_hat = np.zeros_like(source_hat, dtype=complex)
    keep = k2 > 0.0
    potential_hat[keep] = -source_hat[keep] / k2[keep]

    def inverse(multiplier, field_hat=potential_hat):
        return np.fft.ifft2(multiplier * field_hat).real[crop]

    phi_x = inverse(1j * kx)
    phi_y = inverse(1j * ky)
    hxx = inverse(-(kx**2))
    hxy = inverse(-(kx * ky))
    hyy = inverse(-(ky**2))
    source_x = inverse(1j * kx, source_hat)
    source_y = inverse(1j * ky, source_hat)
    return {
        "phi_x": phi_x,
        "phi_y": phi_y,
        "hxx": hxx,
        "hxy": hxy,
        "hyy": hyy,
        "source_x": source_x,
        "source_y": source_y,
    }


def normalize_tensor(qxx, qxy, qyy):
    qxx = np.asarray(qxx, dtype=float)
    qxy = np.asarray(qxy, dtype=float)
    qyy = np.asarray(qyy, dtype=float)
    middle = 0.5 * (qxx + qyy)
    radius = np.sqrt(np.square(0.5 * (qxx - qyy)) + np.square(qxy))
    spectral = np.maximum(np.abs(middle + radius), np.abs(middle - radius))
    scale = np.maximum(1.0, spectral)
    return qxx / scale, qxy / scale, qyy / scale


def spin2_from_vector(x, y):
    norm2 = x * x + y * y
    floor = max(float(np.percentile(norm2[norm2 > 0.0], 1.0)), np.finfo(float).tiny)
    safe = np.maximum(norm2, floor)
    qxx = (x * x - y * y) / safe
    qxy = 2.0 * x * y / safe
    active = norm2 > floor
    return np.where(active, qxx, 0.0), np.where(active, qxy, 0.0)


def subtract_circular_spin2(
    qxx: np.ndarray,
    qxy: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x: float,
    center_y: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx = x_grid - center_x
    dy = y_grid - center_y
    radius = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)
    radial = qxx * cos2 + qxy * sin2
    cross = qxy * cos2 - qxx * sin2
    bins = np.floor(radius / spacing).astype(int)
    count = np.bincount(bins.ravel())
    radial_sum = np.bincount(bins.ravel(), weights=radial.ravel())
    cross_sum = np.bincount(bins.ravel(), weights=cross.ravel())
    radial_mean = np.divide(radial_sum, count, out=np.zeros_like(radial_sum), where=count > 0)
    cross_mean = np.divide(cross_sum, count, out=np.zeros_like(cross_sum), where=count > 0)
    circular_xx = radial_mean[bins] * cos2 - cross_mean[bins] * sin2
    circular_xy = radial_mean[bins] * sin2 + cross_mean[bins] * cos2
    result_xx = 0.5 * (qxx - circular_xx)
    result_xy = 0.5 * (qxy - circular_xy)
    magnitude = np.hypot(result_xx, result_xy)
    scale = np.maximum(1.0, magnitude)
    return result_xx / scale, result_xy / scale


def taper(radius: np.ndarray, start: float, end: float) -> np.ndarray:
    result = np.ones_like(radius)
    transition = (radius > start) & (radius < end)
    result[radius >= end] = 0.0
    result[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (radius[transition] - start) / (end - start))
    )
    return result


def tensor_map(context: Context, source: np.ndarray, family: str, protocol: dict):
    data = context.data
    spacing = float(protocol["grids"]["grid_spacing_kpc"])
    derivatives = baryon_derivatives(source, spacing)
    field_xx, field_xy = spin2_from_vector(derivatives["phi_x"], derivatives["phi_y"])
    gradient_xx, gradient_xy = spin2_from_vector(
        derivatives["source_x"], derivatives["source_y"]
    )
    tidal_xx = 0.5 * (derivatives["hxx"] - derivatives["hyy"])
    tidal_xy = derivatives["hxy"]
    tidal_norm = np.hypot(tidal_xx, tidal_xy)
    tidal_floor = max(
        float(np.percentile(tidal_norm[tidal_norm > 0.0], 1.0)), np.finfo(float).tiny
    )
    tidal_qxx = np.divide(
        tidal_xx, tidal_norm, out=np.zeros_like(tidal_xx), where=tidal_norm > tidal_floor
    )
    tidal_qxy = np.divide(
        tidal_xy, tidal_norm, out=np.zeros_like(tidal_xy), where=tidal_norm > tidal_floor
    )
    positive = source[(source > 0.0) & (data.radius <= 300.0)]
    b50 = float(np.median(positive))
    density_gate = b50 / (source + b50)
    center_x = float(np.sum(data.positions[:, 0] * data.weights))
    center_y = float(np.sum(data.positions[:, 1] * data.weights))
    non_tidal_xx, non_tidal_xy = subtract_circular_spin2(
        tidal_qxx,
        tidal_qxy,
        data.x_grid,
        data.y_grid,
        center_x,
        center_y,
        spacing,
    )
    non_gradient_xx, non_gradient_xy = subtract_circular_spin2(
        gradient_xx,
        gradient_xy,
        data.x_grid,
        data.y_grid,
        center_x,
        center_y,
        spacing,
    )
    if family == "field_aligned":
        qxx, qxy, qyy = field_xx, field_xy, -field_xx
    elif family == "gradient_aligned":
        qxx, qxy, qyy = gradient_xx, gradient_xy, -gradient_xx
    elif family == "tidal_full":
        qxx, qxy, qyy = tidal_qxx, tidal_qxy, -tidal_qxx
    elif family == "tidal_low_density":
        qxx, qxy, qyy = (
            density_gate * tidal_qxx,
            density_gate * tidal_qxy,
            -density_gate * tidal_qxx,
        )
    elif family == "isotropic_low_density":
        qxx, qxy, qyy = density_gate, np.zeros_like(source), density_gate
    elif family == "tidal_gradient_blend":
        qxx = 0.5 * (tidal_qxx + gradient_xx)
        qxy = 0.5 * (tidal_qxy + gradient_xy)
        qyy = -qxx
    elif family == "noncircular_tidal":
        qxx, qxy, qyy = non_tidal_xx, non_tidal_xy, -non_tidal_xx
    elif family == "noncircular_gradient":
        qxx, qxy, qyy = non_gradient_xx, non_gradient_xy, -non_gradient_xx
    elif family == "noncircular_blend":
        qxx = 0.5 * (non_tidal_xx + non_gradient_xx)
        qxy = 0.5 * (non_tidal_xy + non_gradient_xy)
        qyy = -qxx
    else:
        raise ValueError(f"unknown tensor family: {family}")
    radial_taper = taper(
        data.radius,
        float(protocol["grids"]["tensor_taper_start_kpc"]),
        float(protocol["grids"]["tensor_taper_end_kpc"]),
    )
    return normalize_tensor(qxx * radial_taper, qxy * radial_taper, qyy * radial_taper), derivatives


def correction_map(source, tensor, derivatives, spacing):
    qxx, qxy, qyy = tensor
    flux_x = qxx * derivatives["phi_x"] + qxy * derivatives["phi_y"]
    flux_y = qxy * derivatives["phi_x"] + qyy * derivatives["phi_y"]
    padded_x, crop = pad_center(flux_x)
    padded_y, _ = pad_center(flux_y)
    _, kx, ky = padded_fourier_geometry(source.shape, spacing)
    divergence = np.fft.ifft2(
        1j * (kx * np.fft.fft2(padded_x) + ky * np.fft.fft2(padded_y))
    ).real[crop]
    return -divergence


def prediction(source, correction, coupling, aperture):
    raw = source + float(coupling) * correction
    negative_fraction = float(np.sum(np.maximum(-raw[aperture], 0.0))) / max(
        float(np.sum(np.abs(raw[aperture]))), np.finfo(float).tiny
    )
    result = np.maximum(raw, 0.0)
    result[~aperture] = 0.0
    result /= np.sum(result)
    return result, negative_fraction


def central_prediction(context: Context, width: float):
    data = context.data
    center_x = float(np.sum(data.positions[:, 0] * data.weights))
    center_y = float(np.sum(data.positions[:, 1] * data.weights))
    image = np.exp(-0.5 * ((data.x_grid - center_x) ** 2 + (data.y_grid - center_y) ** 2) / width**2)
    image[~context.aperture] = 0.0
    image /= np.sum(image)
    return image


def metric_row(system, cohort, candidate_id, family, width, coupling, metrics, negative):
    return {
        "system": system,
        "cohort": cohort,
        "candidate_id": candidate_id,
        "family": family,
        "source_width_kpc": width,
        "coupling_t": coupling,
        "negative_raw_fraction": negative,
        **metrics,
    }


def build_contexts(protocol, p0567_protocol):
    systems = fresh_systems(p0567_protocol)
    spacing = float(protocol["grids"]["grid_spacing_kpc"])
    result = []
    for data in systems:
        stack = np.asarray(data.range_maps)
        finite = np.sum(np.isfinite(stack), axis=0)
        mean = np.divide(
            np.nansum(stack, axis=0),
            finite,
            out=np.full_like(stack[0], np.nan),
            where=finite > 0,
        )
        target = lens_source_map(mean, data.radius, spacing, 20.0, (250.0, 300.0))
        glafic = lens_source_map(
            data.glafic_map, data.radius, spacing, 20.0, (250.0, 300.0)
        )
        result.append(
            Context(
                data=data,
                target=target,
                glafic_target=glafic,
                aperture=data.radius <= float(protocol["grids"]["score_radius_kpc"]),
            )
        )
    return result


def equal_system_rmse(frame, prediction):
    residual = prediction - frame.velocity_observed_adjusted_km_s.to_numpy(float)
    point = float(np.sqrt(np.mean(np.square(residual))))
    temporary = pd.DataFrame({"system": frame.galaxy.to_numpy(str), "square": residual**2})
    equal = float(np.sqrt(temporary.groupby("system").square.mean().mean()))
    return point, equal


def galaxy_proxy(frame, coupling, q_radial, a0):
    gbar = frame.g_bar_m_s2.to_numpy(float)
    screen = a0**2 / (a0**2 + gbar**2)
    denominator = 1.0 + float(coupling) * float(q_radial) * screen
    if np.any(denominator <= 0.0):
        return np.full(len(frame), np.nan)
    boost = 1.0 / denominator
    radius_m = frame.radius_adjusted_kpc.to_numpy(float) * KPC_M
    return np.sqrt(gbar * boost * radius_m) / 1000.0


def solar_fraction(radius_m, coupling, q_radial, a0):
    gbar = G_SI * M_SUN_KG / np.square(radius_m)
    screen = a0**2 / (a0**2 + gbar**2)
    return 1.0 / (1.0 + coupling * q_radial * screen) - 1.0


def mercury_precession(coupling, q_radial, a0, points=32768):
    semimajor = 0.38709893 * AU_M
    eccentricity = 0.205630
    period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    radius = semimajor * one_minus_e2 / (1.0 + eccentricity * cosine)
    fraction = solar_fraction(radius, coupling, q_radial, a0)
    perturbation = -(G_SI * M_SUN_KG / radius**2) * fraction
    time_weight = one_minus_e2**1.5 / np.square(1.0 + eccentricity * cosine)
    mean_r_cosine = float(np.mean(perturbation * cosine * time_weight))
    period_seconds = period_days * 86400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = -math.sqrt(one_minus_e2) / (mean_motion * semimajor * eccentricity) * mean_r_cosine
    radians_per_orbit = mean_rate * period_seconds
    orbits_per_century = 100.0 * JULIAN_YEAR_DAYS / period_days
    return radians_per_orbit * orbits_per_century * RAD_TO_MAS


def cross_domain_rows(protocol, family_winners, sparc):
    gates = protocol["advance_gates"]
    a0 = 1.2e-10
    rows = []
    noncircular = {"noncircular_tidal", "noncircular_gradient", "noncircular_blend"}
    for row in family_winners.itertuples(index=False):
        q_radial = 0.0 if row.family in noncircular else 1.0
        predicted = galaxy_proxy(sparc, row.coupling_t, q_radial, a0)
        point_rmse, equal_rmse = equal_system_rmse(sparc, predicted)
        radius = np.geomspace(1.6 * 6.957e8, 8.43 * AU_M, 1000)
        fraction = solar_fraction(radius, row.coupling_t, q_radial, a0)
        maximum = float(np.max(np.abs(fraction)))
        earth = float(np.interp(AU_M, radius, fraction))
        mercury = mercury_precession(row.coupling_t, q_radial, a0)
        rows.append(
            {
                "family": row.family,
                "selected_width_kpc": row.source_width_kpc,
                "selected_coupling_t": row.coupling_t,
                "axisymmetric_radial_eigenvalue_proxy": q_radial,
                "SPARC_outer_RMSE_km_s": point_rmse,
                "SPARC_outer_equal_system_RMSE_km_s": equal_rmse,
                "SPARC_pass": point_rmse <= float(gates["SPARC_outer_RMSE_km_s_max"]),
                "solar_maximum_fractional_change": maximum,
                "Earth_fractional_change": earth,
                "Mercury_precession_mas_per_century": mercury,
                "Cassini_pass": maximum <= float(gates["Cassini_fractional_limit"]),
                "Earth_pass": abs(earth) <= float(gates["Earth_fractional_limit"]),
                "Mercury_pass": abs(mercury) <= float(gates["Mercury_mas_per_century_limit"]),
            }
        )
    return pd.DataFrame(rows)


def axisymmetric_audit(protocol, families):
    size = int(protocol["grids"]["grid_pixels"])
    spacing = float(protocol["grids"]["grid_spacing_kpc"])
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    source = np.exp(-radius / 35.0)
    source[radius > 350.0] = 0.0
    source /= np.sum(source)
    dummy_data = type("Dummy", (), {})()
    dummy_data.axis = axis
    dummy_data.x_grid = xx
    dummy_data.y_grid = yy
    dummy_data.radius = radius
    dummy_data.positions = np.asarray([[0.0, 0.0]])
    dummy_data.weights = np.asarray([1.0])
    dummy = Context(dummy_data, source, source, radius <= 250.0)
    rows = []
    for family in families:
        tensor, derivatives = tensor_map(dummy, source, family, protocol)
        correction = correction_map(source, tensor, derivatives, spacing)
        qxx, qxy, qyy = tensor
        middle = 0.5 * (qxx + qyy)
        eig_radius = np.sqrt(np.square(0.5 * (qxx - qyy)) + np.square(qxy))
        spectral = np.maximum(np.abs(middle + eig_radius), np.abs(middle - eig_radius))
        rows.append(
            {
                "family": family,
                "axisymmetric_tensor_RMS": float(np.sqrt(np.mean((qxx[dummy.aperture] ** 2 + 2*qxy[dummy.aperture] ** 2 + qyy[dummy.aperture] ** 2)))),
                "axisymmetric_correction_RMS_over_source_RMS": float(
                    np.sqrt(np.mean(correction[dummy.aperture] ** 2))
                    / np.sqrt(np.mean(source[dummy.aperture] ** 2))
                ),
                "maximum_tensor_spectral_radius": float(np.max(spectral)),
                "correction_integral_fraction": float(abs(np.sum(correction)) / np.sum(np.abs(correction))),
            }
        )
    return pd.DataFrame(rows)


def make_figure(protocol, contexts, scores, impacts, cross_domain, selected, predictions, output):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    holdout_names = set(protocol["validation"]["locked_holdout_systems"])
    holdouts = [context for context in contexts if context.data.label in holdout_names]
    for column, context in enumerate(holdouts):
        extent = [context.data.axis[0], context.data.axis[-1], context.data.axis[0], context.data.axis[-1]]
        axes[0, column].imshow(context.target, origin="lower", extent=extent, cmap="magma")
        axes[0, column].contour(
            context.data.x_grid,
            context.data.y_grid,
            predictions[context.data.label],
            levels=5,
            colors="cyan",
            linewidths=0.8,
        )
        axes[0, column].set_title(f"{context.data.label}\norange target; cyan prediction", fontsize=9)
        axes[0, column].set_xlim(-300, 300)
        axes[0, column].set_ylim(-300, 300)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
    ordered = impacts.sort_values("development_improvement_vs_best_local_fraction")
    axes[0, 3].barh(ordered.family, 100 * ordered.development_improvement_vs_best_local_fraction)
    axes[0, 3].axvline(0, color="black", lw=1)
    axes[0, 3].set_xlabel("best development JS gain vs local (%)")
    axes[0, 3].tick_params(axis="y", labelsize=7)
    family = selected["family"]
    width = selected["source_width_kpc"]
    curve = scores[(scores.family == family) & (scores.source_width_kpc == width)]
    curve = curve.groupby(["cohort", "coupling_t"]).jensen_shannon.mean().reset_index()
    for cohort, block in curve.groupby("cohort"):
        axes[1, 0].plot(block.coupling_t, block.jensen_shannon, marker="o", label=cohort)
    axes[1, 0].axvline(selected["coupling_t"], color="black", ls="--")
    axes[1, 0].set_xlabel("universal coupling t")
    axes[1, 0].set_ylabel("mean JS")
    axes[1, 0].set_title(f"{family}, width {width:g} kpc")
    axes[1, 0].legend(fontsize=7)
    selected_scores = scores[scores.candidate_id == selected["candidate_id"]]
    piv = selected_scores.pivot(index="system", columns="cohort", values="jensen_shannon")
    axes[1, 1].barh(selected_scores.system, selected_scores.jensen_shannon, color=np.where(selected_scores.cohort.eq("holdout"), "tab:orange", "tab:blue"))
    axes[1, 1].set_xlabel("locked tensor JS")
    axes[1, 1].tick_params(axis="y", labelsize=6)
    axes[1, 2].barh(cross_domain.family, cross_domain.SPARC_outer_RMSE_km_s)
    axes[1, 2].axvline(10.8, color="black", ls="--")
    axes[1, 2].set_xlabel("SPARC outer RMSE (km/s)")
    axes[1, 2].tick_params(axis="y", labelsize=7)
    summary = (
        f"Selected: {selected['family']}\n"
        f"t={selected['coupling_t']:+.2f}, width={selected['source_width_kpc']:.0f} kpc\n"
        f"Development JS={selected['development_mean_JS']:.4f}\n"
        f"Holdout JS={selected['holdout_mean_JS']:.4f}\n"
        f"Holdout gain vs local={100*selected['holdout_improvement_vs_best_local_fraction']:.1f}%\n"
        f"Holdout gain vs central={100*selected['holdout_improvement_vs_central_fraction']:.1f}%"
    )
    axes[1, 3].axis("off")
    axes[1, 3].text(0.02, 0.95, summary, va="top", family="monospace", fontsize=11)
    fig.suptitle("P0568 baryon-only tensor forward screen", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summary(report, output):
    selected = report["selected_tensor"]
    gates = report["gates"]
    lines = [
        "# P0568 baryon-only tensor forward screen",
        "",
        "## Outcome",
        "",
        (
            f"The development-selected formula was `{selected['family']}` with coupling "
            f"`t={selected['coupling_t']:+.2f}` and baryon smoothing {selected['source_width_kpc']:.0f} kpc."
        ),
        (
            f"Its locked holdout Jensen-Shannon divergence was {selected['holdout_mean_JS']:.5f}, "
            f"a {100*selected['holdout_improvement_vs_best_local_fraction']:.2f}% change versus the best local-light null "
            f"and a {100*selected['holdout_improvement_vs_central_fraction']:.2f}% change versus the central-halo null."
        ),
        "",
        "## Gates",
        "",
        f"- Cluster morphology versus both nulls: **{gates['cluster_morphology_gate']}**",
        f"- SPARC outer rotation: **{gates['SPARC_gate']}**",
        f"- Solar proxies: **{gates['solar_gate']}**",
        f"- Overall promotion: **{gates['overall_promotion']}**",
        "",
        "The screen is first-order and uses incomplete member-light baryon maps. Earlier exact raw-image member-tensor failure remains authoritative.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    protocol_path = ROOT / "configs/p0568_baryon_only_tensor_forward_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_new_baryon_tensor_forward_map_score":
        raise RuntimeError("P0568 protocol is not frozen for scoring")
    p0567_path = ROOT / protocol["inputs"]["p0567_protocol"]
    p0567_protocol = json.loads(p0567_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    contexts = build_contexts(protocol, p0567_protocol)
    development = set(protocol["validation"]["development_systems"])
    holdout = set(protocol["validation"]["locked_holdout_systems"])
    families = list(protocol["tensor_families"])
    widths = [float(value) for value in protocol["grids"]["source_smoothing_kpc"]]
    couplings = [float(value) for value in protocol["grids"]["coupling_t"]]
    spacing = float(protocol["grids"]["grid_spacing_kpc"])
    rows = []
    field_cache = {}
    prediction_cache = {}
    audit_rows = []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        for width in widths:
            source = deposit_baryons(context.data, width)
            local_id = f"local_w{int(width):03d}"
            local_metrics = shape_metrics(source, context.target, context.aperture)
            rows.append(metric_row(label, cohort, local_id, "local_identity", width, 0.0, local_metrics, 0.0))
            prediction_cache[(label, local_id)] = source
            for family in families:
                tensor, derivatives = tensor_map(context, source, family, protocol)
                correction = correction_map(source, tensor, derivatives, spacing)
                field_cache[(label, family, width)] = (source, correction)
                qxx, qxy, qyy = tensor
                middle = 0.5 * (qxx + qyy)
                eig_radius = np.sqrt(np.square(0.5 * (qxx - qyy)) + np.square(qxy))
                spectral = np.maximum(np.abs(middle + eig_radius), np.abs(middle - eig_radius))
                audit_rows.append(
                    {
                        "system": label,
                        "family": family,
                        "source_width_kpc": width,
                        "maximum_tensor_spectral_radius": float(np.max(spectral)),
                        "correction_integral_fraction": float(abs(np.sum(correction)) / max(np.sum(np.abs(correction)), np.finfo(float).tiny)),
                    }
                )
                for coupling in couplings:
                    candidate_id = f"{family}_w{int(width):03d}_t{coupling:+.2f}"
                    predicted, negative = prediction(source, correction, coupling, context.aperture)
                    metrics = shape_metrics(predicted, context.target, context.aperture)
                    rows.append(metric_row(label, cohort, candidate_id, family, width, coupling, metrics, negative))
                    prediction_cache[(label, candidate_id)] = predicted
        for width in [float(value) for value in protocol["grids"]["central_null_width_kpc"]]:
            candidate_id = f"central_w{int(width):03d}"
            predicted = central_prediction(context, width)
            metrics = shape_metrics(predicted, context.target, context.aperture)
            rows.append(metric_row(label, cohort, candidate_id, "central_null", width, math.nan, metrics, 0.0))
            prediction_cache[(label, candidate_id)] = predicted
        print(f"built forward candidates for {label}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    audit = pd.DataFrame(audit_rows)
    axis_audit = axisymmetric_audit(protocol, families)
    audit.to_csv(output / protocol["outputs"]["numerical_audit"], index=False)
    candidates = (
        scores.groupby(["candidate_id", "family", "source_width_kpc", "coupling_t"], dropna=False)
        .apply(
            lambda block: pd.Series(
                {
                    "development_mean_JS": block.loc[block.cohort.eq("development"), "jensen_shannon"].mean(),
                    "holdout_mean_JS": block.loc[block.cohort.eq("holdout"), "jensen_shannon"].mean(),
                    "development_mean_Pearson": block.loc[block.cohort.eq("development"), "pearson"].mean(),
                    "holdout_mean_Pearson": block.loc[block.cohort.eq("holdout"), "pearson"].mean(),
                    "maximum_negative_raw_fraction": block.negative_raw_fraction.max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    best_local = candidates[candidates.family.eq("local_identity")].sort_values("development_mean_JS").iloc[0]
    best_central = candidates[candidates.family.eq("central_null")].sort_values("development_mean_JS").iloc[0]
    tensor_candidates = candidates[candidates.family.isin(families)]
    selected = tensor_candidates.sort_values("development_mean_JS").iloc[0]
    family_winners = (
        tensor_candidates.sort_values("development_mean_JS").groupby("family", as_index=False).first()
    )
    family_winners["development_improvement_vs_best_local_fraction"] = 1.0 - family_winners.development_mean_JS / float(best_local.development_mean_JS)
    family_winners["holdout_improvement_vs_best_local_fraction"] = 1.0 - family_winners.holdout_mean_JS / float(best_local.holdout_mean_JS)
    family_winners["holdout_improvement_vs_central_fraction"] = 1.0 - family_winners.holdout_mean_JS / float(best_central.holdout_mean_JS)
    amplitude_spans = []
    for family in families:
        block = tensor_candidates[tensor_candidates.family.eq(family)]
        by_t = block.groupby("coupling_t").development_mean_JS.min()
        amplitude_spans.append({"family": family, "development_JS_amplitude_span": float(by_t.max() - by_t.min())})
    family_winners = family_winners.merge(pd.DataFrame(amplitude_spans), on="family")
    family_winners.to_csv(output / protocol["outputs"]["family_impacts"], index=False)
    selected_dict = selected.to_dict()
    selected_dict["holdout_improvement_vs_best_local_fraction"] = 1.0 - float(selected.holdout_mean_JS) / float(best_local.holdout_mean_JS)
    selected_dict["holdout_improvement_vs_central_fraction"] = 1.0 - float(selected.holdout_mean_JS) / float(best_central.holdout_mean_JS)
    selected_prediction = {context.data.label: prediction_cache[(context.data.label, selected.candidate_id)] for context in contexts}
    uncertainty_rows = []
    method_rows = []
    for context in contexts:
        prediction_selected = selected_prediction[context.data.label]
        for index, raw_map in enumerate(context.data.range_maps):
            target = lens_source_map(raw_map, context.data.radius, spacing, 20.0, (250.0, 300.0))
            metric = shape_metrics(prediction_selected, target, context.aperture)
            uncertainty_rows.append({"system": context.data.label, "cohort": "development" if context.data.label in development else "holdout", "realization": index, **metric})
        for name, row in [("selected_tensor", selected), ("best_local", best_local), ("central_null", best_central)]:
            predicted = prediction_cache[(context.data.label, row.candidate_id)]
            metric = shape_metrics(predicted, context.glafic_target, context.aperture)
            method_rows.append({"system": context.data.label, "cohort": "development" if context.data.label in development else "holdout", "model": name, **metric})
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    pd.DataFrame(method_rows).to_csv(output / "glafic_scores.csv", index=False)
    sparc = pd.read_csv(ROOT / protocol["inputs"]["SPARC_points"])
    sparc = sparc[(sparc.model.eq("fixed_RAR")) & sparc.scenario.eq("invariant") & sparc.split.eq("outer_holdout")].copy()
    cross = cross_domain_rows(protocol, family_winners, sparc)
    cross.to_csv(output / protocol["outputs"]["cross_domain"], index=False)
    selected_cross = cross[cross.family.eq(selected.family)].iloc[0]
    null_prediction = galaxy_proxy(sparc, 0.0, 1.0, 1.2e-10)
    newtonian_rmse, newtonian_equal = equal_system_rmse(sparc, null_prediction)
    glafic = pd.DataFrame(method_rows)
    glafic_holdout = glafic[glafic.cohort.eq("holdout")].groupby("model").jensen_shannon.mean()
    gates = protocol["advance_gates"]
    cluster_gate = bool(
        selected_dict["holdout_improvement_vs_best_local_fraction"] >= float(gates["cluster_holdout_improvement_vs_best_local_fraction"])
        and selected_dict["holdout_improvement_vs_central_fraction"] >= float(gates["cluster_holdout_improvement_vs_central_null_fraction"])
    )
    sparc_gate = bool(selected_cross.SPARC_pass)
    solar_gate = bool(selected_cross.Cassini_pass and selected_cross.Earth_pass and selected_cross.Mercury_pass)
    report = {
        "report_version": "P0568-BARYON-ONLY-TENSOR-FORWARD-RESULTS-0.1.0",
        "status": "complete_baryon_only_forward_screen",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {"clusters": len(contexts), "development_clusters": len(development), "holdout_clusters": len(holdout), "tensor_families": len(families), "tensor_candidates": int(len(tensor_candidates)), "system_candidate_scores": int(len(scores)), "lenstool_uncertainty_scores": int(len(uncertainty)), "SPARC_systems": int(sparc.galaxy.nunique()), "SPARC_points": len(sparc)},
        "comparators": {"best_local": json_safe(best_local.to_dict()), "best_central": json_safe(best_central.to_dict()), "Newtonian_SPARC_outer_RMSE_km_s": newtonian_rmse, "Newtonian_SPARC_outer_equal_system_RMSE_km_s": newtonian_equal, "fixed_RAR_SPARC_outer_RMSE_km_s": 10.348465773189679},
        "selected_tensor": json_safe(selected_dict),
        "selected_cross_domain": json_safe(selected_cross.to_dict()),
        "glafic_holdout_mean_JS": json_safe(glafic_holdout.to_dict()),
        "family_impact_ranking": json_safe(family_winners.sort_values("development_improvement_vs_best_local_fraction", ascending=False).to_dict(orient="records")),
        "axisymmetric_numerical_audit": json_safe(axis_audit.to_dict(orient="records")),
        "gates": {"cluster_morphology_gate": cluster_gate, "SPARC_gate": sparc_gate, "solar_gate": solar_gate, "overall_promotion": bool(cluster_gate and sparc_gate and solar_gate), "no_per_cluster_gravity_parameters": True},
        "interpretation": {"strongest_formula_change": str(family_winners.sort_values("development_JS_amplitude_span", ascending=False).iloc[0].family), "meaning": "Impact is a change in normalized standard-model convergence morphology, not absolute lens strength or evidence against dark matter.", "raw_lensing_authority": "The earlier member-only exact image-position tensor test selected zero coupling and remains the stronger lensing result."},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    write_summary(report, output / protocol["outputs"]["summary"])
    make_figure(protocol, contexts, scores, family_winners, cross, selected_dict, selected_prediction, output / protocol["outputs"]["figure"])
    print(json.dumps(report["selected_tensor"], indent=2), flush=True)
    print(json.dumps(report["selected_cross_domain"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
