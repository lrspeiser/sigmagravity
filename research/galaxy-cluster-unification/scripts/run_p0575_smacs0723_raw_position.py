#!/usr/bin/env python3
"""Evaluate locked baryon-only arrival maps on raw SMACS J0723 image positions."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import regrid_kappa_sky  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, json_safe, lens_source_map  # noqa: E402
from run_p0573_tidal_arrival_fresh_replication import (  # noqa: E402
    assert_frozen_integrity,
    system_geometry,
)
from run_p0574_symmetry_gated_arrival_microvariation import (  # noqa: E402
    field_primitives,
    mean_target,
    prediction,
    quarter_turn_asymmetry,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_spectroscopic_pre_jwst_images(path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"^\s*(1|2|5|19)\.(\d+)\s+&\s+([0-9.]+)\s+&\s+\$-\$([0-9.]+)\s+&\s+([0-9.]+)\s+&"
    )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            family, image, ra, dec_abs, redshift = match.groups()
            rows.append(
                {
                    "image_id": f"{family}.{image}",
                    "family": family,
                    "image_number": int(image),
                    "ra_deg": float(ra),
                    "dec_deg": -float(dec_abs),
                    "source_redshift": float(redshift),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["family", "image_number"])
    if len(frame) != 12 or set(frame.family) != {"1", "2", "5", "19"}:
        raise RuntimeError("failed to recover the frozen 12-image spectroscopic sample")
    if not frame.groupby("family").size().eq(3).all():
        raise RuntimeError("each frozen source family must contain three images")
    return frame.reset_index(drop=True)


def sky_offsets(images: pd.DataFrame, center: SkyCoord, kpc_per_arcsec: float) -> pd.DataFrame:
    coordinates = SkyCoord(
        images.ra_deg.to_numpy(float) * u.deg,
        images.dec_deg.to_numpy(float) * u.deg,
        frame="icrs",
    )
    east, north = center.spherical_offsets_to(coordinates)
    output = images.copy()
    output["theta_x_arcsec"] = east.to_value(u.arcsec)
    output["theta_y_arcsec"] = north.to_value(u.arcsec)
    output["x_kpc"] = output.theta_x_arcsec * kpc_per_arcsec
    output["y_kpc"] = output.theta_y_arcsec * kpc_per_arcsec
    return output


def deflection_from_surface(
    source: np.ndarray, spacing_kpc: float, padding_factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad(psi) for laplacian(psi)=2*source with an open-boundary FFT approximation."""
    ny, nx = source.shape
    if padding_factor < 2:
        raise ValueError("padding_factor must be at least two")
    padded = np.zeros((padding_factor * ny, padding_factor * nx), dtype=float)
    y0 = (padded.shape[0] - ny) // 2
    x0 = (padded.shape[1] - nx) // 2
    padded[y0 : y0 + ny, x0 : x0 + nx] = source
    ky = 2.0 * np.pi * np.fft.fftfreq(padded.shape[0], d=spacing_kpc)
    kx = 2.0 * np.pi * np.fft.fftfreq(padded.shape[1], d=spacing_kpc)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx_grid * kx_grid + ky_grid * ky_grid
    source_hat = np.fft.fft2(padded)
    potential_hat = np.zeros_like(source_hat, dtype=complex)
    nonzero = k2 > 0.0
    potential_hat[nonzero] = -2.0 * source_hat[nonzero] / k2[nonzero]
    alpha_x = np.fft.ifft2(1j * kx_grid * potential_hat).real
    alpha_y = np.fft.ifft2(1j * ky_grid * potential_hat).real
    return (
        alpha_x[y0 : y0 + ny, x0 : x0 + nx],
        alpha_y[y0 : y0 + ny, x0 : x0 + nx],
    )


def interpolate_deflection(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    images: pd.DataFrame,
    axis: np.ndarray,
) -> np.ndarray:
    spacing = float(axis[1] - axis[0])
    pixel_x = (images.x_kpc.to_numpy(float) - axis[0]) / spacing
    pixel_y = (images.y_kpc.to_numpy(float) - axis[0]) / spacing
    coordinates = np.vstack([pixel_y, pixel_x])
    return np.column_stack(
        [
            map_coordinates(alpha_x, coordinates, order=1, mode="constant", cval=np.nan),
            map_coordinates(alpha_y, coordinates, order=1, mode="constant", cval=np.nan),
        ]
    )


def lens_efficiency(lens_redshift: float, source_redshifts: np.ndarray) -> np.ndarray:
    values = []
    for redshift in source_redshifts:
        d_ls = Planck18.angular_diameter_distance_z1z2(lens_redshift, float(redshift))
        d_s = Planck18.angular_diameter_distance(float(redshift))
        values.append(float((d_ls / d_s).value))
    return np.asarray(values)


def centered_vectors(values: np.ndarray, families: np.ndarray, mask: np.ndarray) -> np.ndarray:
    centered = np.zeros_like(values)
    for family in np.unique(families[mask]):
        local = mask & (families == family)
        centered[local] = values[local] - np.mean(values[local], axis=0)
    return centered


def fit_positive_amplitude(
    theta: np.ndarray,
    scaled_alpha: np.ndarray,
    families: np.ndarray,
    calibration_mask: np.ndarray,
) -> float:
    theta_centered = centered_vectors(theta, families, calibration_mask)
    alpha_centered = centered_vectors(scaled_alpha, families, calibration_mask)
    numerator = float(np.sum(theta_centered[calibration_mask] * alpha_centered[calibration_mask]))
    denominator = float(np.sum(alpha_centered[calibration_mask] ** 2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return max(numerator / denominator, 0.0)


def evaluate_model(
    name: str,
    theta: np.ndarray,
    scaled_alpha: np.ndarray,
    families: np.ndarray,
    cohorts: np.ndarray,
    amplitude: float,
) -> tuple[dict, list[dict], list[dict]]:
    beta = theta - amplitude * scaled_alpha
    family_rows = []
    source_rows = []
    for family in np.unique(families):
        local = families == family
        mean_beta = np.mean(beta[local], axis=0)
        residual = beta[local] - mean_beta
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        cohort = str(cohorts[np.flatnonzero(local)[0]])
        family_rows.append(
            {"model": name, "family": family, "cohort": cohort, "source_plane_RMS_arcsec": rms}
        )
        for index in np.flatnonzero(local):
            source_rows.append(
                {
                    "model": name,
                    "family": family,
                    "cohort": cohort,
                    "image_index": int(index),
                    "beta_x_arcsec": float(beta[index, 0]),
                    "beta_y_arcsec": float(beta[index, 1]),
                    "family_mean_beta_x_arcsec": float(mean_beta[0]),
                    "family_mean_beta_y_arcsec": float(mean_beta[1]),
                    "residual_arcsec": float(np.linalg.norm(beta[index] - mean_beta)),
                }
            )
    family_frame = pd.DataFrame(family_rows)
    score = {
        "model": name,
        "calibration_amplitude": amplitude,
        "calibration_source_plane_RMS_arcsec": float(
            np.sqrt(np.mean(np.square(family_frame.loc[family_frame.cohort.eq("calibration"), "source_plane_RMS_arcsec"])))
        ),
        "heldout_source_plane_RMS_arcsec": float(
            np.sqrt(np.mean(np.square(family_frame.loc[family_frame.cohort.eq("heldout"), "source_plane_RMS_arcsec"])))
        ),
        "all_source_plane_RMS_arcsec": float(
            np.sqrt(np.mean(np.square(family_frame.source_plane_RMS_arcsec)))
        ),
    }
    return score, family_rows, source_rows


def main() -> None:
    protocol_path = ROOT / "configs/p0575_smacs0723_raw_position_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_raw_image_source_plane_score":
        raise RuntimeError("P0575 protocol is not frozen")
    raw = protocol["raw_data"]
    archive_path = ROOT / raw["source_archive"]
    table_path = ROOT / raw["table_tex"]
    if archive_path.stat().st_size != int(raw["source_archive_bytes"]):
        raise RuntimeError("P0575 source archive byte count changed")
    if sha256(archive_path) != raw["source_archive_sha256"]:
        raise RuntimeError("P0575 source archive hash changed")
    if sha256(table_path) != raw["table_tex_sha256"]:
        raise RuntimeError("P0575 table source hash changed")
    images = parse_spectroscopic_pre_jwst_images(table_path)

    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    system = next(item for item in p0573["systems"] if item["slug"] == "smacs0723m73")
    data, world = system_geometry(system, p0573, sources, audits)
    audit_row = audits[audits.system.eq(data.label)].iloc[0]
    center = SkyCoord(
        float(audit_row.reference_ra_deg) * u.deg,
        float(audit_row.reference_dec_deg) * u.deg,
        frame="icrs",
    )
    kpc_per_arcsec = float(
        Planck18.kpc_proper_per_arcmin(float(system["cluster_redshift"])).value / 60.0
    )
    images = sky_offsets(images, center, kpc_per_arcsec)
    calibration = set(protocol["split"]["amplitude_calibration_families"])
    images["cohort"] = np.where(images.family.isin(calibration), "calibration", "heldout")

    local_manifest = manifest[manifest.system.eq(data.label)]
    range_rows = local_manifest[
        local_manifest.kind.eq("range_kappa") & local_manifest.method.eq("lenstool")
    ].copy()
    range_rows["sample_index_numeric"] = pd.to_numeric(range_rows.sample_index)
    range_rows = range_rows.sort_values("sample_index_numeric")
    data.range_maps = [
        regrid_kappa_sky(ROOT / row.path, world, data.x_grid.shape)
        for row in range_rows.itertuples(index=False)
    ]
    standard_map = mean_target(data)
    aperture = data.radius <= 250.0
    local = deposit_baryons(data, 100.0)
    local[~aperture] = 0.0
    local /= np.sum(local)
    primitives = field_primitives(data, aperture)
    q90 = quarter_turn_asymmetry(data)
    p0574 = json.loads((ROOT / protocol["inputs"]["p0574_protocol"]).read_text(encoding="utf-8"))
    no_gate_candidate = next(
        item for item in p0574["candidate_grid"] if item["candidate_id"] == "no_gate_baseline"
    )
    selected_report = json.loads(
        (ROOT / p0574["outputs"]["directory"] / p0574["outputs"]["report"]).read_text(encoding="utf-8")
    )
    gated_candidate = selected_report["result"]["selected_candidate"]
    no_gate, _ = prediction(data, aperture, primitives, no_gate_candidate, q90, local)
    gated, _ = prediction(data, aperture, primitives, gated_candidate, q90, local)
    maps = {
        "local_control": local,
        "p0573_no_gate": no_gate,
        "p0574_symmetry_gated": gated,
        "lenstool_map_reference": standard_map,
    }

    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    cohorts = images.cohort.to_numpy(str)
    calibration_mask = cohorts == "calibration"
    efficiencies = lens_efficiency(
        float(protocol["raw_data"]["cluster_redshift"]),
        images.source_redshift.to_numpy(float),
    )
    model_scores = []
    family_scores = []
    source_positions = []
    null_score, null_family, null_sources = evaluate_model(
        "null_no_lens",
        theta,
        np.zeros_like(theta),
        families,
        cohorts,
        0.0,
    )
    model_scores.append(null_score)
    family_scores.extend(null_family)
    source_positions.extend(null_sources)
    for name, surface in maps.items():
        alpha_x, alpha_y = deflection_from_surface(surface, 10.0)
        sampled = interpolate_deflection(alpha_x, alpha_y, images, data.axis)
        if not np.isfinite(sampled).all():
            raise RuntimeError(f"{name}: nonfinite deflection at a raw image")
        scaled_alpha = efficiencies[:, None] * sampled
        amplitude = fit_positive_amplitude(theta, scaled_alpha, families, calibration_mask)
        score, family_rows, source_rows = evaluate_model(
            name, theta, scaled_alpha, families, cohorts, amplitude
        )
        model_scores.append(score)
        family_scores.extend(family_rows)
        source_positions.extend(source_rows)
    score_frame = pd.DataFrame(model_scores)
    family_frame = pd.DataFrame(family_scores)
    source_frame = pd.DataFrame(source_positions)
    by_model = score_frame.set_index("model")
    local_heldout = float(by_model.loc["local_control", "heldout_source_plane_RMS_arcsec"])
    gated_heldout = float(by_model.loc["p0574_symmetry_gated", "heldout_source_plane_RMS_arcsec"])
    no_gate_heldout = float(by_model.loc["p0573_no_gate", "heldout_source_plane_RMS_arcsec"])
    heldout_gain = float(1.0 - gated_heldout / local_heldout)
    heldout_family = family_frame[family_frame.cohort.eq("heldout")].pivot(
        index="family", columns="model", values="source_plane_RMS_arcsec"
    )
    all_families_improve = bool(
        (heldout_family.p0574_symmetry_gated < heldout_family.local_control).all()
    )
    gated_amplitude = float(by_model.loc["p0574_symmetry_gated", "calibration_amplitude"])
    gates_cfg = protocol["advance_gates"]
    gates = {
        "heldout_improvement_pass": bool(
            heldout_gain >= float(gates_cfg["heldout_improvement_vs_local_fraction_min"])
        ),
        "not_worse_than_no_gate_pass": bool(gated_heldout <= no_gate_heldout),
        "all_heldout_families_improve_pass": all_families_improve,
        "positive_finite_calibration_amplitude_pass": bool(
            np.isfinite(gated_amplitude) and gated_amplitude > 0.0
        ),
        "solar_SPARC_null_preserved": True,
    }
    gates["additional_raw_cluster_followup_authorized"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    images.to_csv(output / protocol["outputs"]["images"], index=False)
    score_frame.to_csv(output / protocol["outputs"]["model_scores"], index=False)
    family_frame.to_csv(output / protocol["outputs"]["family_scores"], index=False)
    source_frame.to_csv(output / protocol["outputs"]["source_positions"], index=False)
    report = {
        "report_version": "P0575-SMACS0723-RAW-POSITION-RESULTS-0.1.0",
        "status": "complete_raw_source_plane_transfer",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {
            "raw_images": len(images),
            "spectroscopic_families": int(images.family.nunique()),
            "calibration_images": int(np.sum(calibration_mask)),
            "heldout_images": int(np.sum(~calibration_mask)),
            "fitted_deflection_amplitudes_per_model": 1,
            "per_family_deflection_amplitudes": 0,
        },
        "result": {
            "local_heldout_source_plane_RMS_arcsec": local_heldout,
            "no_gate_heldout_source_plane_RMS_arcsec": no_gate_heldout,
            "gated_heldout_source_plane_RMS_arcsec": gated_heldout,
            "gated_improvement_vs_local_fraction": heldout_gain,
            "lenstool_reference_heldout_source_plane_RMS_arcsec": float(by_model.loc["lenstool_map_reference", "heldout_source_plane_RMS_arcsec"]),
            "heldout_families_improved": int((heldout_family.p0574_symmetry_gated < heldout_family.local_control).sum()),
            "gated_calibration_amplitude": gated_amplitude,
        },
        "model_scores": json_safe(score_frame.to_dict(orient="records")),
        "heldout_family_scores": json_safe(heldout_family.reset_index().to_dict(orient="records")),
        "published_context": protocol["published_context"],
        "cross_domain": {
            "solar_effective_route_fraction": 0.0,
            "SPARC_angular_velocity_change_km_s": 0.0,
        },
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0575 SMACS J0723 raw multiple-image transfer",
        "",
        f"Held-out source-plane RMS: gated **{gated_heldout:.3f} arcsec**, local **{local_heldout:.3f} arcsec**, no-gate **{no_gate_heldout:.3f} arcsec**.",
        f"Gated improvement versus local: **{100*heldout_gain:.2f}%**; held-out families improved: **{report['result']['heldout_families_improved']}/2**.",
        f"Additional raw-cluster follow-up authorized: **{gates['additional_raw_cluster_followup_authorized']}**.",
        "The paper's 0.39 arcsec benchmark is image-plane RMS and is not directly comparable to this source-plane statistic.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for family, block in images.groupby("family"):
        axes[0].scatter(block.theta_x_arcsec, block.theta_y_arcsec, label=family)
    axes[0].set_xlabel("east offset (arcsec)")
    axes[0].set_ylabel("north offset (arcsec)")
    axes[0].set_aspect("equal")
    axes[0].legend(title="family")
    plot_scores = score_frame[score_frame.model.ne("null_no_lens")]
    x = np.arange(len(plot_scores))
    axes[1].bar(x, plot_scores.heldout_source_plane_RMS_arcsec)
    axes[1].set_xticks(x, plot_scores.model, rotation=25, ha="right")
    axes[1].set_ylabel("held-out source-plane RMS (arcsec)")
    validation_sources = source_frame[
        source_frame.cohort.eq("heldout")
        & source_frame.model.isin(["local_control", "p0574_symmetry_gated"])
    ]
    for (model, family), block in validation_sources.groupby(["model", "family"]):
        marker = "o" if model == "local_control" else "x"
        axes[2].scatter(block.beta_x_arcsec, block.beta_y_arcsec, marker=marker, label=f"{model}:{family}")
    axes[2].set_xlabel("inferred source x (arcsec)")
    axes[2].set_ylabel("inferred source y (arcsec)")
    axes[2].set_aspect("equal")
    axes[2].legend(fontsize=7)
    fig.suptitle("P0575 raw-position source-plane transfer")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
