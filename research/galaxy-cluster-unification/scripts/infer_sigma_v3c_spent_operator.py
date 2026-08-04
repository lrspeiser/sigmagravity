from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0715_sky_lensing_engine_validation import (
    BARYON_MAPS,
    frozen_sky_field,
    glafic_comparator,
)

from voidscreen.sigma_nonlocal_spectral import entire_ir_transfer
from voidscreen.sigma_operator_inference import (
    angular_wavenumber_grid,
    apodization_window,
    normalized_channel_rmse,
    radial_transfer_spectrum,
    transfer_grid_from_spectrum,
    wavelength_band_mask,
    windowed_fourier,
)
from voidscreen.sky_lensing import lens_invariants


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_cluster(cluster: str, config: dict) -> dict[str, object]:
    measurement = config["map_measurement"]
    sample = config["sample"]
    with np.load(BARYON_MAPS / f"{cluster}_baryons.npz") as data:
        lens_redshift = float(data["redshift"])
        center = SkyCoord(
            float(data["center_ra_deg"]) * u.deg,
            float(data["center_dec_deg"]) * u.deg,
        )
    kpc_per_arcsec = float(
        Planck18.kpc_proper_per_arcmin(lens_redshift).value / 60.0
    )
    half_width = float(measurement["half_width_kpc"])
    points = int(measurement["grid_points"])
    axis_kpc = np.linspace(-half_width, half_width, points)
    east_kpc, north_kpc = np.meshgrid(axis_kpc, axis_kpc)
    east_arcsec = east_kpc / kpc_per_arcsec
    north_arcsec = north_kpc / kpc_per_arcsec
    fields = {
        "AQUAL": frozen_sky_field(cluster, lens_redshift, sample["source_model"]),
        "Newtonian": frozen_sky_field(
            cluster, lens_redshift, sample["null_source_model"]
        ),
        "halo": glafic_comparator(cluster, lens_redshift, center),
    }
    invariants = {
        name: lens_invariants(
            field,
            east_arcsec,
            north_arcsec,
            float(sample["source_redshift"]),
            step_arcsec=float(measurement["jacobian_step_arcsec"]),
        )
        for name, field in fields.items()
    }
    channel_names = tuple(measurement["channels"])
    for model, value in invariants.items():
        for channel in channel_names:
            if np.any(~np.isfinite(getattr(value, channel))):
                raise RuntimeError(f"{cluster} {model} {channel} contains nonfinite pixels")
    window = apodization_window(
        (points, points), float(measurement["tukey_alpha"])
    )
    transforms = {
        model: {
            channel: windowed_fourier(getattr(value, channel), window)
            for channel in channel_names
        }
        for model, value in invariants.items()
    }
    spacing_kpc = float(axis_kpc[1] - axis_kpc[0])
    wavenumber = angular_wavenumber_grid((points, points), spacing_kpc)
    band_config = config["spectral_band"]
    band = wavelength_band_mask(
        wavenumber,
        float(band_config["minimum_wavelength_kpc"]),
        float(band_config["maximum_wavelength_kpc"]),
    )
    return {
        "cluster": cluster,
        "lens_redshift": lens_redshift,
        "kpc_per_arcsec": kpc_per_arcsec,
        "axis_kpc": axis_kpc,
        "invariants": invariants,
        "transforms": transforms,
        "wavenumber": wavenumber,
        "band": band,
        "window": window,
    }


def entire_grid(wavenumber: np.ndarray, amplitude: float, length_kpc: float) -> np.ndarray:
    return entire_ir_transfer(np.square(wavenumber * length_kpc), amplitude)


def fit_entire(
    datasets: list[dict[str, object]],
    config: dict,
    *,
    lower_length_override_kpc: float | None = None,
) -> dict[str, float | bool]:
    fit = config["entire_filter_fit"]
    lower_length = (
        float(fit["L_sigma_kpc_min"])
        if lower_length_override_kpc is None
        else float(lower_length_override_kpc)
    )
    bounds = [
        (float(fit["A_min"]), float(fit["A_max"])),
        (math.log(lower_length), math.log(float(fit["L_sigma_kpc_max"]))),
    ]

    def objective(parameters: np.ndarray) -> float:
        amplitude, log_length = parameters
        length = math.exp(float(log_length))
        squared_errors = []
        for data in datasets:
            transfer = entire_grid(data["wavenumber"], float(amplitude), length)
            error = normalized_channel_rmse(
                data["transforms"]["AQUAL"],
                data["transforms"]["halo"],
                transfer,
                data["band"],
            )
            squared_errors.append(error**2)
        return float(np.mean(squared_errors))

    result = differential_evolution(
        objective,
        bounds,
        seed=20260803,
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-9,
    )
    return {
        "A": float(result.x[0]),
        "L_sigma_kpc": float(math.exp(result.x[1])),
        "normalized_RMSE": float(math.sqrt(result.fun)),
        "success": bool(result.success),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer the spent full-map Sigma v3 baryon-to-Weyl operator target."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3c_spent_operator_inference.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3c_spent_operator_inference",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    clusters = [sample_cluster(name, config) for name in config["sample"]["clusters"]]
    args.output.mkdir(parents=True, exist_ok=True)

    spectra: dict[str, pd.DataFrame] = {}
    spectrum_records: list[pd.DataFrame] = []
    for data in clusters:
        spectrum = radial_transfer_spectrum(
            data["transforms"]["AQUAL"],
            data["transforms"]["halo"],
            data["wavenumber"],
            data["band"],
            bins=int(config["spectral_band"]["radial_bins"]),
        )
        spectrum.insert(0, "cluster", data["cluster"])
        spectra[data["cluster"]] = spectrum
        spectrum_records.append(spectrum)
    spectra_frame = pd.concat(spectrum_records, ignore_index=True)
    spectra_frame.to_csv(args.output / "radial_transfer_spectra.csv", index=False)

    fit_rows = []
    per_cluster_fits = {}
    for data in clusters:
        result = fit_entire([data], config)
        per_cluster_fits[data["cluster"]] = result
        fit_rows.append({"fit_scope": data["cluster"], **result})
    joint_fit = fit_entire(clusters, config)
    primary_lower_length = float(config["entire_filter_fit"]["L_sigma_kpc_min"])
    lower_bound_active = bool(
        joint_fit["L_sigma_kpc"] <= 1.001 * primary_lower_length
    )
    post_failure_sensitivity = None
    if lower_bound_active:
        grid_spacing = 2.0 * float(config["map_measurement"]["half_width_kpc"]) / (
            int(config["map_measurement"]["grid_points"]) - 1
        )
        post_failure_sensitivity = fit_entire(
            clusters,
            config,
            lower_length_override_kpc=0.25 * grid_spacing,
        )
    fit_rows.append({"fit_scope": "JOINT", **joint_fit})
    pd.DataFrame.from_records(fit_rows).to_csv(
        args.output / "entire_filter_fits.csv", index=False
    )

    score_records = []
    for index, data in enumerate(clusters):
        other = clusters[1 - index]
        identity = np.ones_like(data["wavenumber"])
        self_transfer = transfer_grid_from_spectrum(
            spectra[data["cluster"]], data["wavenumber"]
        )
        self_positive = transfer_grid_from_spectrum(
            spectra[data["cluster"]],
            data["wavenumber"],
            clip_nonnegative=True,
        )
        cross_transfer = transfer_grid_from_spectrum(
            spectra[other["cluster"]], data["wavenumber"]
        )
        joint_transfer = entire_grid(
            data["wavenumber"], joint_fit["A"], joint_fit["L_sigma_kpc"]
        )
        null_error = normalized_channel_rmse(
            data["transforms"]["Newtonian"],
            data["transforms"]["halo"],
            identity,
            data["band"],
        )
        score_records.append(
            {
                "cluster": data["cluster"],
                "identity_AQUAL_normalized_RMSE": normalized_channel_rmse(
                    data["transforms"]["AQUAL"],
                    data["transforms"]["halo"],
                    identity,
                    data["band"],
                ),
                "identity_Newtonian_normalized_RMSE": null_error,
                "self_binned_real_oracle_normalized_RMSE": normalized_channel_rmse(
                    data["transforms"]["AQUAL"],
                    data["transforms"]["halo"],
                    self_transfer,
                    data["band"],
                ),
                "self_binned_nonnegative_oracle_normalized_RMSE": normalized_channel_rmse(
                    data["transforms"]["AQUAL"],
                    data["transforms"]["halo"],
                    self_positive,
                    data["band"],
                ),
                "cross_cluster_binned_oracle_normalized_RMSE": normalized_channel_rmse(
                    data["transforms"]["AQUAL"],
                    data["transforms"]["halo"],
                    cross_transfer,
                    data["band"],
                ),
                "joint_entire_normalized_RMSE": normalized_channel_rmse(
                    data["transforms"]["AQUAL"],
                    data["transforms"]["halo"],
                    joint_transfer,
                    data["band"],
                ),
                "median_radial_coherence": float(
                    spectra[data["cluster"]].coherence.median()
                ),
                "negative_best_real_transfer_bin_fraction": float(
                    (spectra[data["cluster"]].best_real_transfer < 0.0).mean()
                ),
            }
        )
    scores = pd.DataFrame.from_records(score_records)
    scores.to_csv(args.output / "operator_scores.csv", index=False)

    gates = config["diagnostic_gates"]
    joint_entire_pass = bool(
        joint_fit["normalized_RMSE"]
        <= gates["maximum_joint_entire_normalized_RMSE_for_linear_plausibility"]
    )
    cross_pass = bool(
        (
            scores.cross_cluster_binned_oracle_normalized_RMSE
            <= gates["maximum_cross_cluster_binned_oracle_normalized_RMSE"]
        ).all()
    )
    coherence_pass = bool(
        (
            scores.median_radial_coherence
            >= gates["minimum_median_radial_coherence_for_isotropic_plausibility"]
        ).all()
    )
    nonnegative_pass = bool(
        (
            scores.negative_best_real_transfer_bin_fraction
            <= gates["maximum_negative_best_real_transfer_bin_fraction"]
        ).all()
    )
    isotropic_plausibility = bool(
        joint_entire_pass and cross_pass and coherence_pass and nonnegative_pass
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, data in enumerate(clusters):
        source = data["invariants"]["AQUAL"].convergence
        target = data["invariants"]["halo"].convergence
        maximum = float(np.nanpercentile(target, 99.0))
        axes[0, column].contour(
            data["axis_kpc"],
            data["axis_kpc"],
            target,
            levels=np.linspace(0.15 * maximum, maximum, 6),
            colors="black",
            linewidths=0.8,
        )
        image = axes[0, column].imshow(
            source,
            origin="lower",
            extent=[-350, 350, -350, 350],
            cmap="viridis",
        )
        axes[0, column].set_title(f"{data['cluster']}: AQUAL color, halo contours")
        axes[0, column].set(xlabel="east kpc", ylabel="north kpc")
        figure.colorbar(image, ax=axes[0, column], shrink=0.8)
    for cluster, spectrum in spectra.items():
        axes[1, 0].semilogx(
            spectrum.wavelength_geometric_unit,
            spectrum.best_real_transfer,
            marker="o",
            label=cluster,
        )
    wavelength = np.geomspace(
        config["spectral_band"]["minimum_wavelength_kpc"],
        config["spectral_band"]["maximum_wavelength_kpc"],
        500,
    )
    k = 2.0 * np.pi / wavelength
    axes[1, 0].semilogx(
        wavelength,
        entire_grid(k, joint_fit["A"], joint_fit["L_sigma_kpc"]),
        color="black",
        linewidth=2,
        label="joint entire",
    )
    axes[1, 0].axhline(0.0, color="gray", linewidth=0.8)
    axes[1, 0].set(
        xlabel="wavelength kpc",
        ylabel="best shared real transfer",
        title="Required isotropic transfer by scale",
    )
    axes[1, 0].legend()
    score_plot = scores.set_index("cluster")[[
        "identity_AQUAL_normalized_RMSE",
        "self_binned_real_oracle_normalized_RMSE",
        "cross_cluster_binned_oracle_normalized_RMSE",
        "joint_entire_normalized_RMSE",
    ]].rename(
        columns={
            "identity_AQUAL_normalized_RMSE": "AQUAL identity",
            "self_binned_real_oracle_normalized_RMSE": "same-cluster radial oracle",
            "cross_cluster_binned_oracle_normalized_RMSE": "other-cluster radial transfer",
            "joint_entire_normalized_RMSE": "joint entire filter",
        }
    )
    score_plot.plot.bar(ax=axes[1, 1])
    axes[1, 1].axhline(
        gates["maximum_cross_cluster_binned_oracle_normalized_RMSE"],
        color="black",
        linestyle="--",
    )
    axes[1, 1].set(ylabel="normalized Fourier RMSE", xlabel="", title="Operator transfer tests")
    axes[1, 1].tick_params(axis="x", rotation=0)
    axes[1, 1].legend(fontsize=8, loc="upper right")
    figure.savefig(args.output / "spent_operator_inference.png", dpi=180)
    plt.close(figure)

    report = {
        "status": "completed Sigma v3C spent operator inference",
        "sample_is_spent": True,
        "input_hashes": {
            "config": sha256(args.config),
            **{
                f"{cluster}_baryon_map": sha256(BARYON_MAPS / f"{cluster}_baryons.npz")
                for cluster in config["sample"]["clusters"]
            },
        },
        "joint_entire_fit": joint_fit,
        "post_failure_lower_length_sensitivity": {
            "primary_fit_at_lower_bound": lower_bound_active,
            "interpretation": "not used by the frozen gate",
            "result": post_failure_sensitivity,
        },
        "per_cluster_entire_fits": per_cluster_fits,
        "scores": scores.to_dict(orient="records"),
        "gate_results": {
            "joint_entire_linear_plausibility": joint_entire_pass,
            "cross_cluster_binned_isotropic_transfer": cross_pass,
            "radial_phase_coherence": coherence_pass,
            "nonnegative_real_transfer": nonnegative_pass,
            "all_isotropic_linear_diagnostics": isotropic_plausibility,
        },
        "inferred_action_requirement": (
            "a wavelength-only real linear kernel is adequate on the spent maps; the remaining obstacle is its causal positive completion"
            if isotropic_plausibility
            else "the next nonlinear action must respond to local tensor orientation, component overlap, or baryonic environment in addition to wavelength; a universal real isotropic convolution is insufficient"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(json.dumps(report["joint_entire_fit"], indent=2, sort_keys=True))
    print(scores.to_string(index=False))
    print(report["inferred_action_requirement"])


if __name__ == "__main__":
    main()
