from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from ppxf import sps_util

try:
    from scripts.reconstruct_m1206_ppxf import (
        _elliptical_coordinates,
        _fit_spectrum,
        _register_center,
        _sha256,
        _source_mask,
        _wavelength,
    )
except ModuleNotFoundError:
    from reconstruct_m1206_ppxf import (
        _elliptical_coordinates,
        _fit_spectrum,
        _register_center,
        _sha256,
        _source_mask,
        _wavelength,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_covariance_protocol.json"


def _display_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _sum_spatial_spectrum(
    data: np.ndarray,
    variance: np.ndarray,
    spatial_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    values = data[:, spatial_mask]
    variances = variance[:, spatial_mask]
    valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0)
    return (
        np.where(valid, values, 0).sum(axis=1),
        np.where(valid, variances, 0).sum(axis=1),
        valid.sum(axis=1),
        int(spatial_mask.sum()),
    )


def _anchored_block_summaries(
    data: np.ndarray,
    variance: np.ndarray,
    spatial_mask: np.ndarray,
    block_shape: tuple[int, int],
    origin: tuple[int, int],
) -> dict[str, np.ndarray]:
    block_height, block_width = block_shape
    origin_y, origin_x = origin
    coordinates = np.argwhere(spatial_mask)
    if not len(coordinates):
        raise ValueError("annulus contains no unmasked spaxels")
    block_y = np.floor_divide(coordinates[:, 0] - origin_y, block_height)
    block_x = np.floor_divide(coordinates[:, 1] - origin_x, block_width)
    keys = np.column_stack([block_y, block_x])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    spectra = []
    variances = []
    valid_counts = []
    member_counts = []
    for block_index in range(len(unique_keys)):
        members = coordinates[inverse == block_index]
        values = data[:, members[:, 0], members[:, 1]]
        errors = variance[:, members[:, 0], members[:, 1]]
        valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        spectra.append(np.where(valid, values, 0).sum(axis=1))
        variances.append(np.where(valid, errors, 0).sum(axis=1))
        valid_counts.append(valid.sum(axis=1))
        member_counts.append(len(members))

    return {
        "keys": unique_keys.astype(int),
        "spectrum": np.asarray(spectra, dtype=float),
        "variance": np.asarray(variances, dtype=float),
        "valid_count": np.asarray(valid_counts, dtype=int),
        "member_count": np.asarray(member_counts, dtype=int),
    }


def _resample_blocks(
    summaries: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    block_count = len(summaries["member_count"])
    draw = rng.integers(0, block_count, size=block_count)
    multiplicity = np.bincount(draw, minlength=block_count)
    return (
        multiplicity @ summaries["spectrum"],
        multiplicity @ summaries["variance"],
        multiplicity @ summaries["valid_count"],
        int(multiplicity @ summaries["member_count"]),
    )


def _fit_aggregated_spectrum(
    spectrum: np.ndarray,
    summed_variance: np.ndarray,
    valid_count: np.ndarray,
    spatial_count: int,
    wavelength: np.ndarray,
    parent: dict,
    template_path: Path,
    galaxy_fwhm: float,
    variance_mask_sigma: float,
    sps: sps_util.sps_lib | None,
) -> tuple[dict, sps_util.sps_lib]:
    wave_range = parent["input"]["observed_wavelength_range_angstrom"]
    enough = valid_count >= (
        parent["spatial_extraction"]["minimum_valid_spaxel_fraction_per_bin"]
        * spatial_count
    )
    selected = (
        (wavelength >= wave_range[0])
        & (wavelength <= wave_range[1])
        & enough
        & np.isfinite(spectrum)
        & np.isfinite(summed_variance)
        & (summed_variance > 0)
    )
    if selected.sum() < 500:
        raise ValueError(f"only {selected.sum()} valid wavelength pixels")
    signal_to_noise = float(
        np.median(spectrum[selected] / np.sqrt(summed_variance[selected]))
    )
    result, sps = _fit_spectrum(
        spectrum[selected],
        summed_variance[selected],
        wavelength[selected],
        parent["input"]["bcg_redshift_initial"],
        template_path,
        galaxy_fwhm,
        parent["spectral_fit"]["additive_polynomial_degree"],
        sps,
        variance_mask_sigma=variance_mask_sigma,
    )
    result.update(
        {
            "spaxels": spatial_count,
            "median_signal_to_noise_per_angstrom": signal_to_noise,
        }
    )
    return result, sps


def _protocols(config: dict) -> list[dict]:
    products = {item["family"]: item for item in config["inputs"]["template_products"]}
    protocols = []
    grid = config["sensitivity_grid"]
    if "protocols" in grid:
        for item in grid["protocols"]:
            protocols.append(
                {
                    **item,
                    "template_path": products[item["template_family"]]["path"],
                    "instrumental_fwhm_angstrom": float(
                        item["instrumental_fwhm_angstrom"]
                    ),
                    "high_variance_mask_sigma": float(
                        item["high_variance_mask_sigma"]
                    ),
                }
            )
        return protocols
    for fwhm in grid["emiles_instrumental_fwhm_angstrom"]:
        for mask_sigma in grid["emiles_high_variance_mask_sigma"]:
            protocols.append(
                {
                    "protocol_id": f"emiles_fwhm{fwhm:.1f}_mask{mask_sigma:.0f}",
                    "template_family": "E-MILES",
                    "template_path": products["E-MILES"]["path"],
                    "instrumental_fwhm_angstrom": float(fwhm),
                    "high_variance_mask_sigma": float(mask_sigma),
                    "baseline": fwhm == 2.6 and mask_sigma == 6.0,
                }
            )
    for run in grid["xsl_runs"]:
        protocols.append(
            {
                "protocol_id": (
                    f"xsl_fwhm{run['instrumental_fwhm_angstrom']:.1f}_"
                    f"mask{run['high_variance_mask_sigma']:.0f}"
                ),
                "template_family": "XSL",
                "template_path": products["XSL"]["path"],
                "instrumental_fwhm_angstrom": float(
                    run["instrumental_fwhm_angstrom"]
                ),
                "high_variance_mask_sigma": float(
                    run["high_variance_mask_sigma"]
                ),
                "baseline": False,
            }
        )
    return protocols


def _construct_covariances(
    bootstrap_vectors: np.ndarray,
    baseline_vector: np.ndarray,
    sensitivity_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bootstrap_covariance = np.cov(bootstrap_vectors, rowvar=False, ddof=1)
    shifts = sensitivity_vectors - baseline_vector[None, :]
    systematic_covariance = np.einsum("ni,nj->ij", shifts, shifts) / len(shifts)
    total_covariance = bootstrap_covariance + systematic_covariance
    return bootstrap_covariance, systematic_covariance, total_covariance


def _load_cube_and_geometry(parent: dict) -> tuple:
    cube_path = ROOT / parent["input"]["cube_path"]
    with fits.open(cube_path, memmap=True, checksum=True) as hdul:
        data = np.asarray(hdul[parent["input"]["data_extension"]].data, dtype=float)
        variance = np.asarray(
            hdul[parent["input"]["variance_extension"]].data, dtype=float
        )
        header = hdul[parent["input"]["data_extension"]].header
        wavelength = _wavelength(header)
        celestial = WCS(header).celestial
        initial_x, initial_y = celestial.world_to_pixel(
            SkyCoord(
                parent["input"]["bcg_center_ra_deg"] * u.deg,
                parent["input"]["bcg_center_dec_deg"] * u.deg,
            )
        )
        white_light = np.nanmedian(
            data[(wavelength >= 6000) & (wavelength <= 7000)], axis=0
        )

    pixel_scale = abs(header["CD2_2"]) * 3600
    spatial = parent["spatial_extraction"]
    registered, registration = _register_center(
        white_light,
        float(initial_x),
        float(initial_y),
        spatial["position_angle_deg_east_of_north"],
        spatial["axis_ratio_b_over_a"],
        pixel_scale,
    )
    y, x = np.mgrid[: white_light.shape[0], : white_light.shape[1]]
    radius, _ = _elliptical_coordinates(
        x,
        y,
        float(registered[0]),
        float(registered[1]),
        pixel_scale,
        spatial["position_angle_deg_east_of_north"],
        spatial["axis_ratio_b_over_a"],
    )
    source_mask, mask_summary = _source_mask(white_light, radius)
    edges = spatial["annulus_semimajor_edges_arcsec"]
    annuli = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        annuli.append(
            (radius >= lower)
            & (radius < upper)
            & ~source_mask
            & np.isfinite(white_light)
        )
    return data, variance, wavelength, annuli, registration, mask_summary


def _write_diagnostic_plot(
    baseline: np.ndarray,
    bootstrap_vectors: np.ndarray,
    sensitivity_vectors: np.ndarray,
    total_covariance: np.ndarray,
    output: Path,
) -> None:
    errors = np.sqrt(np.diag(total_covariance))
    scale = np.sqrt(np.diag(total_covariance))
    correlation = total_covariance / np.outer(scale, scale)
    bins = np.arange(1, len(baseline) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for vector in bootstrap_vectors:
        axes[0].plot(bins, vector, color="0.75", alpha=0.15, linewidth=0.7)
    for vector in sensitivity_vectors:
        axes[0].plot(bins, vector, color="tab:orange", alpha=0.45, linewidth=0.8)
    axes[0].errorbar(
        bins,
        baseline,
        yerr=errors,
        color="black",
        marker="o",
        capsize=3,
        label="baseline + total covariance",
    )
    axes[0].set_xlabel("Frozen annulus bin")
    axes[0].set_ylabel(r"Velocity dispersion (km s$^{-1}$)")
    axes[0].legend(fontsize=8)
    image = axes[1].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    for row in range(len(baseline)):
        for column in range(len(baseline)):
            axes[1].text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    axes[1].set_xticks(range(len(baseline)), bins)
    axes[1].set_yticks(range(len(baseline)), bins)
    axes[1].set_title("Total dispersion correlation")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_covariance(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_profile_protocol"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cube_path = ROOT / config["inputs"]["cube_path"]
    if _sha256(cube_path) != config["inputs"]["cube_sha256"]:
        raise ValueError("cube checksum does not match covariance protocol")
    template_products = config["inputs"]["template_products"]
    for product in template_products:
        path = ROOT / product["path"]
        if _sha256(path) != product["sha256"]:
            raise ValueError(f"template checksum mismatch: {product['family']}")

    baseline_report = json.loads(
        (ROOT / config["inputs"]["baseline_report_path"]).read_text(encoding="utf-8")
    )
    if not baseline_report["gates"]["baseline_internal_consistency_pass"]:
        raise ValueError("baseline profile did not authorize covariance execution")
    baseline_profile = pd.read_csv(ROOT / config["inputs"]["baseline_profile_path"])
    if len(baseline_profile) != 4:
        raise ValueError("frozen baseline profile must contain four bins")

    (
        data,
        variance,
        wavelength,
        annuli,
        registration,
        mask_summary,
    ) = _load_cube_and_geometry(parent)
    if not np.isclose(
        registration["registration_offset_arcsec"],
        baseline_report["registration"]["registration_offset_arcsec"],
        atol=1e-10,
    ):
        raise ValueError("registration changed relative to the frozen baseline")
    if mask_summary != baseline_report["source_mask"]:
        raise ValueError("source mask changed relative to the frozen baseline")

    full_sums = [_sum_spatial_spectrum(data, variance, annulus) for annulus in annuli]
    sensitivity_rows = []
    protocol_vectors = {}
    for protocol in _protocols(config):
        template_path = ROOT / protocol["template_path"]
        sps = None
        values = []
        for bin_index, sums in enumerate(full_sums, start=1):
            row = {
                **protocol,
                "bin": bin_index,
                "status": "success",
                "error": "",
            }
            try:
                result, sps = _fit_aggregated_spectrum(
                    *sums,
                    wavelength,
                    parent,
                    template_path,
                    protocol["instrumental_fwhm_angstrom"],
                    protocol["high_variance_mask_sigma"],
                    sps,
                )
                row.update(result)
                values.append(result["sigma_km_s"])
            except Exception as error:  # keep every failed frozen run visible
                row.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "sigma_km_s": np.nan,
                    }
                )
                values.append(np.nan)
            sensitivity_rows.append(row)
        protocol_vectors[protocol["protocol_id"]] = np.asarray(values, dtype=float)

    baseline_protocol_id = config["sensitivity_grid"].get(
        "baseline_protocol_id", "emiles_fwhm2.6_mask6"
    )
    fitted_baseline = protocol_vectors[baseline_protocol_id]
    frozen_baseline = baseline_profile["sigma_km_s"].to_numpy(dtype=float)
    baseline_reproduction_max_abs_km_s = float(
        np.nanmax(np.abs(fitted_baseline - frozen_baseline))
    )
    for row in sensitivity_rows:
        if np.isfinite(row["sigma_km_s"]):
            bin_index = int(row["bin"]) - 1
            row["shift_from_baseline_km_s"] = (
                row["sigma_km_s"] - fitted_baseline[bin_index]
            )
            row["absolute_fractional_shift_from_baseline"] = abs(
                row["shift_from_baseline_km_s"] / fitted_baseline[bin_index]
            )
        else:
            row["shift_from_baseline_km_s"] = np.nan
            row["absolute_fractional_shift_from_baseline"] = np.nan
    sensitivity = pd.DataFrame(sensitivity_rows)

    bootstrap_config = config["spatial_block_bootstrap"]
    block_shape = tuple(int(value) for value in bootstrap_config["block_shape_spaxels"])
    origin = tuple(int(value) for value in bootstrap_config["block_grid_origin_pixel"])
    block_summaries = [
        _anchored_block_summaries(data, variance, annulus, block_shape, origin)
        for annulus in annuli
    ]
    rng = np.random.default_rng(int(bootstrap_config["random_seed"]))
    spectral = bootstrap_config["spectral_configuration"]
    bootstrap_template = next(
        item
        for item in template_products
        if item["family"] == spectral["template_family"]
    )
    template_path = ROOT / bootstrap_template["path"]
    bootstrap_rows = []
    sps = None
    for replicate in range(1, int(bootstrap_config["replicates"]) + 1):
        for bin_index, summaries in enumerate(block_summaries, start=1):
            row = {
                "replicate": replicate,
                "bin": bin_index,
                "status": "success",
                "error": "",
                "spatial_blocks": int(len(summaries["member_count"])),
            }
            try:
                aggregate = _resample_blocks(summaries, rng)
                result, sps = _fit_aggregated_spectrum(
                    *aggregate,
                    wavelength,
                    parent,
                    template_path,
                    float(spectral["instrumental_fwhm_angstrom"]),
                    float(spectral["high_variance_mask_sigma"]),
                    sps,
                )
                row.update(result)
            except Exception as error:  # keep every failed frozen run visible
                row.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "sigma_km_s": np.nan,
                    }
                )
            bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    bootstrap_matrix = bootstrap.pivot(
        index="replicate", columns="bin", values="sigma_km_s"
    )
    complete_bootstrap = bootstrap_matrix.dropna().to_numpy(dtype=float)
    all_protocols = _protocols(config)
    nonbaseline_protocols = [item for item in all_protocols if not item["baseline"]]
    successful_nonbaseline_protocols = [
        item
        for item in nonbaseline_protocols
        if np.isfinite(protocol_vectors[item["protocol_id"]]).all()
    ]
    covariance_protocols = [
        item
        for item in successful_nonbaseline_protocols
        if item.get("include_in_systematic_covariance", True)
    ]
    sensitivity_vectors = np.asarray(
        [protocol_vectors[item["protocol_id"]] for item in covariance_protocols],
        dtype=float,
    )
    if len(complete_bootstrap) >= 2 and len(sensitivity_vectors) >= 1:
        bootstrap_covariance, systematic_covariance, total_covariance = (
            _construct_covariances(
                complete_bootstrap, fitted_baseline, sensitivity_vectors
            )
        )
    else:
        bootstrap_covariance = np.full((4, 4), np.nan)
        systematic_covariance = np.full((4, 4), np.nan)
        total_covariance = np.full((4, 4), np.nan)

    thresholds = config["success_thresholds"]
    expected_protocols = len(nonbaseline_protocols)
    bootstrap_complete = len(complete_bootstrap) == int(
        thresholds["required_successful_bootstrap_replicates"]
    )
    sensitivity_complete = len(successful_nonbaseline_protocols) == expected_protocols
    systematic_protocol_set_complete = len(sensitivity_vectors) == sum(
        item.get("include_in_systematic_covariance", True)
        for item in nonbaseline_protocols
    )
    covariance_symmetric = bool(
        np.isfinite(total_covariance).all()
        and np.allclose(
            total_covariance,
            total_covariance.T,
            atol=float(thresholds["covariance_symmetric_absolute_tolerance"]),
            rtol=0,
        )
    )
    eigenvalues = (
        np.linalg.eigvalsh(total_covariance)
        if np.isfinite(total_covariance).all()
        else np.full(4, np.nan)
    )
    psd_tolerance = float(
        config["covariance_construction"]["positive_semidefinite_tolerance"]
    )
    covariance_psd = bool(np.isfinite(eigenvalues).all() and eigenvalues.min() >= psd_tolerance)
    total_errors = np.sqrt(np.maximum(np.diag(total_covariance), 0))
    fractional_total_errors = total_errors / fitted_baseline
    fractional_error_pass = bool(
        np.isfinite(fractional_total_errors).all()
        and (
            fractional_total_errors
            <= thresholds["maximum_fractional_total_sigma_uncertainty_each_bin"]
        ).all()
    )
    nonbaseline_sensitivity = sensitivity.loc[~sensitivity["baseline"]]
    maximum_protocol_shift_fraction = float(
        nonbaseline_sensitivity["absolute_fractional_shift_from_baseline"].max()
    )
    protocol_shift_pass = bool(
        sensitivity_complete
        and maximum_protocol_shift_fraction
        <= thresholds["maximum_template_or_mask_shift_fraction_each_bin"]
    )
    baseline_reproduced = baseline_reproduction_max_abs_km_s <= 1e-6
    rest_range = np.asarray(parent["input"]["observed_wavelength_range_angstrom"]) / (
        1 + parent["input"]["bcg_redshift_initial"]
    )
    baseline_protocol = next(
        item for item in all_protocols if item["protocol_id"] == baseline_protocol_id
    )
    baseline_template_product = next(
        item
        for item in template_products
        if item["family"] == baseline_protocol["template_family"]
    )
    template_archive = np.load(ROOT / baseline_template_product["path"])
    template_range = (template_archive["lam"] >= rest_range[0]) & (
        template_archive["lam"] <= rest_range[1]
    )
    maximum_baseline_template_fwhm_rest = float(
        template_archive["fwhm"][template_range].max()
    )
    minimum_baseline_galaxy_fwhm_rest = min(
        item["instrumental_fwhm_angstrom"]
        for item in all_protocols
        if item["template_family"] == baseline_protocol["template_family"]
    ) / (1 + parent["input"]["bcg_redshift_initial"])
    baseline_template_resolution_valid = (
        maximum_baseline_template_fwhm_rest
        < minimum_baseline_galaxy_fwhm_rest
    )
    covariance_gate_pass = all(
        [
            baseline_reproduced,
            bootstrap_complete,
            sensitivity_complete,
            systematic_protocol_set_complete,
            baseline_template_resolution_valid,
            covariance_symmetric,
            covariance_psd,
            fractional_error_pass,
            protocol_shift_pass,
        ]
    )

    outputs = {key: ROOT / value for key, value in config["outputs"].items()}
    for output in outputs.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(outputs["bootstrap_ledger"], index=False)
    sensitivity.to_csv(outputs["sensitivity_ledger"], index=False)

    covariance_rows = []
    correlation = total_covariance / np.outer(total_errors, total_errors)
    for row in range(4):
        for column in range(4):
            covariance_rows.append(
                {
                    "bin_i": row + 1,
                    "bin_j": column + 1,
                    "bootstrap_covariance_km2_s2": bootstrap_covariance[row, column],
                    "systematic_covariance_km2_s2": systematic_covariance[row, column],
                    "total_covariance_km2_s2": total_covariance[row, column],
                    "total_correlation": correlation[row, column],
                }
            )
    pd.DataFrame(covariance_rows).to_csv(
        outputs["covariance_long_table"], index=False
    )
    final_profile = baseline_profile.copy()
    final_profile["sigma_total_error_km_s"] = total_errors
    final_profile["sigma_fractional_total_error"] = fractional_total_errors
    final_profile.to_csv(outputs["profile_with_total_errors"], index=False)
    _write_diagnostic_plot(
        fitted_baseline,
        complete_bootstrap,
        sensitivity_vectors,
        total_covariance,
        outputs["diagnostic_plot"],
    )

    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registration": registration,
        "source_mask": mask_summary,
        "execution": {
            "requested_bootstrap_replicates": int(bootstrap_config["replicates"]),
            "successful_complete_bootstrap_replicates": int(len(complete_bootstrap)),
            "frozen_sensitivity_protocols_including_baseline": len(_protocols(config)),
            "successful_nonbaseline_sensitivity_protocols": len(
                successful_nonbaseline_protocols
            ),
            "required_nonbaseline_sensitivity_protocols": expected_protocols,
            "systematic_covariance_protocols": [
                item["protocol_id"] for item in covariance_protocols
            ],
            "spatial_blocks_per_bin": [
                int(len(summary["member_count"])) for summary in block_summaries
            ],
            "baseline_reproduction_max_abs_km_s": baseline_reproduction_max_abs_km_s,
            "baseline_template_family": baseline_protocol["template_family"],
            "maximum_baseline_template_fwhm_rest_angstrom": maximum_baseline_template_fwhm_rest,
            "minimum_baseline_galaxy_fwhm_rest_angstrom": minimum_baseline_galaxy_fwhm_rest,
        },
        "covariance": {
            "bootstrap_covariance_km2_s2": bootstrap_covariance.tolist(),
            "systematic_covariance_km2_s2": systematic_covariance.tolist(),
            "total_covariance_km2_s2": total_covariance.tolist(),
            "total_correlation": correlation.tolist(),
            "eigenvalues_km2_s2": eigenvalues.tolist(),
            "total_sigma_errors_km_s": total_errors.tolist(),
            "fractional_total_sigma_errors": fractional_total_errors.tolist(),
            "maximum_protocol_shift_fraction": maximum_protocol_shift_fraction,
        },
        "gates": {
            "baseline_reproduced": baseline_reproduced,
            "bootstrap_complete": bootstrap_complete,
            "sensitivity_grid_complete": sensitivity_complete,
            "systematic_protocol_set_complete": systematic_protocol_set_complete,
            "baseline_template_resolution_valid": baseline_template_resolution_valid,
            "covariance_symmetric": covariance_symmetric,
            "covariance_positive_semidefinite": covariance_psd,
            "fractional_total_uncertainty_pass": fractional_error_pass,
            "maximum_protocol_shift_pass": protocol_shift_pass,
            "kinematic_covariance_gate_pass": covariance_gate_pass,
            "strict_r1_ready": False,
        },
        "decision": (
            "Kinematic covariance gate passes. Proceed to baryonic and observable-lens likelihood work; do not fit gravity."
            if covariance_gate_pass
            else "Kinematic covariance gate fails under the frozen protocol. Pause this dynamics branch without tuning the annuli, mask, templates, or thresholds."
        ),
        "outputs": {key: _display_path(path) for key, path in outputs.items()},
    }
    outputs["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_covariance(args.config), indent=2))


if __name__ == "__main__":
    main()
