#!/usr/bin/env python3
"""Fit preregistered relative Fe-K shifts in frozen A2319 detector regions."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import chi2, norm, spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_relative_fe_line_shift.json"
LIGHT_KMS = 299792.458
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-RELATIVE-FE-LINE-SHIFT-1.0.1"
    ):
        raise RuntimeError("unexpected relative-line protocol")
    if config.get("status") != (
        "gate-only correction frozen after version 1.0.0 completed with a false "
        "terminal gate: the implementation checked every fit was inside the shift "
        "bounds but omitted the already-frozen optimizer-convergence requirement; "
        "version 1.0.1 requires both without changing any energy selection, model, "
        "starting value, bound, uncertainty, benchmark, threshold, fitted result, "
        "or failed scientific decision"
    ):
        raise RuntimeError("relative-line protocol is not frozen")
    parent_path = ROOT / config["parents"]["readiness_report"]
    if not parent_path.is_file() or sha256(parent_path) != config["parents"][
        "readiness_report_sha256"
    ]:
        raise RuntimeError("frozen region-readiness parent changed")
    parent = load_json(parent_path)
    if not parent.get("terminal_gate_passed"):
        raise RuntimeError("region-readiness parent did not pass")
    if parent.get("energy_column_or_distribution_read"):
        raise RuntimeError("region-readiness parent crossed the energy boundary")
    if parent.get("validation_or_holdout_accessed"):
        raise RuntimeError("region-readiness parent opened sealed data")
    for key in (
        "generate_response_or_background",
        "fit_bapec_or_claim_absolute_velocity",
        "construct_ssm_sky_region_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if config["authorization"][key]:
            raise RuntimeError(f"sealed relative-line boundary is open: {key}")
    if len(config["regions"]) != 7 or len({item["name"] for item in config["regions"]}) != 7:
        raise RuntimeError("relative-line protocol does not name seven unique regions")
    return config, parent


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    if np.any(expected <= 0) or not np.isfinite(expected).all():
        return math.inf
    positive = observed > 0
    value = 2.0 * np.sum(expected - observed)
    value += 2.0 * np.sum(observed[positive] * np.log(observed[positive] / expected[positive]))
    return float(value)


def line_template(
    other_counts: np.ndarray, bin_width_eV: float, model: dict[str, Any]
) -> np.ndarray:
    narrow = gaussian_filter1d(
        np.asarray(other_counts, dtype=float),
        float(model["narrow_smoothing_sigma_eV"]) / bin_width_eV,
        mode="nearest",
    )
    broad = gaussian_filter1d(
        np.asarray(other_counts, dtype=float),
        float(model["broad_continuum_sigma_eV"]) / bin_width_eV,
        mode="nearest",
    )
    line = np.clip(narrow - broad, 0.0, None)
    total = float(np.sum(line))
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("leave-one-out spectrum has no positive line template")
    return line / total


def fit_one_shift(
    observed: np.ndarray,
    other_counts: np.ndarray,
    centers: np.ndarray,
    window: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    bin_width = float(window["bin_width_eV"])
    template = line_template(other_counts, bin_width, model)
    x = 2.0 * (centers - centers[0]) / (centers[-1] - centers[0]) - 1.0
    shift_min, shift_max = map(float, model["shift_bounds_eV"])

    def objective(parameters: np.ndarray) -> float:
        shift, log_amplitude, log_continuum, slope = parameters
        shifted = np.interp(centers - shift, centers, template, left=0.0, right=0.0)
        expected = np.exp(log_amplitude) * shifted + np.exp(log_continuum + slope * x)
        return poisson_deviance(observed, expected)

    starts = model["initial_shifts_eV"]
    line_counts = max(float(np.sum(observed)) * 0.4, 1.0)
    continuum = max(float(np.mean(observed)) * 0.6, 1e-3)
    bounds = [
        (shift_min, shift_max),
        (math.log(1e-3), math.log(1e9)),
        (math.log(1e-6), math.log(1e6)),
        tuple(map(float, model["continuum_log_slope_bounds"])),
    ]
    attempts = []
    for start in starts:
        result = minimize(
            objective,
            np.array([float(start), math.log(line_counts), math.log(continuum), 0.0]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-11},
        )
        attempts.append(result)
    finite = [item for item in attempts if math.isfinite(float(item.fun))]
    if not finite:
        raise RuntimeError("all relative-shift optimizer starts failed")
    best = min(finite, key=lambda item: float(item.fun))
    shift = float(best.x[0])
    inside = shift_min + 1e-3 < shift < shift_max - 1e-3
    return {
        "shift_eV": shift,
        "poisson_deviance": float(best.fun),
        "optimizer_success": bool(best.success),
        "inside_shift_bounds": inside,
        "amplitude": float(np.exp(best.x[1])),
        "continuum_at_center": float(np.exp(best.x[2])),
        "continuum_log_slope": float(best.x[3]),
        "message": str(best.message),
    }


def fit_dataset(
    histograms: dict[str, np.ndarray],
    centers: np.ndarray,
    window: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    names = list(histograms)
    total = np.sum([histograms[name] for name in names], axis=0)
    fits = {}
    for name in names:
        fits[name] = fit_one_shift(
            histograms[name], total - histograms[name], centers, window, model
        )
    velocities = {
        name: -LIGHT_KMS * item["shift_eV"] / float(window["reference_energy_eV"])
        for name, item in fits.items()
    }
    mean_velocity = float(np.mean(list(velocities.values())))
    for name, item in fits.items():
        item["velocity_raw_kms"] = velocities[name]
        item["velocity_relative_unweighted_mean_kms"] = velocities[name] - mean_velocity
    return fits


def load_region_energies(
    config: dict[str, Any], parent: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    scratch = (ROOT / config["paths"]["readiness_scratch_root"]).resolve()
    expected = {
        (branch["branch"], region["region"]): region
        for branch in parent["branches"]
        for region in branch["regions"]
    }
    energies: dict[str, list[np.ndarray]] = {item["name"]: [] for item in config["regions"]}
    sources = []
    column = config["energy_column"]
    for region in config["regions"]:
        for branch in region["branches"]:
            record = expected.get((branch, region["name"]))
            if record is None:
                raise RuntimeError(f"readiness report lacks {branch}/{region['name']}")
            path = scratch / branch / f"region_{region['name']}.evt"
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"readiness region output changed: {path}")
            with fits.open(path, memmap=True, mode="readonly") as hdus:
                values = np.asarray(hdus["EVENTS"].data[column], dtype=float).copy()
            if len(values) != int(record["rows"]) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid energy vector: {branch}/{region['name']}")
            energies[region["name"]].append(values)
            sources.append(
                {
                    "pointing": region["pointing"],
                    "region": region["name"],
                    "branch": branch,
                    "rows": len(values),
                    "sha256": record["sha256"],
                }
            )
    return {name: np.concatenate(parts) for name, parts in energies.items()}, sources


def histograms_for_window(
    energies: dict[str, np.ndarray], window: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    lo = float(window["minimum_eV"])
    hi = float(window["maximum_eV"])
    width = float(window["bin_width_eV"])
    edges = np.arange(lo, hi + width * 0.5, width)
    if len(edges) < 3 or abs(edges[-1] - hi) > 1e-8:
        raise RuntimeError("energy window is not an exact bin-width multiple")
    histograms = {
        name: np.histogram(values[(values >= lo) & (values < hi)], bins=edges)[0].astype(float)
        for name, values in energies.items()
    }
    return histograms, 0.5 * (edges[:-1] + edges[1:])


def bootstrap_primary(
    histograms: dict[str, np.ndarray],
    centers: np.ndarray,
    window: dict[str, Any],
    model: dict[str, Any],
    uncertainty: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], int]:
    rng = np.random.default_rng(int(uncertainty["seed"]))
    draws = int(uncertainty["draws"])
    samples = {name: [] for name in histograms}
    successes = 0
    for _ in range(draws):
        simulated = {name: rng.poisson(values).astype(float) for name, values in histograms.items()}
        try:
            fitted = fit_dataset(simulated, centers, window, model)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if not all(item["inside_shift_bounds"] for item in fitted.values()):
            continue
        successes += 1
        for name, item in fitted.items():
            samples[name].append(item["velocity_relative_unweighted_mean_kms"])
    result = {}
    for name, values in samples.items():
        if not values:
            result[name] = {"lower_kms": math.nan, "upper_kms": math.nan, "sigma_kms": math.nan}
            continue
        array = np.asarray(values, dtype=float)
        result[name] = {
            "lower_kms": float(np.quantile(array, 0.16)),
            "upper_kms": float(np.quantile(array, 0.84)),
            "sigma_kms": float(np.std(array, ddof=1)) if len(array) > 1 else math.nan,
        }
    return result, successes


def comparison_metrics(
    fitted: dict[str, dict[str, Any]],
    benchmark: dict[str, float],
    total_uncertainties: dict[str, float],
) -> dict[str, Any]:
    names = list(fitted)
    candidate = np.asarray(
        [fitted[name]["velocity_relative_unweighted_mean_kms"] for name in names]
    )
    published = np.asarray([float(benchmark[name]) for name in names])
    published_centered = published - np.mean(published)
    rho = float(spearmanr(candidate, published_centered).statistic)
    sign_agreement = float(np.mean(np.sign(candidate) == np.sign(published_centered)))
    uncertainties = np.asarray([total_uncertainties[name] for name in names])
    constant_chi2 = float(np.sum((candidate / uncertainties) ** 2))
    p_value = float(chi2.sf(constant_chi2, len(names) - 1))
    sigma = float(norm.isf(p_value / 2.0)) if p_value > 0 else math.inf
    return {
        "region_order": names,
        "published_centered_kms": {
            name: float(value) for name, value in zip(names, published_centered, strict=True)
        },
        "centered_rmse_vs_published_kms": float(
            np.sqrt(np.mean((candidate - published_centered) ** 2))
        ),
        "spearman_vs_published": rho,
        "centered_sign_agreement_fraction": sign_agreement,
        "most_blueshifted_region": names[int(np.argmin(candidate))],
        "most_redshifted_region": names[int(np.argmax(candidate))],
        "velocity_range_kms": float(np.max(candidate) - np.min(candidate)),
        "constant_velocity_chi2": constant_chi2,
        "constant_velocity_dof": len(names) - 1,
        "constant_velocity_p_value": p_value,
        "constant_velocity_rejection_sigma": sigma,
    }


def write_plot(
    path: Path,
    windows: dict[str, Any],
    benchmark_centered: dict[str, float],
    primary_errors: dict[str, float],
) -> None:
    names = list(benchmark_centered)
    labels = [name.replace("_prime", "′") for name in names]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, (window_name, record) in zip(axes, windows.items(), strict=True):
        values = [record["fits"][name]["velocity_relative_unweighted_mean_kms"] for name in names]
        errors = [primary_errors[name] for name in names] if window_name == "primary_fe_k" else None
        axis.axhline(0.0, color="0.75", linewidth=1)
        axis.plot(x, [benchmark_centered[name] for name in names], "s--", color="0.45", label="Published, centered")
        axis.errorbar(x, values, yerr=errors, fmt="o-", capsize=3, label="Reduced-data diagnostic")
        axis.set_title(window_name.replace("_", " "))
        axis.set_xticks(x, labels, rotation=45)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Relative line-of-sight velocity (km/s)")
    axes[0].legend(fontsize=8)
    figure.suptitle("A2319 frozen detector regions: empirical Fe-K relative shifts")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, parent = validate_inputs(config_path)
    energies, sources = load_region_energies(config, parent)
    window_records = {}
    histograms_by_window = {}
    centers_by_window = {}
    for name, window in config["windows"].items():
        histograms, centers = histograms_for_window(energies, window)
        fits = fit_dataset(histograms, centers, window, config["empirical_model"])
        window_records[name] = {
            "event_counts": {region: int(np.sum(values)) for region, values in histograms.items()},
            "fits": fits,
        }
        histograms_by_window[name] = histograms
        centers_by_window[name] = centers

    primary_name = "primary_fe_k"
    primary = window_records[primary_name]
    bootstrap, successes = bootstrap_primary(
        histograms_by_window[primary_name],
        centers_by_window[primary_name],
        config["windows"][primary_name],
        config["empirical_model"],
        config["uncertainty"],
    )
    pointing_by_region = {item["name"]: item["pointing"] for item in config["regions"]}
    systematic = config["uncertainty"]["pointing_energy_systematic_eV"]
    reference = float(config["windows"][primary_name]["reference_energy_eV"])
    total_uncertainties = {}
    for region, interval in bootstrap.items():
        systematic_kms = LIGHT_KMS * float(systematic[pointing_by_region[region]]) / reference
        stat_kms = float(interval["sigma_kms"])
        interval["pointing_energy_systematic_kms"] = systematic_kms
        interval["total_uncertainty_kms"] = math.sqrt(stat_kms**2 + systematic_kms**2)
        total_uncertainties[region] = interval["total_uncertainty_kms"]

    metrics = comparison_metrics(
        primary["fits"],
        config["published_no_ssm_benchmark_kms"],
        total_uncertainties,
    )
    gate_config = config["terminal_gate"]
    bootstrap_fraction = successes / int(config["uncertainty"]["draws"])
    gates = {
        "minimum_primary_window_events": all(
            count >= int(gate_config["minimum_primary_window_events_per_region"])
            for count in primary["event_counts"].values()
        ),
        "all_window_fits_converged_inside_shift_bounds": all(
            fit["optimizer_success"] and fit["inside_shift_bounds"]
            for record in window_records.values()
            for fit in record["fits"].values()
        ),
        "bootstrap_success_fraction": bootstrap_fraction
        >= float(gate_config["minimum_bootstrap_success_fraction"]),
        "centered_rmse": metrics["centered_rmse_vs_published_kms"]
        <= float(gate_config["maximum_centered_rmse_vs_published_kms"]),
        "spearman": metrics["spearman_vs_published"]
        >= float(gate_config["minimum_spearman_vs_published"]),
        "sign_agreement": metrics["centered_sign_agreement_fraction"]
        >= float(gate_config["minimum_centered_sign_agreement_fraction"]),
        "most_blueshifted": metrics["most_blueshifted_region"]
        == gate_config["required_most_blueshifted_region"],
        "most_redshifted": metrics["most_redshifted_region"]
        == gate_config["required_most_redshifted_region"],
        "constant_velocity_rejection": metrics["constant_velocity_rejection_sigma"]
        >= float(gate_config["minimum_constant_velocity_rejection_sigma"]),
        "maximum_total_uncertainty": all(
            value <= float(gate_config["maximum_total_uncertainty_kms"])
            for value in total_uncertainties.values()
        ),
    }
    passed = all(gates.values())
    plot_path = ROOT / config["paths"]["plot"]
    write_plot(
        plot_path,
        window_records,
        metrics["published_centered_kms"],
        total_uncertainties,
    )
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_relative_fe_line_shift_diagnostic_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "parent_readiness_report_sha256": sha256(
            ROOT / config["parents"]["readiness_report"]
        ),
        "sources": sources,
        "windows": window_records,
        "primary_bootstrap": {
            "requested_draws": int(config["uncertainty"]["draws"]),
            "successful_draws": successes,
            "success_fraction": bootstrap_fraction,
            "regions": bootstrap,
        },
        "primary_comparison": metrics,
        "gates": gates,
        "terminal_gate_passed": passed,
        "energy_distribution_read_and_fit": True,
        "response_or_background_generated": False,
        "bapec_or_absolute_velocity_fit": False,
        "ssm_sky_region_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "plot": {
            "path": config["paths"]["plot"],
            "bytes": plot_path.stat().st_size,
            "sha256": sha256(plot_path),
        },
        "decision": (
            "authorize_full_response_aware_detector_region_spectral_protocol"
            if passed
            else "do_not_authorize_response_production_until_relative_shift_failure_is_diagnosed"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / config["paths"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_comparison": result["primary_comparison"],
                "primary_bootstrap": result["primary_bootstrap"],
                "gates": result["gates"],
                "terminal_gate_passed": result["terminal_gate_passed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
