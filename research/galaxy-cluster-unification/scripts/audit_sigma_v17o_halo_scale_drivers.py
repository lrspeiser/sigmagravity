"""Audit baryonic predictors of diagnostic NFW scale radii across domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[1]
G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19
A0_SI = 1.2e-10
H0_KM_S_MPC = 67.4
OMEGA_M = 0.315


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_fold(domain: str, system: str, folds: int) -> int:
    digest = hashlib.sha256(f"{domain}:{system}".encode()).hexdigest()
    return int(digest[:16], 16) % folds


def hubble_si(redshift: float) -> float:
    h0_si = H0_KM_S_MPC * 1000.0 / (1000.0 * KPC_M)
    return h0_si * math.sqrt(OMEGA_M * (1.0 + redshift) ** 3 + 1.0 - OMEGA_M)


def nfw_profile(
    radius_kpc: np.ndarray, log10_m200_msun: float, concentration: float, redshift: float
) -> tuple[np.ndarray, float, float]:
    mass_kg = 10.0**log10_m200_msun * M_SUN_KG
    rho_critical = 3.0 * hubble_si(redshift) ** 2 / (8.0 * math.pi * G_SI)
    r200_m = (3.0 * mass_kg / (4.0 * math.pi * 200.0 * rho_critical)) ** (1.0 / 3.0)
    r200_kpc = r200_m / KPC_M
    scale_kpc = r200_kpc / concentration
    x = np.asarray(radius_kpc, dtype=float) / scale_kpc
    normalization = math.log1p(concentration) - concentration / (1.0 + concentration)
    enclosed_fraction = (np.log1p(x) - x / (1.0 + x)) / normalization
    acceleration = G_SI * mass_kg * enclosed_fraction / (np.asarray(radius_kpc) * KPC_M) ** 2
    return acceleration, r200_kpc, scale_kpc


def fit_cluster_nfw(block: pd.DataFrame, redshift: float) -> dict[str, float | bool]:
    radius = block.radius_kpc.to_numpy(float)
    observed = block.log_g_total.to_numpy(float)
    error = np.maximum(block.err_log_g_total.to_numpy(float), 1e-3)

    def residual(theta: np.ndarray) -> np.ndarray:
        acceleration, _, _ = nfw_profile(radius, theta[0], math.exp(theta[1]), redshift)
        return (np.log10(acceleration) - observed) / error

    result = least_squares(
        residual,
        x0=np.array([15.0, math.log(4.0)]),
        bounds=(np.array([13.0, math.log(0.5)]), np.array([16.5, math.log(20.0)])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=10000,
    )
    concentration = math.exp(float(result.x[1]))
    predicted, r200_kpc, scale_kpc = nfw_profile(
        radius, float(result.x[0]), concentration, redshift
    )
    rmse = float(np.sqrt(np.mean((np.log10(predicted) - observed) ** 2)))
    at_boundary = bool(
        abs(result.x[0] - 13.0) < 1e-4
        or abs(result.x[0] - 16.5) < 1e-4
        or abs(result.x[1] - math.log(0.5)) < 1e-4
        or abs(result.x[1] - math.log(20.0)) < 1e-4
    )
    return {
        "fit_success": bool(result.success),
        "fit_at_boundary": at_boundary,
        "fit_rmse_dex": rmse,
        "m200_msun": 10.0 ** float(result.x[0]),
        "concentration": concentration,
        "r200_kpc": r200_kpc,
        "halo_scale_kpc": scale_kpc,
    }


def exponential_cumulative(radius: float, scale: float) -> float:
    x = radius / max(scale, 1e-9)
    return 1.0 - math.exp(-x) * (1.0 + x)


def galaxy_half_mass_radius(row: pd.Series) -> float:
    disk_mass = max(float(row["catalog__disk_mass_solar"]), 0.0)
    bulge_mass = max(float(row["catalog__bulge_mass_solar"]), 0.0)
    gas_mass = max(float(row["catalog__gas_mass_solar"]), 0.0)
    total = disk_mass + bulge_mass + gas_mass
    if not total > 0.0:
        return math.nan
    disk_scale = float(row["catalog__disk_scale_kpc"])
    if not math.isfinite(disk_scale) or disk_scale <= 0.0:
        disk_scale = float(row["catalog__effective_radius_kpc"]) / 1.678
    gas_radius = float(row["catalog__HI_radius_kpc"])
    gas_scale = gas_radius / 3.2 if math.isfinite(gas_radius) and gas_radius > 0.0 else disk_scale
    bulge_scale = float(row["catalog__bulge_scale_fit_kpc"])
    if not math.isfinite(bulge_scale) or bulge_scale <= 0.0:
        bulge_scale = max(0.2 * disk_scale, 1e-3)

    def enclosed(radius: float) -> float:
        disk = disk_mass * exponential_cumulative(radius, disk_scale)
        gas = gas_mass * exponential_cumulative(radius, gas_scale)
        bulge = bulge_mass * radius**2 / (radius + bulge_scale) ** 2
        return disk + gas + bulge - 0.5 * total

    upper = 100.0 * max(disk_scale, gas_scale, bulge_scale, 0.1)
    return float(brentq(enclosed, 1e-8, upper))


def monotonic_half_radius(radius: np.ndarray, enclosed_mass: np.ndarray) -> float:
    order = np.argsort(radius)
    r = np.asarray(radius, dtype=float)[order]
    mass = np.maximum.accumulate(np.asarray(enclosed_mass, dtype=float)[order])
    target = 0.5 * mass[-1]
    if target <= mass[0]:
        return float(r[0] * max(target / mass[0], 0.1))
    unique_mass, indices = np.unique(mass, return_index=True)
    unique_radius = r[indices]
    return float(np.interp(target, unique_mass, unique_radius))


def load_cluster_redshifts(path: Path) -> dict[str, float]:
    redshifts: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3:
            redshifts[parts[-1]] = float(parts[1])
    return redshifts


def build_object_table(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = config["inputs"]
    fits = pd.read_csv(ROOT / inputs["galaxy_nfw_fits"]["path"])
    fits = fits[(fits.model == "NFW") & (fits.scenario == "invariant")].copy()
    features = pd.read_csv(ROOT / inputs["galaxy_baryonic_features"]["path"])
    galaxy = fits.merge(features, on="galaxy", how="inner", validate="one_to_one")
    hubble_km_s_kpc = H0_KM_S_MPC / 1000.0
    galaxy["r200_kpc"] = galaxy.nfw_V200_km_s / (10.0 * hubble_km_s_kpc)
    galaxy["halo_scale_kpc"] = galaxy.r200_kpc / galaxy.nfw_concentration
    galaxy["baryonic_mass_msun"] = galaxy["catalog__baryonic_mass_solar"]
    galaxy["baryonic_half_mass_radius_kpc"] = galaxy.apply(galaxy_half_mass_radius, axis=1)
    galaxy["domain"] = "galaxy"
    galaxy["system"] = galaxy.galaxy
    galaxy["relaxed_target_quality_pass"] = (
        galaxy.optimizer_success.astype(bool)
        & galaxy.finite_fit.astype(bool)
        & np.isfinite(galaxy.halo_scale_kpc)
        & (galaxy.halo_scale_kpc > 0.0)
    )
    galaxy["target_quality_pass"] = (
        galaxy.relaxed_target_quality_pass & ~galaxy.any_parameter_at_boundary.astype(bool)
    )

    profile = pd.read_csv(
        ROOT / inputs["cluster_profiles"]["path"],
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_g_bar",
            "log_g_total",
            "err_log_g_bar",
            "err_log_g_total",
        ],
    )
    redshifts = load_cluster_redshifts(ROOT / inputs["cluster_metadata"]["path"])
    cluster_rows = []
    for system, block in profile.groupby("system", sort=True):
        fit = fit_cluster_nfw(block, redshifts[system])
        radius_m = block.radius_kpc.to_numpy(float) * KPC_M
        mass = 10.0 ** block.log_g_bar.to_numpy(float) * radius_m**2 / G_SI / M_SUN_KG
        mass = np.maximum.accumulate(mass[np.argsort(block.radius_kpc.to_numpy(float))])
        radius_sorted = np.sort(block.radius_kpc.to_numpy(float))
        cluster_rows.append(
            {
                "domain": "cluster",
                "system": system,
                "redshift": redshifts[system],
                "baryonic_mass_msun": float(mass[-1]),
                "baryonic_half_mass_radius_kpc": monotonic_half_radius(radius_sorted, mass),
                "relaxed_target_quality_pass": bool(
                    fit["fit_success"] and math.isfinite(float(fit["halo_scale_kpc"]))
                ),
                "target_quality_pass": bool(fit["fit_success"] and not fit["fit_at_boundary"]),
                **fit,
            }
        )
    cluster = pd.DataFrame(cluster_rows)

    common = [
        "domain",
        "system",
        "baryonic_mass_msun",
        "baryonic_half_mass_radius_kpc",
        "halo_scale_kpc",
        "r200_kpc",
        "target_quality_pass",
        "relaxed_target_quality_pass",
    ]
    all_objects = pd.concat([galaxy[common], cluster[common]], ignore_index=True)
    all_objects = all_objects[
        all_objects.relaxed_target_quality_pass
        & np.isfinite(all_objects.baryonic_mass_msun)
        & (all_objects.baryonic_mass_msun > 0.0)
        & np.isfinite(all_objects.baryonic_half_mass_radius_kpc)
        & (all_objects.baryonic_half_mass_radius_kpc > 0.0)
    ].copy()
    all_objects["mond_radius_kpc"] = (
        np.sqrt(G_SI * all_objects.baryonic_mass_msun * M_SUN_KG / A0_SI) / KPC_M
    )
    all_objects["log_halo_scale"] = np.log10(all_objects.halo_scale_kpc)
    all_objects["log_mass_pivot"] = np.log10(all_objects.baryonic_mass_msun / 1e10)
    all_objects["log_mond_radius"] = np.log10(all_objects.mond_radius_kpc)
    all_objects["log_baryonic_radius"] = np.log10(all_objects.baryonic_half_mass_radius_kpc)
    objects = all_objects[all_objects.target_quality_pass].copy()
    return objects, cluster, all_objects


def equal_domain_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.domain.value_counts()
    domain_weight = 1.0 / len(counts)
    return frame.domain.map(lambda value: domain_weight / counts[value]).to_numpy(float)


def design_matrix(frame: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    ones = np.ones(len(frame))
    if model == "MOND_acceleration_radius":
        return np.column_stack([ones, frame.log_mond_radius]), ["intercept", "fixed"]
    if model == "CRG_density_radius":
        return np.column_stack([ones, frame.log_mass_pivot / 3.0]), ["intercept", "fixed"]
    if model == "AeST_cutoff_radius":
        return np.column_stack([ones, frame.log_mond_radius / 3.0]), ["intercept", "fixed"]
    if model == "baryonic_extent":
        return np.column_stack([ones, frame.log_baryonic_radius]), ["intercept", "fixed"]
    if model == "fixed_geometric_bridge":
        feature = 0.5 * (frame.log_mond_radius + frame.log_baryonic_radius)
        return np.column_stack([ones, feature]), ["intercept", "fixed"]
    if model == "free_mass_exponent_diagnostic":
        return np.column_stack([ones, frame.log_mass_pivot]), ["intercept", "mass_exponent"]
    if model == "mass_extent_bridge_diagnostic":
        delta = frame.log_mond_radius - frame.log_baryonic_radius
        return np.column_stack([ones, delta]), ["intercept", "mass_weight"]
    raise KeyError(model)


def fit_model(train: pd.DataFrame, model: str) -> dict[str, Any]:
    matrix, names = design_matrix(train, model)
    y = train.log_halo_scale.to_numpy(float)
    if model == "mass_extent_bridge_diagnostic":
        y = y - train.log_baryonic_radius.to_numpy(float)
    weights = equal_domain_weights(train)
    if names[1] == "fixed":
        coefficient = float(np.sum(weights * (y - matrix[:, 1])) / np.sum(weights))
        beta = np.array([coefficient, 1.0])
    else:
        root_weights = np.sqrt(weights)
        beta = np.linalg.lstsq(matrix * root_weights[:, None], y * root_weights, rcond=None)[0]
        if names[1] == "mass_weight":
            clipped_weight = float(np.clip(beta[1], -1.0, 2.0))
            beta[1] = clipped_weight
            beta[0] = np.sum(weights * (y - clipped_weight * matrix[:, 1])) / np.sum(weights)
    return {"beta": beta, "names": names}


def predict(frame: pd.DataFrame, model: str, fitted: dict[str, Any]) -> np.ndarray:
    matrix, _ = design_matrix(frame, model)
    prediction = matrix @ np.asarray(fitted["beta"])
    if model == "mass_extent_bridge_diagnostic":
        prediction = prediction + frame.log_baryonic_radius.to_numpy(float)
    return prediction


def score(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    residual = prediction - frame.log_halo_scale.to_numpy(float)
    domains = {}
    for domain in sorted(frame.domain.unique()):
        values = residual[frame.domain.to_numpy() == domain]
        domains[domain] = {
            "systems": len(values),
            "rmse_dex": float(np.sqrt(np.mean(values**2))),
            "median_bias_dex": float(np.median(values)),
        }
    equal_domain_rmse = float(
        math.sqrt(np.mean([entry["rmse_dex"] ** 2 for entry in domains.values()]))
    )
    return {
        "systems": len(frame),
        "equal_system_rmse_dex": float(np.sqrt(np.mean(residual**2))),
        "equal_domain_rmse_dex": equal_domain_rmse,
        "domains": domains,
    }


def cross_validate(
    objects: pd.DataFrame, model: str, folds: int
) -> tuple[dict[str, Any], np.ndarray]:
    assignments = np.array(
        [stable_fold(row.domain, row.system, folds) for row in objects.itertuples()], dtype=int
    )
    prediction = np.full(len(objects), np.nan)
    coefficients = []
    for fold in range(folds):
        train = objects[assignments != fold]
        test = objects[assignments == fold]
        fitted = fit_model(train, model)
        prediction[assignments == fold] = predict(test, model, fitted)
        coefficients.append(
            {
                name: float(value)
                for name, value in zip(fitted["names"], fitted["beta"], strict=True)
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("out-of-fold prediction coverage is incomplete")
    result = score(objects, prediction)
    result["fold_coefficients"] = coefficients
    return result, prediction


def bootstrap_free_slope(objects: pd.DataFrame, draws: int, seed: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    slopes = []
    by_domain = {domain: block for domain, block in objects.groupby("domain")}
    for _ in range(draws):
        parts = []
        for block in by_domain.values():
            indices = generator.integers(0, len(block), len(block))
            parts.append(block.iloc[indices])
        sample = pd.concat(parts, ignore_index=True)
        slopes.append(float(fit_model(sample, "free_mass_exponent_diagnostic")["beta"][1]))
    low, median, high = np.quantile(slopes, [0.025, 0.5, 0.975])
    return {"p2_5": float(low), "median": float(median), "p97_5": float(high)}


def build_report(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    parent = config["parent"]
    parent_hashes_ok = (
        sha256(ROOT / parent["protocol"]) == parent["sha256"]
        and sha256(ROOT / parent["report"]) == parent["report_sha256"]
    )
    input_hashes_ok = all(
        sha256(ROOT / entry["path"]) == entry["sha256"] for entry in config["inputs"].values()
    )
    if not parent_hashes_ok or not input_hashes_ok:
        raise RuntimeError("v17O parent or input hash changed")

    objects, cluster_fits, relaxed_objects = build_object_table(config)
    folds = int(config["evaluation"]["folds"])
    model_ids = [entry["id"] for entry in config["preregistered_relations"]]
    model_results = {}
    prediction_columns = []
    for model in model_ids:
        cv_score, cv_prediction = cross_validate(objects, model, folds)
        fitted = fit_model(objects, model)
        full_prediction = predict(objects, model, fitted)
        transfer = {}
        within_domain_parameters = {}
        for source, target in (("galaxy", "cluster"), ("cluster", "galaxy")):
            source_fit = fit_model(objects[objects.domain == source], model)
            source_parameters = {
                name: float(value)
                for name, value in zip(source_fit["names"], source_fit["beta"], strict=True)
            }
            within_domain_parameters[source] = source_parameters
            target_frame = objects[objects.domain == target]
            transfer_score = score(target_frame, predict(target_frame, model, source_fit))
            transfer_score["source_parameters"] = source_parameters
            transfer[f"{source}_to_{target}"] = transfer_score
        model_results[model] = {
            "parameters": {
                name: float(value)
                for name, value in zip(fitted["names"], fitted["beta"], strict=True)
            },
            "out_of_fold": cv_score,
            "in_sample": score(objects, full_prediction),
            "within_domain_parameters": within_domain_parameters,
            "domain_transfer": transfer,
        }
        column = f"predicted_log_halo_scale__{model}"
        objects[column] = cv_prediction
        prediction_columns.append(column)

    one_parameter = [
        "MOND_acceleration_radius",
        "CRG_density_radius",
        "AeST_cutoff_radius",
        "baryonic_extent",
        "fixed_geometric_bridge",
    ]
    best_one = min(
        one_parameter,
        key=lambda model: model_results[model]["out_of_fold"]["equal_domain_rmse_dex"],
    )
    mass_only = ["MOND_acceleration_radius", "CRG_density_radius", "AeST_cutoff_radius"]
    best_mass = min(
        mass_only,
        key=lambda model: model_results[model]["out_of_fold"]["equal_domain_rmse_dex"],
    )
    hybrid = model_results["mass_extent_bridge_diagnostic"]["out_of_fold"]
    best_mass_rmse = model_results[best_mass]["out_of_fold"]["equal_domain_rmse_dex"]
    geometry_improvement = best_mass_rmse - hybrid["equal_domain_rmse_dex"]
    gates = config["gates"]

    def biases_pass(model_score: dict[str, Any]) -> bool:
        return all(
            abs(entry["median_bias_dex"]) <= gates["maximum_absolute_domain_median_bias_dex"]
            for entry in model_score["domains"].values()
        )

    best_one_score = model_results[best_one]["out_of_fold"]
    one_parameter_pass = best_one_score["equal_domain_rmse_dex"] <= gates[
        "maximum_one_parameter_equal_domain_cv_rmse_dex"
    ] and biases_pass(best_one_score)
    geometry_required = geometry_improvement >= gates[
        "minimum_geometry_improvement_over_best_mass_only_dex"
    ] and biases_pass(hybrid)
    cluster_fit_gate = bool(
        cluster_fits.fit_rmse_dex.max() <= gates["maximum_cluster_nfw_fit_rmse_dex"]
    )
    if one_parameter_pass:
        outcome = f"select_{best_one}_scale_mechanism"
    elif geometry_required:
        outcome = "select_continuous_mass_extent_field_invariant"
    else:
        outcome = "halo_scale_not_identifiable_from_current_diagnostic_products"

    free_slope_interval = bootstrap_free_slope(
        objects,
        int(config["evaluation"]["bootstrap_draws"]),
        int(config["evaluation"]["seed"]),
    )
    relaxed_scores = {}
    for model in model_ids:
        relaxed_score, _ = cross_validate(relaxed_objects, model, folds)
        relaxed_scores[model] = {
            "equal_domain_rmse_dex": relaxed_score["equal_domain_rmse_dex"],
            "domains": relaxed_score["domains"],
            "change_from_strict_rmse_dex": relaxed_score["equal_domain_rmse_dex"]
            - model_results[model]["out_of_fold"]["equal_domain_rmse_dex"],
        }
    relaxed_best_one = min(
        one_parameter,
        key=lambda model: relaxed_scores[model]["equal_domain_rmse_dex"],
    )
    report = {
        "report_version": config["protocol_version"],
        "status": "completed_halo_scale_driver_audit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol": config_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(config_path),
        "parent_hashes_verified": parent_hashes_ok,
        "input_hashes_verified": input_hashes_ok,
        "holdout_opened": False,
        "coverage": {
            "galaxies": int((objects.domain == "galaxy").sum()),
            "clusters": int((objects.domain == "cluster").sum()),
            "cluster_nfw_fit_max_rmse_dex": float(cluster_fits.fit_rmse_dex.max()),
            "cluster_nfw_fit_median_rmse_dex": float(cluster_fits.fit_rmse_dex.median()),
        },
        "models": model_results,
        "free_mass_exponent_bootstrap": free_slope_interval,
        "target_cut_sensitivity": {
            "description": "Repeat the cross-validation while retaining finite galaxy NFW fits whose parameters landed on a prior boundary.",
            "strict_systems": len(objects),
            "relaxed_systems": len(relaxed_objects),
            "best_one_parameter_relation": relaxed_best_one,
            "models": relaxed_scores,
        },
        "selection": {
            "best_one_parameter_relation": best_one,
            "best_mass_only_relation": best_mass,
            "geometry_improvement_over_best_mass_only_dex": geometry_improvement,
            "one_parameter_gate_pass": one_parameter_pass,
            "geometry_required_gate_pass": geometry_required,
            "cluster_target_reconstruction_gate_pass": cluster_fit_gate,
            "outcome": outcome,
            "free_mass_exponent_interpretation": (
                "The pooled exponent is not a within-domain law: compare the separately "
                "fitted galaxy and cluster exponents and their transfer failures."
            ),
        },
        "claim_boundary": [
            "The targets are NFW fit products, not raw observations or evidence that halos are physical.",
            "Galaxy NFW radii inherit inner-radius prior sensitivity; cluster radii inherit the published NFW deprojection.",
            "Domain-transfer failures are evidence against a universal scale law; domain-specific corrections are not authorized.",
            "This result selects at most a mechanism for a covariant action and does not authorize holdout tests.",
        ],
    }
    output_columns = [
        "domain",
        "system",
        "baryonic_mass_msun",
        "baryonic_half_mass_radius_kpc",
        "mond_radius_kpc",
        "halo_scale_kpc",
        *prediction_columns,
    ]
    return report, objects[output_columns], cluster_fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v17o_halo_scale_driver_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "sigma_v17o_halo_scale_driver_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report, predictions, cluster_fits = build_report(args.config, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    cluster_fits.to_csv(args.output_dir / "cluster_nfw_fits.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
