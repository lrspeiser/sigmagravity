#!/usr/bin/env python3
"""Fit the frozen v19H heteroscedastic member phase-space mixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sigma_v19f_chandra_common as common
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19h_causal_observable_protocol.json"
DEFAULT_MAPS = ROOT / "results" / "sigma_v19h_source_maps" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19h_member_phase"


@dataclass(frozen=True)
class Scale:
    center: np.ndarray
    width: np.ndarray


def stable_seed(base: int, cluster: str, suffix: str) -> int:
    payload = f"{base}:{cluster}:{suffix}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def load_catalog(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tangent_phase(rows: list[dict], redshift: float) -> tuple[np.ndarray, np.ndarray, Scale]:
    ra = np.array([float(row["ra_deg"]) for row in rows])
    dec = np.array([float(row["dec_deg"]) for row in rows])
    cz = np.array([float(row["heliocentric_cz_km_s"]) for row in rows])
    cz_error = np.array([float(row["cz_uncertainty_km_s"]) for row in rows])
    ra0 = float(np.median(ra))
    dec0 = float(np.median(dec))
    x = (ra - ra0) * math.cos(math.radians(dec0)) * 3600.0
    y = (dec - dec0) * 3600.0
    rest_velocity = (cz - np.median(cz)) / (1.0 + redshift)
    rest_error = cz_error / (1.0 + redshift)
    observed = np.column_stack([x, y, rest_velocity])
    center = np.median(observed, axis=0)
    q25, q75 = np.percentile(observed, [25.0, 75.0], axis=0)
    width = q75 - q25
    if np.any(~np.isfinite(width)) or np.any(width <= 0.0):
        raise RuntimeError("member phase coordinates have a nonpositive robust scale")
    standardized = (observed - center) / width
    errors = np.zeros_like(standardized)
    errors[:, 2] = rest_error / width[2]
    return standardized, errors, Scale(center=center, width=width)


def parameter_count(components: int, dimensions: int) -> int:
    return (
        components - 1
        + components * dimensions
        + components * dimensions * (dimensions + 1) // 2
    )


def pack(weights: np.ndarray, means: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    components, dimensions = means.shape
    values: list[float] = []
    values.extend(np.log(weights[:-1] / weights[-1]))
    values.extend(means.ravel())
    for index in range(components):
        chol = np.linalg.cholesky(covariance[index])
        for row in range(dimensions):
            for column in range(row + 1):
                values.append(
                    math.log(chol[row, column])
                    if row == column
                    else chol[row, column]
                )
    return np.asarray(values, dtype=float)


def unpack(
    params: np.ndarray, components: int, dimensions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = components - 1
    logits = np.concatenate([params[:offset], [0.0]])
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    means = params[offset : offset + components * dimensions].reshape(
        components, dimensions
    )
    offset += components * dimensions
    covariance = []
    for _ in range(components):
        chol = np.zeros((dimensions, dimensions), dtype=float)
        for row in range(dimensions):
            for column in range(row + 1):
                value = params[offset]
                offset += 1
                chol[row, column] = math.exp(value) if row == column else value
        covariance.append(chol @ chol.T + np.eye(dimensions) * 1e-8)
    return weights, means, np.asarray(covariance)


def likelihood_terms(
    observed: np.ndarray,
    errors: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    rows, dimensions = observed.shape
    components = len(weights)
    values = np.empty((rows, components), dtype=float)
    normalization = dimensions * math.log(2.0 * math.pi)
    error_variance = errors * errors
    for component in range(components):
        total = np.broadcast_to(covariance[component], (rows, dimensions, dimensions)).copy()
        diagonal = np.arange(dimensions)
        total[:, diagonal, diagonal] += error_variance
        sign, logdet = np.linalg.slogdet(total)
        if np.any(sign <= 0) or np.any(~np.isfinite(logdet)):
            return np.full((rows, components), -np.inf)
        delta = observed - means[component]
        solution = np.linalg.solve(total, delta[..., None])[..., 0]
        quadratic = np.einsum("ij,ij->i", delta, solution)
        values[:, component] = math.log(weights[component]) - 0.5 * (
            normalization + logdet + quadratic
        )
    return values


def objective(
    params: np.ndarray,
    observed: np.ndarray,
    errors: np.ndarray,
    components: int,
) -> float:
    weights, means, covariance = unpack(params, components, observed.shape[1])
    terms = likelihood_terms(observed, errors, weights, means, covariance)
    value = -float(np.sum(logsumexp(terms, axis=1)))
    return value if np.isfinite(value) else 1e100


def parameter_bounds(components: int, dimensions: int) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = [(-10.0, 10.0)] * (components - 1)
    bounds.extend([(-10.0, 10.0)] * (components * dimensions))
    for _ in range(components):
        for row in range(dimensions):
            for column in range(row + 1):
                bounds.append((-5.0, 3.0) if row == column else (-5.0, 5.0))
    return bounds


def initialize(observed: np.ndarray, components: int, seed: int) -> np.ndarray:
    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=1e-4,
        n_init=10,
        max_iter=1000,
        random_state=seed,
    ).fit(observed)
    return pack(model.weights_, model.means_, model.covariances_)


def fit_mixture(
    observed: np.ndarray,
    errors: np.ndarray,
    components: int,
    seed: int,
    initial: np.ndarray | None = None,
    maxiter: int = 1000,
) -> dict:
    start = initialize(observed, components, seed) if initial is None else initial.copy()
    result = minimize(
        objective,
        start,
        args=(observed, errors, components),
        method="L-BFGS-B",
        bounds=parameter_bounds(components, observed.shape[1]),
        options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 30},
    )
    weights, means, covariance = unpack(
        result.x, components, observed.shape[1]
    )
    terms = likelihood_terms(observed, errors, weights, means, covariance)
    log_likelihood = float(np.sum(logsumexp(terms, axis=1)))
    responsibilities = np.exp(terms - logsumexp(terms, axis=1)[:, None])
    labels = np.argmax(responsibilities, axis=1)
    count = parameter_count(components, observed.shape[1])
    bic = count * math.log(len(observed)) - 2.0 * log_likelihood
    return {
        "success": bool(result.success and np.isfinite(log_likelihood)),
        "message": str(result.message),
        "iterations": int(result.nit),
        "parameters": result.x,
        "weights": weights,
        "means": means,
        "covariance": covariance,
        "responsibilities": responsibilities,
        "labels": labels,
        "log_likelihood": log_likelihood,
        "parameter_count": count,
        "bic": float(bic),
    }


def physical_parameters(fit: dict, scale: Scale) -> tuple[np.ndarray, np.ndarray]:
    means = fit["means"] * scale.width + scale.center
    diagonal = np.diag(scale.width)
    covariance = np.asarray(
        [diagonal @ value @ diagonal for value in fit["covariance"]]
    )
    return means, covariance


def serialize_fit(fit: dict, scale: Scale) -> dict:
    physical_means, physical_covariance = physical_parameters(fit, scale)
    return {
        "components": len(fit["weights"]),
        "success": fit["success"],
        "message": fit["message"],
        "iterations": fit["iterations"],
        "log_likelihood": fit["log_likelihood"],
        "parameter_count": fit["parameter_count"],
        "bic": fit["bic"],
        "weights": fit["weights"].tolist(),
        "standardized_means": fit["means"].tolist(),
        "standardized_covariance": fit["covariance"].tolist(),
        "physical_means_arcsec_arcsec_km_s": physical_means.tolist(),
        "physical_covariance": physical_covariance.tolist(),
        "assignments": fit["labels"].tolist(),
        "maximum_responsibility": np.max(fit["responsibilities"], axis=1).tolist(),
    }


def select_fit(fits: list[dict], minimum_delta_bic: float = 10.0) -> dict:
    selected = fits[0]
    for candidate in fits[1:]:
        improvement = selected["bic"] - candidate["bic"]
        if improvement >= minimum_delta_bic:
            selected = candidate
        else:
            break
    return selected


def align_components(
    means: np.ndarray, covariance: np.ndarray, weights: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = linear_sum_assignment(
        np.linalg.norm(means[:, None, :] - reference[None, :, :], axis=2)
    )
    order = np.empty(len(rows), dtype=int)
    order[columns] = rows
    return means[order], covariance[order], weights[order]


def bootstrap(
    observed: np.ndarray,
    errors: np.ndarray,
    selected: dict,
    draws: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    generator = np.random.default_rng(seed)
    accepted = []
    failures = []
    components = len(selected["weights"])
    for draw in range(draws):
        indices = generator.integers(0, len(observed), size=len(observed))
        sample = observed[indices].copy()
        sample_errors = errors[indices].copy()
        sample[:, 2] += generator.normal(0.0, sample_errors[:, 2])
        fit = fit_mixture(
            sample,
            sample_errors,
            components,
            seed + draw + 1,
            initial=selected["parameters"],
            maxiter=250,
        )
        if not fit["success"]:
            failures.append(
                {"draw": draw, "message": fit["message"], "bic": fit["bic"]}
            )
            continue
        means, covariance, weights = align_components(
            fit["means"],
            fit["covariance"],
            fit["weights"],
            selected["means"],
        )
        accepted.append(
            {
                "draw": draw,
                "weights": weights.tolist(),
                "standardized_means": means.tolist(),
                "standardized_covariance": covariance.tolist(),
            }
        )
        if (draw + 1) % 100 == 0:
            print(
                f"bootstrap {draw + 1}/{draws}: {len(accepted)} accepted",
                flush=True,
            )
    return accepted, failures


def summarize_bootstrap(draws: list[dict], scale: Scale) -> dict:
    weights = np.asarray([row["weights"] for row in draws])
    means = np.asarray([row["standardized_means"] for row in draws])
    physical_means = means * scale.width[None, None, :] + scale.center[None, None, :]
    return {
        "weights_quantiles_2p5_50_97p5": np.percentile(
            weights, [2.5, 50.0, 97.5], axis=0
        ).tolist(),
        "physical_means_quantiles_2p5_50_97p5": np.percentile(
            physical_means, [2.5, 50.0, 97.5], axis=0
        ).tolist(),
    }


def label_validation(rows: list[dict], assignments: np.ndarray) -> dict:
    declared = [row["subcluster_label"].strip() for row in rows]
    usable = [value not in {"", "unassigned"} for value in declared]
    if sum(usable) == 0:
        return {"available": False, "used_for_fit_or_selection": False}
    mapping = {value: index for index, value in enumerate(sorted(set(declared) - {""}))}
    target = np.asarray([mapping[value] for value, keep in zip(declared, usable) if keep])
    predicted = assignments[np.asarray(usable)]
    return {
        "available": True,
        "rows": int(sum(usable)),
        "adjusted_rand_index": float(adjusted_rand_score(target, predicted)),
        "used_for_fit_or_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    maps_path = args.maps.resolve()
    config = common.load_json(config_path)
    maps = common.load_json(maps_path)
    common.validate_parent_hashes(config)
    if maps["status"] != "both_clusters_passed_frozen_v19h_source_map_gate":
        raise RuntimeError("v19H source-map gate did not pass")
    if maps["lensing_target_opened"] is not False:
        raise RuntimeError("v19H map stage opened a lensing target")
    phase = config["member_phase_structure"]
    required_draws = int(phase["uncertainty"]["catalog_bootstraps"])
    draws = required_draws if args.bootstrap_draws is None else args.bootstrap_draws
    if draws != required_draws:
        raise RuntimeError(f"v19H requires exactly {required_draws} catalog bootstraps")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = []
    for cluster in config["sample"]["clusters"]:
        coord = config["coordinates"]["clusters"][cluster]
        catalog_path = ROOT / coord["member_catalog"]
        if common.sha256(catalog_path) != coord["member_catalog_sha256"]:
            raise RuntimeError(f"{cluster} member catalog hash mismatch")
        rows = load_catalog(catalog_path)
        observed, errors, scale = tangent_phase(rows, float(coord["redshift"]))
        fits = []
        for components in phase["component_counts_tested"]:
            fit = fit_mixture(
                observed,
                errors,
                int(components),
                stable_seed(phase["uncertainty"]["bootstrap_seed"], cluster, str(components)),
            )
            fits.append(fit)
            print(
                f"{cluster} K={components}: BIC={fit['bic']:.3f}, "
                f"logL={fit['log_likelihood']:.3f}, success={fit['success']}",
                flush=True,
            )
        selected = select_fit(fits)
        selected_components = len(selected["weights"])
        print(f"{cluster}: selected K={selected_components}", flush=True)
        bootstrap_draws, failures = bootstrap(
            observed,
            errors,
            selected,
            draws,
            stable_seed(phase["uncertainty"]["bootstrap_seed"], cluster, "bootstrap"),
        )
        gates = {
            "all_primary_optimizations_finite": all(
                np.isfinite(row["log_likelihood"]) for row in fits
            ),
            "selected_fit_converged": bool(selected["success"]),
            "minimum_identifiable_merger_components": selected_components
            >= phase["minimum_identifiable_merger_components"],
            "bootstrap_failure_fraction_at_most_one_percent": len(failures) / draws
            <= 0.01,
        }
        record = {
            "cluster": cluster,
            "rows": len(rows),
            "catalog": coord["member_catalog"],
            "catalog_sha256": common.sha256(catalog_path),
            "standardization": {
                "center_arcsec_arcsec_km_s": scale.center.tolist(),
                "iqr_arcsec_arcsec_km_s": scale.width.tolist(),
            },
            "fits": [serialize_fit(row, scale) for row in fits],
            "selection_rule": "add the next component only when BIC improves by at least 10",
            "selected_components": selected_components,
            "selected_fit": serialize_fit(selected, scale),
            "published_label_validation": label_validation(rows, selected["labels"]),
            "bootstrap": {
                "requested_draws": draws,
                "accepted_draws": len(bootstrap_draws),
                "failed_draws": len(failures),
                "failure_fraction": len(failures) / draws,
                "summary": summarize_bootstrap(bootstrap_draws, scale),
                "failures": failures,
            },
            "gates": gates,
        }
        draw_path = output / f"{cluster.lower()}_bootstrap.json"
        draw_path.write_text(
            json.dumps(bootstrap_draws, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        record["bootstrap"]["draws_file"] = draw_path.relative_to(ROOT).as_posix()
        record["bootstrap"]["draws_sha256"] = common.sha256(draw_path)
        clusters.append(record)

    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    report = {
        "status": (
            "both_clusters_passed_frozen_v19h_member_phase_gate"
            if not failed
            else "frozen_v19h_member_phase_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "source_map_report_sha256": common.sha256(maps_path),
        "failed_clusters": failed,
        "clusters": clusters,
        "published_subcluster_labels_used_for_fit_or_selection": False,
        "registered_science_images_visually_inspected": False,
        "edge_search_run": False,
        "spectrum_or_response_constructed": False,
        "projection_or_clock_drawn": False,
        "causal_source_constructed": False,
        "lensing_target_opened": False,
        "gravity_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
