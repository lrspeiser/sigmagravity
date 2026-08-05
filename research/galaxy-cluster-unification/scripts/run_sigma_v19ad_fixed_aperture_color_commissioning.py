#!/usr/bin/env python3
"""Commission fixed-aperture DECam colors on the frozen V19AB split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ad_fixed_aperture_color_commissioning.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty V19AD output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config["status"] != "frozen_before_fixed_aperture_aggregation_or_validation_scoring":
        raise RuntimeError("V19AD protocol is not frozen")
    for artifact in config["parent_artifacts"]:
        if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"parent artifact hash mismatch: {artifact['path']}")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AD runner hash mismatch")


def feature_vector(row: dict[str, Any], model_spec: dict[str, Any]) -> np.ndarray:
    b_minus_r = float(row["B"]) - float(row["R"])
    r_minus_i = float(row["R"]) - float(row["I"])
    centers = model_spec["feature_centers"]
    scales = model_spec["feature_scales"]
    return np.asarray(
        [
            1.0,
            (b_minus_r - float(centers["B_minus_R"])) / float(scales["B_minus_R"]),
            (r_minus_i - float(centers["R_minus_I"])) / float(scales["R_minus_I"]),
        ]
    )


def color_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(row["g"]) - float(row["r"]), float(row["r"]) - float(row["i"]), float(row["i"]) - float(row["z"])]
    )


def aggregate_measurements(
    sample: list[dict[str, Any]],
    measurements: list[dict[str, str]],
    *,
    aperture_arcsec: int,
    instrument: str,
    filters: list[str],
) -> list[dict[str, Any]]:
    magnitude_column = f"mag_aper{aperture_arcsec}"
    error_column = f"magerr_aper{aperture_arcsec}"
    sample_by_nsc = {row["nsc_id"]: row for row in sample}
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for measurement in measurements:
        if measurement["objectid"] not in sample_by_nsc:
            continue
        if measurement["instrument"] != instrument or measurement["filter"] not in filters:
            continue
        if int(measurement["flags"]) != 0:
            continue
        magnitude = finite_float(measurement.get(magnitude_column))
        error = finite_float(measurement.get(error_column))
        if magnitude is None or error is None or magnitude >= 90.0 or not 0.0 < error <= 1.0:
            continue
        grouped[(measurement["objectid"], measurement["filter"])].append((magnitude, error))

    output: list[dict[str, Any]] = []
    for source in sample:
        row: dict[str, Any] = {
            "cluster": source["cluster"],
            "object_id": source["object_id"],
            "nsc_id": source["nsc_id"],
            "split": source["split"],
            "aperture_diameter_arcsec": aperture_arcsec,
            "B": float(source["B"]),
            "R": float(source["R"]),
            "I": float(source["I"]),
        }
        for band in filters:
            values = grouped.get((source["nsc_id"], band), [])
            if not values:
                row[band] = None
                row[f"{band}_uncertainty"] = None
                row[f"{band}_measurements"] = 0
                continue
            magnitudes = np.asarray([value[0] for value in values])
            errors = np.asarray([value[1] for value in values])
            median = float(np.median(magnitudes))
            mad = float(np.median(np.abs(magnitudes - median)))
            uncertainty = max(
                float(np.median(errors)) / math.sqrt(len(values)),
                1.4826 * mad / math.sqrt(len(values)),
                0.02,
            )
            row[band] = median
            row[f"{band}_uncertainty"] = uncertainty
            row[f"{band}_measurements"] = len(values)
        output.append(row)
    return output


def fit_color_model(rows: list[dict[str, Any]], model_spec: dict[str, Any]) -> dict[str, Any]:
    x = np.vstack([feature_vector(row, model_spec) for row in rows])
    y = np.vstack([color_vector(row) for row in rows])
    ridge = float(model_spec["ridge_penalty"])
    robust_scale = float(model_spec["robust_residual_scale_mag"])
    floor = float(model_spec["predictive_scale_floor_mag"])
    initial = np.linalg.solve(x.T @ x + np.diag([0.0, ridge, ridge]), x.T @ y)
    coefficients: list[np.ndarray] = []
    scales: list[float] = []
    optimizers: list[dict[str, Any]] = []
    for index, name in enumerate(model_spec["outputs"]):
        target = y[:, index]

        def residuals(beta: np.ndarray, target: np.ndarray = target) -> np.ndarray:
            return np.concatenate(
                [(x @ beta - target) / robust_scale, math.sqrt(ridge) * beta[1:]]
            )

        result = least_squares(
            residuals,
            initial[:, index],
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(model_spec["maximum_function_evaluations"]),
        )
        beta = np.asarray(result.x)
        residual = target - x @ beta
        centered = residual - np.median(residual)
        coefficients.append(beta)
        scales.append(max(floor, 1.4826 * float(np.median(np.abs(centered)))))
        optimizers.append({"output": name, "success": bool(result.success), "nfev": int(result.nfev)})
    return {
        "coefficient_matrix": np.column_stack(coefficients),
        "predictive_scales": np.asarray(scales),
        "optimizers": optimizers,
    }


def predict(model: dict[str, Any], row: dict[str, Any], model_spec: dict[str, Any]) -> np.ndarray:
    return feature_vector(row, model_spec) @ model["coefficient_matrix"]


def evaluate_aperture(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    development = [row for row in rows if row["split"] == "development"]
    validation = [row for row in rows if row["split"] == "validation"]
    model_spec = config["model"]
    model = fit_color_model(development, model_spec)
    absolute_errors: dict[str, list[float]] = {name: [] for name in model_spec["outputs"]}
    retrieval: list[dict[str, Any]] = []
    true_ranks: list[int] = []
    for member in validation:
        predicted = predict(model, member, model_spec)
        observed = color_vector(member)
        for index, name in enumerate(model_spec["outputs"]):
            absolute_errors[name].append(abs(float(observed[index] - predicted[index])))
        candidates = []
        for candidate in validation:
            score = float(
                np.sum(((color_vector(candidate) - predicted) / model["predictive_scales"]) ** 2)
            )
            candidates.append(
                {
                    "member_object_id": member["object_id"],
                    "candidate_object_id": candidate["object_id"],
                    "candidate_nsc_id": candidate["nsc_id"],
                    "is_provisional_singleton_pair": candidate["object_id"] == member["object_id"],
                    "color_score": score,
                }
            )
        ordered = sorted(candidates, key=lambda row: (row["color_score"], row["candidate_nsc_id"]))
        ranks = {row["candidate_nsc_id"]: index + 1 for index, row in enumerate(ordered)}
        for candidate in candidates:
            candidate["rank"] = ranks[candidate["candidate_nsc_id"]]
            retrieval.append(candidate)
            if candidate["is_provisional_singleton_pair"]:
                true_ranks.append(candidate["rank"])
    return {
        "model": model,
        "median_absolute_error_mag": {
            name: float(np.median(values)) for name, values in absolute_errors.items()
        },
        "top1": sum(rank == 1 for rank in true_ranks),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in true_ranks])),
        "true_ranks": true_ranks,
        "retrieval_rows": retrieval,
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    sample = read_csv_rows(ROOT / config["inputs"]["commissioning_sample"])
    measurements = read_csv_rows(ROOT / config["inputs"]["measurements"])
    if len(sample) != int(config["inputs"]["expected_rows"]):
        raise RuntimeError("V19AD sample count changed")
    for split, expected in (
        ("development", config["inputs"]["expected_development_ids"]),
        ("validation", config["inputs"]["expected_validation_ids"]),
    ):
        if [row["object_id"] for row in sample if row["split"] == split] != expected:
            raise RuntimeError(f"V19AD {split} split changed")

    aggregation = config["aggregation"]
    apertures = [int(aggregation["primary_aperture_diameter_arcsec"])] + [
        int(value) for value in aggregation["sensitivity_aperture_diameters_arcsec"]
    ]
    aggregated_by_aperture: dict[int, list[dict[str, Any]]] = {}
    evaluations: dict[int, dict[str, Any]] = {}
    all_output_rows: list[dict[str, Any]] = []
    for aperture in apertures:
        rows = aggregate_measurements(
            sample,
            measurements,
            aperture_arcsec=aperture,
            instrument=aggregation["instrument"],
            filters=list(aggregation["filters"]),
        )
        aggregated_by_aperture[aperture] = rows
        all_output_rows.extend(rows)
        if all(row[band] is not None for row in rows for band in aggregation["filters"]):
            evaluations[aperture] = evaluate_aperture(rows, config)

    primary_aperture = int(aggregation["primary_aperture_diameter_arcsec"])
    primary_rows = aggregated_by_aperture[primary_aperture]
    if primary_aperture not in evaluations:
        raise RuntimeError("V19AD primary aperture lacks complete griz")
    primary = evaluations[primary_aperture]
    gate_spec = config["validation_gates"]
    gates = {
        "exact_sample_and_split": len(sample) == 15,
        "all_primary_rows_have_griz": all(
            row[band] is not None for row in primary_rows for band in aggregation["filters"]
        ),
        "all_optimizers_succeeded": all(row["success"] for row in primary["model"]["optimizers"]),
        "validation_median_absolute_error_each_color": all(
            value <= float(gate_spec["maximum_validation_median_absolute_error_each_color_mag"])
            for value in primary["median_absolute_error_mag"].values()
        ),
        "color_only_top1_retrieval": primary["top1"]
        >= int(gate_spec["minimum_color_only_top1_retrievals_out_of_5"]),
        "color_only_mean_reciprocal_rank": primary["mean_reciprocal_rank"]
        >= float(gate_spec["minimum_color_only_mean_reciprocal_rank"]),
        "no_ambiguous_association_mass_lensing_or_gravity": True,
    }
    gates["all_commissioning_gates_pass"] = all(gates.values())
    aggregated_path = ROOT / config["outputs"]["aggregated_sample"]
    retrieval_path = ROOT / config["outputs"]["primary_validation_retrieval"]
    write_csv(aggregated_path, all_output_rows)
    write_csv(retrieval_path, primary["retrieval_rows"])

    def summarized_evaluation(aperture: int, evaluation: dict[str, Any]) -> dict[str, Any]:
        return {
            "aperture_diameter_arcsec": aperture,
            "median_absolute_error_mag": evaluation["median_absolute_error_mag"],
            "top1": evaluation["top1"],
            "mean_reciprocal_rank": evaluation["mean_reciprocal_rank"],
            "true_ranks": evaluation["true_ranks"],
            "coefficient_matrix": evaluation["model"]["coefficient_matrix"],
            "predictive_scales_mag": evaluation["model"]["predictive_scales"],
            "optimizers": evaluation["model"]["optimizers"],
        }

    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AD-FIXED-APERTURE-COLOR-COMMISSIONING-1.0.0",
        "status": "completed_fixed_aperture_color_commissioning",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "implementation": config["implementation"],
        "parent_hashes": {
            artifact["path"]: artifact["sha256"] for artifact in config["parent_artifacts"]
        },
        "primary": summarized_evaluation(primary_aperture, primary),
        "sensitivity": [
            summarized_evaluation(aperture, evaluations[aperture])
            for aperture in apertures
            if aperture != primary_aperture and aperture in evaluations
        ],
        "gates": gates,
        "ambiguous_likelihood_application_authorized": bool(gates["all_commissioning_gates_pass"]),
        "outputs": {
            "aggregated_sample": aggregated_path.relative_to(ROOT).as_posix(),
            "aggregated_sample_sha256": sha256(aggregated_path),
            "primary_validation_retrieval": retrieval_path.relative_to(ROOT).as_posix(),
            "primary_validation_retrieval_sha256": sha256(retrieval_path),
        },
        "claim_boundary": config["claim_boundary"],
        "ambiguous_candidate_photometry_scored": False,
        "counterpart_selected": False,
        "stellar_mass_inferred": False,
        "mass_current_constructed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.write_text(json.dumps(strict_json(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(strict_json(run(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
