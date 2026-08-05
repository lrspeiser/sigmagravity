#!/usr/bin/env python3
"""Commission a frozen Bessel B/R/I to NSC g/r/i/z transformation.

Only the preregistered 15 Bullet singleton/full-color cones are opened.  No
ambiguous member candidate, lensing observable, mass, current, or gravity
quantity is read or produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from run_sigma_v19aa_member_counterpart_association import (
    finite_float,
    load_survey_detections,
    nsc_has_band,
)
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ab_bessel_nsc_transform_commissioning.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_artifacts(config: dict[str, Any]) -> None:
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"parent artifact hash mismatch: {path}")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AB runner hash mismatch")


def probable_foreground_star(payload: dict[str, str]) -> bool:
    pmra = finite_float(payload.get("pmra"))
    pmdec = finite_float(payload.get("pmdec"))
    pmraerr = finite_float(payload.get("pmraerr"))
    pmdecerr = finite_float(payload.get("pmdecerr"))
    class_star = finite_float(payload.get("class_star"))
    if None in (pmra, pmdec, pmraerr, pmdecerr, class_star):
        return False
    if pmraerr <= 0.0 or pmdecerr <= 0.0:
        return False
    significance = math.hypot(pmra / pmraerr, pmdec / pmdecerr)
    return bool(significance >= 5.0 and class_star >= 0.8)


def photometric_features(row: dict[str, Any], model_spec: dict[str, Any]) -> np.ndarray:
    b_minus_r = float(row["B"]) - float(row["R"])
    r_minus_i = float(row["R"]) - float(row["I"])
    centers = model_spec["feature_centers"]
    scales = model_spec["feature_scales"]
    return np.asarray(
        [
            1.0,
            (b_minus_r - float(centers["B_minus_R"])) / float(scales["B_minus_R"]),
            (r_minus_i - float(centers["R_minus_I"])) / float(scales["R_minus_I"]),
        ],
        dtype=float,
    )


def observed_offsets(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row["g"]) - float(row["B"]),
            float(row["r"]) - float(row["R"]),
            float(row["i"]) - float(row["I"]),
            float(row["z"]) - float(row["I"]),
        ],
        dtype=float,
    )


def predicted_colors(row: dict[str, Any], offsets: np.ndarray) -> np.ndarray:
    b_minus_r = float(row["B"]) - float(row["R"])
    r_minus_i = float(row["R"]) - float(row["I"])
    return np.asarray(
        [
            b_minus_r + offsets[0] - offsets[1],
            r_minus_i + offsets[1] - offsets[2],
            offsets[2] - offsets[3],
        ],
        dtype=float,
    )


def observed_colors(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row["g"]) - float(row["r"]),
            float(row["r"]) - float(row["i"]),
            float(row["i"]) - float(row["z"]),
        ],
        dtype=float,
    )


def fit_transform(rows: list[dict[str, Any]], model_spec: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < 4:
        raise ValueError("at least four rows are required")
    x = np.vstack([photometric_features(row, model_spec) for row in rows])
    y = np.vstack([observed_offsets(row) for row in rows])
    robust_scale = float(model_spec["robust_residual_scale_mag"])
    ridge = float(model_spec["ridge_penalty"])
    floor = float(model_spec["predictive_scale_floor_mag"])
    max_nfev = int(model_spec["maximum_function_evaluations"])
    output_names = list(model_spec["output_offsets"])
    coefficients: list[np.ndarray] = []
    predictive_scales: list[float] = []
    optimizer_records: list[dict[str, Any]] = []

    penalty = np.diag([0.0, ridge, ridge])
    initial_matrix = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    for index, output_name in enumerate(output_names):
        target = y[:, index]

        def residuals(beta: np.ndarray, target: np.ndarray = target) -> np.ndarray:
            data = (x @ beta - target) / robust_scale
            regularization = math.sqrt(ridge) * beta[1:]
            return np.concatenate([data, regularization])

        result = least_squares(
            residuals,
            initial_matrix[:, index],
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=max_nfev,
        )
        beta = np.asarray(result.x, dtype=float)
        residual = target - x @ beta
        centered = residual - float(np.median(residual))
        scale = max(floor, 1.4826 * float(np.median(np.abs(centered))))
        coefficients.append(beta)
        predictive_scales.append(scale)
        optimizer_records.append(
            {
                "output": output_name,
                "success": bool(result.success),
                "status": int(result.status),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
            }
        )
    coefficient_matrix = np.column_stack(coefficients)
    return {
        "coefficient_matrix": coefficient_matrix,
        "predictive_scales": np.asarray(predictive_scales, dtype=float),
        "optimizer_records": optimizer_records,
    }


def predict_transform(model: dict[str, Any], row: dict[str, Any], model_spec: dict[str, Any]) -> np.ndarray:
    return photometric_features(row, model_spec) @ model["coefficient_matrix"]


def load_commissioning_sample(config: dict[str, Any]) -> list[dict[str, Any]]:
    nsc_report = json.loads((ROOT / config["inputs"]["nsc_report"]).read_text(encoding="utf-8"))
    detections, member_ids = load_survey_detections(
        nsc_report,
        survey="NSC",
        astrometric_floor_arcsec=0.02,
    )
    bri_rows = read_csv_rows(ROOT / config["inputs"]["bullet_bri"])
    eligible: list[dict[str, Any]] = []
    for paper in bri_rows:
        if paper["cluster"] != config["inputs"]["cluster"]:
            continue
        if paper["published_bri_available"].strip().lower() != "true":
            continue
        object_id = paper["object_id"].zfill(2)
        ids = member_ids.get((paper["cluster"], object_id), set())
        if len(ids) != 1:
            continue
        detection = detections[next(iter(ids))]
        if not all(nsc_has_band(detection.payload, band) for band in ("g", "r", "i", "z")):
            continue
        if probable_foreground_star(detection.payload):
            continue
        values = {
            "B": finite_float(paper["b_bessel_mag"]),
            "R": finite_float(paper["r_bessel_mag"]),
            "I": finite_float(paper["i_bessel_mag"]),
            "g": finite_float(detection.payload["gmag"]),
            "r": finite_float(detection.payload["rmag"]),
            "i": finite_float(detection.payload["imag"]),
            "z": finite_float(detection.payload["zmag"]),
        }
        if any(value is None for value in values.values()):
            raise RuntimeError(f"eligible row has non-finite photometry: {object_id}")
        eligible.append(
            {
                "cluster": paper["cluster"],
                "object_id": object_id,
                "nsc_id": detection.survey_id,
                **values,
            }
        )
    if len(eligible) != int(config["inputs"]["expected_eligible_rows"]):
        raise RuntimeError(f"eligible sample mismatch: {len(eligible)}")
    salt = config["split"]["salt"]
    eligible.sort(key=lambda row: hashlib.sha256(f"{salt}{row['object_id']}".encode()).hexdigest())
    development_count = int(config["split"]["development_count"])
    for index, row in enumerate(eligible):
        row["split"] = "development" if index < development_count else "validation"
        row["split_hash"] = hashlib.sha256(f"{salt}{row['object_id']}".encode()).hexdigest()
    development_ids = [row["object_id"] for row in eligible if row["split"] == "development"]
    validation_ids = [row["object_id"] for row in eligible if row["split"] == "validation"]
    if development_ids != config["split"]["expected_development_ids"]:
        raise RuntimeError(f"development split mismatch: {development_ids}")
    if validation_ids != config["split"]["expected_validation_ids"]:
        raise RuntimeError(f"validation split mismatch: {validation_ids}")
    return eligible


def retrieval_rows(
    validation: list[dict[str, Any]],
    model: dict[str, Any],
    model_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scales = model["predictive_scales"]
    color_scales = np.asarray(
        [
            math.hypot(scales[0], scales[1]),
            math.hypot(scales[1], scales[2]),
            math.hypot(scales[2], scales[3]),
        ]
    )
    output: list[dict[str, Any]] = []
    full_true_ranks: list[int] = []
    color_true_ranks: list[int] = []
    for member in validation:
        prediction = predict_transform(model, member, model_spec)
        predicted_color = predicted_colors(member, prediction)
        candidates: list[dict[str, Any]] = []
        for candidate in validation:
            candidate_offsets = np.asarray(
                [
                    candidate["g"] - member["B"],
                    candidate["r"] - member["R"],
                    candidate["i"] - member["I"],
                    candidate["z"] - member["I"],
                ]
            )
            full_score = float(np.sum(((candidate_offsets - prediction) / scales) ** 2))
            color_score = float(
                np.sum(((observed_colors(candidate) - predicted_color) / color_scales) ** 2)
            )
            candidates.append(
                {
                    "member_object_id": member["object_id"],
                    "candidate_object_id": candidate["object_id"],
                    "candidate_nsc_id": candidate["nsc_id"],
                    "is_provisional_singleton_pair": candidate["object_id"] == member["object_id"],
                    "full_offset_score": full_score,
                    "color_only_score": color_score,
                }
            )
        full_order = sorted(candidates, key=lambda row: (row["full_offset_score"], row["candidate_nsc_id"]))
        color_order = sorted(candidates, key=lambda row: (row["color_only_score"], row["candidate_nsc_id"]))
        full_rank = {row["candidate_nsc_id"]: index + 1 for index, row in enumerate(full_order)}
        color_rank = {row["candidate_nsc_id"]: index + 1 for index, row in enumerate(color_order)}
        for row in candidates:
            row["full_offset_rank"] = full_rank[row["candidate_nsc_id"]]
            row["color_only_rank"] = color_rank[row["candidate_nsc_id"]]
            output.append(row)
            if row["is_provisional_singleton_pair"]:
                full_true_ranks.append(row["full_offset_rank"])
                color_true_ranks.append(row["color_only_rank"])
    metrics = {
        "full_offset_top1": sum(rank == 1 for rank in full_true_ranks),
        "full_offset_mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in full_true_ranks])),
        "color_only_top1": sum(rank == 1 for rank in color_true_ranks),
        "color_only_mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in color_true_ranks])),
        "full_offset_true_ranks": full_true_ranks,
        "color_only_true_ranks": color_true_ranks,
    }
    return output, metrics


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_transform_fit_or_validation_scoring":
        raise RuntimeError("V19AB protocol is not in its frozen pre-fit state")
    validate_artifacts(config)
    sample = load_commissioning_sample(config)
    development = [row for row in sample if row["split"] == "development"]
    validation = [row for row in sample if row["split"] == "validation"]
    model_spec = config["model"]
    model = fit_transform(development, model_spec)

    output_names = list(model_spec["output_offsets"])
    validation_rows: list[dict[str, Any]] = []
    validation_errors: dict[str, list[float]] = {name: [] for name in output_names}
    for row in validation:
        observed = observed_offsets(row)
        predicted = predict_transform(model, row, model_spec)
        record: dict[str, Any] = {
            "cluster": row["cluster"],
            "object_id": row["object_id"],
            "nsc_id": row["nsc_id"],
        }
        for index, name in enumerate(output_names):
            error = float(observed[index] - predicted[index])
            record[f"observed_{name}"] = observed[index]
            record[f"predicted_{name}"] = predicted[index]
            record[f"error_{name}"] = error
            record[f"standardized_error_{name}"] = error / model["predictive_scales"][index]
            validation_errors[name].append(abs(error))
        validation_rows.append(record)

    retrieval, retrieval_metrics = retrieval_rows(validation, model, model_spec)
    median_absolute_error = {
        name: float(np.median(values)) for name, values in validation_errors.items()
    }
    coefficient_rows = []
    for index, name in enumerate(output_names):
        coefficient_rows.append(
            {
                "output": name,
                "intercept": model["coefficient_matrix"][0, index],
                "B_minus_R_standardized_slope": model["coefficient_matrix"][1, index],
                "R_minus_I_standardized_slope": model["coefficient_matrix"][2, index],
                "predictive_scale_mag": model["predictive_scales"][index],
            }
        )

    gates_spec = config["validation_gates"]
    gates = {
        "exact_eligible_and_split_counts": len(sample) == 15
        and len(development) == int(config["split"]["development_count"])
        and len(validation) == int(config["split"]["validation_count"]),
        "all_optimizers_succeeded": all(row["success"] for row in model["optimizer_records"]),
        "validation_median_absolute_error_each_offset": all(
            value <= float(gates_spec["maximum_validation_median_absolute_error_each_offset_mag"])
            for value in median_absolute_error.values()
        ),
        "full_offset_top1_retrieval": retrieval_metrics["full_offset_top1"]
        >= int(gates_spec["minimum_full_offset_top1_retrievals_out_of_5"]),
        "full_offset_mean_reciprocal_rank": retrieval_metrics["full_offset_mean_reciprocal_rank"]
        >= float(gates_spec["minimum_full_offset_mean_reciprocal_rank"]),
        "color_only_top1_retrieval": retrieval_metrics["color_only_top1"]
        >= int(gates_spec["minimum_color_only_top1_retrievals_out_of_5"]),
        "color_only_mean_reciprocal_rank": retrieval_metrics["color_only_mean_reciprocal_rank"]
        >= float(gates_spec["minimum_color_only_mean_reciprocal_rank"]),
        "no_ambiguous_association_mass_lensing_or_gravity": True,
    }
    gates["all_commissioning_gates_pass"] = all(gates.values())

    outputs = config["outputs"]
    sample_path = ROOT / outputs["commissioning_sample"]
    validation_path = ROOT / outputs["validation_predictions"]
    retrieval_path = ROOT / outputs["validation_retrieval"]
    sample_rows = []
    for row in sample:
        offsets = observed_offsets(row)
        sample_rows.append(
            {
                **row,
                **{name: offsets[index] for index, name in enumerate(output_names)},
            }
        )
    write_csv(sample_path, sample_rows)
    write_csv(validation_path, validation_rows)
    write_csv(retrieval_path, retrieval)

    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AB-BESSEL-NSC-TRANSFORM-COMMISSIONING-1.0.0",
        "status": "completed_internal_transform_commissioning",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "implementation": config["implementation"],
        "parent_hashes": {artifact["path"]: artifact["sha256"] for artifact in config["parent_artifacts"]},
        "sample": {
            "eligible_rows": len(sample),
            "development_ids": [row["object_id"] for row in development],
            "validation_ids": [row["object_id"] for row in validation],
            "singleton_rows_are_provisional_anchors_not_ground_truth": True,
        },
        "fit": {
            "coefficients": coefficient_rows,
            "optimizers": model["optimizer_records"],
        },
        "validation": {
            "median_absolute_error_mag": median_absolute_error,
            "retrieval": retrieval_metrics,
        },
        "gates": gates,
        "likelihood_application_authorized": bool(gates["all_commissioning_gates_pass"]),
        "outputs": {
            "commissioning_sample": sample_path.relative_to(ROOT).as_posix(),
            "commissioning_sample_sha256": sha256(sample_path),
            "validation_predictions": validation_path.relative_to(ROOT).as_posix(),
            "validation_predictions_sha256": sha256(validation_path),
            "validation_retrieval": retrieval_path.relative_to(ROOT).as_posix(),
            "validation_retrieval_sha256": sha256(retrieval_path),
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
