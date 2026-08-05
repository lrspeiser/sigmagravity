#!/usr/bin/env python3
"""Run the frozen V19AV signed-flux multi-epoch stack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19av_signed_flux_stack.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


V19AS = load_module(
    "sigma_v19as_frozen_for_v19av",
    ROOT / "scripts" / "run_sigma_v19as_decam_forced_photometry_development.py",
)
V19AT = load_module(
    "sigma_v19at_frozen_for_v19av",
    ROOT / "scripts" / "run_sigma_v19at_decam_forced_photometry_validation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flux_uncertainty(row: dict[str, str]) -> float:
    if row.get("flux_uncertainty"):
        value = float(row["flux_uncertainty"])
        if np.isfinite(value) and value > 0:
            return value
    usable = int(float(row["usable_aperture_pixels"]))
    total = int(float(row["total_aperture_pixels"]))
    noise = float(row["background_noise"])
    fraction = usable / total if total else 0.0
    return noise * math.sqrt(max(usable, 1)) / fraction if fraction > 0 else float("nan")


def normalize_flux(row: dict[str, str], reference_zero_point: float) -> tuple[float, float]:
    flux = float(row["flux"])
    sigma = flux_uncertainty(row)
    magzero = float(row["magzero"])
    factor = 10.0 ** (0.4 * (reference_zero_point - magzero))
    return flux * factor, sigma * factor


def robust_stack(
    fluxes: np.ndarray,
    uncertainties: np.ndarray,
    huber_threshold: float,
    iterations: int,
) -> dict[str, float | int]:
    finite = np.isfinite(fluxes) & np.isfinite(uncertainties) & (uncertainties > 0)
    values = fluxes[finite]
    sigma = uncertainties[finite]
    if values.size == 0:
        return {
            "finite_exposures": 0,
            "stacked_flux": float("nan"),
            "stacked_flux_uncertainty": float("nan"),
            "stacked_signal_to_noise": float("nan"),
            "robust_flux_scatter": float("nan"),
            "huber_downweighted_exposures": 0,
        }
    base_weight = 1.0 / sigma**2
    mean = float(np.sum(base_weight * values) / np.sum(base_weight))
    robust_weight = np.ones(values.size, dtype=float)
    for _ in range(iterations):
        residual = (values - mean) / sigma
        robust_weight = np.minimum(1.0, huber_threshold / np.maximum(np.abs(residual), 1e-12))
        weight = base_weight * robust_weight
        updated = float(np.sum(weight * values) / np.sum(weight))
        if math.isclose(updated, mean, rel_tol=1e-10, abs_tol=1e-12):
            mean = updated
            break
        mean = updated
    weight = base_weight * robust_weight
    formal = math.sqrt(1.0 / float(np.sum(weight)))
    scatter = float(V19AS.robust_sigma(values)) if values.size >= 2 else float("nan")
    scatter_error = scatter / math.sqrt(values.size) if np.isfinite(scatter) else 0.0
    uncertainty = max(formal, scatter_error)
    return {
        "finite_exposures": int(values.size),
        "stacked_flux": mean,
        "stacked_flux_uncertainty": uncertainty,
        "stacked_signal_to_noise": mean / uncertainty if uncertainty > 0 else float("nan"),
        "robust_flux_scatter": scatter,
        "huber_downweighted_exposures": int(np.sum(robust_weight < 1.0)),
    }


def stack_rows(
    rows: list[dict[str, str]],
    id_field: str,
    ids: list[str],
    filters: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row[id_field], row["filter"])].append(
            normalize_flux(row, float(config["stack"]["reference_zero_point_mag"]))
        )
    output: list[dict[str, Any]] = []
    for object_id in ids:
        for band in filters:
            pairs = grouped.get((object_id, band), [])
            fluxes = np.asarray([pair[0] for pair in pairs])
            uncertainties = np.asarray([pair[1] for pair in pairs])
            result = robust_stack(
                fluxes,
                uncertainties,
                float(config["stack"]["huber_threshold_sigma"]),
                int(config["stack"]["maximum_iterations"]),
            )
            flux = float(result["stacked_flux"])
            sigma = float(result["stacked_flux_uncertainty"])
            magnitude = (
                float(config["stack"]["reference_zero_point_mag"]) - 2.5 * math.log10(flux)
                if np.isfinite(flux) and flux > 0
                else float("nan")
            )
            magnitude_uncertainty = (
                2.5 / math.log(10.0) * sigma / flux
                if np.isfinite(sigma) and np.isfinite(flux) and flux > 0
                else float("nan")
            )
            output.append(
                {
                    id_field: object_id,
                    "filter": band,
                    "planned_exposures": len(pairs),
                    **result,
                    "stacked_magnitude": magnitude,
                    "stacked_magnitude_uncertainty": magnitude_uncertainty,
                }
            )
    return output


def fit_anchor_color_model(
    config: dict[str, Any],
    stacks: list[dict[str, Any]],
    bri: dict[str, dict[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, float], list[dict[str, Any]]]:
    magnitudes = {
        (row["member_id"], row["filter"]): float(row["stacked_magnitude"])
        for row in stacks
    }
    development_ids = config["split"]["development_ids"]
    parameters: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    fit_rows: list[dict[str, Any]] = []
    for first, second in (("g", "r"), ("r", "i"), ("i", "z")):
        name = f"{first}_minus_{second}"
        features = np.vstack([V19AT.feature(bri[member]) for member in development_ids])
        observed = np.asarray(
            [magnitudes[(member, first)] - magnitudes[(member, second)] for member in development_ids]
        )
        fitted = V19AS.affine_fit(features, observed, float(config["color_model"]["ridge_penalty"]))
        residual = observed - features @ fitted
        scale = max(
            float(config["color_model"]["predictive_scale_floor_mag"]),
            float(V19AS.robust_sigma(residual)),
        )
        parameters[name] = fitted
        scales[name] = scale
        for member, value, prediction, difference in zip(
            development_ids, observed, features @ fitted, residual
        ):
            fit_rows.append(
                {
                    "color": name,
                    "member_id": member,
                    "observed_color": value,
                    "fitted_color": prediction,
                    "residual": difference,
                    "predictive_scale": scale,
                }
            )
    return parameters, scales, fit_rows


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for parent in config["parent_artifacts"]:
        path = ROOT / parent["path"]
        if sha256(path) != parent["sha256"]:
            raise RuntimeError(f"parent artifact hash changed: {parent['path']}")

    development = [
        row
        for row in read_csv(ROOT / config["inputs"]["development_measurements"])
        if row["variant"] == "area_scaled" and float(row["aperture_diameter_arcsec"]) == 4.0
    ]
    validation = read_csv(ROOT / config["inputs"]["validation_measurements"])
    candidates = read_csv(ROOT / config["inputs"]["candidate_measurements"])
    hypotheses = read_csv(ROOT / config["inputs"]["candidate_hypotheses"])
    bri_rows = read_csv(ROOT / config["inputs"]["commissioning_sample"])
    bri = {row["object_id"]: row for row in bri_rows}
    if len(development) != 670 or len(validation) != 362 or len(candidates) != 40812:
        raise RuntimeError("frozen measurement row count changed")

    filters = list(config["stack"]["filters"])
    anchor_ids = config["split"]["development_ids"] + config["split"]["validation_ids"]
    anchor_stacks = stack_rows(
        development + validation,
        "member_id",
        anchor_ids,
        filters,
        config,
    )
    candidate_ids = sorted({row["candidate_id"] for row in candidates})
    candidate_stacks = stack_rows(candidates, "candidate_id", candidate_ids, filters, config)

    parameters, scales, fit_rows = fit_anchor_color_model(config, anchor_stacks, bri)
    anchor_magnitudes = {
        (row["member_id"], row["filter"]): float(row["stacked_magnitude"])
        for row in anchor_stacks
    }
    predictions, retrieval, validation_metrics = V19AT.score_validation(
        config,
        anchor_magnitudes,
        bri,
        parameters,
        scales,
    )

    detection_threshold = float(config["stack"]["candidate_detection_signal_to_noise"])
    candidate_snr = {
        (row["candidate_id"], row["filter"]): float(row["stacked_signal_to_noise"])
        for row in candidate_stacks
    }
    complete_griz = {
        candidate_id
        for candidate_id in candidate_ids
        if all(candidate_snr[(candidate_id, band)] >= detection_threshold for band in "griz")
    }
    member_candidates: dict[str, set[str]] = defaultdict(set)
    for row in hypotheses:
        member_candidates[row["member_id"]].add(row["candidate_id"])
    members_with_complete_candidate = sum(
        bool(candidates_for_member & complete_griz)
        for candidates_for_member in member_candidates.values()
    )

    gates = config["gates"]
    gate_results = {
        "all_stack_rows_present": len(candidate_stacks) == 568 * 5
        and len(anchor_stacks) == 15 * 5,
        "validation_color_error": all(
            value <= float(gates["maximum_validation_median_absolute_error_each_color_mag"])
            for value in validation_metrics["median_absolute_error_mag"].values()
        ),
        "validation_top1": validation_metrics["top1_retrievals"]
        >= int(gates["minimum_validation_top1_retrievals"]),
        "validation_mrr": validation_metrics["mean_reciprocal_rank"]
        >= float(gates["minimum_validation_mean_reciprocal_rank"]),
        "candidate_complete_griz_fraction": len(complete_griz) / len(candidate_ids)
        >= float(gates["minimum_candidate_complete_griz_fraction"]),
        "every_member_has_complete_candidate": members_with_complete_candidate
        == int(gates["exact_members"]),
        "no_candidate_association_scored": True,
    }
    passed = all(gate_results.values())

    outputs = config["outputs"]
    products = {
        "anchor_stacks": (anchor_stacks, list(anchor_stacks[0])),
        "candidate_stacks": (candidate_stacks, list(candidate_stacks[0])),
        "development_color_fit": (fit_rows, list(fit_rows[0])),
        "validation_predictions": (predictions, list(predictions[0])),
        "validation_retrieval": (retrieval, list(retrieval[0])),
    }
    output_metadata: dict[str, str] = {}
    for name, (rows_to_write, fields) in products.items():
        path = ROOT / outputs[name]
        write_csv(path, rows_to_write, fields)
        output_metadata[name] = path.relative_to(ROOT).as_posix()
        output_metadata[f"{name}_sha256"] = sha256(path)

    report = {
        "protocol_version": config["protocol_version"],
        "decision": "passed" if passed else "failed_closed",
        "counts": {
            "anchor_stack_rows": len(anchor_stacks),
            "candidate_stack_rows": len(candidate_stacks),
            "candidate_complete_griz": len(complete_griz),
            "members_with_at_least_one_complete_candidate": members_with_complete_candidate,
        },
        "candidate_complete_griz_fraction": len(complete_griz) / len(candidate_ids),
        "development_color_model": {
            "parameters": {name: values.tolist() for name, values in parameters.items()},
            "predictive_scales": scales,
        },
        "validation_metrics": validation_metrics,
        "gate_results": gate_results,
        "candidate_association_or_bri_score_computed": False,
        "outputs": output_metadata,
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
