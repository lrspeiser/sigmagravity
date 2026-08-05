#!/usr/bin/env python3
"""Validate a signed-flux, amplitude-profiled color likelihood on frozen anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ay_signed_flux_likelihood_validation.json"
SCORE_COLUMNS = [
    "target_member_id",
    "flux_member_id",
    "is_true_pair",
    "log_photometric_score",
    "central_template_amplitude",
    "central_template_chi2",
    "rank_for_target",
]


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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def features(b_mag: float, r_mag: float, i_mag: float) -> np.ndarray:
    return np.asarray([1.0, (b_mag - r_mag - 2.4) / 1.0, (r_mag - i_mag - 1.1) / 0.5])


def color_model(report: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    parameters = report["development_color_model"]["parameters"]
    scales = report["development_color_model"]["predictive_scales"]
    return (
        {key: np.asarray(value, dtype=float) for key, value in parameters.items()},
        {key: float(value) for key, value in scales.items()},
    )


def predict_colors(
    b_mag: float,
    r_mag: float,
    i_mag: float,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    x = features(b_mag, r_mag, i_mag)
    return np.asarray(
        [
            x @ parameters["g_minus_r"],
            x @ parameters["r_minus_i"],
            x @ parameters["i_minus_z"],
        ]
    )


def flux_template(colors: np.ndarray) -> np.ndarray:
    g_minus_r, r_minus_i, i_minus_z = colors
    g_over_r = 10.0 ** (-0.4 * g_minus_r)
    i_over_r = 10.0 ** (0.4 * r_minus_i)
    z_over_r = i_over_r * 10.0 ** (0.4 * i_minus_z)
    return np.asarray([g_over_r, 1.0, i_over_r, z_over_r])


def profile_amplitude(
    flux: np.ndarray, uncertainty: np.ndarray, template: np.ndarray
) -> tuple[float, float]:
    if flux.shape != (4,) or uncertainty.shape != (4,) or template.shape != (4,):
        raise ValueError("profile_amplitude requires four griz values")
    if not (
        np.all(np.isfinite(flux))
        and np.all(np.isfinite(uncertainty))
        and np.all(uncertainty > 0)
        and np.all(np.isfinite(template))
        and np.all(template > 0)
    ):
        raise ValueError("non-finite or non-positive flux likelihood input")
    inverse_variance = uncertainty**-2
    denominator = float(np.sum(template * template * inverse_variance))
    amplitude = max(0.0, float(np.sum(flux * template * inverse_variance)) / denominator)
    chi2 = float(np.sum(((flux - amplitude * template) / uncertainty) ** 2))
    return amplitude, chi2


def quadrature_templates(
    mean_colors: np.ndarray, color_scales: np.ndarray, order: int
) -> list[tuple[float, np.ndarray]]:
    nodes, weights = hermgauss(order)
    templates: list[tuple[float, np.ndarray]] = []
    normalizer = 1.5 * math.log(math.pi)
    for indices in itertools.product(range(order), repeat=3):
        colors = mean_colors + math.sqrt(2.0) * color_scales * nodes[list(indices)]
        log_weight = sum(math.log(float(weights[index])) for index in indices) - normalizer
        templates.append((log_weight, flux_template(colors)))
    return templates


def signed_flux_log_score(
    flux: np.ndarray,
    uncertainty: np.ndarray,
    mean_colors: np.ndarray,
    color_scales: np.ndarray,
    quadrature_order: int,
) -> float:
    terms = []
    for log_weight, template in quadrature_templates(
        mean_colors, color_scales, quadrature_order
    ):
        _, chi2 = profile_amplitude(flux, uncertainty, template)
        terms.append(log_weight - 0.5 * chi2)
    return float(logsumexp(terms))


def anchor_fluxes(
    rows: list[dict[str, str]], member_ids: list[str], bands: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    indexed = {(row["member_id"], row["filter"]): row for row in rows}
    result = {}
    for member_id in member_ids:
        selected = [indexed[(member_id, band)] for band in bands]
        result[member_id] = (
            np.asarray([float(row["stacked_flux"]) for row in selected]),
            np.asarray([float(row["stacked_flux_uncertainty"]) for row in selected]),
        )
    return result


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_flux_space_validation_scoring":
        raise RuntimeError("V19AY validation is not frozen")
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"parent artifact hash changed: {artifact['path']}")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AY runner hash changed")
    prohibited = (
        "score_ambiguous_candidates",
        "select_or_rank_counterparts",
        "infer_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    )
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AY authorizes a prohibited action")

    v19av_report = json.loads((ROOT / config["inputs"]["v19av_report"]).read_text())
    parameters, scales = color_model(v19av_report)
    bands = list(config["likelihood"]["bands"])
    validation_ids = list(config["split"]["validation_ids"])
    color_scales = np.asarray(
        [scales["g_minus_r"], scales["r_minus_i"], scales["i_minus_z"]]
    )
    commissioning = {
        row["object_id"]: row
        for row in read_csv(ROOT / config["inputs"]["commissioning_sample"])
    }
    fluxes = anchor_fluxes(
        read_csv(ROOT / config["inputs"]["anchor_stacks"]), validation_ids, bands
    )

    scores: list[dict[str, Any]] = []
    true_ranks: dict[str, int] = {}
    order = int(config["likelihood"]["gauss_hermite_order_per_color"])
    for target_id in validation_ids:
        source = commissioning[target_id]
        colors = predict_colors(
            float(source["B"]), float(source["R"]), float(source["I"]), parameters
        )
        target_rows = []
        for flux_member_id in validation_ids:
            flux, uncertainty = fluxes[flux_member_id]
            template = flux_template(colors)
            amplitude, chi2 = profile_amplitude(flux, uncertainty, template)
            target_rows.append(
                {
                    "target_member_id": target_id,
                    "flux_member_id": flux_member_id,
                    "is_true_pair": target_id == flux_member_id,
                    "log_photometric_score": signed_flux_log_score(
                        flux, uncertainty, colors, color_scales, order
                    ),
                    "central_template_amplitude": amplitude,
                    "central_template_chi2": chi2,
                }
            )
        target_rows.sort(
            key=lambda row: (-row["log_photometric_score"], row["flux_member_id"])
        )
        for rank, row in enumerate(target_rows, start=1):
            row["rank_for_target"] = rank
            if row["is_true_pair"]:
                true_ranks[target_id] = rank
        scores.extend(target_rows)

    top1 = sum(rank == 1 for rank in true_ranks.values())
    mrr = sum(1.0 / rank for rank in true_ranks.values()) / len(true_ranks)
    gates = config["validation_gates"]
    gate_results = {
        "exact_score_rows": len(scores) == int(gates["exact_score_rows"]),
        "all_scores_finite": all(
            math.isfinite(float(row["log_photometric_score"])) for row in scores
        ),
        "minimum_top1": top1 >= int(gates["minimum_top1_retrievals"]),
        "minimum_mrr": mrr >= float(gates["minimum_mean_reciprocal_rank"]),
        "no_ambiguous_candidate_scoring": True,
    }
    passed = all(gate_results.values())
    scores_path = ROOT / config["outputs"]["validation_scores"]
    write_csv(scores_path, scores, SCORE_COLUMNS)
    report = {
        "protocol_version": config["protocol_version"],
        "decision": "passed" if passed else "failed_closed",
        "validation": {
            "top1_retrievals": top1,
            "mean_reciprocal_rank": mrr,
            "true_pair_ranks": true_ranks,
        },
        "likelihood": {
            "bands": bands,
            "gauss_hermite_order_per_color": order,
            "quadrature_templates_per_score": order**3,
            "color_predictive_scales_mag": color_scales.tolist(),
            "brightness_amplitude_profiled_nonnegative": True,
            "signed_flux_retained": True,
        },
        "gate_results": gate_results,
        "ambiguous_candidate_scoring_performed": False,
        "outputs": {
            "validation_scores": scores_path.relative_to(ROOT).as_posix(),
            "validation_scores_sha256": sha256(scores_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / config["outputs"]["report"]
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
