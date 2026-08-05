#!/usr/bin/env python3
"""Run the frozen V19L fixture-corrected smooth-null front test."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np
import sigma_v19f_chandra_common as common
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import closing, disk

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19l_fixture_corrected_front_likelihood.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19l_fixture_corrected_fronts"
V19K_RUNNER = ROOT / "scripts" / "run_sigma_v19k_smooth_null_fronts.py"


def modules():
    v19k = common.load_module(V19K_RUNNER, "sigma_v19k_for_v19l")
    return v19k, v19k.v19j_module()


def fit_profile(profile: dict, config: dict, v19k) -> dict:
    positive = profile["positive"]
    centers = profile["centers_kpc"][positive]
    tau = centers / 200.0
    counts = profile["counts"][positive]
    background = profile["background"][positive]
    exposure = profile["exposure"][positive]
    inherited = config["poisson_models"]
    correction = config["v19l_correction"]["continuous_null_correction"]
    inherited_bounds = inherited["parameter_bounds"]
    correction_bounds = correction["new_nuisance_bounds"]
    null_bounds = [
        tuple(inherited_bounds["a"]),
        tuple(inherited_bounds["b"]),
        tuple(inherited_bounds["c"]),
        tuple(correction_bounds["e"]),
        tuple(correction_bounds["f"]),
    ]
    alternative_bounds = null_bounds + [tuple(inherited_bounds["d"])]
    net_rate = max(float(np.sum(counts - background) / np.sum(exposure)), math.exp(-40.0))
    initial_a = float(np.clip(math.log(net_rate), *null_bounds[0]))
    null_design = np.column_stack(
        [np.ones(len(tau)), tau, tau**2, tau**3, tau**4]
    )
    null = v19k.fit_design(
        np.asarray([initial_a, 0.0, 0.0, 0.0, 0.0]),
        null_bounds,
        null_design,
        counts,
        background,
        exposure,
    )
    alternatives = []
    if null["success"]:
        for delta in inherited["step_location_grid_kpc"]:
            step = (centers >= float(delta)).astype(float)
            design = np.column_stack([null_design, step])
            initial = np.concatenate([null["parameters"], [math.log(4.0)]])
            fit = v19k.fit_design(
                initial,
                alternative_bounds,
                design,
                counts,
                background,
                exposure,
            )
            fit["delta_kpc"] = float(delta)
            alternatives.append(fit)
    finite = [row for row in alternatives if row["success"]]
    if not null["success"] or not finite:
        return {
            "success": False,
            "null": v19k.serialize_fit(null),
            "alternatives": [v19k.serialize_fit(row) for row in alternatives],
        }
    finite.sort(
        key=lambda row: (-row["log_likelihood"], abs(row["delta_kpc"]), row["delta_kpc"])
    )
    best = finite[0]
    delta_cash = max(0.0, 2.0 * (best["log_likelihood"] - null["log_likelihood"]))
    return {
        "success": True,
        "null": v19k.serialize_fit(null),
        "best_alternative": v19k.serialize_fit(best),
        "delta_cash": float(delta_cash),
        "step_score_sigma": float(math.sqrt(delta_cash)),
        "density_compression": float(math.exp(best["parameters"][5] / 2.0)),
        "tested_step_locations_kpc": [float(row["delta_kpc"]) for row in alternatives],
        "finite_alternative_count": len(finite),
    }


def fit_seeds(
    seeds: list[dict],
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    pixel_kpc: float,
    config: dict,
    v19k,
) -> list[dict]:
    records = []
    for index, seed in enumerate(seeds, start=1):
        profile = v19k.extract_profile(
            seed,
            counts,
            background,
            background_variance,
            exposure,
            mask,
            pixel_kpc,
            config["profile_extraction"],
        )
        record = dict(seed)
        record["profile_valid"] = bool(profile["valid"])
        record["profile_valid_fraction"] = float(profile.get("valid_fraction", 0.0))
        if profile["valid"]:
            fit = fit_profile(profile, config, v19k)
            record["fit"] = fit
            record["passes_step_score"] = bool(
                fit.get("success", False)
                and fit.get("step_score_sigma", 0.0)
                >= float(config["poisson_models"]["minimum_discontinuity_score_sigma"])
            )
        else:
            record["profile_failure"] = profile["failure"]
            record["fit"] = {"success": False}
            record["passes_step_score"] = False
        records.append(record)
        if index % 100 == 0:
            print(f"seed fits {index}/{len(seeds)}", flush=True)
    return records


def segment_is_continuous(
    first: dict, second: dict, closed_candidate: np.ndarray
) -> bool:
    distance_pixels = math.hypot(
        second["row"] - first["row"], second["column"] - first["column"]
    )
    samples = max(2, math.ceil(2.0 * distance_pixels) + 1)
    rows = np.linspace(first["row"], second["row"], samples)
    columns = np.linspace(first["column"], second["column"], samples)
    values = ndimage.map_coordinates(
        closed_candidate.astype(float),
        [rows, columns],
        order=0,
        mode="constant",
        cval=0.0,
    )
    return bool(np.all(values >= 0.5))


def link_arcs(
    nodes: list[dict],
    candidate: np.ndarray,
    pixel_kpc: float,
    config: dict,
    v19k,
    v19j,
) -> list[dict]:
    passed = [node for node in nodes if node["passes_step_score"]]
    if not passed:
        return []
    correction = config["v19l_correction"]["ridge_continuity_correction"]
    closing_radius = math.floor(
        (float(correction["physical_maximum_empty_gap_kpc"]) / 2.0) / pixel_kpc
    )
    closed_candidate = closing(candidate, disk(max(0, closing_radius)))
    coordinates = np.asarray([[node["x_kpc"], node["y_kpc"]] for node in passed])
    pairs = cKDTree(coordinates).query_pairs(
        float(correction["maximum_seed_neighbor_distance_kpc"])
    )
    union = v19k.UnionFind(len(passed))
    max_normal = math.radians(
        float(config["arc_linking"]["maximum_normal_difference_deg"])
    )
    max_tangent = math.radians(30.0)
    for first, second in sorted(pairs):
        if v19k.circular_difference(
            passed[first]["normal_rad"], passed[second]["normal_rad"]
        ) > max_normal:
            continue
        if v19k.tangent_alignment(passed[first], passed[second]) > max_tangent:
            continue
        if v19k.tangent_alignment(passed[second], passed[first]) > max_tangent:
            continue
        if not segment_is_continuous(passed[first], passed[second], closed_candidate):
            continue
        union.union(first, second)
    components: dict[int, list[dict]] = {}
    for index, node in enumerate(passed):
        components.setdefault(union.find(index), []).append(node)
    arcs = []
    arc_config = config["arc_linking"]
    for component in components.values():
        if len(component) < int(arc_config["minimum_component_nodes"]):
            continue
        x = np.asarray([node["x_kpc"] for node in component])
        y = np.asarray([node["y_kpc"] for node in component])
        circle = v19j.circle_fit(x, y)
        if circle is None:
            continue
        center_x, center_y, radius = circle
        if not (
            float(arc_config["curvature_radius_kpc"][0])
            <= radius
            <= float(arc_config["curvature_radius_kpc"][1])
        ):
            continue
        radial = np.hypot(x - center_x, y - center_y)
        rms_residual = float(np.sqrt(np.mean((radial - radius) ** 2)))
        if rms_residual > float(arc_config["maximum_rms_radial_residual_kpc"]):
            continue
        angular = np.arctan2(y - center_y, x - center_x)
        span = v19k.arc_span(angular)
        length = radius * span
        if length < float(arc_config["minimum_projected_length_kpc"]):
            continue
        weights = np.asarray([node["fit"]["delta_cash"] for node in component])
        total_weight = float(np.sum(weights))
        normal_value = np.sum(
            weights
            * np.exp(1j * np.asarray([node["normal_rad"] for node in component]))
        )
        if total_weight <= 0.0 or abs(normal_value) <= np.finfo(float).eps:
            continue
        arcs.append(
            {
                "arc_id": 0,
                "node_count": len(component),
                "node_seed_ids": [int(node["seed_id"]) for node in component],
                "circle_center_x_kpc": float(center_x),
                "circle_center_y_kpc": float(center_y),
                "curvature_radius_kpc": float(radius),
                "rms_radial_residual_kpc": rms_residual,
                "angular_span_deg": math.degrees(span),
                "projected_length_kpc": float(length),
                "representative_x_kpc": float(np.sum(weights * x) / total_weight),
                "representative_y_kpc": float(np.sum(weights * y) / total_weight),
                "normal_faint_to_bright_deg": float(
                    math.degrees(np.angle(normal_value) % (2.0 * math.pi))
                ),
                "component_delta_cash": total_weight,
                "median_step_score_sigma": float(
                    np.median([node["fit"]["step_score_sigma"] for node in component])
                ),
                "median_density_compression": float(
                    np.median([node["fit"]["density_compression"] for node in component])
                ),
            }
        )
    arcs.sort(
        key=lambda row: (
            -row["component_delta_cash"],
            -row["projected_length_kpc"],
            row["representative_x_kpc"],
            row["representative_y_kpc"],
        )
    )
    for index, arc in enumerate(arcs, start=1):
        arc["arc_id"] = index
    return arcs


def analyze_arrays(
    counts: np.ndarray,
    background: np.ndarray,
    background_variance: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    front: dict,
    pixel_kpc: float,
    center_x: float,
    center_y: float,
    config: dict,
    v19k,
    v19j,
) -> tuple[list[dict], list[dict]]:
    seeds = v19k.local_maximum_seeds(
        front["candidate"],
        front["best_score"],
        front["best_angle"],
        front["best_scale"],
        pixel_kpc,
        10.0,
        center_x,
        center_y,
    )
    fitted = fit_seeds(
        seeds,
        counts,
        background,
        background_variance,
        exposure,
        mask,
        pixel_kpc,
        config,
        v19k,
    )
    return fitted, link_arcs(
        fitted, front["candidate"], pixel_kpc, config, v19k, v19j
    )


def mandatory_fixtures(config: dict, v19k, v19j) -> dict:
    v19k_config = config["v19k_base"]
    v19j_config = common.load_json(ROOT / v19k_config["parents"]["v19j_config"])
    results = {}
    for kind in ("uniform", "linear", "radial", "step", "masked_step"):
        counts, background, background_variance, exposure, mask = v19k.fixture_maps(kind)
        front = v19j.detect_fronts(
            counts,
            background,
            background_variance,
            exposure,
            mask,
            4.0,
            v19j_config,
            192.5,
            192.5,
        )
        seeds, arcs = analyze_arrays(
            counts,
            background,
            background_variance,
            exposure,
            mask,
            front,
            4.0,
            192.5,
            192.5,
            config,
            v19k,
            v19j,
        )
        passing = [seed for seed in seeds if seed["passes_step_score"]]
        record = {
            "v19j_candidate_pixels": int(np.sum(front["candidate"])),
            "v19l_seed_count": len(seeds),
            "passing_seed_count": len(passing),
            "retained_arc_count": len(arcs),
        }
        if kind == "step":
            if arcs:
                distances = [
                    abs(math.hypot(seed["x_kpc"], seed["y_kpc"]) - 300.0)
                    for seed in passing
                ]
                compressions = [
                    seed["fit"]["density_compression"] for seed in passing
                ]
                record["median_passing_seed_distance_to_circle_kpc"] = float(
                    np.median(distances)
                )
                record["median_passing_seed_density_compression"] = float(
                    np.median(compressions)
                )
                record["passed"] = bool(
                    record["median_passing_seed_distance_to_circle_kpc"] <= 16.0
                    and 1.5
                    <= record["median_passing_seed_density_compression"]
                    <= 2.5
                )
            else:
                record["passed"] = False
        elif kind in {"uniform", "linear", "radial"}:
            record["passed"] = len(passing) == 0 and len(arcs) == 0
        else:
            record["passed"] = len(arcs) == 0
        results[kind] = record
        print(f"fixture {kind}: {record}", flush=True)
    return results


def run_cluster(
    config: dict,
    v19j_report: dict,
    cluster: str,
    output: Path,
    v19k,
    v19j,
) -> dict:
    values = v19k.load_science_inputs(config["v19k_base"], v19j_report, cluster, v19j)
    counts, background, background_variance, exposure, mask, front, pixel_kpc, center_x, center_y = values
    seeds, arcs = analyze_arrays(
        counts,
        background,
        background_variance,
        exposure,
        mask,
        front,
        pixel_kpc,
        center_x,
        center_y,
        config,
        v19k,
        v19j,
    )
    stem = cluster.lower()
    seed_path = output / f"{stem}_seed_fits.json"
    seed_path.write_text(json.dumps(seeds, separators=(",", ":")) + "\n", encoding="utf-8")
    arc_path = output / f"{stem}_arc_catalog.json"
    arc_path.write_text(json.dumps(arcs, indent=2) + "\n", encoding="utf-8")
    figure_path = output / f"{stem}_diagnostic.png"
    v19k.render_diagnostic(
        figure_path, counts, background, exposure, mask, seeds, arcs, cluster
    )
    products = []
    for kind, path in (
        ("seed_profile_likelihoods", seed_path),
        ("retained_arc_catalog", arc_path),
        ("post_likelihood_diagnostic", figure_path),
    ):
        products.append(
            {
                "kind": kind,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    attempted = [seed for seed in seeds if seed["profile_valid"]]
    failed = [seed for seed in attempted if not seed["fit"].get("success", False)]
    failed_fraction = len(failed) / len(attempted) if attempted else 1.0
    gates = {
        "maximum_failed_seed_fraction": failed_fraction
        <= float(config["v19l_correction"]["science_gate"]["maximum_failed_seed_fraction"]),
        "minimum_retained_arcs": len(arcs)
        >= int(config["v19l_correction"]["science_gate"]["minimum_retained_arcs_per_cluster"]),
        "every_retained_arc_passes_geometry": all(
            arc["node_count"] >= int(config["arc_linking"]["minimum_component_nodes"])
            and arc["projected_length_kpc"]
            >= float(config["arc_linking"]["minimum_projected_length_kpc"])
            and arc["rms_radial_residual_kpc"]
            <= float(config["arc_linking"]["maximum_rms_radial_residual_kpc"])
            for arc in arcs
        ),
    }
    print(
        f"{cluster}: {len(seeds)} seeds, {len(attempted)} profiles, "
        f"{sum(seed['passes_step_score'] for seed in seeds)} passing, {len(arcs)} arcs",
        flush=True,
    )
    return {
        "cluster": cluster,
        "pixel_scale_kpc": pixel_kpc,
        "seed_count": len(seeds),
        "profile_valid_seed_count": len(attempted),
        "successful_seed_fit_count": len(attempted) - len(failed),
        "failed_seed_fit_count": len(failed),
        "failed_seed_fit_fraction": failed_fraction,
        "passing_step_seed_count": sum(seed["passes_step_score"] for seed in seeds),
        "retained_arc_count": len(arcs),
        "arcs": arcs,
        "products": products,
        "gates": gates,
    }


def resolved_config(config: dict) -> dict:
    base = common.load_json(ROOT / config["parents"]["v19k_config"])
    merged = dict(base)
    merged["v19l_correction"] = {
        "continuous_null_correction": config["continuous_null_correction"],
        "ridge_continuity_correction": config["ridge_continuity_correction"],
        "science_gate": config["science_gate"],
    }
    merged["v19k_base"] = base
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    frozen = common.load_json(config_path)
    common.validate_parent_hashes(frozen)
    config = resolved_config(frozen)
    v19k, v19j = modules()
    fixtures = mandatory_fixtures(config, v19k, v19j)
    if not all(row["passed"] for row in fixtures.values()):
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "mandatory_pre_science_fixture_failure",
            "protocol_version": frozen["protocol_version"],
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": common.sha256(config_path),
            "runner_sha256": common.sha256(Path(__file__).resolve()),
            "mandatory_pre_science_fixtures": fixtures,
            "cluster_science_array_read": False,
            "science_seed_fitted": False,
            "science_arc_linked": False,
            "lensing_target_opened": False,
            "gravity_formula_or_parameter_changed": False,
        }
        failure_path = output / "fixture_failure.json"
        failure_path.write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(f"fixture failure: {failure_path}")
        print(f"sha256: {common.sha256(failure_path)}")
        raise RuntimeError(f"a mandatory pre-science V19L fixture failed: {fixtures}")
    v19j_report = common.load_json(ROOT / frozen["parents"]["v19j_report"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clusters = [
        run_cluster(config, v19j_report, cluster, output, v19k, v19j)
        for cluster in config["sample"]["clusters"]
    ]
    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    report = {
        "status": (
            "both_clusters_passed_frozen_v19l_fixture_corrected_front_map_gate"
            if not failed
            else "frozen_v19l_fixture_corrected_front_map_gate_failed"
        ),
        "protocol_version": frozen["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "runner_sha256": common.sha256(Path(__file__).resolve()),
        "mandatory_pre_science_fixtures": fixtures,
        "failed_clusters": failed,
        "clusters": clusters,
        "final_broken_power_law_profile_fit_run": False,
        "parametric_bootstrap_run": False,
        "spectrum_or_response_constructed": False,
        "shock_classification_run": False,
        "post_hash_visual_audit_run": False,
        "published_front_coordinate_used": False,
        "lensing_target_opened": False,
        "gravity_formula_selected": False,
        "gravity_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"report: {report_path}")
    print(f"sha256: {common.sha256(report_path)}")


if __name__ == "__main__":
    main()
