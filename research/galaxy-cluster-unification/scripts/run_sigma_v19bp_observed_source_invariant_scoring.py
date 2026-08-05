#!/usr/bin/env python3
"""Execute the frozen source-only I4/I5 decision after V19X4 and V19BM pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_gas_source_stream import (
    REQUIRED_GAS_FIELDS,
    append_feature_batch,
    concatenate_feature_batches,
    gas_feature_batch,
)
from voidscreen.sigma_source_invariants import (
    axial_angle_deg,
    axial_difference_deg,
    symmetric_fractional_change,
)
from voidscreen.sigma_source_score_engine import (
    gradient_support_mask,
    i4_draw_summary,
    i5_draw_summary,
    leave_one_region_out_stability,
    posterior_feature_summary,
    posterior_novelty_scores,
)

FloatArray = NDArray[np.float64]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19bp_observed_source_invariant_scoring.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_static(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19BP parent changed: {name}")
        hashes[name] = actual
    implementation = config["implementation"]
    for role in ("runner", "gas_stream_module", "score_engine_module"):
        path = ROOT / implementation[role]
        actual = sha256(path)
        if actual != implementation[f"{role}_sha256"]:
            raise RuntimeError(f"V19BP implementation changed: {role}")
        hashes[role] = actual
    if (ROOT / implementation["runner"]).resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BP configuration names another runner")
    return hashes


def build_preflight_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    hashes = validate_static(config)
    variants = config["variants"]
    thresholds = config["thresholds"]
    v19bn = load_json(ROOT / config["parents"]["v19bn_config"]["path"])
    preflight_reports = [
        load_json(ROOT / config["parents"][name]["path"])
        for name in (
            "v19bl_report",
            "v19bm_preflight_report",
            "v19bn_report",
            "v19bo_report",
        )
    ]
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hashes),
        "all_parent_preflight_gates_pass": all(
            report.get("gates") and all(report["gates"].values())
            for report in preflight_reports
        ),
        "two_clusters_three_branches_and_six_variants_frozen": (
            config["registered_inputs"]["clusters"] == ["BULLET", "ABELL2146"]
            and config["registered_inputs"]["rank_correlations"] == [-0.9, 0.0, 0.9]
            and len(variants["smoothing_fwhm_kpc"]) * len(variants["radii_kpc"])
            == 6
        ),
        "primary_variant_is_50kpc_r350kpc": (
            variants["primary_smoothing_fwhm_kpc"] == 50.0
            and variants["primary_radius_kpc"] == 350.0
        ),
        "all_v19bl_thresholds_copied_exactly": thresholds
        == config["inherited_v19bl_thresholds"]
        == v19bn["inherited_thresholds"],
        "i4_direction_required_before_any_action_derivation": config[
            "decision_rule"
        ]["requires_I4_direction_in_both_clusters_all_branches"],
        "lensing_halo_action_gravity_and_holdout_sealed": (
            not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["derive_action_now"]
            and not config["authorization"]["change_gravity_formula_or_parameter"]
            and not config["authorization"]["open_holdout"]
        ),
        "terminal_inputs_not_authorized_before_both_pass": (
            config["authorization"]["run_after_terminal_v19x4_and_v19bm"]
            and not config["authorization"]["run_before_terminal_inputs"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "decision": (
            "passed_observed_source_executor_preflight_awaiting_terminal_inputs"
            if all(gates.values())
            else "failed_closed"
        ),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": hashes,
        "gates": gates,
        "terminal_v19x4_or_v19bm_opened": False,
        "observed_source_score_computed": False,
        "lensing_halo_action_or_gravity_payload_opened": False,
        "claim_boundary": config["claim_boundary"],
    }


def validate_product(path: Path, product: dict[str, Any], authority: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"V19BP missing {authority} product: {path}")
    if path.stat().st_size != int(product["bytes"]):
        raise RuntimeError(f"V19BP resized {authority} product: {path}")
    if sha256(path) != product["sha256"]:
        raise RuntimeError(f"V19BP changed {authority} product: {path}")


def validate_terminal_reports(
    config: dict[str, Any], x4_report_path: Path, bm_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    x4 = load_json(x4_report_path)
    bm = load_json(bm_report_path)
    runtime = config["terminal_authorization"]
    if (
        x4.get("status") != runtime["required_v19x4_status"]
        or x4.get("config_sha256")
        != config["parents"]["v19x4_config"]["sha256"]
        or not x4.get("source_invariant_scoring_authorized")
        or not x4.get("gates")
        or not all(x4["gates"].values())
        or x4.get("lensing_or_halo_payload_opened") is not False
    ):
        raise RuntimeError("V19BP requires a passing target-sealed V19X4 report")
    if (
        bm.get("status") != runtime["required_v19bm_status"]
        or bm.get("config_sha256")
        != config["parents"]["v19bm_config"]["sha256"]
        or not bm.get("invariant_scoring_ready")
        or not bm.get("gates")
        or not all(bm["gates"].values())
        or bm.get("lensing_halo_action_or_gravity_payload_opened") is not False
    ):
        raise RuntimeError("V19BP requires a passing target-sealed V19BM report")
    if len(x4.get("products", [])) != 12 or len(bm.get("products", [])) != 2:
        raise RuntimeError("V19BP terminal product inventory changed")
    for product in x4["products"]:
        validate_product(ROOT / product["relative_path"], product, "V19X4")
    for product in bm["products"]:
        validate_product(ROOT / product["relative_path"], product, "V19BM")
    return x4, bm


def product_for(
    products: list[dict[str, Any]],
    *,
    cluster: str,
    role: str,
    rank_correlation: float | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in products
        if row["cluster"] == cluster
        and row["role"] == role
        and (
            rank_correlation is None
            or math.isclose(
                float(row["rank_correlation"]), float(rank_correlation), abs_tol=1.0e-12
            )
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"V19BP expected one {cluster} {role} rho={rank_correlation}: {len(matches)}"
        )
    return matches[0]


def variant_token(fwhm_kpc: float, radius_kpc: float) -> str:
    return f"{float(fwhm_kpc):g}kpc_r{float(radius_kpc):g}kpc"


def feature(features: dict[str, FloatArray], name: str, token: str) -> FloatArray:
    return np.asarray(features[f"{name}_{token}"], dtype=float)


def component_loo_pass_fractions(
    response: FloatArray,
    support: NDArray[np.bool_],
    *,
    candidate: str,
    maximum_activation_change_fraction: float,
    maximum_axis_change_deg: float,
) -> dict[str, float]:
    values = np.asarray(response, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    indices = np.flatnonzero(support)
    median = np.median(values[:, indices], axis=0)
    if candidate == "I4":
        base_activation = math.sqrt(float(np.mean(2.0 * np.sum(median**2, axis=1))))
        base_axis = float(axial_angle_deg(np.mean(median[:, 0]), np.mean(median[:, 1])))
    else:
        base_activation = float(np.mean(median[:, 0]))
        base_axis = math.nan
    activation_passes: list[bool] = []
    axis_passes: list[bool] = []
    for omitted in range(indices.size):
        retained = np.ones(indices.size, dtype=bool)
        retained[omitted] = False
        subset = median[retained]
        if candidate == "I4":
            activation = math.sqrt(float(np.mean(2.0 * np.sum(subset**2, axis=1))))
            axis = float(axial_angle_deg(np.mean(subset[:, 0]), np.mean(subset[:, 1])))
            axis_change = float(axial_difference_deg(axis, base_axis))
        else:
            activation = float(np.mean(subset[:, 0]))
            axis_change = 0.0
        activation_passes.append(
            float(symmetric_fractional_change(activation, base_activation))
            <= maximum_activation_change_fraction
        )
        axis_passes.append(axis_change <= maximum_axis_change_deg)
    return {
        "activation_pass_fraction": float(np.mean(activation_passes)),
        "axis_pass_fraction": float(np.mean(axis_passes)),
    }


def posterior_region_summaries(
    arrays: dict[str, FloatArray], prefix: str
) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for name, values in arrays.items():
        for label, percentile in (("q16", 16.0), ("median", 50.0), ("q84", 84.0)):
            result[f"{prefix}_{name}_{label}"] = np.percentile(
                np.asarray(values, dtype=float), percentile, axis=0
            ).astype(np.float32)
    return result


def score_variant(
    features: dict[str, FloatArray],
    stellar_rank: FloatArray,
    *,
    fwhm_kpc: float,
    radius_kpc: float,
    candidate: str,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    token = variant_token(fwhm_kpc, radius_kpc)
    density_gradient = (
        feature(features, "electron_density_gradient_east_kpc_inv", token),
        feature(features, "electron_density_gradient_north_kpc_inv", token),
    )
    if candidate == "I4":
        second_gradient = (
            feature(features, "entropy_gradient_east_kpc_inv", token),
            feature(features, "entropy_gradient_north_kpc_inv", token),
        )
        response = np.stack(
            (
                feature(features, "i4_q_plus", token),
                feature(features, "i4_q_cross", token),
            ),
            axis=-1,
        )
    elif candidate == "I5":
        second_gradient = (
            feature(features, "pressure_gradient_east_kpc_inv", token),
            feature(features, "pressure_gradient_north_kpc_inv", token),
        )
        response = feature(features, "i5_baroclinicity", token)
    else:
        raise ValueError("candidate must be I4 or I5")
    support = gradient_support_mask(
        (density_gradient, second_gradient),
        minimum_detection_sigma=thresholds["minimum_gradient_detection_sigma"],
    )
    controls = np.stack(
        (
            feature(features, "control_log_gas_surface_density", token),
            np.asarray(stellar_rank, dtype=float),
            feature(features, "control_log_surface_gradient", token),
            feature(features, "control_surface_hessian_trace", token),
            feature(features, "control_surface_hessian_anisotropy", token),
        ),
        axis=-1,
    )
    if np.count_nonzero(support) < int(thresholds["minimum_supported_regions"]):
        raise RuntimeError(
            f"V19BP {candidate} {token} has only {np.count_nonzero(support)} regions"
        )
    draw_summary = (
        i4_draw_summary(response[..., 0], response[..., 1], support)
        if candidate == "I4"
        else i5_draw_summary(response, support)
    )
    posterior = posterior_feature_summary(draw_summary)
    novelty = posterior_novelty_scores(
        controls,
        response,
        support,
        minimum_unexplained_fraction=thresholds[
            "minimum_unexplained_variance_fraction"
        ],
    )
    loo = leave_one_region_out_stability(
        response,
        support,
        candidate=candidate,
        maximum_activation_change_fraction=thresholds[
            "maximum_activation_change_fraction"
        ],
        maximum_axis_change_deg=thresholds["maximum_axis_change_deg"],
    )
    component_loo = component_loo_pass_fractions(
        response,
        support,
        candidate=candidate,
        maximum_activation_change_fraction=thresholds[
            "maximum_activation_change_fraction"
        ],
        maximum_axis_change_deg=thresholds["maximum_axis_change_deg"],
    )
    common = {
        "minimum_supported_regions": int(np.count_nonzero(support))
        >= int(thresholds["minimum_supported_regions"]),
        "minimum_detection_sigma": posterior["detection_sigma"]
        >= thresholds["minimum_detection_sigma"],
        "minimum_novelty_draw_pass_fraction": novelty["pass_fraction"]
        >= thresholds["minimum_novelty_draw_pass_fraction"],
        "minimum_leave_one_region_out_activation_pass_fraction": component_loo[
            "activation_pass_fraction"
        ]
        >= thresholds["minimum_leave_one_region_out_pass_fraction"],
    }
    if candidate == "I4":
        direction_gates = {
            "minimum_supported_regions": common["minimum_supported_regions"],
            "maximum_axial_width_deg": posterior["axial_posterior"][
                "width_95_deg"
            ]
            <= thresholds["maximum_I4_axial_width_deg"],
            "minimum_leave_one_region_out_axis_pass_fraction": component_loo[
                "axis_pass_fraction"
            ]
            >= thresholds["minimum_leave_one_region_out_pass_fraction"],
        }
        amplitude_gates = common
    else:
        direction_gates = {}
        amplitude_gates = common
    result = {
        "candidate": candidate,
        "variant": token,
        "supported_regions": int(np.count_nonzero(support)),
        "posterior": posterior,
        "novelty": {
            "pass_fraction": novelty["pass_fraction"],
            "unexplained_fraction_percentiles": {
                label: float(value)
                for label, value in zip(
                    ("q05", "median", "q95"),
                    np.percentile(novelty["unexplained_fraction"], [5, 50, 95]),
                    strict=True,
                )
            },
            "maximum_leverage": float(np.max(novelty["maximum_leverage"])),
        },
        "leave_one_region_out": {**loo, **component_loo},
        "direction_gates": direction_gates,
        "amplitude_or_scalar_gates": amplitude_gates,
    }
    output = {
        "support": support,
        "draw_summary": draw_summary,
        "novelty": novelty,
        "response": response,
        "controls": controls,
    }
    return result, output


def variant_stability(
    primary: dict[str, FloatArray],
    alternatives: list[dict[str, FloatArray]],
    thresholds: dict[str, float],
) -> dict[str, float]:
    activation = np.asarray(primary["activation"], dtype=float)
    activation_pass = np.ones(activation.shape, dtype=bool)
    axis_pass = np.ones(activation.shape, dtype=bool)
    for alternative in alternatives:
        activation_pass &= symmetric_fractional_change(
            activation, np.asarray(alternative["activation"], dtype=float)
        ) <= thresholds["maximum_activation_change_fraction"]
        if "axis_deg" in primary:
            axis_pass &= axial_difference_deg(
                np.asarray(primary["axis_deg"], dtype=float),
                np.asarray(alternative["axis_deg"], dtype=float),
            ) <= thresholds["maximum_axis_change_deg"]
    return {
        "activation_draw_pass_fraction": float(np.mean(activation_pass)),
        "axis_draw_pass_fraction": float(np.mean(axis_pass)),
        "joint_draw_pass_fraction": float(np.mean(activation_pass & axis_pass)),
    }


def score_branch(
    regional_fields: dict[str, FloatArray],
    *,
    region_ids: NDArray[np.int64],
    label_grid: NDArray[np.int64],
    east_axis_kpc: FloatArray,
    north_axis_kpc: FloatArray,
    stellar_ranks: dict[float, FloatArray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    variants = config["variants"]
    thresholds = config["thresholds"]
    batch_size = int(config["execution"]["draw_batch_size"])
    draws = next(iter(regional_fields.values())).shape[0]
    variant_reports: dict[str, dict[str, Any]] = {"I4": {}, "I5": {}}
    draw_summaries: dict[str, dict[str, dict[str, FloatArray]]] = {
        "I4": {},
        "I5": {},
    }
    output_arrays: dict[str, Any] = {"bin_id": region_ids}
    for fwhm in variants["smoothing_fwhm_kpc"]:
        for radius in variants["radii_kpc"]:
            token = variant_token(float(fwhm), float(radius))
            accumulated: dict[str, list[FloatArray]] = {}
            for start in range(0, draws, batch_size):
                stop = min(start + batch_size, draws)
                batch = gas_feature_batch(
                    {name: values[start:stop] for name, values in regional_fields.items()},
                    region_ids=region_ids,
                    label_grid=label_grid,
                    east_axis_kpc=east_axis_kpc,
                    north_axis_kpc=north_axis_kpc,
                    spacing_kpc=float(variants["grid_spacing_kpc"]),
                    smoothing_fwhm_kpc=[float(fwhm)],
                    radii_kpc=[float(radius)],
                )
                append_feature_batch(accumulated, batch)
            features = concatenate_feature_batches(accumulated, draws)
            for candidate in ("I4", "I5"):
                report, arrays = score_variant(
                    features,
                    stellar_ranks[float(fwhm)],
                    fwhm_kpc=float(fwhm),
                    radius_kpc=float(radius),
                    candidate=candidate,
                    thresholds=thresholds,
                )
                variant_reports[candidate][token] = report
                draw_summaries[candidate][token] = arrays["draw_summary"]
                prefix = f"{candidate.lower()}_{token}"
                output_arrays[f"{prefix}_support"] = arrays["support"]
                for name, values in arrays["draw_summary"].items():
                    output_arrays[f"{prefix}_{name}"] = np.asarray(values).astype(
                        np.float32
                    )
                output_arrays[f"{prefix}_novelty_unexplained_fraction"] = np.asarray(
                    arrays["novelty"]["unexplained_fraction"]
                ).astype(np.float32)
                summarized = {
                    "response": (
                        arrays["response"]
                        if arrays["response"].ndim == 2
                        else arrays["response"].reshape(draws, region_ids.size * 2)
                    ),
                }
                summarized.update(
                    {
                        f"control_{index}": arrays["controls"][..., index]
                        for index in range(5)
                    }
                )
                region_summary = posterior_region_summaries(summarized, prefix)
                if arrays["response"].ndim == 3:
                    for key, value in list(region_summary.items()):
                        if key.startswith(f"{prefix}_response_"):
                            region_summary[key] = value.reshape(region_ids.size, 2)
                output_arrays.update(region_summary)

    primary = variant_token(
        variants["primary_smoothing_fwhm_kpc"], variants["primary_radius_kpc"]
    )
    candidate_results: dict[str, Any] = {}
    for candidate in ("I4", "I5"):
        alternatives = [
            summary
            for token, summary in draw_summaries[candidate].items()
            if token != primary
        ]
        stability = variant_stability(
            draw_summaries[candidate][primary], alternatives, thresholds
        )
        reports = variant_reports[candidate]
        amplitude_core = all(
            all(report["amplitude_or_scalar_gates"].values())
            for report in reports.values()
        )
        amplitude_pass = amplitude_core and stability[
            "activation_draw_pass_fraction"
        ] >= thresholds["minimum_variant_draw_pass_fraction"]
        if candidate == "I4":
            direction_core = all(
                all(report["direction_gates"].values()) for report in reports.values()
            )
            direction_pass = direction_core and stability[
                "axis_draw_pass_fraction"
            ] >= thresholds["minimum_variant_draw_pass_fraction"]
        else:
            direction_pass = False
        candidate_results[candidate] = {
            "primary_variant": primary,
            "variants": reports,
            "variant_stability": stability,
            "direction_pass": direction_pass,
            "amplitude_or_scalar_pass": amplitude_pass,
        }
    return candidate_results, output_arrays


def aggregate_source_decision(
    branch_reports: list[dict[str, Any]], expected_branches: int
) -> dict[str, bool]:
    if len(branch_reports) != expected_branches:
        raise ValueError("source decision does not contain every registered branch")
    i4_direction = all(
        row["candidates"]["I4"]["direction_pass"] for row in branch_reports
    )
    i4_amplitude = all(
        row["candidates"]["I4"]["amplitude_or_scalar_pass"]
        for row in branch_reports
    )
    i5_scalar = all(
        row["candidates"]["I5"]["amplitude_or_scalar_pass"]
        for row in branch_reports
    )
    return {
        "I4_direction_pass": i4_direction,
        "I4_amplitude_pass": i4_amplitude,
        "I5_scalar_pass": i5_scalar,
        "action_derivation_authorized": i4_direction
        and (i4_amplitude or i5_scalar),
    }


def execute(
    config: dict[str, Any], x4_report_path: Path, bm_report_path: Path
) -> dict[str, Any]:
    validate_static(config)
    x4, bm = validate_terminal_reports(config, x4_report_path, bm_report_path)
    output_root = ROOT / config["outputs"]["root"]
    branch_reports: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    for cluster in config["registered_inputs"]["clusters"]:
        stellar_product = product_for(
            bm["products"], cluster=cluster, role="stellar_morphology_control"
        )
        with np.load(ROOT / stellar_product["relative_path"]) as payload:
            stellar_ids = np.asarray(payload["bin_id"], dtype=np.int64)
            sample_id = np.asarray(payload["sample_id"], dtype=np.int64)
            stellar_ranks = {
                float(fwhm): np.asarray(
                    payload[f"light_percentile_rank_{float(fwhm):g}kpc"],
                    dtype=float,
                )
                for fwhm in config["variants"]["smoothing_fwhm_kpc"]
            }
        if not np.array_equal(sample_id, np.arange(sample_id.size)):
            raise RuntimeError(f"V19BP {cluster} stellar draw sequence changed")
        for correlation in config["registered_inputs"]["rank_correlations"]:
            regional_product = product_for(
                x4["products"],
                cluster=cluster,
                role="regional_posterior",
                rank_correlation=float(correlation),
            )
            grid_product = product_for(
                x4["products"],
                cluster=cluster,
                role="common_grid_summary",
                rank_correlation=float(correlation),
            )
            with np.load(ROOT / regional_product["relative_path"]) as payload:
                region_ids = np.asarray(payload["bin_id"], dtype=np.int64)
                regional_fields = {
                    name: np.asarray(payload[name], dtype=float).T
                    for name in REQUIRED_GAS_FIELDS
                }
                stored_correlation = float(payload["rank_correlation"])
            with np.load(ROOT / grid_product["relative_path"]) as payload:
                east_axis = np.asarray(payload["axis_east_kpc"], dtype=float)
                north_axis = np.asarray(payload["axis_north_kpc"], dtype=float)
                labels = np.asarray(payload["bin_id"], dtype=np.int64)
            if (
                not np.array_equal(region_ids, stellar_ids)
                or not math.isclose(
                    stored_correlation, float(correlation), abs_tol=1.0e-12
                )
                or any(values.shape[0] != sample_id.size for values in regional_fields.values())
            ):
                raise RuntimeError(f"V19BP {cluster} rho={correlation} pairing changed")
            candidates, arrays = score_branch(
                regional_fields,
                region_ids=region_ids,
                label_grid=labels,
                east_axis_kpc=east_axis,
                north_axis_kpc=north_axis,
                stellar_ranks=stellar_ranks,
                config=config,
            )
            arrays["sample_id"] = sample_id
            arrays["rank_correlation"] = np.asarray(float(correlation))
            token = f"rho_{float(correlation):+.1f}".replace("+", "p").replace(
                "-", "m"
            ).replace(".", "p")
            output = output_root / cluster / f"source_invariant_scores_{token}.npz"
            atomic_npz(output, arrays)
            products.append(
                {
                    "cluster": cluster,
                    "rank_correlation": float(correlation),
                    "role": "source_invariant_region_summary",
                    "relative_path": output.relative_to(ROOT).as_posix(),
                    "bytes": output.stat().st_size,
                    "sha256": sha256(output),
                }
            )
            branch_reports.append(
                {
                    "cluster": cluster,
                    "rank_correlation": float(correlation),
                    "regions": int(region_ids.size),
                    "draws": int(sample_id.size),
                    "candidates": candidates,
                }
            )
    expected_branches = len(config["registered_inputs"]["clusters"]) * len(
        config["registered_inputs"]["rank_correlations"]
    )
    decision = aggregate_source_decision(branch_reports, expected_branches)
    i4_direction = decision["I4_direction_pass"]
    i4_amplitude = decision["I4_amplitude_pass"]
    i5_scalar = decision["I5_scalar_pass"]
    authorize_action = decision["action_derivation_authorized"]
    gates = {
        "six_cluster_correlation_branches_scored": len(branch_reports)
        == expected_branches
        == 6,
        "six_hash_bound_source_only_products": len(products) == 6,
        "I4_direction_passes_both_clusters_all_branches": i4_direction,
        "I4_amplitude_or_I5_scalar_passes_both_clusters_all_branches": i4_amplitude
        or i5_scalar,
        "lensing_halo_action_and_gravity_payload_not_opened": True,
    }
    return {
        "status": (
            "observed_source_invariant_gates_passed_action_derivation_authorized"
            if authorize_action and all(gates.values())
            else "observed_source_invariant_gates_failed_no_action_authorized"
        ),
        "x4_report_sha256": sha256(x4_report_path),
        "stellar_control_report_sha256": sha256(bm_report_path),
        "branch_reports": branch_reports,
        "products": products,
        "aggregate_decision": decision,
        "gates": gates,
        "lensing_halo_action_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--x4-report", type=Path)
    parser.add_argument("--stellar-report", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if args.preflight_only:
        report = build_preflight_report(config_path)
        output = ROOT / config["outputs"]["preflight_report"]
    else:
        x4_report = args.x4_report or (ROOT / config["terminal_authorization"]["v19x4_report"])
        bm_report = args.stellar_report or (
            ROOT / config["terminal_authorization"]["v19bm_report"]
        )
        try:
            result = execute(config, x4_report.resolve(), bm_report.resolve())
        except Exception as exc:  # noqa: BLE001 - preserve terminal failure evidence
            result = {
                "status": "observed_source_invariant_execution_failed_closed",
                "execution_exception": f"{type(exc).__name__}: {exc}",
                "branch_reports": [],
                "products": [],
                "aggregate_decision": {
                    "I4_direction_pass": False,
                    "I4_amplitude_pass": False,
                    "I5_scalar_pass": False,
                    "action_derivation_authorized": False,
                },
                "gates": {"execution_completed": False},
                "lensing_halo_action_or_gravity_payload_opened": False,
                "gravity_formula_or_parameter_changed": False,
            }
        report = {
            "protocol_version": config["protocol_version"],
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            **result,
            "claim_boundary": config["claim_boundary"],
        }
        output = ROOT / config["outputs"]["terminal_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report.get("decision", report.get("status")))
    if report.get("decision") == "failed_closed" or report.get("status") == (
        "observed_source_invariant_execution_failed_closed"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
