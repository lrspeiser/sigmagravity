#!/usr/bin/env python3
"""Commission the frozen integrated Bullet two-temperature redshift profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19de_bullet_integrated_redshift_profile.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19de_bullet_integrated_redshift_profile"
PASS_STATUS = "bullet_integrated_two_temperature_redshift_profile_passed"
PREFLIGHT_STATUS = "v19de_payload_blind_integrated_profile_plan_passed"
C_KM_S = 299792.458


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_record(record: dict[str, Any], *, hash_payload: bool) -> Path:
    path = ROOT / record["path"]
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"V19DE frozen product changed size: {path}")
    if hash_payload and sha256(path) != record["sha256"]:
        raise RuntimeError(f"V19DE frozen product changed hash: {path}")
    return path


def validate_frozen(config: dict[str, Any], *, hash_payload: bool) -> dict[str, Path]:
    if config.get("freeze_state") != "frozen_after_v19dd_transport_pass_before_integrated_source_line_access":
        raise RuntimeError("V19DE is not frozen before integrated source-line access")
    implementation = config["implementation"]
    if implementation["runner"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise RuntimeError("V19DE config names another runner")
    if implementation["runner_sha256"] != sha256(Path(__file__).resolve()):
        raise RuntimeError("V19DE runner changed after freeze")
    parents: dict[str, Path] = {}
    for key, record in config["parents"].items():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"V19DE frozen parent changed: {path}")
        parents[key] = path
    if load_json(parents["v19dc_report"]).get("status") != config["parents"]["v19dc_report"]["required_status"]:
        raise RuntimeError("V19DE gain parent is not a terminal pass")
    if load_json(parents["v19dd_report"]).get("status") != config["parents"]["v19dd_report"]["required_status"]:
        raise RuntimeError("V19DE transport parent is not a terminal pass")
    auth = config["authorization"]
    if not (
        auth["open_integrated_source_pha_and_response_after_committed_preflight"]
        and auth["fit_integrated_temperature_abundance_redshift"]
        and auth["run_integrated_apec_and_mekal_profiles"]
        and not auth["run_posterior_predictive_or_thermal_sobol"]
        and not auth["open_any_regional_source_line_or_velocity"]
        and not auth["open_obsid554_or_abell2146"]
        and not auth["open_lensing_halo_gravity_or_action"]
    ):
        raise RuntimeError("V19DE authorization boundary is open")
    products = {key: validate_record(record, hash_payload=hash_payload) for key, record in config["data"].items() if isinstance(record, dict)}
    return {**parents, **products}


def inclusive_grid(center: float, half_width: float, step: float) -> list[float]:
    low = center - half_width
    count = round(2.0 * half_width / step)
    values = [round(low + index * step, 10) for index in range(count + 1)]
    if abs(values[-1] - (center + half_width)) > 1e-9:
        raise RuntimeError("V19DE profile grid does not close")
    return values


def evaluation_order(values: list[float], center: float) -> list[float]:
    return sorted(values, key=lambda value: (abs(value - center), value))


def canonical_state(state: dict[str, float]) -> dict[str, float]:
    result = dict(state)
    if result["T1"] > result["T2"]:
        for left, right in (("T1", "T2"), ("Z1", "Z2"), ("norm1", "norm2")):
            result[left], result[right] = result[right], result[left]
    return result


def profile_crossing(rows: list[dict[str, Any]], best_z: float, delta: float, side: str) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row["redshift"]))
    minimum = min(float(row["statistic"]) for row in ordered)
    target = minimum + delta
    if side == "lower":
        candidates = [row for row in ordered if float(row["redshift"]) <= best_z]
        candidates = list(reversed(candidates))
    elif side == "upper":
        candidates = [row for row in ordered if float(row["redshift"]) >= best_z]
    else:
        raise ValueError(side)
    for first, second in pairwise(candidates):
        y1 = float(first["statistic"]) - target
        y2 = float(second["statistic"]) - target
        if y1 == 0:
            return float(first["redshift"])
        if y1 * y2 <= 0 and y1 != y2:
            x1, x2 = float(first["redshift"]), float(second["redshift"])
            return x1 + (x2 - x1) * (-y1) / (y2 - y1)
    return None


def distinct_secondary_minima(
    rows: list[dict[str, Any]], best_z: float, separation: float
) -> list[dict[str, float]]:
    ordered = sorted(rows, key=lambda row: float(row["redshift"]))
    best_stat = min(float(row["statistic"]) for row in ordered)
    minima: list[dict[str, float]] = []
    for before, row, after in zip(ordered, ordered[1:], ordered[2:], strict=False):
        stat = float(row["statistic"])
        if stat <= float(before["statistic"]) and stat < float(after["statistic"]):
            redshift = float(row["redshift"])
            if abs(redshift - best_z) >= separation:
                minima.append({"redshift": redshift, "delta_statistic": stat - best_stat})
    return minima


def effective_integrated_gain(parents: dict[str, Path]) -> dict[str, Any]:
    transport = load_json(parents["v19dd_report"])
    gain_report = load_json(parents["v19dc_report"])
    contributions = {
        int(row["obsid"]): float(row["direct_cell_sum_cm2_s"])
        for row in transport["integrated_weight_equivalence"]["obsids"]
    }
    total = sum(contributions.values())
    weights = {obsid: value / total for obsid, value in contributions.items()}
    gains = {int(row["obsid"]): row["gain"] for row in gain_report["obsids"]}
    if set(weights) != set(gains):
        raise RuntimeError("V19DE integrated gain observation set changed")
    observed_fe = float(transport["observed_fe_energy_keV"])
    parameters = np.zeros(2, dtype=float)
    covariance = np.zeros((2, 2), dtype=float)
    corrections: list[tuple[float, float]] = []
    for obsid in sorted(weights):
        weight = weights[obsid]
        gain = gains[obsid]
        parameters += weight * np.array([gain["intercept_keV"], gain["slope"]], dtype=float)
        covariance += weight * weight * np.asarray(gain["covariance_intercept_slope"], dtype=float)
        correction = float(gain["intercept_keV"]) + (float(gain["slope"]) - 1.0) * observed_fe
        corrections.append((weight, correction))
    vector = np.array([1.0, observed_fe], dtype=float)
    sigma_energy = math.sqrt(max(0.0, float(vector @ covariance @ vector)))
    mean_correction = float(parameters[0] + (parameters[1] - 1.0) * observed_fe)
    dispersion = math.sqrt(
        max(0.0, sum(weight * (correction - mean_correction) ** 2 for weight, correction in corrections))
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "weights_by_obsid": {str(key): value for key, value in sorted(weights.items())},
        "intercept_keV": float(parameters[0]),
        "slope": float(parameters[1]),
        "covariance_intercept_slope": covariance.tolist(),
        "minimum_covariance_eigenvalue": float(np.min(eigenvalues)),
        "correction_at_observed_fe_keV": mean_correction,
        "one_sigma_energy_uncertainty_at_observed_fe_keV": sigma_energy,
        "one_sigma_equivalent_velocity_uncertainty_km_s": C_KM_S * sigma_energy / observed_fe,
        "weighted_rms_obsid_correction_dispersion_keV": dispersion,
        "weighted_rms_obsid_correction_dispersion_km_s": C_KM_S * dispersion / observed_fe,
        "finite_symmetric_positive_semidefinite": bool(
            np.isfinite(covariance).all()
            and np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14)
            and float(np.min(eigenvalues)) >= -1e-14
        ),
    }


def set_parameter(parameter: Any, value: float, low: float, high: float) -> None:
    parameter.min = low
    parameter.max = high
    parameter.val = min(max(value, low), high)


def configure_session(config: dict[str, Any], products: dict[str, Path], branch: str) -> dict[str, Any]:
    from sherpa.astro import ui

    ui.clean()
    ui.set_xsabund(config["model"]["abundance_table"])
    ui.set_xsxsect(config["model"]["cross_sections"])
    ui.load_pha(1, str(products["source"]))
    ui.ungroup(1)
    ui.set_analysis(1, "energy", "counts")
    low, high = map(float, config["model"]["fit_band_keV"])
    ui.ignore_id(1, None, low)
    ui.ignore_id(1, high, None)
    data = ui.get_data(1)
    if not getattr(data, "background_ids", []):
        raise RuntimeError("V19DE integrated source lacks an associated background")
    if bool(getattr(data, "subtracted", False)):
        raise RuntimeError("V19DE integrated source is background-subtracted")
    absorption = ui.create_model_component("xstbabs", f"v19de_tbabs_{branch}")
    thermal_name = "xsapec" if branch == "apec" else "xsmekal"
    first = ui.create_model_component(thermal_name, f"v19de_{branch}_1")
    second = ui.create_model_component(thermal_name, f"v19de_{branch}_2")
    ui.set_source(1, absorption * (first + second))
    nh = float(config["model"]["galactic_nh_1e22_cm2"])
    fraction = float(config["model"]["galactic_nh_fractional_interval"])
    set_parameter(absorption.nH, nh, nh * (1.0 - fraction), nh * (1.0 + fraction))
    ui.thaw(absorption.nH)
    t_low, t_high = map(float, config["model"]["temperature_range_keV"])
    z_low, z_high = map(float, config["model"]["abundance_range_solar"])
    n_low, n_high = map(float, config["model"]["normalization_range"])
    for component in (first, second):
        set_parameter(component.kT, 10.0, t_low, t_high)
        set_parameter(component.Abundanc, 0.3, z_low, z_high)
        set_parameter(component.norm, 0.01, n_low, n_high)
        ui.thaw(component.kT, component.Abundanc, component.norm)
        component.Redshift = float(config["profile"]["optical_redshift_center"])
    ui.link(second.Redshift, first.Redshift)
    ui.freeze(first.Redshift)
    if branch == "mekal":
        first.nH = second.nH = float(config["model"]["mekal_fixed_hydrogen_density_cm3"])
        first.switch = second.switch = int(config["model"]["mekal_fixed_switch"])
        ui.freeze(first.nH, second.nH, first.switch, second.switch)
    ui.set_stat(config["model"]["statistic"])
    ui.set_method(config["optimization"]["method"])
    ui.set_method_opt("maxfev", int(config["optimization"]["maximum_function_evaluations"]))
    counts = float(np.asarray(data.get_dep(filter=True), dtype=float).sum())
    return {
        "ui": ui,
        "absorption": absorption,
        "first": first,
        "second": second,
        "filtered_source_counts": counts,
        "background_ids": [int(value) for value in data.background_ids],
        "data_subtracted": bool(getattr(data, "subtracted", False)),
    }


def apply_state(session: dict[str, Any], state: dict[str, float], redshift: float) -> None:
    absorption, first, second = session["absorption"], session["first"], session["second"]
    absorption.nH = float(state["nH"])
    for component, suffix in ((first, "1"), (second, "2")):
        component.kT = float(state[f"T{suffix}"])
        component.Abundanc = float(state[f"Z{suffix}"])
        component.norm = float(state[f"norm{suffix}"])
    first.Redshift = redshift


def session_state(session: dict[str, Any]) -> dict[str, float]:
    absorption, first, second = session["absorption"], session["first"], session["second"]
    return canonical_state(
        {
            "nH": float(absorption.nH.val),
            "T1": float(first.kT.val),
            "T2": float(second.kT.val),
            "Z1": float(first.Abundanc.val),
            "Z2": float(second.Abundanc.val),
            "norm1": float(first.norm.val),
            "norm2": float(second.norm.val),
        }
    )


def anchor_state(config: dict[str, Any], index: int) -> dict[str, float]:
    anchor = dict(config["optimization"]["anchor_states"][index % len(config["optimization"]["anchor_states"])])
    anchor["nH"] = float(config["model"]["galactic_nh_1e22_cm2"])
    return canonical_state({key: float(value) for key, value in anchor.items()})


def fit_profile_point(
    config: dict[str, Any], session: dict[str, Any], branch: str, redshift: float, warm: dict[str, float], index: int
) -> dict[str, Any]:
    ui = session["ui"]
    starts = [("warm", canonical_state(warm)), ("anchor", anchor_state(config, index))]
    attempts: list[dict[str, Any]] = []
    for label, state in starts:
        apply_state(session, state, redshift)
        try:
            ui.fit(1)
            fit = ui.get_fit_results()
            statistic = float(fit.statval)
            result_state = session_state(session)
            finite = math.isfinite(statistic) and all(math.isfinite(value) for value in result_state.values())
            attempts.append(
                {
                    "start": label,
                    "finite": finite,
                    "succeeded": bool(getattr(fit, "succeeded", True)),
                    "statistic": statistic,
                    "state": result_state,
                    "nfev": int(getattr(fit, "nfev", -1)),
                    "message": str(getattr(fit, "message", "")),
                }
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append({"start": label, "finite": False, "exception": f"{type(exc).__name__}: {exc}"})
    finite_attempts = [row for row in attempts if row.get("finite")]
    if not finite_attempts:
        return {"branch": branch, "redshift": redshift, "finite": False, "attempts": attempts}
    best = min(finite_attempts, key=lambda row: float(row["statistic"]))
    return {
        "branch": branch,
        "redshift": redshift,
        "finite": True,
        "statistic": float(best["statistic"]),
        "state": best["state"],
        "selected_start": best["start"],
        "attempts": attempts,
    }


def checkpoint(output: Path, config_path: Path, branch_results: dict[str, Any]) -> None:
    atomic_json(
        output / "checkpoint.json",
        {
            "protocol_version": "SIGMA-V19DE-BULLET-INTEGRATED-REDSHIFT-PROFILE-CHECKPOINT-1.0.0",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha256(config_path),
            "branches": branch_results,
            "terminal_gate_passed": False,
            "regional_source_line_or_velocity_opened": False,
            "abell2146_opened": False,
        },
    )


def run_grid(
    config: dict[str, Any], session: dict[str, Any], branch: str, values: list[float], center: float,
    warm: dict[str, float], existing: list[dict[str, Any]], output: Path, config_path: Path,
    branch_results: dict[str, Any], phase: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_z = {float(row["redshift"]): row for row in existing}
    ordered = evaluation_order(values, center)
    for index, redshift in enumerate(ordered):
        if redshift in by_z and by_z[redshift].get("finite"):
            warm = canonical_state(by_z[redshift]["state"])
            continue
        row = fit_profile_point(config, session, branch, redshift, warm, index)
        by_z[redshift] = row
        if row.get("finite"):
            warm = canonical_state(row["state"])
        branch_results.setdefault(branch, {})[phase] = sorted(by_z.values(), key=lambda item: float(item["redshift"]))
        checkpoint(output, config_path, branch_results)
    return sorted(by_z.values(), key=lambda row: float(row["redshift"])), warm


def summarize_profile(config: dict[str, Any], coarse: list[dict[str, Any]], fine: list[dict[str, Any]]) -> dict[str, Any]:
    finite_fine = [row for row in fine if row.get("finite")]
    if not finite_fine:
        return {"finite": False}
    best = min(finite_fine, key=lambda row: float(row["statistic"]))
    best_z = float(best["redshift"])
    delta = float(config["profile"]["statistical_delta"])
    lower = profile_crossing(finite_fine, best_z, delta, "lower")
    upper = profile_crossing(finite_fine, best_z, delta, "upper")
    secondary = distinct_secondary_minima(
        [row for row in coarse if row.get("finite")],
        best_z,
        float(config["profile"]["distinct_minimum_separation"]),
    )
    optical = float(config["profile"]["optical_redshift_center"])
    return {
        "finite": True,
        "best_redshift": best_z,
        "best_statistic": float(best["statistic"]),
        "best_state": best["state"],
        "delta1_interval": [lower, upper],
        "difference_from_optical_redshift": best_z - optical,
        "velocity_relative_to_optical_km_s": C_KM_S * (best_z - optical) / (1.0 + optical),
        "secondary_minima": secondary,
        "profile_points": len(coarse) + len(fine),
        "finite_profile_points": sum(row.get("finite", False) for row in coarse + fine),
    }


def execute(config: dict[str, Any], config_path: Path, output: Path) -> dict[str, Any]:
    products = validate_frozen(config, hash_payload=True)
    gain = effective_integrated_gain(products)
    branch_results: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    optical = float(config["profile"]["optical_redshift_center"])
    coarse_values = inclusive_grid(optical, float(config["profile"]["half_range"]), float(config["profile"]["coarse_step"]))
    for branch in config["model"]["branches"]:
        session = configure_session(config, products, branch)
        warm = anchor_state(config, 0)
        coarse, warm = run_grid(
            config, session, branch, coarse_values, optical, warm, [], output, config_path, branch_results, "coarse"
        )
        finite_coarse = [row for row in coarse if row.get("finite")]
        if not finite_coarse:
            branch_results[branch]["fine"] = []
            summaries[branch] = {"finite": False}
            continue
        coarse_best = min(finite_coarse, key=lambda row: float(row["statistic"]))
        fine_center = float(coarse_best["redshift"])
        fine_values = inclusive_grid(
            fine_center, float(config["profile"]["fine_half_width"]), float(config["profile"]["fine_step"])
        )
        fine, warm = run_grid(
            config, session, branch, fine_values, fine_center, warm, [], output, config_path, branch_results, "fine"
        )
        branch_results[branch]["session"] = {
            "filtered_source_counts_2_10_keV": session["filtered_source_counts"],
            "background_ids": session["background_ids"],
            "data_subtracted": session["data_subtracted"],
        }
        summaries[branch] = summarize_profile(config, coarse, fine)
        checkpoint(output, config_path, branch_results)
    expected_points = len(coarse_values) + len(
        inclusive_grid(optical, float(config["profile"]["fine_half_width"]), float(config["profile"]["fine_step"]))
    )
    complete = all(
        summaries.get(branch, {}).get("finite")
        and summaries[branch]["finite_profile_points"] == expected_points
        for branch in config["model"]["branches"]
    )
    intervals = [summaries.get(branch, {}).get("delta1_interval", [None, None]) for branch in config["model"]["branches"]]
    interior = all(interval[0] is not None and interval[1] is not None for interval in intervals)
    no_secondary = all(
        not any(
            float(row["delta_statistic"]) < float(config["profile"]["secondary_minimum_delta"])
            for row in summaries.get(branch, {}).get("secondary_minima", [])
        )
        for branch in config["model"]["branches"]
    )
    near_optical = all(
        abs(float(summaries.get(branch, {}).get("difference_from_optical_redshift", math.inf))) <= 0.01
        for branch in config["model"]["branches"]
    )
    model_difference = (
        abs(float(summaries["apec"]["best_redshift"]) - float(summaries["mekal"]["best_redshift"]))
        if summaries.get("apec", {}).get("finite") and summaries.get("mekal", {}).get("finite")
        else math.inf
    )
    gates = {
        "both_model_profiles_complete": complete,
        "every_profile_point_has_a_finite_multistart_fit": complete,
        "best_redshift_and_delta1_interval_interior": interior,
        "no_distinct_secondary_minimum_within_delta_6p63": no_secondary,
        "each_best_redshift_within_0p01_of_optical": near_optical,
        "apec_mekal_best_redshift_difference_at_most_0p003": model_difference
        <= float(config["gates"]["apec_mekal_best_redshift_difference_at_most"]),
        "integrated_gain_covariance_finite_psd": gain["finite_symmetric_positive_semidefinite"],
    }
    return {
        "status": PASS_STATUS if all(gates.values()) else "bullet_integrated_redshift_profile_gate_failed",
        "products": {
            key: {"path": value.relative_to(ROOT).as_posix(), "bytes": value.stat().st_size, "sha256": sha256(value)}
            for key, value in products.items()
            if key in config["data"]
        },
        "gain": gain,
        "summaries": summaries,
        "profiles": branch_results,
        "gates": gates,
        "integrated_systematic_and_goodness_stage_authorized": all(gates.values()),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    products = validate_frozen(config, hash_payload=False)
    coarse = inclusive_grid(
        float(config["profile"]["optical_redshift_center"]),
        float(config["profile"]["half_range"]),
        float(config["profile"]["coarse_step"]),
    )
    fine = inclusive_grid(
        float(config["profile"]["optical_redshift_center"]),
        float(config["profile"]["fine_half_width"]),
        float(config["profile"]["fine_step"]),
    )
    return {
        "status": PREFLIGHT_STATUS,
        "branches": list(config["model"]["branches"]),
        "coarse_points_per_branch": len(coarse),
        "fine_points_per_branch": len(fine),
        "multistarts_per_point": 2,
        "integrated_product_sizes_verified": {key: products[key].stat().st_size for key in config["data"] if isinstance(config["data"][key], dict)},
        "source_pha_response_scientific_arrays_opened": False,
        "source_line_temperature_abundance_redshift_or_velocity_fitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_json(config_path)
    try:
        result = preflight(config) if args.preflight_only else execute(config, config_path, output)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "v19de_execution_failed_closed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "integrated_systematic_and_goodness_stage_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "posterior_predictive_or_thermal_sobol_run": False,
        "regional_source_line_or_velocity_opened": False,
        "obsid554_or_abell2146_opened": False,
        "lensing_halo_gravity_or_action_opened": False,
    }
    report_name = "preflight_report.json" if args.preflight_only else "report.json"
    atomic_json(output / report_name, report)
    print(json.dumps({key: report.get(key) for key in ("status", "execution_exception")}, indent=2, sort_keys=True))
    required = PREFLIGHT_STATUS if args.preflight_only else PASS_STATUS
    if report["status"] != required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
