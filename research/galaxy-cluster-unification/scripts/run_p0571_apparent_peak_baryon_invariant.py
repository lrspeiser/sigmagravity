#!/usr/bin/env python3
"""Search for baryon-only invariants at P0567 apparent-dark residual peaks."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import build_source_context  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import json_safe  # noqa: E402


@dataclass
class SourceSystem:
    label: str
    positions: np.ndarray
    weights: np.ndarray
    center: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_systems(protocol: dict, p0567: dict) -> dict[str, SourceSystem]:
    acquisition = json.loads(
        (ROOT / p0567["inputs"]["fresh_acquisition"]).read_text(encoding="utf-8")
    )
    sources = pd.read_csv(ROOT / p0567["inputs"]["fresh_sources"])
    audits = pd.read_csv(ROOT / p0567["inputs"]["fresh_systems_audit"])
    settings = {
        "pixels_per_axis": int(p0567["preprocessing"]["grid_pixels"]),
        "grid_spacing_kpc": float(p0567["preprocessing"]["grid_spacing_kpc"]),
        "common_radius_kpc": float(p0567["preprocessing"]["baryon_source_radius_kpc"]),
    }
    result = {}
    for system in acquisition["systems"]:
        label = system["label"]
        audit = audits[audits.system.eq(label)].iloc[0]
        context, _ = build_source_context(system, audit, sources, settings)
        weights = np.asarray(context.hard_weights, dtype=float)
        weights /= np.sum(weights)
        positions = np.asarray(context.positions, dtype=float)
        center = np.sum(weights[:, None] * positions, axis=0)
        result[label] = SourceSystem(label, positions, weights, center)
    expected = set(protocol["data"]["development_systems"] + protocol["data"]["validation_systems"])
    if set(result) != expected:
        raise RuntimeError("P0571 source-system coverage differs from the frozen ten-system set")
    return result


def invariant_values(
    system: SourceSystem,
    points: np.ndarray,
    force_exponent: float,
    softening_kpc: float,
    weight_power: float,
) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=float)
    delta = system.positions[None, :, :] - points[:, None, :]
    radius2 = np.sum(delta * delta, axis=2)
    softened = radius2 + float(softening_kpc) ** 2
    source_weight = np.power(system.weights, float(weight_power))[None, :]
    k = 0.5 * (float(force_exponent) + 1.0)
    coefficient = source_weight * np.power(softened, -k)
    vectors = coefficient[:, :, None] * delta
    magnitudes = np.linalg.norm(vectors, axis=2)
    scalar_sum = np.sum(magnitudes, axis=1)
    net = np.sum(vectors, axis=1)
    net_norm = np.linalg.norm(net, axis=1)
    tiny = np.finfo(float).tiny
    coherence = np.divide(net_norm, scalar_sum, out=np.zeros_like(net_norm), where=scalar_sum > tiny)
    cancellation = np.clip(1.0 - coherence, 0.0, 1.0)
    effective = np.divide(
        scalar_sum * scalar_sum,
        np.sum(magnitudes * magnitudes, axis=1),
        out=np.ones_like(scalar_sum),
        where=np.sum(magnitudes * magnitudes, axis=1) > tiny,
    )

    inward = system.center[None, :] - points
    inward_norm = np.linalg.norm(inward, axis=1)
    cosine = np.divide(
        np.sum(net * inward, axis=1),
        net_norm * inward_norm,
        out=np.ones_like(net_norm),
        where=(net_norm > tiny) & (inward_norm > tiny),
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    cross = net[:, 0] * inward[:, 1] - net[:, 1] * inward[:, 0]
    transverse = np.divide(
        np.abs(cross),
        net_norm * inward_norm,
        out=np.zeros_like(net_norm),
        where=(net_norm > tiny) & (inward_norm > tiny),
    )

    inverse_k = np.power(softened, -k)
    inverse_k1 = np.power(softened, -k - 1.0)
    dx = delta[:, :, 0]
    dy = delta[:, :, 1]
    txx = np.sum(source_weight * (-inverse_k + 2.0 * k * dx * dx * inverse_k1), axis=1)
    tyy = np.sum(source_weight * (-inverse_k + 2.0 * k * dy * dy * inverse_k1), axis=1)
    txy = np.sum(source_weight * (2.0 * k * dx * dy * inverse_k1), axis=1)
    trace = txx + tyy
    shear = np.sqrt(np.square(0.5 * (txx - tyy)) + np.square(txy))
    tidal_norm = np.sqrt(txx * txx + 2.0 * txy * txy + tyy * tyy)
    tidal_balance = np.divide(
        shear,
        shear + 0.5 * np.abs(trace),
        out=np.zeros_like(shear),
        where=(shear + 0.5 * np.abs(trace)) > tiny,
    )
    local_density = np.sum(
        source_weight * np.exp(-0.5 * radius2 / float(softening_kpc) ** 2), axis=1
    ) / (2.0 * np.pi * float(softening_kpc) ** 2)
    nearest = np.sqrt(np.min(radius2, axis=1))
    floor = np.finfo(float).tiny
    return {
        "vector_cancellation": cancellation,
        "radial_misalignment": 1.0 - cosine,
        "transverse_fraction": transverse,
        "log_effective_sources": np.log10(np.maximum(effective, 1.0)),
        "tidal_balance": tidal_balance,
        "log_tidal_norm": np.log10(np.maximum(tidal_norm, floor)),
        "log_local_density": np.log10(np.maximum(local_density, floor)),
        "log_nearest_distance": np.log10(np.maximum(nearest, 1.0e-6)),
    }


def sample_points(row, rotations: np.ndarray) -> np.ndarray:
    x = float(row.peak_x_kpc)
    y = float(row.peak_y_kpc)
    radius = math.hypot(x, y)
    phase = math.atan2(y, x)
    angles = np.concatenate([[phase], phase + rotations])
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def equal_system_effect(frame: pd.DataFrame, cohort: str, signed: float = 1.0) -> tuple[float, dict]:
    local = frame[frame.cohort.eq(cohort)]
    per_system = local.groupby("system").centered_rank.mean()
    return float(signed * per_system.mean()), {str(k): float(signed * v) for k, v in per_system.items()}


def main() -> None:
    protocol_path = ROOT / "configs/p0571_apparent_peak_baryon_invariant_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_apparent_peak_invariant_scores":
        raise RuntimeError("P0571 protocol is not frozen")
    p0567 = json.loads((ROOT / protocol["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    systems = load_source_systems(protocol, p0567)
    peaks = pd.read_csv(ROOT / protocol["inputs"]["p0567_peaks"])
    peaks = peaks[peaks.cohort.ne("spent_pilot")].copy()
    primary = peaks[peaks.method.eq("lenstool_ensemble")].copy()
    method = peaks[peaks.method.eq("glafic_best")].copy()
    cohort = {
        label: ("development" if label in protocol["data"]["development_systems"] else "validation")
        for label in systems
    }
    primary["cohort"] = primary.system.map(cohort)
    method["cohort"] = "method_control"
    rotations = np.radians(
        np.arange(1, int(protocol["factorial"]["same_radius_rotations"]) + 1)
        * float(protocol["factorial"]["rotation_step_degrees"])
    )
    all_peaks = pd.concat([primary, method], ignore_index=True)
    records = []
    rank_cube = []
    candidate_keys = []
    primary_index = {(row.system, row.peak_rank): i for i, row in enumerate(primary.itertuples(index=False))}
    for p in map(float, protocol["factorial"]["force_exponent_p"]):
        for softening in map(float, protocol["factorial"]["softening_kpc"]):
            for weight_power in map(float, protocol["factorial"]["source_weight_power_gamma"]):
                values_by_peak = {}
                for row in all_peaks.itertuples(index=False):
                    points = sample_points(row, rotations)
                    values_by_peak[(row.method, row.system, row.peak_rank)] = invariant_values(
                        systems[row.system], points, p, softening, weight_power
                    )
                for feature in protocol["factorial"]["features"]:
                    candidate_id = f"{feature}__p{p:g}__s{softening:g}__w{weight_power:g}"
                    candidate_keys.append((candidate_id, feature, p, softening, weight_power))
                    primary_ranks = np.empty((len(primary), len(rotations) + 1), dtype=float)
                    for row in all_peaks.itertuples(index=False):
                        values = values_by_peak[(row.method, row.system, row.peak_rank)][feature]
                        ranks = (rankdata(values, method="average") - 0.5) / len(values)
                        if row.method == "lenstool_ensemble":
                            primary_ranks[primary_index[(row.system, row.peak_rank)]] = ranks
                        records.append(
                            {
                                "candidate_id": candidate_id,
                                "feature": feature,
                                "force_exponent_p": p,
                                "softening_kpc": softening,
                                "source_weight_power_gamma": weight_power,
                                "method": row.method,
                                "cohort": row.cohort,
                                "system": row.system,
                                "peak_rank": int(row.peak_rank),
                                "peak_value": float(values[0]),
                                "rank_percentile": float(ranks[0]),
                                "centered_rank": float(ranks[0] - 0.5),
                            }
                        )
                    rank_cube.append(primary_ranks - 0.5)
    if len(candidate_keys) != int(protocol["factorial"]["candidate_count"]):
        raise RuntimeError("P0571 candidate count differs from the frozen count")
    peak_scores = pd.DataFrame(records)
    cube = np.asarray(rank_cube)
    primary_rows = list(primary.itertuples(index=False))
    dev_labels = protocol["data"]["development_systems"]
    val_labels = protocol["data"]["validation_systems"]
    candidate_rows = []
    for candidate_index, (candidate_id, feature, p, softening, weight_power) in enumerate(candidate_keys):
        local = peak_scores[peak_scores.candidate_id.eq(candidate_id)]
        dev_effect, dev_system = equal_system_effect(local, "development")
        direction = 1.0 if dev_effect >= 0.0 else -1.0
        validation_effect, validation_system = equal_system_effect(local, "validation", direction)
        method_effect, method_system = equal_system_effect(local, "method_control", direction)
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": candidate_id,
                "feature": feature,
                "force_exponent_p": p,
                "softening_kpc": softening,
                "source_weight_power_gamma": weight_power,
                "development_direction": "high" if direction > 0 else "low",
                "development_effect": dev_effect,
                "development_absolute_effect": abs(dev_effect),
                "validation_signed_effect": validation_effect,
                "validation_systems_same_direction": int(sum(value > 0.0 for value in validation_system.values())),
                "method_control_signed_effect": method_effect,
                "null_safe_feature": feature in protocol["cross_domain"]["null_safe_features"],
                "development_system_effects": json.dumps(dev_system, sort_keys=True),
                "validation_system_effects": json.dumps(validation_system, sort_keys=True),
                "method_system_effects": json.dumps(method_system, sort_keys=True),
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["development_absolute_effect", "candidate_id"], ascending=[False, True]
    )
    selected = candidates.iloc[0]

    # Search-aware null: choose one same-radius rotation per development system,
    # retain that rotation for every peak in a system, and record the largest
    # absolute equal-system effect across the complete frozen library.
    rng = np.random.default_rng(20260811)
    null_maxima = []
    system_peak_indices = {
        label: np.asarray([i for i, row in enumerate(primary_rows) if row.system == label], dtype=int)
        for label in dev_labels
    }
    for trial in range(256):
        effects = np.zeros(len(candidate_keys), dtype=float)
        for label in dev_labels:
            sample_index = int(rng.integers(1, len(rotations) + 1))
            effects += np.mean(cube[:, system_peak_indices[label], sample_index], axis=1) / len(dev_labels)
        null_maxima.append({"trial": trial, "maximum_absolute_effect": float(np.max(np.abs(effects)))})
    null_frame = pd.DataFrame(null_maxima)
    empirical_p = float(
        (1 + np.sum(null_frame.maximum_absolute_effect >= float(selected.development_absolute_effect)))
        / (1 + len(null_frame))
    )
    gates = protocol["advance_gates"]
    gate_values = {
        "development_effect_pass": bool(selected.development_absolute_effect >= float(gates["development_absolute_effect_min"])),
        "search_control_pass": bool(empirical_p <= float(gates["max_search_empirical_p_max"])),
        "validation_effect_pass": bool(selected.validation_signed_effect >= float(gates["validation_signed_effect_min"])),
        "validation_direction_pass": bool(selected.validation_systems_same_direction >= int(gates["validation_systems_same_direction_min"])),
        "method_control_pass": bool(selected.method_control_signed_effect >= float(gates["method_control_signed_effect_min"])),
        "cross_domain_null_pass": bool(selected.null_safe_feature),
    }
    gate_values["forward_activation_authorized"] = bool(all(gate_values.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    peak_scores.to_csv(output / protocol["outputs"]["peak_scores"], index=False)
    null_frame.to_csv(output / protocol["outputs"]["null_maxima"], index=False)
    report = {
        "report_version": "P0571-APPARENT-PEAK-BARYON-INVARIANT-RESULTS-0.1.0",
        "status": "complete_baryon_invariant_peak_audit",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {
            "systems": len(systems),
            "development_systems": len(dev_labels),
            "validation_systems": len(val_labels),
            "primary_peaks": len(primary),
            "method_control_peaks": len(method),
            "candidates": len(candidates),
            "same_radius_controls_per_peak": len(rotations),
            "search_null_trials": len(null_frame),
        },
        "selected": json_safe(selected.to_dict()),
        "search_control": {
            "empirical_max_search_p": empirical_p,
            "null_maximum_effect_median": float(null_frame.maximum_absolute_effect.median()),
            "null_maximum_effect_p90": float(null_frame.maximum_absolute_effect.quantile(0.9)),
        },
        "gates": gate_values,
        "cross_domain": {
            "isolated_point_source_activation": 0.0 if bool(selected.null_safe_feature) else None,
            "axisymmetric_coarse_grained_activation": 0.0 if bool(selected.null_safe_feature) else None,
            "solar_fractional_change": 0.0 if bool(selected.null_safe_feature) else None,
            "galaxy_interpretation": "angular null only; this audit does not supply the radial force needed by galaxy rotation curves",
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "# P0571 apparent-peak baryon invariant audit",
        "",
        f"Selected `{selected.candidate_id}` with development absolute centered-rank effect **{selected.development_absolute_effect:.3f}**.",
        f"Search-aware empirical p: **{empirical_p:.4f}**.",
        f"Held-out signed effect: **{selected.validation_signed_effect:.3f}**; same-direction systems: **{int(selected.validation_systems_same_direction)}/3**.",
        f"GLAFIC method-control signed effect: **{selected.method_control_signed_effect:.3f}**.",
        f"Forward activation authorized: **{gate_values['forward_activation_authorized']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    top = candidates.head(15).sort_values("development_absolute_effect")
    axes[0].barh(top.candidate_id, top.development_absolute_effect)
    axes[0].set_xlabel("development |centered rank effect|")
    axes[0].tick_params(axis="y", labelsize=5)
    axes[1].scatter(candidates.development_absolute_effect, candidates.validation_signed_effect, s=9, alpha=0.45)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("development absolute effect")
    axes[1].set_ylabel("held-out signed effect")
    axes[2].hist(null_frame.maximum_absolute_effect, bins=24, color="0.55")
    axes[2].axvline(selected.development_absolute_effect, color="tab:red", lw=2, label="selected real")
    axes[2].set_xlabel("maximum effect under same-radius null")
    axes[2].legend()
    fig.suptitle("P0571 baryon-only invariants at apparent-dark residual peaks")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["search_control"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
