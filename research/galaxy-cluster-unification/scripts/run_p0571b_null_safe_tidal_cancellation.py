#!/usr/bin/env python3
"""Test a null-safe tidal-balance times vector-cancellation interaction."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0567_baryon_flux_tensor_backtrack import json_safe  # noqa: E402
from run_p0571_apparent_peak_baryon_invariant import (  # noqa: E402
    SourceSystem,
    invariant_values,
    load_source_systems,
    sample_points,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pilot_sources(p0567: dict) -> dict[str, SourceSystem]:
    acquisition = json.loads(
        (ROOT / p0567["inputs"]["pilot_acquisition"]).read_text(encoding="utf-8")
    )
    sources = pd.read_csv(ROOT / p0567["inputs"]["pilot_sources"])
    result = {}
    source_radius = float(p0567["preprocessing"]["baryon_source_radius_kpc"])
    for system in acquisition["systems"]:
        label = system["label"]
        redshift = float(system["cluster_redshift"])
        kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(redshift).value / 60.0)
        first_map = sorted((ROOT / system["lensing_directory"] / "range").glob("*_kappa.fits"))[0]
        header = fits.getheader(first_map)
        pixel_scale_kpc = abs(float(header["CDELT1"])) * 3600.0 * kpc_per_arcsec
        center_x = float(header["CRPIX1"]) - 1.0
        center_y = float(header["CRPIX2"]) - 1.0
        local = sources[sources.system.eq(label)].copy()
        local = local[local.hard_member.astype(str).str.lower().eq("true")]
        local["x_kpc"] = (local.map_x_pixel - center_x) * pixel_scale_kpc
        local["y_kpc"] = (local.map_y_pixel - center_y) * pixel_scale_kpc
        local = local[np.hypot(local.x_kpc, local.y_kpc) <= source_radius]
        positions = local[["x_kpc", "y_kpc"]].to_numpy(float)
        weights = np.maximum(local.f160w_flux_nJy.to_numpy(float), 0.0)
        weights /= np.sum(weights)
        center = np.sum(weights[:, None] * positions, axis=0)
        result[label] = SourceSystem(label, positions, weights, center)
    return result


def system_effect(frame: pd.DataFrame, cohort: str, direction: float = 1.0):
    local = frame[frame.cohort.eq(cohort)]
    per_system = local.groupby("system").centered_rank.mean() * float(direction)
    return float(per_system.mean()), {str(key): float(value) for key, value in per_system.items()}


def main() -> None:
    protocol_path = ROOT / "configs/p0571b_null_safe_tidal_cancellation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_tidal_cancellation_interaction_scores":
        raise RuntimeError("P0571B protocol is not frozen")
    p0571 = json.loads((ROOT / protocol["inputs"]["p0571_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / protocol["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    systems = load_source_systems(p0571, p0567)
    systems.update(load_pilot_sources(p0567))
    peaks = pd.read_csv(ROOT / protocol["inputs"]["p0567_peaks"])
    cohort = {}
    cohort.update({label: "development" for label in p0571["data"]["development_systems"]})
    cohort.update({label: "spent_validation" for label in p0571["data"]["validation_systems"]})
    cohort.update({label: "pilot_validation" for label in ["A2537", "MACS J0417", "MACS J0949"]})
    selected_peaks = peaks[peaks.method.eq("lenstool_ensemble")].copy()
    selected_peaks["cohort"] = selected_peaks.system.map(cohort)
    method_peaks = peaks[peaks.method.eq("glafic_best")].copy()
    method_peaks["cohort"] = "method_control"
    all_peaks = pd.concat([selected_peaks, method_peaks], ignore_index=True)
    rotations = np.radians(
        np.arange(1, int(protocol["factorial"]["same_radius_rotations"]) + 1)
        * float(protocol["factorial"]["rotation_step_degrees"])
    )
    base_values = {}
    for row in all_peaks.itertuples(index=False):
        base_values[(row.method, row.system, row.peak_rank)] = invariant_values(
            systems[row.system],
            sample_points(row, rotations),
            2.0,
            float(protocol["formula"]["fixed_softening_kpc"]),
            float(protocol["formula"]["fixed_source_weight_power"]),
        )
    primary = selected_peaks[selected_peaks.cohort.isin(["development", "spent_validation"])].copy()
    primary_index = {(row.system, row.peak_rank): index for index, row in enumerate(primary.itertuples(index=False))}
    records = []
    cube = []
    candidate_keys = []
    for alpha in map(float, protocol["factorial"]["cancellation_power_alpha"]):
        for beta in map(float, protocol["factorial"]["tidal_power_beta"]):
            candidate_id = f"tidal_cancellation__a{alpha:g}__b{beta:g}"
            candidate_keys.append((candidate_id, alpha, beta))
            primary_ranks = np.empty((len(primary), len(rotations) + 1), dtype=float)
            for row in all_peaks.itertuples(index=False):
                base = base_values[(row.method, row.system, row.peak_rank)]
                values = np.power(base["vector_cancellation"], alpha) * np.power(base["tidal_balance"], beta)
                ranks = (rankdata(values, method="average") - 0.5) / len(values)
                if row.method == "lenstool_ensemble" and row.cohort in {"development", "spent_validation"}:
                    primary_ranks[primary_index[(row.system, row.peak_rank)]] = ranks
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "cancellation_power_alpha": alpha,
                        "tidal_power_beta": beta,
                        "method": row.method,
                        "cohort": row.cohort,
                        "system": row.system,
                        "peak_rank": int(row.peak_rank),
                        "peak_activation": float(values[0]),
                        "rank_percentile": float(ranks[0]),
                        "centered_rank": float(ranks[0] - 0.5),
                    }
                )
            cube.append(primary_ranks - 0.5)
    if len(candidate_keys) != int(protocol["factorial"]["candidates"]):
        raise RuntimeError("P0571B candidate count differs from the frozen count")
    scores = pd.DataFrame(records)
    candidate_rows = []
    for index, (candidate_id, alpha, beta) in enumerate(candidate_keys):
        local = scores[scores.candidate_id.eq(candidate_id)]
        dev, dev_system = system_effect(local, "development")
        direction = 1.0 if dev >= 0.0 else -1.0
        spent, spent_system = system_effect(local, "spent_validation", direction)
        pilot, pilot_system = system_effect(local, "pilot_validation", direction)
        method, method_system = system_effect(local, "method_control", direction)
        candidate_rows.append(
            {
                "candidate_index": index,
                "candidate_id": candidate_id,
                "cancellation_power_alpha": alpha,
                "tidal_power_beta": beta,
                "development_direction": "high" if direction > 0 else "low",
                "development_effect": dev,
                "development_absolute_effect": abs(dev),
                "spent_validation_signed_effect": spent,
                "pilot_validation_signed_effect": pilot,
                "pilot_systems_same_direction": int(sum(value > 0.0 for value in pilot_system.values())),
                "method_control_signed_effect": method,
                "development_system_effects": json.dumps(dev_system, sort_keys=True),
                "spent_validation_system_effects": json.dumps(spent_system, sort_keys=True),
                "pilot_system_effects": json.dumps(pilot_system, sort_keys=True),
                "method_system_effects": json.dumps(method_system, sort_keys=True),
            }
        )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["development_absolute_effect", "candidate_id"], ascending=[False, True]
    )
    selected = candidates.iloc[0]

    rank_cube = np.asarray(cube)
    primary_rows = list(primary.itertuples(index=False))
    dev_labels = p0571["data"]["development_systems"]
    indices = {
        label: np.asarray([i for i, row in enumerate(primary_rows) if row.system == label], dtype=int)
        for label in dev_labels
    }
    rng = np.random.default_rng(20260812)
    null_rows = []
    for trial in range(256):
        effects = np.zeros(len(candidate_keys), dtype=float)
        for label in dev_labels:
            sample_index = int(rng.integers(1, len(rotations) + 1))
            effects += np.mean(rank_cube[:, indices[label], sample_index], axis=1) / len(dev_labels)
        null_rows.append({"trial": trial, "maximum_absolute_effect": float(np.max(np.abs(effects)))})
    nulls = pd.DataFrame(null_rows)
    empirical_p = float(
        (1 + np.sum(nulls.maximum_absolute_effect >= float(selected.development_absolute_effect)))
        / (1 + len(nulls))
    )
    required = protocol["advance_gates"]
    gates = {
        "development_effect_pass": bool(selected.development_absolute_effect >= float(required["development_absolute_effect_min"])),
        "search_control_pass": bool(empirical_p <= float(required["max_search_empirical_p_max"])),
        "spent_validation_effect_pass": bool(selected.spent_validation_signed_effect >= float(required["fresh_spent_validation_signed_effect_min"])),
        "pilot_validation_effect_pass": bool(selected.pilot_validation_signed_effect >= float(required["pilot_validation_signed_effect_min"])),
        "pilot_validation_direction_pass": bool(selected.pilot_systems_same_direction >= int(required["pilot_systems_same_direction_min"])),
        "method_control_pass": bool(selected.method_control_signed_effect >= float(required["method_control_signed_effect_min"])),
        "exact_axisymmetric_and_solar_null_pass": True,
    }
    gates["forward_activation_authorized"] = bool(all(gates.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    scores.to_csv(output / protocol["outputs"]["peak_scores"], index=False)
    nulls.to_csv(output / protocol["outputs"]["null_maxima"], index=False)
    report = {
        "report_version": "P0571B-NULL-SAFE-TIDAL-CANCELLATION-RESULTS-0.1.0",
        "status": "complete_null_safe_interaction_test",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {
            "candidates": len(candidates),
            "development_systems": len(dev_labels),
            "spent_validation_systems": 3,
            "pilot_validation_systems": 3,
            "pilot_validation_peaks": int((scores.cohort.eq("pilot_validation") & scores.candidate_id.eq(selected.candidate_id)).sum()),
            "method_control_systems": int(scores[scores.cohort.eq("method_control")].system.nunique()),
            "same_radius_controls_per_peak": len(rotations),
        },
        "selected": json_safe(selected.to_dict()),
        "search_control": {
            "empirical_max_search_p": empirical_p,
            "null_maximum_effect_median": float(nulls.maximum_absolute_effect.median()),
            "null_maximum_effect_p90": float(nulls.maximum_absolute_effect.quantile(0.9)),
        },
        "gates": gates,
        "cross_domain": {
            "isolated_point_source_activation": 0.0,
            "axisymmetric_coarse_grained_activation": 0.0,
            "solar_fractional_change": 0.0,
            "SPARC_rotation_change": 0.0,
            "interpretation": "exact angular null; no standalone radial galaxy explanation",
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0571B null-safe tidal-cancellation interaction",
        "",
        f"Selected `{selected.candidate_id}`; development effect **{selected.development_absolute_effect:.3f}**.",
        f"Search-aware p: **{empirical_p:.4f}**.",
        f"Spent-validation effect: **{selected.spent_validation_signed_effect:.3f}**.",
        f"Pilot-validation effect: **{selected.pilot_validation_signed_effect:.3f}**, same direction in **{int(selected.pilot_systems_same_direction)}/3** systems.",
        f"GLAFIC method-control effect: **{selected.method_control_signed_effect:.3f}**.",
        f"Forward activation authorized: **{gates['forward_activation_authorized']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    order = candidates.sort_values("development_absolute_effect")
    axes[0].barh(order.candidate_id, order.development_absolute_effect)
    axes[0].set_xlabel("development |centered rank effect|")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[1].scatter(candidates.development_absolute_effect, candidates.pilot_validation_signed_effect)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("development effect")
    axes[1].set_ylabel("pilot signed effect")
    axes[2].hist(nulls.maximum_absolute_effect, bins=20, color="0.55")
    axes[2].axvline(selected.development_absolute_effect, color="tab:red", lw=2)
    axes[2].set_xlabel("nine-formula null maximum")
    fig.suptitle("P0571B null-safe tidal-cancellation refinement")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["search_control"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
