#!/usr/bin/env python3
"""Stress-test P0575 over every family split and several FFT padding factors."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import regrid_kappa_sky  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, json_safe  # noqa: E402
from run_p0573_tidal_arrival_fresh_replication import assert_frozen_integrity, system_geometry  # noqa: E402
from run_p0574_symmetry_gated_arrival_microvariation import (  # noqa: E402
    field_primitives,
    mean_target,
    prediction,
    quarter_turn_asymmetry,
)
from run_p0575_smacs0723_raw_position import (  # noqa: E402
    deflection_from_surface,
    evaluate_model,
    fit_positive_amplitude,
    interpolate_deflection,
    lens_efficiency,
    sha256,
)


def build_maps(protocol: dict, images: pd.DataFrame):
    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    system = next(item for item in p0573["systems"] if item["slug"] == "smacs0723m73")
    data, world = system_geometry(system, p0573, sources, audits)
    local_manifest = manifest[manifest.system.eq(data.label)]
    range_rows = local_manifest[
        local_manifest.kind.eq("range_kappa") & local_manifest.method.eq("lenstool")
    ].copy()
    range_rows["sample_index_numeric"] = pd.to_numeric(range_rows.sample_index)
    range_rows = range_rows.sort_values("sample_index_numeric")
    data.range_maps = [
        regrid_kappa_sky(ROOT / row.path, world, data.x_grid.shape)
        for row in range_rows.itertuples(index=False)
    ]
    standard = mean_target(data)
    aperture = data.radius <= 250.0
    local = deposit_baryons(data, 100.0)
    local[~aperture] = 0.0
    local /= np.sum(local)
    primitives = field_primitives(data, aperture)
    q90 = quarter_turn_asymmetry(data)
    p0574 = json.loads((ROOT / protocol["inputs"]["p0574_protocol"]).read_text(encoding="utf-8"))
    no_gate_candidate = next(
        item for item in p0574["candidate_grid"] if item["candidate_id"] == "no_gate_baseline"
    )
    p0574_report = json.loads(
        (ROOT / p0574["outputs"]["directory"] / p0574["outputs"]["report"]).read_text(encoding="utf-8")
    )
    gated_candidate = p0574_report["result"]["selected_candidate"]
    no_gate, _ = prediction(data, aperture, primitives, no_gate_candidate, q90, local)
    gated, _ = prediction(data, aperture, primitives, gated_candidate, q90, local)
    return data, {
        "local_control": local,
        "p0573_no_gate": no_gate,
        "p0574_symmetry_gated": gated,
        "lenstool_map_reference": standard,
    }


def sampled_fields(data, maps: dict, images: pd.DataFrame, padding_factor: int) -> dict:
    output = {}
    for name, surface in maps.items():
        alpha_x, alpha_y = deflection_from_surface(surface, 10.0, padding_factor)
        sampled = interpolate_deflection(alpha_x, alpha_y, images, data.axis)
        if not np.isfinite(sampled).all():
            raise RuntimeError(f"{name}: nonfinite sampled field")
        output[name] = sampled
    return output


def score_split(images, sampled, calibration_families, padding_factor):
    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.astype(str).to_numpy()
    calibration_mask = np.isin(families, list(calibration_families))
    cohorts = np.where(calibration_mask, "calibration", "heldout")
    efficiency = lens_efficiency(0.39, images.source_redshift.to_numpy(float))
    rows = []
    for name, alpha in sampled.items():
        scaled = efficiency[:, None] * alpha
        amplitude = fit_positive_amplitude(theta, scaled, families, calibration_mask)
        score, _, _ = evaluate_model(name, theta, scaled, families, cohorts, amplitude)
        rows.append(
            {
                "calibration_families": "+".join(sorted(calibration_families, key=int)),
                "heldout_families": "+".join(sorted(set(families) - set(calibration_families), key=int)),
                "padding_factor": padding_factor,
                **score,
            }
        )
    return rows


def main() -> None:
    protocol_path = ROOT / "configs/p0575b_raw_position_robustness_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_primary_P0575_before_split_or_padding_robustness_scores":
        raise RuntimeError("P0575B protocol is not frozen")
    p0575 = json.loads((ROOT / protocol["inputs"]["p0575_protocol"]).read_text(encoding="utf-8"))
    images = pd.read_csv(ROOT / protocol["inputs"]["p0575_images"], dtype={"family": str})
    data, maps = build_maps(protocol, images)
    sampled_two = sampled_fields(data, maps, images, 2)
    families = list(protocol["split_robustness"]["families"])
    split_rows = []
    for calibration in itertools.combinations(families, 2):
        split_rows.extend(score_split(images, sampled_two, calibration, 2))
    split_scores = pd.DataFrame(split_rows)
    split_pivot = split_scores.pivot(
        index="calibration_families", columns="model", values="heldout_source_plane_RMS_arcsec"
    )
    split_gain = 1.0 - split_pivot.p0574_symmetry_gated / split_pivot.local_control
    splits_improved = int((split_gain > 0.0).sum())
    median_gain = float(split_gain.median())
    lenstool_best_count = int(
        (
            split_pivot.lenstool_map_reference
            < split_pivot[["local_control", "p0573_no_gate", "p0574_symmetry_gated"]].min(axis=1)
        ).sum()
    )

    primary = tuple(protocol["solver_robustness"]["primary_calibration_families"])
    padding_rows = []
    for padding in map(int, protocol["solver_robustness"]["padding_factors"]):
        sampled = sampled_two if padding == 2 else sampled_fields(data, maps, images, padding)
        padding_rows.extend(score_split(images, sampled, primary, padding))
    padding_scores = pd.DataFrame(padding_rows)
    padding_pivot = padding_scores.pivot(
        index="padding_factor", columns="model", values="heldout_source_plane_RMS_arcsec"
    )
    padding_gain = 1.0 - padding_pivot.p0574_symmetry_gated / padding_pivot.local_control
    gates_cfg = protocol["robustness_gates"]
    gates = {
        "median_split_improvement_pass": bool(
            median_gain >= float(gates_cfg["median_heldout_improvement_vs_local_fraction_min"])
        ),
        "split_count_pass": bool(splits_improved >= int(gates_cfg["splits_improved_vs_local_min"])),
        "padding_robustness_pass": bool((padding_gain > 0.0).all()),
        "lenstool_reference_sanity_pass": bool(lenstool_best_count == len(split_pivot)),
    }
    gates["raw_failure_survives_robustness"] = bool(
        not gates["median_split_improvement_pass"]
        and not gates["split_count_pass"]
        and not gates["padding_robustness_pass"]
        and gates["lenstool_reference_sanity_pass"]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    split_scores.to_csv(output / protocol["outputs"]["split_scores"], index=False)
    padding_scores.to_csv(output / protocol["outputs"]["padding_scores"], index=False)
    report = {
        "report_version": "P0575B-RAW-POSITION-ROBUSTNESS-RESULTS-0.1.0",
        "status": "complete_raw_position_robustness",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {"two_family_splits": len(split_pivot), "padding_factors": len(padding_pivot), "models": len(maps)},
        "result": {
            "median_gated_improvement_vs_local_fraction": median_gain,
            "splits_gated_improves_vs_local": splits_improved,
            "splits_total": len(split_pivot),
            "lenstool_reference_best_splits": lenstool_best_count,
            "padding_gated_improvement_vs_local_fraction": {str(index): float(value) for index, value in padding_gain.items()},
        },
        "split_gains": [
            {
                "calibration_families": index,
                "gated_improvement_vs_local_fraction": float(split_gain.loc[index]),
                "gated_heldout_RMS_arcsec": float(split_pivot.loc[index, "p0574_symmetry_gated"]),
                "local_heldout_RMS_arcsec": float(split_pivot.loc[index, "local_control"]),
                "lenstool_heldout_RMS_arcsec": float(split_pivot.loc[index, "lenstool_map_reference"]),
            }
            for index in split_pivot.index
        ],
        "gates": gates,
        "interpretation": "If the failure survives, normalized convergence morphology is not enough: the formula must change the radial potential/deflection structure or acquire an absolute baryonic normalization before another cluster-map fit.",
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0575B raw-position robustness",
        "",
        f"Median gated improvement versus local over all six family splits: **{100*median_gain:.2f}%**.",
        f"Splits improved: **{splits_improved}/6**; Lenstool reference best: **{lenstool_best_count}/6**.",
        f"Raw failure survives robustness: **{gates['raw_failure_survives_robustness']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    split_x = np.arange(len(split_gain))
    axes[0].bar(split_x, 100.0 * split_gain.values)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xticks(split_x, split_gain.index, rotation=30, ha="right")
    axes[0].set_ylabel("gated improvement vs local (%)")
    axes[0].set_title("all two-family calibration splits")
    axes[1].plot(padding_gain.index, 100.0 * padding_gain.values, marker="o")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("zero-padding factor")
    axes[1].set_ylabel("gated improvement vs local (%)")
    axes[1].set_title("primary split boundary robustness")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
