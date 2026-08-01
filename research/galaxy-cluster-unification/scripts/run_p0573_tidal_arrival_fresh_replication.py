#!/usr/bin/env python3
"""Run the locked P0572B arrival law on three genuinely fresh RELICS systems."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import build_source_context, regrid_kappa_sky  # noqa: E402
from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import (  # noqa: E402
    SystemData,
    deposit_baryons,
    json_safe,
    lens_source_map,
)
from run_p0572_tidal_cancellation_arrival_forward import (  # noqa: E402
    activation_and_carriers,
    destination_map,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8", newline="") as handle:
        return pd.DataFrame(list(csv.DictReader(handle)))


def assert_frozen_integrity(protocol_path: Path, protocol: dict) -> tuple[Path, pd.DataFrame]:
    acquisition = protocol["acquisition"]
    provenance_path = ROOT / acquisition["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["protocol_sha256"] != sha256(protocol_path):
        raise RuntimeError("P0573 protocol changed after acquisition")
    manifest_path = ROOT / acquisition["manifest"]
    if provenance["manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError("P0573 manifest changed after acquisition")
    audit_path = ROOT / protocol["outputs"]["input_audit_directory"] / "report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["coverage_gate_passed"]:
        raise RuntimeError("P0573 pre-score spatial coverage gate failed")
    if audit["protocol_sha256"] != sha256(protocol_path):
        raise RuntimeError("P0573 input audit did not use the frozen protocol")
    return manifest_path, read_manifest(manifest_path)


def system_geometry(system: dict, protocol: dict, sources: pd.DataFrame, audits: pd.DataFrame):
    settings = protocol["spatial_preprocessing"]
    source_settings = {
        "pixels_per_axis": int(settings["pixels_per_axis"]),
        "grid_spacing_kpc": float(settings["grid_spacing_kpc"]),
        "common_radius_kpc": float(settings["common_radius_kpc"]),
    }
    audit_row = audits[audits.system.eq(system["label"])].iloc[0]
    context, world = build_source_context(system, audit_row, sources, source_settings)
    return SystemData(
        label=system["label"],
        cohort="fresh_replication",
        redshift=float(system["cluster_redshift"]),
        axis=context.axis_kpc,
        x_grid=context.x_grid,
        y_grid=context.y_grid,
        radius=context.radius_grid,
        positions=context.positions,
        weights=context.hard_weights,
        range_maps=[],
        glafic_map=None,
    ), world


def axisymmetric_disk_audit() -> dict:
    """Numerically test the claimed null on an extended circular exponential disk."""
    size = 256
    spacing = 10.0
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    positions = [[0.0, 0.0]]
    weights = [1.0]
    for radius in np.arange(15.0, 301.0, 15.0):
        count = 96
        angles = 2.0 * np.pi * np.arange(count) / count
        ring_weight = radius * np.exp(-radius / 75.0)
        positions.extend(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
        weights.extend(np.full(count, ring_weight / count))
    dummy = type("Dummy", (), {})()
    dummy.x_grid = xx
    dummy.y_grid = yy
    dummy.positions = np.asarray(positions, dtype=float)
    dummy.weights = np.asarray(weights, dtype=float)
    dummy.weights /= np.sum(dummy.weights)
    aperture = np.hypot(xx, yy) <= 250.0
    activation, _, audit = activation_and_carriers(dummy, aperture)
    return {
        "source_points": int(len(dummy.positions)),
        "scale_length_kpc": 75.0,
        "activation_RMS": float(np.sqrt(np.mean(activation[aperture] ** 2))),
        "activation_maximum": float(np.max(np.abs(activation[aperture]))),
        "median_coherence": audit["median_coherence"],
        "median_tidal_balance": audit["median_tidal_balance"],
    }


def solar_point_audit() -> dict:
    size = 128
    axis = (np.arange(size) - (size - 1) / 2.0) * 10.0
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    dummy = type("Dummy", (), {})()
    dummy.x_grid = xx
    dummy.y_grid = yy
    dummy.positions = np.asarray([[0.0, 0.0]])
    dummy.weights = np.asarray([1.0])
    aperture = np.hypot(xx, yy) <= 250.0
    activation, _, _ = activation_and_carriers(dummy, aperture)
    return {
        "activation_RMS": float(np.sqrt(np.mean(activation[aperture] ** 2))),
        "activation_maximum": float(np.max(np.abs(activation[aperture]))),
        "force_change": 0.0,
        "Mercury_precession_change_mas_per_century": 0.0,
    }


def main() -> None:
    protocol_path = ROOT / "configs/p0573_tidal_arrival_fresh_replication_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_download_or_fresh_map_spatial_inspection":
        raise RuntimeError("P0573 protocol is not frozen")
    _, manifest = assert_frozen_integrity(protocol_path, protocol)
    audit_directory = ROOT / protocol["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    spacing = float(protocol["spatial_preprocessing"]["grid_spacing_kpc"])
    locked = protocol["locked_formula"]
    primary_rows: list[dict] = []
    uncertainty_rows: list[dict] = []
    glafic_rows: list[dict] = []
    activation_rows: list[dict] = []
    plot_rows = []

    for system in protocol["systems"]:
        data, world = system_geometry(system, protocol, sources, audits)
        aperture = data.radius <= float(protocol["spatial_preprocessing"]["score_radius_kpc"])
        local = deposit_baryons(data, float(locked["local_control_width_kpc"]))
        local[~aperture] = 0.0
        local /= np.sum(local)
        _, carriers, activation_audit = activation_and_carriers(data, aperture)
        destination = destination_map(
            carriers["tidal_weighted"],
            float(locked["arrival_smoothing_kpc"]),
            spacing,
            aperture,
        )
        fraction = float(locked["route_fraction_f"])
        selected = (1.0 - fraction) * local + fraction * destination
        selected /= np.sum(selected)
        activation_rows.append({"system": data.label, **activation_audit})

        local_manifest = manifest[manifest.system.eq(data.label)]
        range_rows = local_manifest[
            local_manifest.kind.eq("range_kappa") & local_manifest.method.eq("lenstool")
        ].copy()
        range_rows["sample_index_numeric"] = pd.to_numeric(range_rows.sample_index)
        range_rows = range_rows.sort_values("sample_index_numeric")
        if len(range_rows) != 100:
            raise RuntimeError(f"{data.label}: expected 100 frozen Lenstool maps")
        raw_sum = np.zeros_like(data.x_grid, dtype=float)
        raw_count = np.zeros_like(data.x_grid, dtype=int)
        for number, row in enumerate(range_rows.itertuples(index=False), start=1):
            path = ROOT / row.path
            if sha256(path) != row.sha256:
                raise RuntimeError(f"hash mismatch: {row.path}")
            raw = regrid_kappa_sky(path, world, data.x_grid.shape)
            finite = np.isfinite(raw)
            raw_sum[finite] += raw[finite]
            raw_count[finite] += 1
            target = lens_source_map(raw, data.radius, spacing, 20.0, (250.0, 300.0))
            local_js = shape_metrics(local, target, aperture)["jensen_shannon"]
            selected_js = shape_metrics(selected, target, aperture)["jensen_shannon"]
            uncertainty_rows.append(
                {
                    "system": data.label,
                    "realization": int(number - 1),
                    "local_JS": local_js,
                    "selected_JS": selected_js,
                    "selected_improves": bool(selected_js < local_js),
                }
            )
        raw_mean = np.divide(
            raw_sum,
            raw_count,
            out=np.full_like(raw_sum, np.nan),
            where=raw_count > 0,
        )
        mean_target = lens_source_map(raw_mean, data.radius, spacing, 20.0, (250.0, 300.0))
        for model, prediction in (("local_control", local), ("locked_arrival", selected)):
            primary_rows.append(
                {
                    "system": data.label,
                    "model": model,
                    **shape_metrics(prediction, mean_target, aperture),
                }
            )

        glafic_row = local_manifest[
            local_manifest.kind.eq("best_kappa") & local_manifest.method.eq("glafic")
        ].iloc[0]
        glafic_path = ROOT / glafic_row.path
        if sha256(glafic_path) != glafic_row.sha256:
            raise RuntimeError(f"hash mismatch: {glafic_row.path}")
        glafic_raw = regrid_kappa_sky(glafic_path, world, data.x_grid.shape)
        glafic_target = lens_source_map(
            glafic_raw, data.radius, spacing, 20.0, (250.0, 300.0)
        )
        for model, prediction in (("local_control", local), ("locked_arrival", selected)):
            glafic_rows.append(
                {
                    "system": data.label,
                    "model": model,
                    **shape_metrics(prediction, glafic_target, aperture),
                }
            )
        plot_rows.append((data.label, data.axis, mean_target, selected))
        print(f"P0573 scored {data.label}: 100 Lenstool + GLAFIC", flush=True)

    primary = pd.DataFrame(primary_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    glafic = pd.DataFrame(glafic_rows)
    primary_js = primary.pivot(index="system", columns="model", values="jensen_shannon")
    primary_pearson = primary.pivot(index="system", columns="model", values="pearson")
    glafic_js = glafic.pivot(index="system", columns="model", values="jensen_shannon")
    local_mean = float(primary_js.local_control.mean())
    selected_mean = float(primary_js.locked_arrival.mean())
    gain = float(1.0 - selected_mean / local_mean)
    systems_improved = int((primary_js.locked_arrival < primary_js.local_control).sum())
    realization_fraction = float(uncertainty.selected_improves.mean())
    glafic_local = float(glafic_js.local_control.mean())
    glafic_selected = float(glafic_js.locked_arrival.mean())
    glafic_gain = float(1.0 - glafic_selected / glafic_local)
    glafic_improved = int((glafic_js.locked_arrival < glafic_js.local_control).sum())
    local_pearson = float(primary_pearson.local_control.mean())
    selected_pearson = float(primary_pearson.locked_arrival.mean())
    solar = solar_point_audit()
    disk = axisymmetric_disk_audit()
    gates_cfg = protocol["advance_gates"]
    gates = {
        "primary_improvement_pass": bool(
            gain >= float(gates_cfg["equal_system_improvement_vs_local_fraction_min"])
        ),
        "primary_system_count_pass": bool(
            systems_improved >= int(gates_cfg["systems_improved_min"])
        ),
        "lenstool_uncertainty_pass": bool(
            realization_fraction
            >= float(gates_cfg["lenstool_realizations_improved_fraction_min"])
        ),
        "glafic_improvement_pass": bool(
            glafic_gain
            >= float(gates_cfg["glafic_equal_system_improvement_vs_local_fraction_min"])
        ),
        "glafic_system_count_pass": bool(
            glafic_improved >= int(gates_cfg["glafic_systems_improved_min"])
        ),
        "Pearson_pass": bool(selected_pearson >= local_pearson),
        "solar_point_null_pass": bool(solar["activation_RMS"] <= 1.0e-12),
        "extended_axisymmetric_disk_null_pass": bool(disk["activation_RMS"] <= 1.0e-12),
    }
    gates["raw_lensing_followup_authorized"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["result_directory"]
    output.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output / "system_scores.csv", index=False)
    uncertainty.to_csv(output / "uncertainty.csv", index=False)
    glafic.to_csv(output / "glafic_scores.csv", index=False)
    pd.DataFrame(activation_rows).to_csv(output / "activation_audit.csv", index=False)
    result = {
        "local_equal_system_JS": local_mean,
        "locked_equal_system_JS": selected_mean,
        "improvement_vs_local_fraction": gain,
        "systems_improved": systems_improved,
        "lenstool_realizations_improved_fraction": realization_fraction,
        "local_mean_Pearson": local_pearson,
        "locked_mean_Pearson": selected_pearson,
        "glafic_local_equal_system_JS": glafic_local,
        "glafic_locked_equal_system_JS": glafic_selected,
        "glafic_improvement_vs_local_fraction": glafic_gain,
        "glafic_systems_improved": glafic_improved,
    }
    report = {
        "report_version": "P0573-TIDAL-ARRIVAL-FRESH-REPLICATION-RESULTS-0.1.0",
        "status": "complete_locked_fresh_replication",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "fresh_clusters": len(protocol["systems"]),
            "lenstool_realizations": len(uncertainty),
            "glafic_method_controls": len(protocol["systems"]),
            "parameters_fit_on_fresh_systems": 0,
        },
        "locked_formula": locked,
        "result": result,
        "per_system": [
            {
                "system": label,
                "local_JS": float(primary_js.loc[label, "local_control"]),
                "locked_JS": float(primary_js.loc[label, "locked_arrival"]),
                "improvement_fraction": float(
                    1.0
                    - primary_js.loc[label, "locked_arrival"]
                    / primary_js.loc[label, "local_control"]
                ),
                "glafic_improvement_fraction": float(
                    1.0
                    - glafic_js.loc[label, "locked_arrival"]
                    / glafic_js.loc[label, "local_control"]
                ),
            }
            for label in primary_js.index
        ],
        "cross_domain": {
            "solar_point_source": solar,
            "extended_axisymmetric_exponential_disk": disk,
            "SPARC_speed_prediction": "undefined: the arrival law redistributes a normalized projected map but supplies no radial acceleration law",
        },
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0573 fresh tidal-arrival replication",
        "",
        f"Locked equal-system JS: **{selected_mean:.5f}** versus local **{local_mean:.5f}**; gain **{100*gain:.2f}%**.",
        f"Fresh systems improved: **{systems_improved}/3**; Lenstool realizations improved: **{100*realization_fraction:.1f}%**.",
        f"Independent GLAFIC gain: **{100*glafic_gain:.2f}%** on **{glafic_improved}/3** systems.",
        f"Extended axisymmetric-disk activation RMS: **{disk['activation_RMS']:.5f}** (an exact null would be zero).",
        f"Raw-lensing follow-up authorized: **{gates['raw_lensing_followup_authorized']}**.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis_plot, (label, grid_axis, target, selected) in zip(axes[:3], plot_rows):
        extent = [grid_axis[0], grid_axis[-1], grid_axis[0], grid_axis[-1]]
        axis_plot.imshow(target, origin="lower", extent=extent, cmap="magma")
        axis_plot.contour(
            *np.meshgrid(grid_axis, grid_axis, indexing="xy"),
            selected,
            levels=5,
            colors="cyan",
            linewidths=0.8,
        )
        axis_plot.set_title(label)
        axis_plot.set_xlim(-300, 300)
        axis_plot.set_ylim(-300, 300)
        axis_plot.set_xticks([])
        axis_plot.set_yticks([])
    x = np.arange(len(primary_js))
    axes[3].bar(x - 0.18, primary_js.local_control, 0.36, label="local")
    axes[3].bar(x + 0.18, primary_js.locked_arrival, 0.36, label="locked")
    axes[3].set_xticks(x, primary_js.index, rotation=25, ha="right")
    axes[3].set_ylabel("Jensen-Shannon")
    axes[3].legend()
    fig.suptitle("P0573 genuinely fresh locked tidal-arrival replication")
    fig.savefig(output / "p0573_tidal_arrival_fresh_replication.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print(json.dumps(report["cross_domain"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
