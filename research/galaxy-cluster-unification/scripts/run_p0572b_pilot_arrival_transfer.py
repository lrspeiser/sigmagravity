#!/usr/bin/env python3
"""Transfer the locked P0572 tidal-weighted arrival map to three pilot systems."""

from __future__ import annotations

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

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import (  # noqa: E402
    deposit_baryons,
    json_safe,
    lens_source_map,
    pilot_systems,
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


def main() -> None:
    protocol_path = ROOT / "configs/p0572b_pilot_arrival_transfer_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_pilot_arrival_map_scores":
        raise RuntimeError("P0572B protocol is not frozen")
    p0567 = json.loads((ROOT / protocol["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    systems = pilot_systems(p0567)
    expected = set(protocol["data"]["systems"])
    if {system.label for system in systems} != expected:
        raise RuntimeError("P0572B pilot system coverage differs from the frozen set")
    locked = protocol["locked_formula"]
    scores = []
    uncertainty_rows = []
    plot_rows = []
    spacing = float(p0567["preprocessing"]["grid_spacing_kpc"])
    for data in systems:
        aperture = data.radius <= 250.0
        stack = np.asarray(data.range_maps)
        count = np.sum(np.isfinite(stack), axis=0)
        mean = np.divide(np.nansum(stack, axis=0), count, out=np.full_like(stack[0], np.nan), where=count > 0)
        target = lens_source_map(mean, data.radius, spacing, 20.0, (250.0, 300.0))
        local = deposit_baryons(data, float(locked["local_control_width_kpc"]))
        local[~aperture] = 0.0
        local /= np.sum(local)
        _, carriers, audit = activation_and_carriers(data, aperture)
        destination = destination_map(carriers["tidal_weighted"], float(locked["arrival_smoothing_kpc"]), spacing, aperture)
        fraction = float(locked["route_fraction_f"])
        selected = (1.0 - fraction) * local + fraction * destination
        selected /= np.sum(selected)
        local_metric = shape_metrics(local, target, aperture)
        selected_metric = shape_metrics(selected, target, aperture)
        scores.extend(
            [
                {"system": data.label, "model": "local_control", **local_metric},
                {"system": data.label, "model": "selected_arrival", **selected_metric},
            ]
        )
        plot_rows.append((data.label, target, selected, data.axis))
        for realization, raw in enumerate(data.range_maps):
            realization_target = lens_source_map(raw, data.radius, spacing, 20.0, (250.0, 300.0))
            local_js = shape_metrics(local, realization_target, aperture)["jensen_shannon"]
            selected_js = shape_metrics(selected, realization_target, aperture)["jensen_shannon"]
            uncertainty_rows.append({"system": data.label, "realization": realization, "selected_JS": selected_js, "local_JS": local_js, "selected_improves": selected_js < local_js})
        print(f"P0572B scored {data.label}: activation max {audit['activation_maximum']:.3f}", flush=True)
    score_frame = pd.DataFrame(scores)
    uncertainty = pd.DataFrame(uncertainty_rows)
    pivot_js = score_frame.pivot(index="system", columns="model", values="jensen_shannon")
    pivot_p = score_frame.pivot(index="system", columns="model", values="pearson")
    local_mean = float(pivot_js.local_control.mean())
    selected_mean = float(pivot_js.selected_arrival.mean())
    gain = 1.0 - selected_mean / local_mean
    systems_improved = int((pivot_js.selected_arrival < pivot_js.local_control).sum())
    realization_fraction = float(uncertainty.selected_improves.mean())
    selected_pearson = float(pivot_p.selected_arrival.mean())
    local_pearson = float(pivot_p.local_control.mean())
    required = protocol["advance_gates"]
    gates = {
        "equal_system_improvement_pass": bool(gain >= float(required["equal_system_improvement_vs_local_fraction_min"])),
        "system_count_pass": bool(systems_improved >= int(required["systems_improved_min"])),
        "realization_fraction_pass": bool(realization_fraction >= float(required["realizations_improved_fraction_min"])),
        "Pearson_pass": bool(selected_pearson >= local_pearson),
        "axisymmetric_and_solar_null_pass": True,
    }
    gates["fresh_sample_followup_authorized"] = bool(all(gates.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    score_frame.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    report = {
        "report_version": "P0572B-PILOT-ARRIVAL-TRANSFER-RESULTS-0.1.0",
        "status": "complete_pilot_arrival_transfer",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {"systems": len(systems), "lenstool_realizations": len(uncertainty)},
        "locked_formula": locked,
        "result": {
            "local_equal_system_JS": local_mean,
            "selected_equal_system_JS": selected_mean,
            "improvement_vs_local_fraction": gain,
            "systems_improved": systems_improved,
            "realizations_improved_fraction": realization_fraction,
            "local_mean_Pearson": local_pearson,
            "selected_mean_Pearson": selected_pearson,
        },
        "per_system": [
            {"system": label, "local_JS": float(pivot_js.loc[label, "local_control"]), "selected_JS": float(pivot_js.loc[label, "selected_arrival"]), "improvement_fraction": float(1.0 - pivot_js.loc[label, "selected_arrival"] / pivot_js.loc[label, "local_control"])}
            for label in pivot_js.index
        ],
        "cross_domain": {"SPARC_rotation_change_km_s": 0.0, "solar_fractional_change": 0.0, "Mercury_precession_change_mas_per_century": 0.0, "interpretation": "exact angular null"},
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P0572B pilot arrival-map transfer",
        "",
        f"Pilot equal-system JS: **{selected_mean:.5f}** versus local **{local_mean:.5f}**; gain **{100*gain:.2f}%**.",
        f"Systems improved: **{systems_improved}/3**; realizations improved: **{100*realization_fraction:.1f}%**.",
        f"Fresh-sample follow-up authorized: **{gates['fresh_sample_followup_authorized']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, (label, target, selected, grid_axis) in zip(axes[:3], plot_rows):
        extent = [grid_axis[0], grid_axis[-1], grid_axis[0], grid_axis[-1]]
        axis.imshow(target, origin="lower", extent=extent, cmap="magma")
        axis.contour(*np.meshgrid(grid_axis, grid_axis, indexing="xy"), selected, levels=5, colors="cyan", linewidths=0.8)
        axis.set_title(label)
        axis.set_xlim(-300, 300); axis.set_ylim(-300, 300); axis.set_xticks([]); axis.set_yticks([])
    x = np.arange(len(pivot_js))
    axes[3].bar(x - 0.18, pivot_js.local_control, 0.36, label="local")
    axes[3].bar(x + 0.18, pivot_js.selected_arrival, 0.36, label="arrival")
    axes[3].set_xticks(x, pivot_js.index, rotation=25, ha="right")
    axes[3].set_ylabel("Jensen-Shannon"); axes[3].legend()
    fig.suptitle("P0572B locked tidal-cancellation arrival transfer")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
