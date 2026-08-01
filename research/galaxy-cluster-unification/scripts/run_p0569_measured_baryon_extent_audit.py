#!/usr/bin/env python3
"""Compare measured CLASH baryon extents with the P0568 effective width."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import build_contexts  # noqa: E402
from run_p0557_baryon_proxy_tidal import compressed_map_catalog, json_safe  # noqa: E402
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(weights)
    return float(np.interp(float(quantile) * cumulative[-1], cumulative, values))


def extent_row(system, component, catalog, scale, role, mass_fraction=None):
    weights = catalog.normalized_light_weight.to_numpy(float)
    weights /= np.sum(weights)
    x = catalog.x_arcsec.to_numpy(float) * scale
    y = catalog.y_arcsec.to_numpy(float) * scale
    center_x = float(np.sum(weights * x))
    center_y = float(np.sum(weights * y))
    radius = np.hypot(x - center_x, y - center_y)
    rms = float(np.sqrt(np.sum(weights * radius**2)))
    r50 = weighted_quantile(radius, weights, 0.5)
    r80 = weighted_quantile(radius, weights, 0.8)
    r90 = weighted_quantile(radius, weights, 0.9)
    sigma_rms = rms / math.sqrt(2.0)
    sigma_r80 = r80 / math.sqrt(2.0 * math.log(5.0))
    return {
        "system": system,
        "component": component,
        "role": role,
        "pseudo_sources": len(catalog),
        "angular_scale_kpc_per_arcsec": scale,
        "centroid_x_kpc": center_x,
        "centroid_y_kpc": center_y,
        "weighted_RMS_radius_kpc": rms,
        "R50_kpc": r50,
        "R80_kpc": r80,
        "R90_kpc": r90,
        "equivalent_sigma_RMS_kpc": sigma_rms,
        "equivalent_sigma_R80_kpc": sigma_r80,
        "mass_fraction": mass_fraction,
    }


def main():
    protocol_path = ROOT / "configs/p0569_measured_baryon_extent_audit_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_new_component_extent_computation":
        raise RuntimeError("P0569 protocol is not frozen")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text(encoding="utf-8"))
    p0557 = json.loads((ROOT / p0559["inputs"]["p0557_protocol"]).read_text(encoding="utf-8"))
    member = json.loads((ROOT / p0559["inputs"]["member_tidal_protocol"]).read_text(encoding="utf-8"))
    member["optimization"]["maximum_function_evaluations"] = 10
    contexts, _, _ = build_contexts(
        member, softening_kpc=float(p0559["locked_field"]["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    catalogs, physical_audits = physical_catalogs(p0559, contexts, registered)
    audit_by_system = physical_audits.set_index("system_label")
    block = int(p0557["proxy_maps"]["compression_block_pixels"])
    rows = []
    for context in contexts:
        label = context.system["label"]
        scale = float(
            context.local_protocol["cosmology_and_coordinates"][
                "angular_scale_kpc_per_arcsec"
            ]
        )
        star_catalog = compressed_map_catalog(
            registered[label]["axis"],
            registered[label]["star"],
            block_pixels=block,
            transform="linear",
        )
        gas_fraction = float(audit_by_system.loc[label, "absolute_projected_map_gas_fraction"])
        rows.append(extent_row(label, "members", context.members, scale, "light_proxy"))
        rows.append(extent_row(label, "registered_starlight", star_catalog, scale, "stellar_proxy", 1.0 - gas_fraction))
        component_keys = [
            ("accept_gas_spherical", ("accept_absolute", 0.0, False), "gas"),
            ("accept_gas_sqrt_morphology", ("accept_absolute", 0.5, False), "gas"),
            ("accept_gas_linear_morphology", ("accept_absolute", 1.0, False), "gas"),
            ("stars_plus_gas_spherical", ("accept_absolute", 0.0, True), "primary_sensitivity"),
            ("stars_plus_gas_sqrt_morphology", ("accept_absolute", 0.5, True), "primary"),
            ("stars_plus_gas_linear_morphology", ("accept_absolute", 1.0, True), "primary_sensitivity"),
            ("tian_anchor_stars_plus_gas_sqrt", ("renormalize_accept_to_tian_spherical_anchor", 0.5, True), "normalization_sensitivity"),
        ]
        for name, key, role in component_keys:
            fraction = 1.0 if role == "gas" else gas_fraction
            rows.append(extent_row(label, name, catalogs[label][key], scale, role, fraction))
    frame = pd.DataFrame(rows)
    band = protocol["extent_metrics"]["p0568_preferred_band_kpc"]
    frame["RMS_sigma_in_P0568_band"] = frame.equivalent_sigma_RMS_kpc.between(*band)
    frame["R80_sigma_in_P0568_band"] = frame.equivalent_sigma_R80_kpc.between(*band)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / protocol["outputs"]["component_extents"], index=False)
    primary = frame[frame.component.eq("stars_plus_gas_sqrt_morphology")]
    median_rms = float(primary.equivalent_sigma_RMS_kpc.median())
    median_r80 = float(primary.equivalent_sigma_R80_kpc.median())
    systems_matching = int(
        (primary.RMS_sigma_in_P0568_band | primary.R80_sigma_in_P0568_band).sum()
    )
    extent_match = bool(
        float(band[0]) <= median_rms <= float(band[1])
        or float(band[0]) <= median_r80 <= float(band[1])
    )
    coverage_match = systems_matching >= 3
    component_summary = (
        frame.groupby("component")
        .agg(
            systems=("system", "nunique"),
            median_sigma_RMS_kpc=("equivalent_sigma_RMS_kpc", "median"),
            median_sigma_R80_kpc=("equivalent_sigma_R80_kpc", "median"),
            median_R80_kpc=("R80_kpc", "median"),
            systems_RMS_in_band=("RMS_sigma_in_P0568_band", "sum"),
            systems_R80_in_band=("R80_sigma_in_P0568_band", "sum"),
        )
        .reset_index()
    )
    report = {
        "report_version": "P0569-MEASURED-BARYON-EXTENT-AUDIT-RESULTS-0.1.0",
        "status": "complete_extent_audit",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {"systems": primary.system.nunique(), "component_system_rows": len(frame), "components": frame.component.nunique()},
        "primary": {
            "component": "stars_plus_gas_sqrt_morphology",
            "median_equivalent_sigma_RMS_kpc": median_rms,
            "median_equivalent_sigma_R80_kpc": median_r80,
            "systems_matching_either_sigma_definition": systems_matching,
            "systems": json_safe(primary.to_dict(orient="records")),
        },
        "component_summary": json_safe(component_summary.to_dict(orient="records")),
        "gates": {
            "median_extent_matches_P0568_band": extent_match,
            "at_least_three_systems_match": coverage_match,
            "measured_baryonic_extent_is_sufficient_scale_explanation": bool(extent_match and coverage_match),
        },
        "interpretation": {
            "if_pass": "The phenomenological P0568 width is commensurate with measured projected baryon extent and should not be interpreted as a gravity propagation length until the measured map is used directly.",
            "if_fail": "The P0568 width exceeds or undershoots measured baryonic extent and remains an independent phenomenological coordinate.",
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0569 measured baryon extent audit",
        "",
        f"Primary median sigma from RMS: **{median_rms:.1f} kpc**.",
        f"Primary median sigma from R80: **{median_r80:.1f} kpc**.",
        f"Systems matching the P0568 75-125 kpc band: **{systems_matching}/4**.",
        f"Measured extent is a sufficient scale explanation: **{extent_match and coverage_match}**.",
        "",
        "This is a scale match only; raw-lensing amplitude and angular structure remain to be tested.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    colors = {
        "members": "tab:gray",
        "registered_starlight": "goldenrod",
        "accept_gas_sqrt_morphology": "tab:blue",
        "stars_plus_gas_sqrt_morphology": "tab:purple",
        "tian_anchor_stars_plus_gas_sqrt": "tab:red",
    }
    shown = list(colors)
    for axis, system in zip(axes[:4], protocol["systems"]):
        local = frame[(frame.system.eq(system)) & frame.component.isin(shown)]
        for index, row in enumerate(local.itertuples(index=False)):
            axis.scatter(row.equivalent_sigma_RMS_kpc, index, color=colors[row.component], s=45)
            axis.text(row.equivalent_sigma_RMS_kpc + 2, index, row.component.replace("_", " "), va="center", fontsize=7)
        axis.axvspan(float(band[0]), float(band[1]), color="green", alpha=0.15)
        axis.set_title(system)
        axis.set_xlabel("equivalent Gaussian sigma from RMS (kpc)")
        axis.set_yticks([])
    summary_plot = component_summary[component_summary.component.isin(shown)].sort_values("median_sigma_RMS_kpc")
    axes[4].barh(summary_plot.component, summary_plot.median_sigma_RMS_kpc, color=[colors[name] for name in summary_plot.component])
    axes[4].axvspan(float(band[0]), float(band[1]), color="green", alpha=0.15)
    axes[4].set_xlabel("four-system median sigma (kpc)")
    axes[4].tick_params(axis="y", labelsize=7)
    axes[5].axis("off")
    axes[5].text(0.03, 0.95, f"P0568 preferred band\n{band[0]:g}-{band[1]:g} kpc\n\nprimary median RMS sigma\n{median_rms:.1f} kpc\n\nprimary median R80 sigma\n{median_r80:.1f} kpc\n\nmatching systems\n{systems_matching}/4", va="top", family="monospace", fontsize=12)
    fig.suptitle("P0569 measured baryon extent versus P0568 effective width")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["primary"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
