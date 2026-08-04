#!/usr/bin/env python3
"""Render outcome-blind audit panels for the frozen Sigma v17B regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pycrates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "results" / "sigma_v17b_temperature_regions" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17b_temperature_regions" / "audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_values(path: Path) -> np.ndarray:
    return np.asarray(pycrates.read_file(str(path)).get_image().values, dtype=float)


def positive_limits(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = values[mask & np.isfinite(values) & (values > 0)]
    if not selected.size:
        raise RuntimeError("audit image has no positive finite pixels")
    low, high = np.percentile(selected, [2.0, 99.5])
    return max(float(low), np.finfo(float).tiny), max(float(high), float(low) * 1.01)


def region_boundaries(binmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(binmap.shape, dtype=bool)
    boundaries[:, 1:] |= binmap[:, 1:] != binmap[:, :-1]
    boundaries[1:, :] |= binmap[1:, :] != binmap[:-1, :]
    return boundaries & mask


def component_count(selection: np.ndarray) -> int:
    """Count four-connected components without adding a SciPy dependency."""
    remaining = set(map(tuple, np.argwhere(selection)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def audit_region_geometry(binmap: np.ndarray, mask: np.ndarray, expected: int) -> dict:
    finite_inside = mask & np.isfinite(binmap)
    integer_inside = finite_inside & np.isclose(binmap, np.rint(binmap), atol=1e-9)
    assigned_inside = integer_inside & (binmap >= 0)
    identifiers = sorted(np.unique(np.rint(binmap[assigned_inside]).astype(int)).tolist())
    components = {
        str(identifier): component_count(assigned_inside & (np.rint(binmap) == identifier))
        for identifier in identifiers
    }
    result = {
        "mask_pixels": int(mask.sum()),
        "assigned_pixels": int(assigned_inside.sum()),
        "unassigned_mask_pixels": int((mask & ~assigned_inside).sum()),
        "region_identifiers": identifiers,
        "identifiers_contiguous_from_zero": identifiers == list(range(expected)),
        "connected_components_per_region": components,
        "all_regions_four_connected": all(value == 1 for value in components.values()),
    }
    if result["unassigned_mask_pixels"] != 0:
        raise RuntimeError("frozen contbin map does not cover every analysis-mask pixel")
    if not result["identifiers_contiguous_from_zero"]:
        raise RuntimeError("frozen contbin identifiers are incomplete or non-contiguous")
    if not result["all_regions_four_connected"]:
        raise RuntimeError("a frozen contbin region is not four-connected")
    return result


def render_cluster(cluster: dict, output: Path, products: dict[str, Path]) -> dict:
    science = image_values(products["science_counts"])
    background = image_values(products["scaled_background"])
    exposure = image_values(products["exposure_map"])
    analysis_mask = image_values(products["analysis_mask"])
    binmap = image_values(products["binmap"])
    arrays = [science, background, exposure, analysis_mask, binmap]
    if len({array.shape for array in arrays}) != 1:
        raise RuntimeError(f"{cluster['cluster']} audit products do not share one grid")

    mask = np.isfinite(analysis_mask) & (analysis_mask > 0)
    geometry = audit_region_geometry(binmap, mask, int(cluster["valid_region_count"]))
    net = science - background
    surface_brightness = np.full(science.shape, np.nan)
    usable = mask & np.isfinite(exposure) & (exposure > 0)
    surface_brightness[usable] = np.maximum(net[usable], 0) / exposure[usable]
    boundaries = region_boundaries(binmap, mask)
    science_low, science_high = positive_limits(science, mask)
    sb_low, sb_high = positive_limits(surface_brightness, mask)
    relative_exposure = exposure / np.nanmax(exposure)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    extent = (0.5, science.shape[1] + 0.5, 0.5, science.shape[0] + 0.5)
    center = cluster["final_center"]
    marker = (center["logicalx"], center["logicaly"])

    image = axes[0, 0].imshow(
        science,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=science_low, vmax=science_high),
    )
    axes[0, 0].plot(*marker, marker="+", color="cyan", markersize=13, markeredgewidth=2)
    axes[0, 0].set_title("Point-source-excluded 0.5–7 keV counts")
    fig.colorbar(image, ax=axes[0, 0], label="counts per output pixel")

    image = axes[0, 1].imshow(
        relative_exposure,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].contour(mask.astype(int), levels=[0.5], colors="white", linewidths=1.0)
    axes[0, 1].plot(*marker, marker="+", color="red", markersize=13, markeredgewidth=2)
    axes[0, 1].set_title("Relative exposure and frozen analysis mask")
    fig.colorbar(image, ax=axes[0, 1], label="exposure / maximum")

    image = axes[1, 0].imshow(
        surface_brightness,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=sb_low, vmax=sb_high),
    )
    axes[1, 0].contour(boundaries.astype(int), levels=[0.5], colors="cyan", linewidths=0.45)
    axes[1, 0].plot(*marker, marker="+", color="white", markersize=13, markeredgewidth=2)
    axes[1, 0].set_title("Background-subtracted exposure-corrected signal + bin edges")
    fig.colorbar(image, ax=axes[1, 0], label="net counts / exposure")

    display_binmap = np.where(mask, binmap, np.nan)
    image = axes[1, 1].imshow(
        display_binmap,
        origin="lower",
        extent=extent,
        cmap="turbo",
        interpolation="nearest",
    )
    axes[1, 1].contour(boundaries.astype(int), levels=[0.5], colors="black", linewidths=0.35)
    axes[1, 1].plot(*marker, marker="+", color="white", markersize=13, markeredgewidth=2)
    axes[1, 1].set_title(f"Frozen contbin map: {cluster['valid_region_count']} valid regions")
    fig.colorbar(image, ax=axes[1, 1], label="region identifier")

    for axis in axes.flat:
        axis.set_xlabel("logical X pixel")
        axis.set_ylabel("logical Y pixel")
        axis.set_aspect("equal")

    fig.suptitle(
        f"Sigma v17B outcome-blind region audit — {cluster['cluster']}\n"
        f"minimum net counts {cluster['minimum_net_counts']:.1f}; "
        f"minimum S/N {cluster['minimum_signal_to_noise']:.2f}; "
        f"minimum source fraction {cluster['minimum_source_fraction']:.3f}",
        fontsize=14,
    )
    destination = output / f"{cluster['cluster']}_temperature_region_audit.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return {
        "cluster": cluster["cluster"],
        "relative_path": destination.relative_to(ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "array_shape": list(science.shape),
        "masked_pixels": int(mask.sum()),
        "regions": int(cluster["valid_region_count"]),
        "geometry": geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report_path = args.report.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "both_clusters_passed_frozen_temperature_region_gate":
        raise RuntimeError("temperature-region gate has not passed")

    records = []
    for cluster in report["clusters"]:
        products = {}
        for product in cluster["frozen_snapshot"]["products"]:
            path = ROOT / product["relative_path"]
            if path.stat().st_size != product["bytes"] or sha256(path) != product["sha256"]:
                raise RuntimeError(f"frozen product does not match report: {path}")
            products[product["role"]] = path
        records.append(render_cluster(cluster, output, products))

    manifest = {
        "status": "rendered_pending_manual_visual_audit",
        "generated_utc": datetime.now(UTC).isoformat(),
        "temperature_region_report_sha256": sha256(report_path),
        "outcome_data_opened": False,
        "renders": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
