#!/usr/bin/env python3
"""Render and machine-audit the already-hashed V19M region products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v19m_adaptive_thermodynamic_regions" / "report.json"
SOURCE_REPORT = ROOT / "results" / "sigma_v19h_source_maps" / "report.json"
OUTPUT = ROOT / "results" / "sigma_v19m_adaptive_thermodynamic_regions"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image(path: Path) -> np.ndarray:
    with fits.open(path) as payload:
        return np.asarray(payload[0].data, dtype=float)


def source_product(row: dict, role: str) -> Path:
    matches = [item for item in row["frozen_snapshot"]["products"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {role} source product")
    path = ROOT / matches[0]["relative_path"]
    if sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"source product changed: {path}")
    return path


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report_hash = sha256(REPORT)
    sources = {
        row["cluster"]: row
        for row in json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))["clusters"]
    }
    diagnostics = []
    for record in report["clusters"]:
        cluster = record["cluster"]
        for product in record["products"]:
            path = ROOT / product["relative_path"]
            if path.stat().st_size != product["bytes"] or sha256(path) != product["sha256"]:
                raise RuntimeError(f"V19M product changed before audit: {path}")
        roles = {row["role"]: row for row in record["products"] if row["role"] != "spectral_region"}
        binmap = image(ROOT / roles["binmap"]["relative_path"])
        counts = image(source_product(sources[cluster], "broad_counts"))
        background = image(source_product(sources[cluster], "broad_scaled_background"))
        exposure = image(source_product(sources[cluster], "broad_exposure"))
        mask = image(source_product(sources[cluster], "analysis_mask")) > 0.5
        labels = sorted(int(value) for value in np.unique(binmap[mask]) if value >= 0)
        components = {}
        for label in labels:
            _, component_count = ndimage.label(mask & (binmap == label))
            components[label] = int(component_count)
        admitted = set(record["valid_region_ids"])
        disconnected_admitted = sorted(
            label for label in admitted if components.get(label, 0) != 1
        )
        net_rate = np.divide(
            counts - background,
            exposure,
            out=np.full_like(counts, np.nan),
            where=mask & (exposure > 0.0),
        )
        positive = net_rate[np.isfinite(net_rate) & (net_rate > 0.0)]
        lower, upper = np.percentile(positive, [2.0, 99.5])
        display = np.log10(np.clip(net_rate, lower, upper))
        boundaries = find_boundaries(binmap.astype(int), connectivity=1, mode="thick") & mask
        invalid = mask & np.isin(binmap.astype(int), list(set(labels) - admitted))

        figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        first = axes[0].imshow(display, origin="lower", cmap="magma")
        axes[0].imshow(np.ma.masked_where(~boundaries, boundaries), origin="lower", cmap="gray", alpha=0.8)
        axes[0].imshow(np.ma.masked_where(~invalid, invalid), origin="lower", cmap="winter", alpha=0.8)
        figure.colorbar(first, ax=axes[0], fraction=0.046, label="log10 net count rate")
        axes[0].set_title("Broad-band source with region boundaries")
        label_display = np.ma.masked_where(~mask, binmap)
        second = axes[1].imshow(label_display, origin="lower", cmap="turbo")
        axes[1].imshow(np.ma.masked_where(~invalid, invalid), origin="lower", cmap="gray", alpha=0.9)
        figure.colorbar(second, ax=axes[1], fraction=0.046, label="contbin ID")
        axes[1].set_title("Region topology; gray regions fail admission")
        for axis in axes:
            axis.set_xlabel("registered output pixel")
            axis.set_ylabel("registered output pixel")
        figure.suptitle(f"{cluster}: frozen V19M adaptive thermodynamic regions")
        figure_path = OUTPUT / f"{cluster.lower()}_region_audit.png"
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        diagnostics.append(
            {
                "cluster": cluster,
                "bin_labels_inside_mask": len(labels),
                "admitted_region_count": len(admitted),
                "disconnected_admitted_region_ids": disconnected_admitted,
                "maximum_connected_components_any_region": max(components.values()),
                "all_admitted_regions_one_connected_component": not disconnected_admitted,
                "diagnostic_path": figure_path.relative_to(ROOT).as_posix(),
                "diagnostic_sha256": sha256(figure_path),
                "diagnostic_bytes": figure_path.stat().st_size,
            }
        )
    payload = {
        "status": "machine_topology_audit_complete_visual_inspection_pending",
        "v19m_report_sha256": report_hash,
        "clusters": diagnostics,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    path = OUTPUT / "topology_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(sha256(path))


if __name__ == "__main__":
    main()
