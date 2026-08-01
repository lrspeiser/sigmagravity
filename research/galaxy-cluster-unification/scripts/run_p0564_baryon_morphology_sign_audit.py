#!/usr/bin/env python3
"""Audit observed star-gas morphology against the spent P0563 sign split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import build_contexts  # noqa: E402
from run_p0557_baryon_proxy_tidal import json_safe, sha256  # noqa: E402
from run_p0559_accept_projected_gas_tidal import prepare_registered_maps  # noqa: E402


def component_descriptors(axis, image, aperture):
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    mask = radius <= float(aperture)
    weight = np.maximum(np.asarray(image, dtype=float), 0.0) * mask
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("morphology aperture has no positive weight")
    weight /= total
    cx = float(np.sum(weight * xx))
    cy = float(np.sum(weight * yy))
    second = float(np.sum(weight * (xx * xx + yy * yy)))
    qxx = float(np.sum(weight * (xx * xx - yy * yy)) / max(second, 1e-30))
    qxy = float(np.sum(weight * (2.0 * xx * yy)) / max(second, 1e-30))
    qmag = float(np.hypot(qxx, qxy))
    qangle = float(np.degrees(0.5 * np.arctan2(qxy, qxx)) % 180.0)
    peak = np.unravel_index(np.argmax(weight), weight.shape)
    peak_radius = float(np.hypot(xx[peak], yy[peak]))
    concentration = float(weight[radius <= 0.5 * aperture].sum())
    rotated = np.flip(np.flip(weight, axis=0), axis=1)
    asymmetry = float(0.5 * np.sum(np.abs(weight - rotated)))
    return {
        "centroid_x_over_aperture": cx / aperture,
        "centroid_y_over_aperture": cy / aperture,
        "centroid_radius_over_aperture": float(np.hypot(cx, cy) / aperture),
        "peak_radius_over_aperture": peak_radius / aperture,
        "quadrupole_magnitude": qmag,
        "quadrupole_angle_deg": qangle,
        "half_aperture_concentration": concentration,
        "rot180_asymmetry": asymmetry,
        "_centroid_x": cx,
        "_centroid_y": cy,
        "_qxx": qxx,
        "_qxy": qxy,
        "_normalized_image": weight,
        "_mask": mask,
    }


def acute_quadrupole_misalignment(angle_a, angle_b):
    difference = abs(float(angle_a) - float(angle_b)) % 180.0
    return min(difference, 180.0 - difference)


def main():
    config_path = ROOT / "configs/p0564_baryon_morphology_sign_audit_protocol.json"
    protocol = json.loads(config_path.read_text())
    if not protocol["status"].startswith("frozen_after_p0563_"):
        raise RuntimeError("P0564 is not frozen with its post-hoc disclosure")
    p0559_path = ROOT / protocol["inputs"]["p0559_protocol"]
    member_path = ROOT / protocol["inputs"]["member_tidal_protocol"]
    sign_path = ROOT / protocol["inputs"]["p0563_report"]
    p0559 = json.loads(p0559_path.read_text())
    member = json.loads(member_path.read_text())
    contexts, _, _ = build_contexts(member, softening_kpc=20.0)
    registered = prepare_registered_maps(p0559, contexts)
    sign_report = json.loads(sign_path.read_text())
    sign_frame = pd.DataFrame(sign_report["per_system_summary"])
    sign_map = (
        sign_frame.drop_duplicates(["system_label", "near_zero_preferred_sign"])
        .set_index("system_label")
        .near_zero_preferred_sign.to_dict()
    )
    if sign_map != {
        "MACS0329": "positive",
        "MACS0429": "negative",
        "MACS1115": "positive",
        "MACS1931": "positive",
    }:
        raise RuntimeError("P0563 sign labels changed")

    rows = []
    for label, maps in registered.items():
        axis = maps["axis"]
        spacing = float(axis[1] - axis[0])
        star = np.maximum(maps["star"], 0.0)
        gas = np.sqrt(
            np.maximum(
                gaussian_filter(
                    np.maximum(maps["gas"], 0.0),
                    sigma=float(p0559["gas_map"]["smoothing_sigma_arcsec"]) / spacing,
                    mode="nearest",
                ),
                0.0,
            )
        )
        for aperture in map(float, protocol["maps"]["apertures_arcsec"]):
            components = {
                "star": component_descriptors(axis, star, aperture),
                "gas": component_descriptors(axis, gas, aperture),
            }
            for component, values in components.items():
                for descriptor in [
                    "centroid_x_over_aperture",
                    "centroid_y_over_aperture",
                    "centroid_radius_over_aperture",
                    "peak_radius_over_aperture",
                    "quadrupole_magnitude",
                    "quadrupole_angle_deg",
                    "half_aperture_concentration",
                    "rot180_asymmetry",
                ]:
                    rows.append(
                        {
                            "system_label": label,
                            "response_sign": sign_map[label],
                            "aperture_arcsec": aperture,
                            "component": component,
                            "descriptor": descriptor,
                            "value": values[descriptor],
                        }
                    )
            star_values, gas_values = components["star"], components["gas"]
            centroid_offset = np.hypot(
                star_values["_centroid_x"] - gas_values["_centroid_x"],
                star_values["_centroid_y"] - gas_values["_centroid_y"],
            ) / aperture
            misalignment = acute_quadrupole_misalignment(
                star_values["quadrupole_angle_deg"],
                gas_values["quadrupole_angle_deg"],
            )
            cos2 = float(np.cos(np.radians(2.0 * misalignment)))
            mask = star_values["_mask"] & gas_values["_mask"]
            star_pixels = star_values["_normalized_image"][mask]
            gas_pixels = gas_values["_normalized_image"][mask]
            correlation = float(np.corrcoef(star_pixels, gas_pixels)[0, 1])
            for descriptor, value in {
                "star_gas_centroid_offset_over_aperture": centroid_offset,
                "star_gas_quadrupole_misalignment_deg": misalignment,
                "star_gas_quadrupole_cos2_alignment": cos2,
                "star_gas_normalized_correlation": correlation,
            }.items():
                rows.append(
                    {
                        "system_label": label,
                        "response_sign": sign_map[label],
                        "aperture_arcsec": aperture,
                        "component": "joint",
                        "descriptor": descriptor,
                        "value": value,
                    }
                )
    descriptors = pd.DataFrame(rows)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(output / protocol["outputs"]["descriptors"], index=False)

    negative_label = protocol["ranking"]["negative_system"]
    positive_labels = protocol["ranking"]["positive_systems"]
    ranked_rows = []
    for keys, group in descriptors.groupby(
        ["aperture_arcsec", "component", "descriptor"], sort=True
    ):
        values = group.set_index("system_label").value
        negative = float(values.loc[negative_label])
        positive = values.loc[positive_labels].to_numpy(float)
        mean = float(positive.mean())
        std = float(positive.std(ddof=1))
        signed_z = (negative - mean) / max(std, 1.0e-6)
        ranked_rows.append(
            {
                "aperture_arcsec": keys[0],
                "component": keys[1],
                "descriptor": keys[2],
                "MACS0429_value": negative,
                "positive_group_mean": mean,
                "positive_group_sample_std": std,
                "signed_separation_z": signed_z,
                "absolute_separation_z": abs(signed_z),
                "positive_group_min": float(positive.min()),
                "positive_group_max": float(positive.max()),
                "outside_positive_range": bool(
                    negative < float(positive.min()) or negative > float(positive.max())
                ),
            }
        )
    ranked = pd.DataFrame(ranked_rows).sort_values(
        ["absolute_separation_z", "aperture_arcsec"], ascending=[False, True]
    )
    ranked.to_csv(output / protocol["outputs"]["ranked_separators"], index=False)
    top = ranked.head(12)
    report = {
        "report_version": "P0564-BARYON-MORPHOLOGY-SIGN-AUDIT-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "input_hashes": {
            "p0559_protocol": sha256(p0559_path),
            "member_tidal_protocol": sha256(member_path),
            "p0563_report": sha256(sign_path),
        },
        "descriptor_rows": int(len(descriptors)),
        "ranked_candidates": top.to_dict("records"),
        "primary": {
            "top_descriptor": top.iloc[0].descriptor,
            "top_component": top.iloc[0].component,
            "top_aperture_arcsec": float(top.iloc[0].aperture_arcsec),
            "top_signed_separation_z": float(top.iloc[0].signed_separation_z),
            "top_outside_positive_range": bool(top.iloc[0].outside_positive_range),
            "candidate_gate_nominated": True,
            "candidate_gate_validated": False,
        },
        "verdict": {"formula_promoted": False},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    display = top.iloc[::-1]
    labels = [
        f"{row.component}:{row.descriptor}@{row.aperture_arcsec:g}"
        for row in display.itertuples(index=False)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].barh(labels, display.signed_separation_z)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set(
        xlabel="MACS0429 separation from positive-sign mean (descriptive z)",
        title="Post-hoc morphology candidates",
    )
    joint_60 = descriptors[
        descriptors.aperture_arcsec.eq(60.0)
        & descriptors.component.eq("joint")
    ].pivot(index="system_label", columns="descriptor", values="value")
    axes[1].scatter(
        joint_60.star_gas_quadrupole_misalignment_deg,
        joint_60.star_gas_centroid_offset_over_aperture,
    )
    for label, row in joint_60.iterrows():
        axes[1].annotate(label, (row.star_gas_quadrupole_misalignment_deg, row.star_gas_centroid_offset_over_aperture))
    axes[1].set(
        xlabel="star-gas quadrupole misalignment (deg, 60 arcsec)",
        ylabel="star-gas centroid offset / aperture",
        title="Two interpretable joint descriptors",
    )
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    first = top.iloc[0]
    (output / protocol["outputs"]["summary"]).write_text(
        f"""# P0564 morphology-sign audit

Top post-hoc separator: `{first.component}:{first.descriptor}` at
{float(first.aperture_arcsec):g} arcsec, signed descriptive separation
{float(first.signed_separation_z):+.3f}. This nominates a measurable candidate
gate but does not validate one. No formula is promoted.
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["primary"]), indent=2), flush=True)
    print(top.head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
