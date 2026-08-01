#!/usr/bin/env python3
"""Fit and audit Gaia astrometry for every sealed P0633 optical map."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.astrometric_registration import read_vizier_gaia_tsv, solve_foreground_star_wcs

DEFAULT_CONFIG = ROOT / "configs" / "p0638_gaia_astrometric_registration.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0638_gaia_astrometric_registration"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_header(wcs: WCS) -> dict[str, str | float | int | bool]:
    header = wcs.to_header(relax=True)
    return {card.keyword: card.value for card in header.cards if card.keyword}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    maps = json.loads((ROOT / config["parent_maps"]).read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        ROOT / "results" / "p0637_little_things_photometric_metadata" / "photometric_inputs.csv"
    ).set_index("galaxy")
    raw_gaia = ROOT / config["raw_directory"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "wcs").mkdir(exist_ok=True)
    rows = []
    residual_sets = []
    failures = []
    acceptance = config["acceptance"]
    for target in maps["targets"]:
        galaxy = target["id"]
        metadata_row = metadata.loc[galaxy]
        v_path = (
            ROOT
            / maps["raw_directory"]
            / galaxy
            / f"{target['optical_prefix']}v.fits"
        )
        image = np.squeeze(fits.getdata(v_path)).astype(float)
        image_header = fits.getheader(v_path)
        gaia_path = raw_gaia / f"{galaxy}_gaia_dr3.tsv"
        gaia = read_vizier_gaia_tsv(gaia_path)
        gaia_sky = SkyCoord(
            gaia["RA_ICRS"].to_numpy() * u.deg,
            gaia["DE_ICRS"].to_numpy() * u.deg,
        )
        center = SkyCoord(
            str(metadata_row["photometric_center_ra_j2000"]),
            str(metadata_row["photometric_center_dec_j2000"]),
            unit=(u.hourangle, u.deg),
        )
        fit = solve_foreground_star_wcs(
            image,
            catalog_center=center,
            catalog_pixel_scale_arcsec=float(metadata_row["optical_pixel_scale_arcsec"]),
            gaia_sky=gaia_sky,
            settings=config["algorithm"],
        )
        diagnostics = dict(fit.diagnostics)
        scale_error = abs(float(diagnostics["similarity_scale"]) - 1.0)
        existing_center_delta = np.nan
        if str(image_header.get("CTYPE1", "")).startswith("RA"):
            archived = WCS(image_header).celestial
            old_x, old_y = archived.world_to_pixel(center)
            new_x, new_y = fit.wcs.world_to_pixel(center)
            existing_center_delta = float(np.hypot(old_x - new_x, old_y - new_y))
        p95_arcsec = float(diagnostics["p95_residual_pixel"]) * float(
            metadata_row["optical_pixel_scale_arcsec"]
        )
        gates = {
            "minimum_inliers": int(diagnostics["gaia_inliers"])
            >= int(acceptance["minimum_gaia_inliers_per_image"]),
            "median_residual": float(diagnostics["median_residual_pixel"])
            <= float(acceptance["maximum_median_residual_pixel"]),
            "p95_residual_pixel": float(diagnostics["p95_residual_pixel"])
            <= float(acceptance["maximum_p95_residual_pixel"]),
            "p95_residual_arcsec": p95_arcsec
            <= float(acceptance["maximum_p95_residual_arcsec"]),
            "scale": scale_error <= float(acceptance["maximum_fractional_scale_error"]),
            "existing_wcs_center": bool(
                np.isnan(existing_center_delta)
                or existing_center_delta
                <= float(
                    acceptance["maximum_catalog_center_disagreement_for_existing_wcs_pixel"]
                )
            ),
        }
        if not all(gates.values()):
            failures.append({"galaxy": galaxy, "gates": gates})
        wcs_path = output / "wcs" / f"{galaxy}.json"
        wcs_path.write_text(
            json.dumps(
                {
                    "galaxy": galaxy,
                    "wcs_header": json_header(fit.wcs),
                    "diagnostics": diagnostics,
                    "gates": gates,
                    "sealed_target_observables_opened": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "galaxy": galaxy,
                **diagnostics,
                "p95_residual_arcsec": p95_arcsec,
                "fractional_scale_error": scale_error,
                "existing_wcs_center_disagreement_pixel": existing_center_delta,
                "all_gates_pass": all(gates.values()),
                "gaia_sha256": sha256(gaia_path),
                "v_image_sha256": sha256(v_path),
                "wcs_relative_path": wcs_path.relative_to(ROOT).as_posix(),
            }
        )
        residual_sets.append((galaxy, fit.residual_pixel))
        print(
            f"{galaxy}: {diagnostics['gaia_inliers']} stars, "
            f"median={diagnostics['median_residual_pixel']:.3f} px, "
            f"p95={diagnostics['p95_residual_pixel']:.3f} px"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "astrometry_audit.csv", index=False)
    figure, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True)
    for axis, (galaxy, residual) in zip(axes.ravel(), residual_sets, strict=False):
        axis.hist(residual, bins=np.linspace(0.0, 4.0, 33), color="#285f8f")
        axis.axvline(3.0, color="#b22222", linestyle="--", linewidth=1)
        axis.set_title(galaxy)
    for axis in axes.ravel()[len(residual_sets) :]:
        axis.axis("off")
    figure.supxlabel("Gaia-to-image residual (pixel)")
    figure.supylabel("Matched stars")
    figure.suptitle("P0638 foreground-star astrometric registration")
    figure.tight_layout()
    figure.savefig(output / "astrometric_residuals.png", dpi=180)
    plt.close(figure)
    report = {
        "status": "pass" if not failures and len(frame) == 13 else "fail",
        "protocol_version": config["protocol_version"],
        "targets": len(frame),
        "all_gates_pass": bool(not failures and frame["all_gates_pass"].all()),
        "minimum_inliers": int(frame["gaia_inliers"].min()),
        "maximum_median_residual_pixel": float(frame["median_residual_pixel"].max()),
        "maximum_p95_residual_pixel": float(frame["p95_residual_pixel"].max()),
        "maximum_p95_residual_arcsec": float(frame["p95_residual_arcsec"].max()),
        "maximum_fractional_scale_error": float(frame["fractional_scale_error"].max()),
        "maximum_existing_wcs_center_disagreement_pixel": float(
            frame["existing_wcs_center_disagreement_pixel"].max()
        ),
        "failures": failures,
        "config_sha256": sha256(config_path),
        "sealed_target_observables_opened": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SUMMARY.md").write_text(
        f"""# P0638 Gaia astrometric registration

- Status: **{report['status'].upper()}**
- Images passing every gate: {int(frame['all_gates_pass'].sum())} / 13
- Minimum Gaia inliers: {report['minimum_inliers']}
- Worst median residual: {report['maximum_median_residual_pixel']:.3f} pixel
- Worst 95th-percentile residual: {report['maximum_p95_residual_pixel']:.3f} pixel / {report['maximum_p95_residual_arcsec']:.3f} arcsec
- Largest fractional scale correction: {report['maximum_fractional_scale_error']:.4f}
- Largest catalog-center difference on the four archived-WCS images: {report['maximum_existing_wcs_center_disagreement_pixel']:.3f} pixel
- Sealed target observables opened: `{str(report['sealed_target_observables_opened']).lower()}`

Foreground Gaia stars, rather than galaxy morphology or kinematics, now place
every V-band image on the same celestial frame as its H I map.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
