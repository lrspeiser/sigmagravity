#!/usr/bin/env python3
"""Freeze theory-blind photometric geometry and normalization for P0633 targets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0637_little_things_photometric_metadata.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0637_little_things_photometric_metadata"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_vizier_tsv(path: Path) -> pd.DataFrame:
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) < 3 or "Name" not in lines[0].split("\t"):
        raise ValueError(f"{path.name} is not a VizieR TSV table")
    frame = pd.read_csv(io.StringIO("\n".join([lines[0], *lines[2:]])), sep="\t", dtype=str)
    frame.columns = [column.strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].fillna("").str.strip()
    frame["target_id"] = frame["Name"].str.replace(" ", "", regex=False)
    return frame


def number(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"missing or invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def inclination_from_axis_ratio(axis_ratio: float, intrinsic_axis_ratio: float) -> float:
    if not 0.0 < axis_ratio <= 1.0 or not 0.0 <= intrinsic_axis_ratio < 1.0:
        raise ValueError("axis ratios are outside their physical range")
    cosine_squared = (axis_ratio**2 - intrinsic_axis_ratio**2) / (
        1.0 - intrinsic_axis_ratio**2
    )
    cosine_squared = float(np.clip(cosine_squared, 0.0, 1.0))
    return float(np.degrees(np.arccos(np.sqrt(cosine_squared))))


def unique_row(frame: pd.DataFrame, target: str, columns: list[str]) -> pd.Series:
    rows = frame.loc[frame["target_id"] == target, columns].drop_duplicates()
    if len(rows) != 1:
        raise ValueError(f"{target}: expected one unique {columns}, found {len(rows)}")
    return rows.iloc[0]


def audit(config: dict) -> tuple[dict, pd.DataFrame]:
    raw = ROOT / config["raw_directory"]
    paths = {table: raw / f"{table}.tsv" for table in config["catalog"]["tables"]}
    tables = {name: read_vizier_tsv(path) for name, path in paths.items()}
    q0 = float(config["universal_geometry"]["intrinsic_axis_ratio_q0"])
    solar_mv = float(config["stellar_normalization"]["solar_absolute_v_magnitude"])
    nominal_ml = float(
        config["stellar_normalization"]["nominal_v_band_mass_to_light_solar"]
    )
    rows: list[dict] = []
    errors: list[str] = []
    for target in config["targets"]:
        try:
            basic = unique_row(
                tables["table1"], target, ["Dist", "E(B-V)f", "logMHI"]
            )
            optical = tables["table2"].loc[
                (tables["table2"]["target_id"] == target)
                & tables["table2"]["Filt"].str.contains("V", regex=False),
                ["Scale", "PA", "b/a", "RAJ2000", "DEJ2000"],
            ].drop_duplicates()
            if len(optical) != 1:
                raise ValueError(
                    f"expected one unique V-band geometry, found {len(optical)}"
                )
            optical = optical.iloc[0]
            photometry = tables["table3"].loc[
                tables["table3"]["target_id"] == target
            ].copy()
            photometry["radius_numeric"] = pd.to_numeric(photometry["Rad"], errors="raise")
            integrated = photometry.loc[photometry["radius_numeric"].idxmax()]
            published = unique_row(tables["table5"], target, ["Inc"])

            distance = number(basic["Dist"], "distance")
            axis_ratio = number(optical["b/a"], "axis ratio")
            inclination = inclination_from_axis_ratio(axis_ratio, q0)
            published_inclination = number(published["Inc"], "published inclination")
            inclination_delta = inclination - published_inclination
            if abs(inclination_delta) > 1.0:
                raise ValueError(
                    f"derived inclination differs from catalog audit by {inclination_delta:.3f} deg"
                )
            absolute_v = number(integrated["VMAG"], "absolute V magnitude")
            luminosity_v = 10.0 ** (-0.4 * (absolute_v - solar_mv))
            rows.append(
                {
                    "galaxy": target,
                    "distance_mpc": distance,
                    "foreground_ebv_mag": number(basic["E(B-V)f"], "foreground reddening"),
                    "catalog_log_hi_plus_helium_mass_solar": number(
                        basic["logMHI"], "catalog gas mass"
                    ),
                    "optical_pixel_scale_arcsec": number(optical["Scale"], "pixel scale"),
                    "photometric_pa_deg": number(optical["PA"], "position angle"),
                    "photometric_axis_ratio": axis_ratio,
                    "intrinsic_axis_ratio_q0": q0,
                    "derived_photometric_inclination_deg": inclination,
                    "catalog_rounding_inclination_deg": published_inclination,
                    "inclination_rounding_delta_deg": inclination_delta,
                    "photometric_center_ra_j2000": optical["RAJ2000"],
                    "photometric_center_dec_j2000": optical["DEJ2000"],
                    "integrated_aperture_radius_arcmin": number(
                        integrated["Rad"], "integrated aperture"
                    ),
                    "absolute_v_magnitude": absolute_v,
                    "dereddened_b_minus_v_mag": number(
                        integrated["(B-V)0"], "integrated B-V color"
                    ),
                    "v_band_luminosity_solar": luminosity_v,
                    "nominal_universal_v_band_mass_to_light": nominal_ml,
                    "nominal_stellar_mass_solar": nominal_ml * luminosity_v,
                }
            )
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"{target}: {exc}")
    frame = pd.DataFrame(rows)
    report = {
        "status": "ready" if not errors and len(frame) == len(config["targets"]) else "failure",
        "protocol_version": config["protocol_version"],
        "targets": len(frame),
        "expected_targets": len(config["targets"]),
        "errors": errors,
        "universal_intrinsic_axis_ratio_q0": q0,
        "nominal_universal_v_band_mass_to_light": nominal_ml,
        "maximum_inclination_rounding_error_deg": (
            float(frame["inclination_rounding_delta_deg"].abs().max())
            if not frame.empty
            else None
        ),
        "source_files": {
            name: {"relative_path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "sealed_target_observables_opened": False,
        "per_galaxy_gravity_parameters_fit": False,
    }
    return report, frame


def write_outputs(report: dict, frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "photometric_inputs.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = f"""# P0637 theory-blind photometric metadata

- Status: **{report['status'].upper()}**
- Frozen galaxies: {report['targets']} / {report['expected_targets']}
- Universal intrinsic dwarf thickness: `q0={report['universal_intrinsic_axis_ratio_q0']}`
- Nominal universal V-band mass-to-light ratio: `{report['nominal_universal_v_band_mass_to_light']}`
- Largest inclination rounding difference: `{report['maximum_inclination_rounding_error_deg']:.3f} deg`
- Sealed rotation observables opened: `{str(report['sealed_target_observables_opened']).lower()}`
- Per-galaxy gravity settings fitted: `{str(report['per_galaxy_gravity_parameters_fit']).lower()}`

These inputs were selected with predeclared catalog rules. They provide physical
scale, photometric orientation, and a universal stellar-normalization baseline
without reading any target velocity field or rotation curve.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_photometric_inputs_before_candidate_lock":
        raise RuntimeError("P0637 metadata config is not frozen")
    report, frame = audit(config)
    report["config_sha256"] = sha256(config_path)
    write_outputs(report, frame, args.output.resolve())
    print(json.dumps(report, indent=2))
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
