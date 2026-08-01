from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.sparc_morphology import build_sparc_morphology_catalog, parse_sparc_profile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    labels = ["minimum", "p05", "median", "p95", "maximum"]
    return dict(zip(labels, np.quantile(finite, [0.0, 0.05, 0.5, 0.95, 1.0]), strict=True))


def assign_folds(names: pd.Series, folds: int) -> np.ndarray:
    order = sorted(
        range(len(names)),
        key=lambda index: hashlib.sha256(str(names.iloc[index]).encode()).hexdigest(),
    )
    assignment = np.empty(len(names), dtype=int)
    for rank, index in enumerate(order):
        assignment[index] = rank % folds
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbp0_morphology_protocol.json",
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=ROOT / "data" / "derived" / "nbp0_sparc_morphology.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "nbp0_sparc_morphology_audit" / "report.json",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    settings = protocol["SPARC_inputs"]
    data_directory = ROOT / settings["directory"]
    catalog = build_sparc_morphology_catalog(
        data_directory,
        disk_mass_to_light=settings["disk_mass_to_light"],
        bulge_mass_to_light=settings["bulge_mass_to_light"],
        helium_factor=settings["helium_factor"],
    )

    point_counts = []
    outer_point_counts = []
    for row in catalog.itertuples(index=False):
        profile = parse_sparc_profile(
            data_directory / "rotmod" / f"{row.galaxy}_rotmod.dat"
        )
        point_counts.append(len(profile.radius_kpc))
        outer_point_counts.append(int(np.sum(profile.radius_kpc >= 3.0 * row.disk_scale_kpc)))
    catalog["rotation_points"] = point_counts
    catalog["outer_rotation_points"] = outer_point_counts

    has_bulge = catalog["bulge_luminosity_fit_solar"] > 0.0
    disk_fit_pass = (
        np.isfinite(catalog["disk_velocity_fractional_rms"])
        & (
            catalog["disk_velocity_fractional_rms"]
            <= settings["disk_velocity_profile_fractional_rms_max"]
        )
    )
    bulge_fit_pass = (~has_bulge) | (
        np.isfinite(catalog["bulge_velocity_fractional_rms"])
        & (
            catalog["bulge_velocity_fractional_rms"]
            <= settings["bulge_velocity_profile_fractional_rms_max"]
        )
    )
    scale_min, scale_max = settings["bulge_scale_over_disk_scale_range"]
    scale_pass = (~has_bulge) | (
        np.isfinite(catalog["bulge_scale_over_disk_scale"])
        & (catalog["bulge_scale_over_disk_scale"] >= scale_min)
        & (catalog["bulge_scale_over_disk_scale"] <= scale_max)
    )
    catalog["morphology_input_pass"] = (
        (catalog["quality"] <= settings["quality_max"])
        & (catalog["inclination_deg"] >= settings["inclination_min_deg"])
        & (catalog["rotation_points"] >= settings["minimum_rotation_points"])
        & (catalog["baryonic_mass_solar"] > 0.0)
        & disk_fit_pass
        & bulge_fit_pass
        & scale_pass
    )
    catalog["fold"] = -1
    selected = catalog.loc[catalog["morphology_input_pass"]].copy()
    selected_folds = assign_folds(
        selected["galaxy"], protocol["empirical_morphology_test"]["whole_galaxy_folds"]
    )
    catalog.loc[selected.index, "fold"] = selected_folds

    bulge_selected = selected.loc[selected["bulge_luminosity_fit_solar"] > 0.0]
    report = {
        "report_version": "NBP0-M1-SPARC-morphology-audit-0.1",
        "status": "completed residual-blind SPARC disk/bulge input audit",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "observed_velocity_inspected": False,
        "all_systems": int(len(catalog)),
        "selected_systems": int(catalog["morphology_input_pass"].sum()),
        "selected_with_bulge": int(
            (
                catalog["morphology_input_pass"]
                & (catalog["bulge_luminosity_fit_solar"] > 0.0)
            ).sum()
        ),
        "fit_pass_counts": {
            "disk": int(disk_fit_pass.sum()),
            "bulge_or_no_bulge": int(bulge_fit_pass.sum()),
            "bulge_scale_or_no_bulge": int(scale_pass.sum()),
        },
        "selected_quantiles": {
            "baryonic_bulge_fraction": quantiles(
                selected["baryonic_bulge_fraction"].to_numpy(dtype=float)
            ),
            "stellar_bulge_fraction": quantiles(
                selected["stellar_bulge_fraction"].to_numpy(dtype=float)
            ),
            "gas_fraction": quantiles(selected["gas_fraction"].to_numpy(dtype=float)),
            "disk_velocity_fractional_rms": quantiles(
                selected["disk_velocity_fractional_rms"].to_numpy(dtype=float)
            ),
            "bulge_velocity_fractional_rms": quantiles(
                bulge_selected["bulge_velocity_fractional_rms"].to_numpy(dtype=float)
            ),
            "bulge_scale_over_disk_scale": quantiles(
                bulge_selected["bulge_scale_over_disk_scale"].to_numpy(dtype=float)
            ),
            "outer_rotation_points": quantiles(
                selected["outer_rotation_points"].to_numpy(dtype=float)
            ),
        },
        "fold_counts": {
            str(key): int(value)
            for key, value in catalog.loc[catalog["morphology_input_pass"], "fold"]
            .value_counts()
            .sort_index()
            .items()
        },
        "limitations": [
            "SPARC supplies gas rotation contributions and total HI mass but not the gas surface-density profile in this local snapshot.",
            "The exponential-disk and Hernquist fits are residual-blind geometry approximations to the published baryonic component fields.",
            "Vertical disk and gas thicknesses are not measured and remain explicit synthetic sensitivity dimensions.",
        ],
    }
    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.catalog_output, index=False)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
