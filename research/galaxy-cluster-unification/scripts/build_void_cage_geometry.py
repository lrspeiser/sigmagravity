from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.environment import GRID_SPECS, load_density_grid
from voidscreen.void_cage import cage_geometry_for_grid, kernel_label

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build residual-blind exterior-void cage geometry for SPARC galaxies."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "void_cage_test_protocol.json",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT / "data" / "derived" / "void_scores_cf4.csv",
    )
    parser.add_argument(
        "--cf4",
        type=Path,
        default=ROOT / "data" / "raw" / "cosmicflows4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "derived" / "void_cage_geometry.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "void_cage_geometry" / "report.json",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    environment = pd.read_csv(args.environment).sort_values("galaxy", kind="stable")
    points = environment[["sgx_hmpc", "sgy_hmpc", "sgz_hmpc"]].to_numpy(dtype=float)
    names = tuple(environment["galaxy"].astype(str))
    source = protocol["source_definition"]
    primary_power = float(protocol["kernels"]["power_law_primary"]["force_power_p"])
    primary_range = float(protocol["kernels"]["yukawa_primary"]["range_h100_inverse_mpc"])
    robustness_ranges = tuple(
        float(value)
        for value in protocol["kernels"]["yukawa_range_robustness_h100_inverse_mpc"]
    )
    ranges = tuple(dict.fromkeys([*robustness_ranges, primary_range]))

    frame = environment[
        [
            "galaxy",
            "ra_deg",
            "dec_deg",
            "distance_mpc",
            "sgx_hmpc",
            "sgy_hmpc",
            "sgz_hmpc",
        ]
    ].copy()
    input_grids: list[dict[str, object]] = []
    for spec in GRID_SPECS:
        print(f"grid={spec.key}", flush=True)
        grid = load_density_grid(args.cf4, spec)
        geometry = cage_geometry_for_grid(
            grid,
            points,
            names,
            grid_key=spec.key,
            box_size_hmpc=spec.box_size_hmpc,
            inner_hmpc=float(source["local_exclusion_h100_inverse_mpc"]),
            outer_hmpc=float(source["outer_shell_h100_inverse_mpc"]),
            power_law_values=(primary_power,),
            yukawa_ranges_hmpc=ranges,
        )
        frame = frame.merge(geometry, on="galaxy", how="inner", validate="one_to_one")
        path = args.cf4 / spec.filename
        input_grids.append(
            {
                "key": spec.key,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "shape": list(spec.shape),
                "box_size_hmpc": spec.box_size_hmpc,
                "voxel_size_hmpc": spec.voxel_size_hmpc,
            }
        )

    if len(frame) != len(environment):
        raise RuntimeError("Geometry merge did not preserve every environment row")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
        raise ValueError(f"Non-finite geometry columns: {bad}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.12g")
    primary_suffix = f"yukawa_l{kernel_label(primary_range)}"
    primary_columns = [
        f"{spec.key}_{primary_suffix}_kappa_unit" for spec in GRID_SPECS
    ]
    power_columns = [
        f"{spec.key}_power_p{kernel_label(primary_power)}_kappa_unit" for spec in GRID_SPECS
    ]
    summaries = {
        column: describe(frame[column])
        for column in [
            *primary_columns,
            *power_columns,
            *[f"{spec.key}_shell_dipole" for spec in GRID_SPECS],
            *[f"{spec.key}_shell_quadrupole" for spec in GRID_SPECS],
        ]
    }
    compressive = {
        spec.key: {
            "fully_compressive_fraction": float(
                frame[f"{spec.key}_{primary_suffix}_fully_compressive"].mean()
            ),
            "median_compressive_directions": float(
                frame[f"{spec.key}_{primary_suffix}_compressive_directions"].median()
            ),
            "median_anisotropy": float(
                frame[f"{spec.key}_{primary_suffix}_anisotropy"].median()
            ),
        }
        for spec in GRID_SPECS
    }
    report = {
        "status": "completed residual-blind exterior-void cage geometry",
        "report_version": "void-cage-geometry-0.1",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "version": protocol["protocol_version"],
        },
        "guardrail": "No SPARC rotation velocity, residual, or fitted gravity quantity entered this calculation.",
        "rows": len(frame),
        "source_shell_hmpc": {
            "inner": source["local_exclusion_h100_inverse_mpc"],
            "outer": source["outer_shell_h100_inverse_mpc"],
        },
        "primary_kernel": {
            "family": "Yukawa",
            "range_hmpc": primary_range,
            "score_columns": primary_columns,
        },
        "power_law_kernel": {
            "force_power": primary_power,
            "score_columns": power_columns,
        },
        "analytic_inverse_square_null": "p=2 gives zero isotropic compressive trace for every exterior source element; it is not fitted as a cage score.",
        "input_environment": {
            "path": str(args.environment.relative_to(ROOT)),
            "sha256": sha256(args.environment),
        },
        "input_grids": input_grids,
        "summaries": summaries,
        "primary_compression_geometry": compressive,
        "spearman_primary_score_correlations": frame[primary_columns]
        .corr(method="spearman")
        .to_dict(),
        "output": {
            "path": str(args.output.relative_to(ROOT)),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
