#!/usr/bin/env python3
"""Post-failure higher-order harmonic sensitivity on the spent v16B maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from infer_sigma_v16_spent_boundary import sample_cluster

from voidscreen.sigma_boundary_inference import (
    decompose_boundary_shear,
    harmonic_shear_basis,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit sensitivity of the spent boundary oracle to harmonic order."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v16c_harmonic_order_sensitivity.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v16c_harmonic_order_sensitivity",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "frozen before fitting any higher-order harmonic basis":
        raise RuntimeError("the v16C order protocol is not frozen")
    base_config_path = ROOT / config["base_config"]
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    sampled = [sample_cluster(cluster, base) for cluster in base["sample"]["clusters"]]
    target_axis = np.linspace(
        -float(base["map_measurement"]["target_half_width_kpc"]),
        float(base["map_measurement"]["target_half_width_kpc"]),
        int(base["map_measurement"]["target_grid_points"]),
    )
    east, north = np.meshgrid(target_axis, target_axis)
    radius = np.hypot(east, north)
    rows = []
    for dataset, _, _ in sampled:
        missing = tuple(
            target - source for target, source in zip(dataset.target, dataset.base, strict=True)
        )
        for maximum_order in config["maximum_orders"]:
            basis = harmonic_shear_basis(
                east,
                north,
                minimum_order=int(config["minimum_order"]),
                maximum_order=int(maximum_order),
                reference_radius_kpc=float(
                    base["boundary_decomposition"]["harmonic_reference_radius_kpc"]
                ),
            )
            decomposition = decompose_boundary_shear(
                missing[0],
                missing[1],
                missing[2],
                radius,
                dataset.mask,
                basis,
                taper_start_kpc=float(base["map_measurement"]["internal_taper_start_kpc"]),
                taper_end_kpc=float(base["map_measurement"]["internal_taper_end_kpc"]),
                padding_factor=int(
                    base["boundary_decomposition"]["primary_fourier_padding_factor"]
                ),
            )
            rows.append(
                {
                    "cluster": dataset.name,
                    "maximum_order": int(maximum_order),
                    "coefficient_count": len(decomposition.harmonic_fit.coefficients),
                    "harmonic_oracle_NRMSE": decomposition.harmonic_fit.normalized_RMSE,
                    "harmonic_oracle_power_closed": decomposition.harmonic_fit.power_closed,
                    "boundary_to_total_shear_power_ratio": decomposition.boundary_to_total_shear_power_ratio,
                }
            )

    by_cluster = {
        cluster: [row for row in rows if row["cluster"] == cluster]
        for cluster in base["sample"]["clusters"]
    }
    gates = config["decision_gates"]
    maximum_order = max(int(value) for value in config["maximum_orders"])
    maximum_rows = {
        cluster: next(row for row in values if row["maximum_order"] == maximum_order)
        for cluster, values in by_cluster.items()
    }
    both_half_power = bool(
        all(
            row["harmonic_oracle_power_closed"] >= gates["minimum_power_closed_each_cluster_at_m12"]
            for row in maximum_rows.values()
        )
    )
    plck_rows = by_cluster["PLCKG287"]
    plck_gain = float(
        next(row for row in plck_rows if row["maximum_order"] == maximum_order)[
            "harmonic_oracle_power_closed"
        ]
        - next(row for row in plck_rows if row["maximum_order"] == 6)[
            "harmonic_oracle_power_closed"
        ]
    )
    truncation_material = bool(
        plck_gain >= gates["minimum_PLCKG287_gain_from_m6_to_m12_to_call_truncation_material"]
    )
    if both_half_power and truncation_material:
        decision = "higher harmonic order rescues the descriptive boundary oracle, but not measured-baryon transfer; no boundary theory advances"
    else:
        decision = "the static harmonic-boundary failure remains robust through m=12; advance to a baryon-unique dynamical-state question"

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for cluster, values in by_cluster.items():
        axis.plot(
            [row["maximum_order"] for row in values],
            [row["harmonic_oracle_power_closed"] for row in values],
            marker="o",
            label=cluster,
        )
    axis.axhline(
        gates["minimum_power_closed_each_cluster_at_m12"],
        color="black",
        linestyle="--",
        label="half-power gate",
    )
    axis.set(
        xlabel="maximum harmonic potential order",
        ylabel="boundary-shear power closed",
        ylim=(0.0, 1.0),
        title="Spent harmonic-order sensitivity",
    )
    axis.legend()
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "harmonic_order_sensitivity.png", dpi=180)
    plt.close(figure)
    report = {
        "status": "completed Sigma v16C harmonic-order sensitivity",
        "protocol_version": config["protocol_version"],
        "sample_is_spent": True,
        "observational_validation_claim": False,
        "input_hashes": {
            "config": sha256(args.config),
            "base_config": sha256(base_config_path),
        },
        "rows": rows,
        "maximum_order_rows": maximum_rows,
        "PLCKG287_power_gain_m6_to_m12": plck_gain,
        "gate_results": {
            "both_clusters_half_power_at_m12": both_half_power,
            "PLCKG287_material_order_gain": truncation_material,
        },
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    for row in rows:
        print(
            f"{row['cluster']} m<={row['maximum_order']}: "
            f"power={row['harmonic_oracle_power_closed']:.6f}, "
            f"NRMSE={row['harmonic_oracle_NRMSE']:.6f}"
        )
    print(decision)


if __name__ == "__main__":
    main()
