#!/usr/bin/env python3
"""Construct gas-state posteriors from a terminal V19X3B regional archive."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x4_gas_state_posterior as inherited_v19x4

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_X3B_STATUS = inherited_v19x4.AUTHORIZED_X3_STATUS
FROZEN_STATE = "frozen_after_terminal_v19x3b_pass"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return inherited_v19x4.sha256(path)


def validate_frozen_runner(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != FROZEN_STATE:
        raise RuntimeError("V19X4B configuration is not frozen after V19X3B")
    implementation = config["implementation"]
    runner = ROOT / implementation["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19X4B configuration names another runner")
    for name in ("runner", "inherited_v19x4_runner", "posterior_module"):
        path = ROOT / implementation[name]
        if not path.is_file() or sha256(path) != implementation[f"{name}_sha256"]:
            raise RuntimeError(f"V19X4B implementation changed: {name}")


def validate_preconditions(
    config: dict[str, Any], x3b_config_path: Path, x3b_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            path = ROOT / value
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"V19X4B parent changed: {value}")
    validate_frozen_runner(config)
    if not x3b_config_path.is_file() or not x3b_report_path.is_file():
        raise RuntimeError("V19X4B requires terminal V19X3B config and report")
    x3b_config = load_json(x3b_config_path)
    x3b_report = load_json(x3b_report_path)
    if x3b_report.get("status") != AUTHORIZED_X3B_STATUS:
        raise RuntimeError("V19X3B did not authorize gas-source construction")
    if x3b_report.get("source_map_construction_authorized") is not True:
        raise RuntimeError("V19X3B source-map authorization is false")
    if not x3b_report.get("gates") or not all(x3b_report["gates"].values()):
        raise RuntimeError("V19X3B contains a failed production gate")
    if x3b_report.get("config_sha256") != sha256(x3b_config_path):
        raise RuntimeError("V19X3B report names another frozen config")
    expected_runner = config["parents"]["v19x3b_runner_sha256"]
    if x3b_report.get("runner_sha256") != expected_runner:
        raise RuntimeError("V19X3B report names another runner")
    if x3b_config.get("implementation", {}).get("runner_sha256") != expected_runner:
        raise RuntimeError("V19X3B frozen config names another runner")
    if x3b_report.get("lensing_or_halo_payload_opened") is not False:
        raise RuntimeError("V19X3B opened a prohibited target")
    v19m_report = load_json(ROOT / config["parents"]["v19m_region_report"])
    source_report = load_json(ROOT / config["parents"]["source_map_report"])
    return x3b_report, v19m_report, source_report


def execute(
    config: dict[str, Any],
    x3b_config_path: Path,
    x3b_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    x3b_report, v19m_report, source_report = validate_preconditions(
        config, x3b_config_path, x3b_report_path
    )
    x3_by_cluster: dict[str, list[dict[str, Any]]] = {
        cluster: [] for cluster in config["geometry"]["clusters"]
    }
    for row in x3b_report["regions"]:
        x3_by_cluster[row["cluster"]].append(row)
    source_by_cluster = {row["cluster"]: row for row in source_report["clusters"]}
    correlations = [float(value) for value in config["posterior"]["rank_correlations"]]
    products: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []

    for cluster, cluster_config in config["geometry"]["clusters"].items():
        geometry = inherited_v19x4.load_valid_region_geometry(
            ROOT / cluster_config["region_statistics"]
        )
        binmap_path = inherited_v19x4.product_path(v19m_report, cluster, "binmap")
        with fits.open(binmap_path, memmap=False) as handle:
            binmap = np.asarray(handle[0].data, dtype=np.int64)
        for correlation in correlations:
            token = (
                f"rho_{correlation:+.1f}"
                .replace("+", "p")
                .replace("-", "m")
                .replace(".", "p")
            )
            arrays, regional_summary = inherited_v19x4.cluster_branch(
                config,
                cluster,
                x3_by_cluster[cluster],
                geometry,
                correlation,
            )
            regional_path = output / cluster / f"regional_posterior_{token}.npz"
            inherited_v19x4.atomic_npz(regional_path, arrays)
            maps, map_summary = inherited_v19x4.build_common_maps(
                config,
                cluster,
                arrays,
                binmap,
                source_by_cluster[cluster]["final_center"],
            )
            map_path = output / cluster / f"common_grid_summary_{token}.npz"
            inherited_v19x4.atomic_npz(map_path, maps)
            for role, path in (
                ("regional_posterior", regional_path),
                ("common_grid_summary", map_path),
            ):
                products.append(
                    {
                        "cluster": cluster,
                        "rank_correlation": correlation,
                        "role": role,
                        "relative_path": path.resolve()
                        .relative_to(ROOT.resolve())
                        .as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            branch_summaries.append(
                {**regional_summary, "common_grid": map_summary}
            )

    minimum_quality = int(
        config["future_runtime_gates"][
            "minimum_individual_quality_passes_per_cluster"
        ]
    )
    expected_draws = int(config["posterior"]["draws"])
    gates = {
        "three_registered_dependence_branches_per_cluster": len(branch_summaries)
        == 3 * len(config["geometry"]["clusters"]),
        "all_494_regions_reconstructed_in_every_branch": all(
            row["regions"]
            == int(
                config["geometry"]["clusters"][row["cluster"]][
                    "expected_valid_regions"
                ]
            )
            for row in branch_summaries
        ),
        "exact_4096_draws_per_region": all(
            row["draws_per_region"] == expected_draws == 4096
            for row in branch_summaries
        ),
        "minimum_quality_passes_per_cluster": all(
            row["individual_quality_passes"] >= minimum_quality
            for row in branch_summaries
        ),
        "every_stored_draw_finite_positive": all(
            row["all_draws_finite_positive"] for row in branch_summaries
        ),
        "common_grid_represents_every_region": all(
            row["common_grid"]["represented_region_ids"] == row["regions"]
            for row in branch_summaries
        ),
        "surface_density_smoothing_mass_conserved_to_1e_6": all(
            row["common_grid"][
                "maximum_surface_density_smoothing_mass_relative_error"
            ]
            <= 1.0e-6
            for row in branch_summaries
        ),
        "all_products_hash_bound": len(products) == 2 * len(branch_summaries),
    }
    return {
        "status": (
            "gas_state_posterior_and_common_grids_passed_source_invariant_scoring_authorized"
            if all(gates.values())
            else "gas_state_posterior_or_common_grid_gate_failed"
        ),
        "x3b_config_sha256": sha256(x3b_config_path),
        "x3b_report_sha256": sha256(x3b_report_path),
        "branch_summaries": branch_summaries,
        "products": products,
        "gates": gates,
        "source_invariant_scoring_authorized": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--x3b-config", type=Path, required=True)
    parser.add_argument("--x3b-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(
            config,
            args.x3b_config.resolve(),
            args.x3b_report.resolve(),
            output,
        )
    except Exception as exc:  # noqa: BLE001 - retain terminal admission failure
        result = {
            "status": "v19x4b_gas_state_posterior_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "gates": {"execution_completed": False},
            "source_invariant_scoring_authorized": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "lensing_or_halo_payload_opened": False,
        "source_invariant_or_action_selected": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["source_invariant_scoring_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
