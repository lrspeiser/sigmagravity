from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.unified import (  # noqa: E402
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_matrix(config: dict, actual_columns: dict[str, set[str]]) -> pd.DataFrame:
    compact = pd.DataFrame(config["records"])
    key = ["dataset", "source_variant", "scored_column"]
    if compact.duplicated(key).any():
        raise ValueError("duplicate provenance matrix keys")
    unknown = set(compact["lineage_id"]) - set(config["lineages"])
    if unknown:
        raise ValueError(f"unknown lineage ids: {sorted(unknown)}")
    lineage = pd.json_normalize(compact["lineage_id"].map(config["lineages"]))
    matrix = pd.concat([compact.reset_index(drop=True), lineage], axis=1)
    required_fields = {
        "raw_observable",
        "local_file",
        "publication",
        "transformation",
        "metric_or_dynamics_assumptions",
        "covariance_status",
        "evidence_level",
        "alternative_theory_forward_modeling",
    }
    missing_fields = required_fields - set(matrix.columns)
    if missing_fields:
        raise ValueError(f"expanded matrix fields missing: {sorted(missing_fields)}")
    for dataset, required in config["required_score_columns"].items():
        configured = set(matrix.loc[matrix["dataset"] == dataset, "scored_column"])
        required_set = set(required)
        if configured != required_set:
            raise ValueError(
                f"{dataset} matrix mismatch: missing={sorted(required_set-configured)}, "
                f"extra={sorted(configured-required_set)}"
            )
        absent_from_data = required_set - actual_columns[dataset]
        if absent_from_data:
            raise ValueError(f"{dataset} data missing scored columns: {sorted(absent_from_data)}")
    return matrix.sort_values(key, kind="stable").reset_index(drop=True)


def _instance_provenance(
    *,
    config: dict,
    matrix: pd.DataFrame,
    sparc: pd.DataFrame,
    clash: pd.DataFrame,
    bcg: pd.DataFrame,
    sparc_dir: Path,
    clash_path: Path,
    bcg_path: Path,
) -> pd.DataFrame:
    """Expand column lineage to one row for every scalar that enters a score."""
    frames = {
        "SPARC": sparc.copy(),
        "CLASH": clash.copy(),
        "BCG": bcg.rename(columns={"plateifu": "system"}).copy(),
    }
    for frame in frames.values():
        frame["score_input_row_index_zero_based"] = range(len(frame))
        frame["system_point_index_zero_based"] = frame.groupby("system", sort=False).cumcount()
    file_hashes: dict[Path, str] = {}

    def exact_input(dataset: str, system: str) -> Path:
        if dataset == "SPARC":
            path = sparc_dir / "rotmod" / f"{system}_rotmod.dat"
        elif dataset == "CLASH":
            path = clash_path
        else:
            path = bcg_path
        if not path.is_file():
            raise FileNotFoundError(f"missing exact score input for {dataset}/{system}: {path}")
        return path

    rows: list[pd.DataFrame] = []
    lineage_columns = [
        "raw_observable",
        "local_file",
        "publication",
        "transformation",
        "metric_or_dynamics_assumptions",
        "covariance_status",
        "evidence_level",
        "alternative_theory_forward_modeling",
    ]
    for record in matrix.itertuples(index=False):
        frame = frames[record.dataset]
        if record.dataset == "BCG":
            frame = frame.loc[frame["measurement_source"] == record.source_variant]
        selected = frame[
            [
                "system",
                "score_input_row_index_zero_based",
                "system_point_index_zero_based",
                record.scored_column,
            ]
        ].copy()
        selected = selected.rename(columns={record.scored_column: "scored_value"})
        selected.insert(0, "lineage_id", record.lineage_id)
        selected.insert(0, "score_role", record.score_role)
        selected.insert(0, "scored_column", record.scored_column)
        selected.insert(0, "source_variant", record.source_variant)
        selected.insert(0, "dataset", record.dataset)
        selected["score_unit"] = config["score_units"][record.dataset][record.scored_column]
        selected["score_input_file"] = [
            _display_path(exact_input(record.dataset, str(system)))
            for system in selected["system"]
        ]
        hashes = []
        for path_text in selected["score_input_file"]:
            path = ROOT / path_text
            if path not in file_hashes:
                file_hashes[path] = _sha256(path)
            hashes.append(file_hashes[path])
        selected["score_input_sha256"] = hashes
        selected["source_row_locator"] = [
            f"{system}:point:{point}:loaded_row:{row}"
            for system, point, row in zip(
                selected["system"],
                selected["system_point_index_zero_based"],
                selected["score_input_row_index_zero_based"],
            )
        ]
        selected["full_covariance_artifact_ingested_for_this_score"] = False
        for column in lineage_columns:
            selected[column] = getattr(record, column)
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    key = [
        "dataset",
        "source_variant",
        "system",
        "system_point_index_zero_based",
        "scored_column",
    ]
    if result.duplicated(key).any():
        raise ValueError("duplicate scalar provenance keys")
    if result["scored_value"].isna().any():
        raise ValueError("a scored scalar has no value")
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def _system_coverage(clash: pd.DataFrame, bcg: pd.DataFrame, gate: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for system, group in clash.groupby("system", sort=True):
        summary_points = len(group)
        rows.append(
            {
                "sample": "CLASH_Tian2020",
                "system": str(system),
                "measurement_source": "NFW_deprojected_GR_lensing_summary",
                "dynamics_summary_points": 0,
                "lensing_summary_points": summary_points,
                "raw_or_likelihood_dynamics_points": 0,
                "raw_or_likelihood_lensing_points": 0,
                "baryonic_summary_points": summary_points,
                "forward_model_baryonic_profile_points": 0,
                "full_covariance_ingested": False,
                "same_object_pilot_eligible": False,
                "shortfall_reason": (
                    "no same-object dynamics; lensing values are GR+NFW deprojections; "
                    "source radial covariance/likelihood is not ingested"
                ),
            }
        )
    for row in bcg.itertuples(index=False):
        direct = str(row.measurement_source) == "Tian2024_direct"
        rows.append(
            {
                "sample": "SPIDERS_MaNGA_BCG",
                "system": str(row.plateifu),
                "measurement_source": str(row.measurement_source),
                "dynamics_summary_points": 1,
                "lensing_summary_points": 0,
                "raw_or_likelihood_dynamics_points": 0,
                "raw_or_likelihood_lensing_points": 0,
                "baryonic_summary_points": 1,
                "forward_model_baryonic_profile_points": 0,
                "full_covariance_ingested": False,
                "same_object_pilot_eligible": False,
                "shortfall_reason": (
                    "one Jeans/Abel summary point and no same-object lensing or radial likelihood"
                    if direct
                    else "one calibrated JAM/NSA proxy point and no same-object lensing or radial likelihood"
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    computed = (
        (coverage["raw_or_likelihood_dynamics_points"] >= gate["minimum_dynamics_radial_points"])
        & (coverage["raw_or_likelihood_lensing_points"] >= gate["minimum_lensing_radial_points"])
        & (
            coverage["forward_model_baryonic_profile_points"]
            >= gate["minimum_forward_model_baryonic_radial_points"]
        )
        & coverage["full_covariance_ingested"]
    )
    if not computed.equals(coverage["same_object_pilot_eligible"]):
        raise AssertionError("stored eligibility disagrees with the frozen gate")
    return coverage.sort_values(["sample", "system"], kind="stable").reset_index(drop=True)


def build_audit(
    *,
    config_path: Path,
    sparc_dir: Path,
    clash_path: Path,
    bcg_path: Path,
    matrix_output: Path,
    instance_output: Path,
    coverage_output: Path,
    report_output: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sparc = load_sparc_acceleration_frame(sparc_dir)
    clash = load_clash_acceleration_frame(clash_path)
    bcg = pd.read_csv(bcg_path)
    matrix = _validate_matrix(
        config,
        {"SPARC": set(sparc.columns), "CLASH": set(clash.columns), "BCG": set(bcg.columns)},
    )
    instances = _instance_provenance(
        config=config,
        matrix=matrix,
        sparc=sparc,
        clash=clash,
        bcg=bcg,
        sparc_dir=sparc_dir,
        clash_path=clash_path,
        bcg_path=bcg_path,
    )
    coverage = _system_coverage(clash, bcg, config["pilot_gate"])

    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    instance_output.parent.mkdir(parents=True, exist_ok=True)
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_output, index=False)
    instances.to_csv(instance_output, index=False)
    coverage.to_csv(coverage_output, index=False)

    eligible = int(coverage["same_object_pilot_eligible"].sum())
    direct = int((bcg["measurement_source"] == "Tian2024_direct").sum())
    proxy = int((bcg["measurement_source"] == "DynPop_NSA_calibrated_proxy").sum())
    report = {
        "audit_version": config["audit_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_rule": config["freeze_rule"],
        "inputs": {
            "config": {"path": _display_path(config_path), "sha256": _sha256(config_path)},
            "clash": {"path": _display_path(clash_path), "sha256": _sha256(clash_path)},
            "bcg": {"path": _display_path(bcg_path), "sha256": _sha256(bcg_path)},
        },
        "provenance_matrix": {
            "rows": len(matrix),
            "rows_by_dataset": {
                str(name): int(count) for name, count in matrix.groupby("dataset").size().items()
            },
            "all_required_scored_columns_covered": True,
            "output": _display_path(matrix_output),
            "bytes": matrix_output.stat().st_size,
            "sha256": _sha256(matrix_output),
        },
        "instance_provenance": {
            "rows": len(instances),
            "rows_by_dataset": {
                str(name): int(count)
                for name, count in instances.groupby("dataset").size().items()
            },
            "systems_by_dataset": {
                str(name): int(count)
                for name, count in instances.groupby("dataset")["system"].nunique().items()
            },
            "unique_score_input_files": int(instances["score_input_file"].nunique()),
            "all_values_finite": bool(
                np.isfinite(pd.to_numeric(instances["scored_value"]).to_numpy(dtype=float)).all()
            ),
            "all_exact_input_hashes_present": bool(
                instances["score_input_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
            ),
            "every_scalar_has_lineage_and_covariance_disposition": bool(
                instances[
                    [
                        "raw_observable",
                        "publication",
                        "transformation",
                        "metric_or_dynamics_assumptions",
                        "covariance_status",
                        "alternative_theory_forward_modeling",
                    ]
                ].notna().all().all()
            ),
            "output": _display_path(instance_output),
            "bytes": instance_output.stat().st_size,
            "sha256": _sha256(instance_output),
        },
        "clash": {
            "systems": int(clash["system"].nunique()),
            "scored_summary_points": len(clash),
            "summary_points_per_system_min": int(clash.groupby("system").size().min()),
            "summary_points_per_system_max": int(clash.groupby("system").size().max()),
            "raw_or_likelihood_level_systems_ingested": 0,
            "alternative_metric_forward_model_ready_systems": 0,
            "classification": "Tian2020 accelerations are NFW-deprojected GR lensing reconstructions",
        },
        "bcg": {
            "frozen_systems": len(bcg),
            "direct_single_radius_Jeans_summaries": direct,
            "calibrated_single_radius_proxies": proxy,
            "resolved_dynamics_likelihood_systems_ingested": 0,
            "resolved_lensing_likelihood_systems_ingested": 0,
        },
        "same_object_pilot_gate": {
            **config["pilot_gate"],
            "eligible_systems": eligible,
            "passes": eligible >= config["pilot_gate"]["minimum_systems"],
        },
        "public_data_shortfall": {
            "hard_shortfall_documented": True,
            "clash": (
                "CDS contains Tian table1/fig2 summaries. MAST exposes reduced imaging, photometric "
                "catalogs and lens-model products, but the per-cluster Umetsu surface-density "
                "likelihood/covariance was not located in those public endpoints or article source."
            ),
            "bcg": (
                "The frozen bridge has no same-object lensing or ingested radial stellar-kinematic "
                "likelihood; 23/34 dynamics targets are calibrated proxies."
            ),
        },
        "stage_decision": {
            "R0_provenance_matrix": "complete_for_every_current_scored_scalar_and_column_lineage",
            "R0_raw_likelihood_acquisition": "fails_current_data_requirement",
            "R1_sample_freeze": "not_authorized",
            "R2_two_potential_reconstruction": "not_authorized",
            "new_force_law": "prohibited_until_R0_R2_pass",
            "next_action": (
                "Obtain Umetsu per-cluster likelihood/covariance and same-object radial kinematics, "
                "or freeze a public replacement sample satisfying the 10-system gate."
            ),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit observable provenance and R1 coverage.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "r0_observable_audit.json")
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--clash", type=Path, default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat"
    )
    parser.add_argument(
        "--bcg", type=Path, default=ROOT / "data" / "derived" / "measured_host_profile_sample.csv"
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r0_observable_provenance.csv",
    )
    parser.add_argument(
        "--instance-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r0_scored_observable_instance_provenance.csv",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r0_same_object_coverage.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "r0_observable_audit" / "report.json",
    )
    args = parser.parse_args()
    report = build_audit(
        config_path=args.config,
        sparc_dir=args.sparc,
        clash_path=args.clash,
        bcg_path=args.bcg,
        matrix_output=args.matrix_output,
        instance_output=args.instance_output,
        coverage_output=args.coverage_output,
        report_output=args.report_output,
    )
    print(json.dumps(report["stage_decision"], indent=2))


if __name__ == "__main__":
    main()
