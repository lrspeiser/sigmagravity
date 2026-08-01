#!/usr/bin/env python3
"""Audit the public Caminha+2019 observables and model-nuisance packages."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/r1_clash_caminha2019"
PROTOCOL_PATH = ROOT / "configs/r1_clash_caminha2019_ingest_protocol.json"
PROVENANCE_PATH = RAW / "provenance.json"
TABLE_PATH = RAW / "tablea2_multiple_images.dat"
IMAGE_OUTPUT = ROOT / "data/derived/r1_clash_caminha2019_image_observables.csv"
SYSTEM_OUTPUT = ROOT / "data/derived/r1_clash_caminha2019_system_summary.csv"
COVARIANCE_OUTPUT = ROOT / "data/derived/r1_clash_caminha2019_coordinate_covariances.npz"
REPORT_PATH = ROOT / "results/r1_clash_caminha2019/report.json"


MODELS = [
    ("MACS J0329.7-0211", "MACS J0329", "MACSJ0329-P2-shear_mcmc", "obs_arcs_v3.dat", "members_lenstool_v2.dat", True, "MACS0329"),
    ("MACS J0429.6-0253", "MACS J0429", "MACSJ0429-P1_mcmc", "obs_arcs_v1.dat", "members_lenstool_v1.dat", True, "MACS0429"),
    ("MACS J1115.9+0129", "MACS J1115", "MACSJ1115-P1_mcmc", "obs_arcs_v1.dat", "members_lenstool_v2.dat", True, "MACS1115"),
    ("MACS J1311.0-0310", "MACS 1311", "MACSJ1311-P1_mcmc", "obs_arcs_v3.dat", "members_lenstool_v1.dat", False, "MACS1311"),
    ("MACS J1931.8-2635", "MACS J1931", "MACSJ1931-P2_circular_mcmc", "obs_arcs_v3.dat", "members_lenstool_v2.dat", True, "MACS1931"),
    ("MACS J2129.4-0741", "MACS J2129", "MACSJ2129-P2_mcmc", "obs_arcs_v02.dat", "members_lenstool_v2.dat", False, "MACS2129"),
    ("RX J1347.5-1145", "RX J1347", "RXJ1347-P2-shear_mcmc", "obs_arcs_v4.dat", "members_lenstool_v2.dat", True, "RXJ1347"),
    ("RX J2129.7+0005", "RX J2129", "RXJ2129-P1_mcmc", "obs_arcs_v3.dat", "members_lenstool_v2.dat", True, "RXJ2129"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def optional_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == "---":
        return None
    return float(value)


def family_from_id(image_id: str) -> int:
    match = re.search(r"-?([0-9]+)[a-z]$", image_id)
    if match is None:
        raise ValueError(f"Cannot parse family from {image_id!r}")
    return int(match.group(1))


def parse_tablea2() -> pd.DataFrame:
    records = []
    for line in TABLE_PATH.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        cluster = line[0:10].strip()
        image_id = line[11:20].strip()
        muse = optional_float(line[50:56])
        previous_1 = optional_float(line[58:64])
        reference_1 = line[64:67].strip()
        previous_2 = optional_float(line[68:73])
        reference_2 = line[75:76].strip()
        redshift = None
        kind = None
        source = None
        if muse is not None:
            redshift = muse
            kind = "MUSE_spectroscopic"
            source = "Caminha2019_MUSE"
        elif previous_1 is not None and reference_1 in {"a", "b", "c", "d"}:
            redshift = previous_1
            kind = "prior_spectroscopic"
            source = f"Caminha2019_tableA2_reference_{reference_1}"
        elif previous_2 is not None and reference_2 in {"f", "g", "h"}:
            redshift = previous_2
            kind = "prior_spectroscopic"
            source = f"Caminha2019_tableA2_reference_{reference_2}"
        records.append(
            {
                "table_cluster": cluster,
                "table_image_id": image_id,
                "family_id": family_from_id(image_id),
                "ra_deg_tablea2": float(line[21:33]),
                "dec_deg_tablea2": float(line[34:49]),
                "row_spectroscopic_redshift": redshift,
                "row_redshift_kind": kind,
                "row_redshift_source": source,
            }
        )
    return pd.DataFrame(records)


def spectroscopic_families(table: pd.DataFrame) -> dict[tuple[str, int], dict]:
    output = {}
    for (cluster, family), group in table.groupby(["table_cluster", "family_id"]):
        measured = group.dropna(subset=["row_spectroscopic_redshift"])
        if measured.empty:
            continue
        rounded = [round(float(value), 4) for value in measured["row_spectroscopic_redshift"]]
        redshift = Counter(rounded).most_common(1)[0][0]
        matched = measured.loc[
            np.isclose(measured["row_spectroscopic_redshift"].astype(float), redshift, atol=0.01)
        ]
        output[(cluster, int(family))] = {
            "spectroscopic_redshift": redshift,
            "spectroscopic_redshift_kind": "+".join(sorted(set(matched["row_redshift_kind"]))),
            "spectroscopic_redshift_source": "+".join(sorted(set(matched["row_redshift_source"]))),
        }
    return output


def parse_arc_catalog(path: Path) -> pd.DataFrame:
    records = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise RuntimeError(f"Unexpected observed-arc row in {path}: {line!r}")
        records.append(
            {
                "image_id": fields[0],
                "family_id": family_from_id(fields[0]),
                "ra_deg": float(fields[1]),
                "dec_deg": float(fields[2]),
                "position_sigma_axis_1_arcsec": float(fields[3]),
                "position_sigma_axis_2_arcsec": float(fields[4]),
                "position_error_angle_deg": float(fields[5]),
                "model_catalog_family_redshift": float(fields[6]),
                "lenstool_type_code": int(fields[7]),
            }
        )
    frame = pd.DataFrame(records)
    if frame["image_id"].duplicated().any() or frame[["ra_deg", "dec_deg"]].duplicated().any():
        raise RuntimeError(f"Duplicate image row in {path}")
    return frame


def inspect_chain(path: Path) -> dict:
    parameter_headers = []
    sample_rows = 0
    column_count = None
    finite_log_likelihood = True
    with path.open("r", encoding="ascii", errors="strict") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped not in {"#Nsample", "#ln(Lhood)"}:
                    parameter_headers.append(stripped[1:])
                continue
            fields = stripped.split()
            if column_count is None:
                column_count = len(fields)
            elif len(fields) != column_count:
                raise RuntimeError(f"Inconsistent chain width in {path}")
            if len(fields) < 2 or not math.isfinite(float(fields[1])):
                finite_log_likelihood = False
            sample_rows += 1
    return {
        "chain_samples": sample_rows,
        "chain_columns": int(column_count or 0),
        "chain_parameter_headers": len(parameter_headers),
        "chain_finite_log_likelihood": finite_log_likelihood,
        "chain_schema_consistent": bool(column_count and len(parameter_headers) == column_count - 2),
    }


def covariance_block(sigma_1: float, sigma_2: float, angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return rotation @ np.diag([sigma_1**2, sigma_2**2]) @ rotation.T


def build_audit() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    for record in provenance["records"]:
        path = ROOT / record["local_path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Provenance mismatch for {path}")

    table = parse_tablea2()
    family_redshifts = spectroscopic_families(table)
    all_images = []
    summaries = []
    covariance_payload = {}
    for system, table_cluster, folder, arcs_name, members_name, tian_target, key in MODELS:
        package = RAW / folder
        required = [package / arcs_name, package / "lenstool_in.par", package / members_name, package / "bayes.dat"]
        if not all(path.exists() and path.stat().st_size > 0 for path in required):
            raise RuntimeError(f"Incomplete public model package for {system}")
        arcs = parse_arc_catalog(package / arcs_name)
        arcs.insert(0, "system", system)
        arcs.insert(1, "tian2020_target", tian_target)
        arcs["table_cluster"] = table_cluster
        arcs["observed_arc_catalog"] = str((package / arcs_name).relative_to(ROOT)).replace("\\", "/")
        arcs["model_chain"] = str((package / "bayes.dat").relative_to(ROOT)).replace("\\", "/")
        for field in ["spectroscopic_redshift", "spectroscopic_redshift_kind", "spectroscopic_redshift_source"]:
            arcs[field] = arcs["family_id"].map(
                lambda family: family_redshifts.get((table_cluster, int(family)), {}).get(field)
            )
        arcs["spectroscopic_family"] = arcs["spectroscopic_redshift"].notna()
        arcs["metric_neutral_likelihood_row"] = arcs["spectroscopic_family"]
        arcs["exclusion_reason"] = np.where(
            arcs["spectroscopic_family"],
            "",
            "family has no independent spectroscopic redshift in published table A.2; model-catalog redshift is retained for audit only",
        )
        arcs["model_catalog_redshift_used_as_metric_neutral_input"] = False
        arcs["position_error_source"] = "published Lenstool observed-arc input catalog"
        arcs["position_error_cross_image_covariance_published"] = False
        arcs["model_chain_metric_dependent"] = True
        arcs["gravity_target_used"] = False

        eligible = arcs.loc[arcs["metric_neutral_likelihood_row"]].reset_index(drop=True)
        covariance = np.zeros((2 * len(eligible), 2 * len(eligible)))
        for index, row in eligible.iterrows():
            covariance[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = covariance_block(
                float(row["position_sigma_axis_1_arcsec"]),
                float(row["position_sigma_axis_2_arcsec"]),
                float(row["position_error_angle_deg"]),
            )
        covariance_payload[f"{key}_covariance_arcsec2"] = covariance
        covariance_payload[f"{key}_image_ids"] = eligible["image_id"].astype(str).to_numpy()
        chain = inspect_chain(package / "bayes.dat")
        member_rows = sum(
            1
            for line in (package / members_name).read_text(encoding="ascii").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        summaries.append(
            {
                "system": system,
                "tian2020_target": tian_target,
                "all_model_arc_rows": int(len(arcs)),
                "spectroscopic_likelihood_rows": int(len(eligible)),
                "all_families": int(arcs["family_id"].nunique()),
                "spectroscopic_families": int(eligible["family_id"].nunique()),
                "declared_position_errors": bool(
                    (arcs["position_sigma_axis_1_arcsec"] > 0).all()
                    and (arcs["position_sigma_axis_2_arcsec"] > 0).all()
                ),
                "minimum_position_sigma_arcsec": float(
                    min(arcs["position_sigma_axis_1_arcsec"].min(), arcs["position_sigma_axis_2_arcsec"].min())
                ),
                "maximum_position_sigma_arcsec": float(
                    max(arcs["position_sigma_axis_1_arcsec"].max(), arcs["position_sigma_axis_2_arcsec"].max())
                ),
                "member_catalog_rows": member_rows,
                "metric_neutral_coordinate_covariance_rank": int(np.linalg.matrix_rank(covariance)),
                "metric_neutral_coordinate_covariance_dimension": int(len(covariance)),
                "observable_catalog_acquired": bool(len(eligible) > 0),
                "rerunnable_lenstool_package_acquired": True,
                "model_chain_metric_dependent": True,
                "metric_neutral_weyl_posterior_acquired": False,
                **chain,
            }
        )
        all_images.append(arcs)

    images = pd.concat(all_images, ignore_index=True)
    summary = pd.DataFrame(summaries)
    IMAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images.to_csv(IMAGE_OUTPUT, index=False)
    summary.to_csv(SYSTEM_OUTPUT, index=False)
    np.savez_compressed(COVARIANCE_OUTPUT, **covariance_payload)

    checks = protocol["pre_registered_checks"]
    package_pass = bool(
        len(summary) == checks["exact_cluster_count"]
        and int(summary["tian2020_target"].sum()) == checks["exact_tian2020_intersection_count"]
        and len(table) == checks["published_tablea2_record_count"]
        and summary["rerunnable_lenstool_package_acquired"].all()
        and summary["chain_schema_consistent"].all()
        and summary["chain_finite_log_likelihood"].all()
    )
    observable_pass = bool(summary["observable_catalog_acquired"].all() and summary["declared_position_errors"].all())
    tian_package = summary.loc[summary["tian2020_target"], "system"].tolist()
    report = {
        "report_version": "R1-CLASH-Caminha2019-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "published_tablea2_rows": int(len(table)),
        "systems": int(len(summary)),
        "systems_with_observable_catalog": int(summary["observable_catalog_acquired"].sum()),
        "systems_with_complete_rerunnable_lenstool_package": int(summary["rerunnable_lenstool_package_acquired"].sum()),
        "systems_with_local_model_chain": int((summary["chain_samples"] > 0).sum()),
        "all_model_arc_rows": int(summary["all_model_arc_rows"].sum()),
        "spectroscopic_metric_neutral_likelihood_rows": int(summary["spectroscopic_likelihood_rows"].sum()),
        "spectroscopic_metric_neutral_families": int(summary["spectroscopic_families"].sum()),
        "total_chain_samples": int(summary["chain_samples"].sum()),
        "tian2020_intersection": {
            "count": len(tian_package),
            "systems": tian_package,
            "preexisting_local_tian_observable_catalogs": [
                "Abell 383",
                "Abell 611",
                "MS 2137",
                "MACS J0416",
                "MACS J1206",
                "RX J2248",
                "RX J2129",
                "Abell 2261"
            ],
            "new_catalogs_from_this_package": 5,
            "confirmed_local_catalog_count_after_ingest": 13,
            "target_count": 20
        },
        "gates": {
            "package_integrity_passed": package_pass,
            "observable_coordinate_likelihood_acquired_for_all_eight": observable_pass,
            "metric_neutral_weyl_posterior_acquired": False,
            "gravity_response_fit_authorized": False
        },
        "interpretation": "The image positions, independent spectroscopic family redshifts, and declared per-image errors are forward-model inputs. The Lenstool chains are preserved as model-dependent nuisance/control products and are not a metric-neutral Weyl-response posterior.",
        "authorization": {
            "count_all_eight_as_observable_catalogs_acquired": package_pass and observable_pass,
            "count_six_tian2020_systems_toward_20_catalog_target": package_pass and observable_pass,
            "reuse_lenstool_chain_as_alternative_metric_posterior": False,
            "infer_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False
        },
        "outputs": {
            "image_catalog": str(IMAGE_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "system_summary": str(SYSTEM_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "coordinate_covariances": str(COVARIANCE_OUTPUT.relative_to(ROOT)).replace("\\", "/")
        }
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
