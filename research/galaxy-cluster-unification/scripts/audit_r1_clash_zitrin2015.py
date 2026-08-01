#!/usr/bin/env python3
"""Audit the complete Zitrin+2015 CLASH image table for the seven coverage gaps."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/r1_clash_zitrin2015"
PROTOCOL_PATH = ROOT / "configs/r1_clash_zitrin2015_ingest_protocol.json"
PROVENANCE_PATH = RAW / "provenance.json"
TABLE_PATH = RAW / "zitrin2015_table2_multiple_images.dat"
IMAGE_OUTPUT = ROOT / "data/derived/r1_clash_zitrin2015_image_observables.csv"
SYSTEM_OUTPUT = ROOT / "data/derived/r1_clash_zitrin2015_system_summary.csv"
CONTROL_COVARIANCE_OUTPUT = ROOT / "data/derived/r1_clash_zitrin2015_model_control_covariances.npz"
REPORT_PATH = ROOT / "results/r1_clash_zitrin2015/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def optional_float(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def parse_family(arc_id: str) -> int:
    match = re.match(r"[cp]?(\d+)", arc_id)
    if match is None:
        raise RuntimeError(f"Cannot parse family from arc identifier {arc_id!r}")
    return int(match.group(1))


def parse_table() -> pd.DataFrame:
    records = []
    for line in TABLE_PATH.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        line = line.ljust(120)
        arc_id = line[8:19].strip()
        records.append(
            {
                "catalog_id": line[0:7].strip(),
                "arc_id": arc_id,
                "family_id": parse_family(arc_id),
                "ra_deg": float(line[20:30]),
                "dec_deg": float(line[31:41]),
                "z_phot": optional_float(line[42:46]),
                "z_phot_95_lower": optional_float(line[47:51]),
                "z_phot_95_upper": optional_float(line[52:56]),
                "z_input_signed": optional_float(line[57:63]),
                "z_ltm_predicted": optional_float(line[64:68]),
                "z_ltm_95_lower": optional_float(line[69:73]),
                "z_ltm_95_upper": optional_float(line[74:78]),
                "z_ltm_underconstrained": line[79:80].strip() == "u",
                "z_nfw_predicted": optional_float(line[81:85]),
                "z_nfw_95_lower": optional_float(line[86:90]),
                "z_nfw_95_upper": optional_float(line[91:95]),
                "z_nfw_underconstrained": line[96:97].strip() == "u",
                "redshift_comment": line[98:120].strip(),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != 579:
        raise RuntimeError(f"Published table row-count mismatch: {len(frame)}")
    return frame


def inspect_readme(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "model_position_sigma_arcsec": r"sigma_pos=([0-9.]+)\"",
        "archived_sl_constraints": r"SL constraints we use is n_c=\s*([0-9]+)",
        "archived_sl_chi2": r"strong-lensing \(SL\) part chi\^2 is ([0-9.]+)",
        "archived_image_rms_arcsec": r"reproduction rms is ([0-9.]+)\"",
        "archived_free_source_redshifts": r"as well as ([0-9]+) free source redshifts",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise RuntimeError(f"Could not extract {key} from {path}")
        result[key] = float(match.group(1)) if key in {
            "model_position_sigma_arcsec", "archived_sl_chi2", "archived_image_rms_arcsec"
        } else int(match.group(1))
    return result


def add_family_redshift_audit(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["family_independent_redshift_anchor"] = False
    frame["family_independent_redshift_kind"] = ""
    frame["family_spectroscopic_redshift"] = np.nan
    for (system, family), group in frame.loc[frame["measured_position_row"]].groupby(["system", "family_id"]):
        spectroscopic = group.loc[group["z_input_signed"].fillna(0) < 0, "z_input_signed"].abs()
        if not spectroscopic.empty:
            if float(spectroscopic.max() - spectroscopic.min()) > 0.01:
                raise RuntimeError(f"Conflicting spectroscopic redshifts for {system} family {family}")
            mask = (frame["system"] == system) & (frame["family_id"] == family)
            frame.loc[mask, "family_independent_redshift_anchor"] = True
            frame.loc[mask, "family_independent_redshift_kind"] = "spectroscopic_negative_z_input_flag"
            frame.loc[mask, "family_spectroscopic_redshift"] = float(spectroscopic.median())
            continue
        photometric = group.loc[
            group["z_phot"].notna()
            & group["z_phot_95_lower"].notna()
            & group["z_phot_95_upper"].notna()
        ]
        if not photometric.empty:
            mask = (frame["system"] == system) & (frame["family_id"] == family)
            frame.loc[mask, "family_independent_redshift_anchor"] = True
            frame.loc[mask, "family_independent_redshift_kind"] = "row_level_clash_photometric_95pct_intervals"
    return frame


def build_audit() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    expected_records = 5 + 2 * protocol["pre_registered_checks"]["exact_system_count"]
    if len(provenance["records"]) != expected_records:
        raise RuntimeError("Expected the five primary/release products plus fourteen MAST audit files")
    for record in provenance["records"]:
        path = ROOT / record["local_path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Provenance mismatch for {path}")

    published = parse_table()
    systems = pd.DataFrame(protocol["systems"])
    frame = published.merge(systems, on="catalog_id", how="inner", validate="many_to_one")
    if frame["system"].nunique() != protocol["pre_registered_checks"]["exact_system_count"]:
        raise RuntimeError("One or more frozen target IDs are absent from the published table")
    if frame.duplicated(["system", "arc_id"]).any():
        raise RuntimeError("Duplicate cluster-plus-arc identifiers in target rows")
    if frame.duplicated(["system", "ra_deg", "dec_deg"]).any():
        raise RuntimeError("Duplicate coordinates within a target cluster")

    frame["candidate_flag"] = frame["arc_id"].str.contains("c", regex=False)
    frame["predicted_missing_flag"] = frame["arc_id"].str.contains("p", regex=False)
    frame["additional_ambiguity_flag"] = frame["arc_id"].str.contains("?", regex=False)
    frame["measured_position_row"] = ~(
        frame["candidate_flag"] | frame["predicted_missing_flag"] | frame["additional_ambiguity_flag"]
    )
    frame = add_family_redshift_audit(frame)
    frame["metric_neutral_observable_row"] = (
        frame["measured_position_row"] & frame["family_independent_redshift_anchor"]
    )
    frame["positive_z_input_used_as_metric_neutral_input"] = False
    frame["ltm_or_nfw_redshift_used_as_metric_neutral_input"] = False
    frame["metric_neutral_coordinate_likelihood_ready"] = False
    frame["gravity_target_used"] = False
    frame["exclusion_reason"] = ""
    frame.loc[frame["candidate_flag"], "exclusion_reason"] = "published candidate identifier c"
    frame.loc[frame["predicted_missing_flag"], "exclusion_reason"] = "published predicted/missing identifier p"
    frame.loc[frame["additional_ambiguity_flag"], "exclusion_reason"] = "published additionally ambiguous identifier ?"
    no_anchor = frame["measured_position_row"] & ~frame["family_independent_redshift_anchor"]
    frame.loc[no_anchor, "exclusion_reason"] = "no independent spectroscopic or bounded photometric family redshift summary"

    summaries = []
    covariance_payload = {}
    for spec in protocol["systems"]:
        system = spec["system"]
        rows = frame.loc[frame["system"] == system].copy()
        readme = RAW / f"mast_{spec['mast_alias']}_zitrin_ltm_v2_readme.txt"
        params = RAW / f"mast_{spec['mast_alias']}_zitrin_ltm_v2_params.txt"
        archived = inspect_readme(readme)
        sigma = archived["model_position_sigma_arcsec"]
        rows["model_control_position_sigma_arcsec"] = sigma
        frame.loc[rows.index, "model_control_position_sigma_arcsec"] = sigma
        eligible = rows.loc[rows["metric_neutral_observable_row"]]
        control_covariance = np.eye(2 * len(eligible)) * sigma**2
        key = system.replace(" ", "_")
        covariance_payload[f"{key}_model_control_covariance_arcsec2"] = control_covariance
        covariance_payload[f"{key}_arc_ids"] = eligible["arc_id"].astype(str).to_numpy()
        measured_families = rows.loc[rows["measured_position_row"], "family_id"].nunique()
        anchored_families = eligible["family_id"].nunique()
        raw_acquired = bool(len(eligible) > 0)
        summaries.append(
            {
                "system": system,
                "catalog_id": spec["catalog_id"],
                "published_rows": int(len(rows)),
                "measured_position_rows": int(rows["measured_position_row"].sum()),
                "candidate_or_ambiguous_rows": int((~rows["measured_position_row"]).sum()),
                "measured_families": int(measured_families),
                "independently_redshift_anchored_families": int(anchored_families),
                "metric_neutral_observable_rows": int(len(eligible)),
                "raw_observable_catalog_acquired": raw_acquired,
                "metric_neutral_coordinate_likelihood_ready": False,
                "model_control_position_covariance_dimension": int(len(control_covariance)),
                "model_control_covariance_metric_dependent": True,
                "model_parameter_summary_local": params.exists() and params.stat().st_size > 0,
                "metric_neutral_weyl_posterior_local": False,
                "hard_shortfall": "" if raw_acquired else "all published rows are candidate or additionally ambiguous",
                **archived,
            }
        )

    frame = frame.sort_values(["system", "family_id", "arc_id"]).reset_index(drop=True)
    summary = pd.DataFrame(summaries)
    IMAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(IMAGE_OUTPUT, index=False)
    summary.to_csv(SYSTEM_OUTPUT, index=False)
    np.savez_compressed(CONTROL_COVARIANCE_OUTPUT, **covariance_payload)

    acquired = int(summary["raw_observable_catalog_acquired"].sum())
    shortfalls = summary.loc[~summary["raw_observable_catalog_acquired"], "system"].tolist()
    report = {
        "report_version": "R1-CLASH-Zitrin2015-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "published_table_rows": int(len(published)),
        "target_table_rows": int(len(frame)),
        "target_systems": int(len(summary)),
        "systems_with_raw_observable_catalog": acquired,
        "systems_with_metric_neutral_coordinate_likelihood": 0,
        "measured_position_rows": int(summary["measured_position_rows"].sum()),
        "metric_neutral_observable_rows": int(summary["metric_neutral_observable_rows"].sum()),
        "independently_redshift_anchored_families": int(summary["independently_redshift_anchored_families"].sum()),
        "hard_shortfall_systems": shortfalls,
        "hard_shortfall_primary_url": "https://archive.stsci.edu/prepds/clash/",
        "hard_shortfall_primary_statement": "The official CLASH release says the RXJ1532 mass model is based on only one candidate multiply-imaged system, not confirmed, plus weak-lensing constraints.",
        "next_cycle_threshold_met": acquired >= 4,
        "all_seven_success_outcome_met": acquired == 7,
        "gates": {
            "download_integrity_passed": True,
            "six_unambiguous_catalogs_added": acquired == 6,
            "rxj1532_candidate_only_shortfall_preserved": shortfalls == ["RXJ1532"],
            "metric_neutral_coordinate_covariance_acquired": False,
            "metric_neutral_weyl_posterior_acquired": False,
            "gravity_response_fit_authorized": False,
        },
        "interpretation": "Six systems provide non-ambiguous measured image coordinates with at least one independently redshift-anchored family. RXJ1532 has only three c/?-flagged entries and remains a hard shortfall. The archived 1.4-arcsec sigma_pos values are GR/LTM optimization conventions, not astrometric covariances.",
        "authorization": {
            "count_six_systems_toward_20_catalog_target": acquired == 6,
            "count_rxj1532_toward_20_catalog_target": False,
            "reuse_model_control_covariance_as_metric_neutral": False,
            "infer_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "outputs": {
            "image_catalog": str(IMAGE_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "system_summary": str(SYSTEM_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "model_control_covariances": str(CONTROL_COVARIANCE_OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2))
