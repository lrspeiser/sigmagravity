from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ca_skymapper_gaia_foreground_diagnostics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_query(config: dict[str, Any], object_ids: list[int]) -> str:
    if not object_ids or len(object_ids) > int(config["query_policy"]["batch_size"]):
        raise ValueError("invalid V19CA query batch")
    projection = config["query_policy"]["projection"]
    projection_text = " ".join(projection).lower()
    for token in config["query_policy"]["forbidden_projection_tokens"]:
        if token.lower() in projection_text:
            raise ValueError(f"forbidden projection token: {token}")
    ids = ",".join(str(value) for value in sorted(object_ids))
    return (
        f"SELECT {','.join(projection)} FROM {config['primary_source']['master_table']} AS m "
        f"{config['query_policy']['join']} WHERE m.object_id IN ({ids}) "
        "ORDER BY m.object_id"
    )


def request_batch(
    config: dict[str, Any], object_ids: list[int]
) -> tuple[str, list[dict[str, str]]]:
    query = build_query(config, object_ids)
    payload = urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    ).encode("utf-8")
    request = Request(
        config["primary_source"]["tap_endpoint"],
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "SigmaGravity-V19CA/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(int(config["query_policy"]["retries"])):
        try:
            with urlopen(request, timeout=120) as response:
                text = response.read().decode("utf-8-sig")
            if text.lstrip().startswith("<?xml"):
                raise RuntimeError(f"SkyMapper TAP returned XML: {text[:500]}")
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames is None:
                raise RuntimeError("SkyMapper TAP response has no header")
            return query, list(reader)
        except Exception as error:  # pragma: no cover - live network branch
            last_error = error
            if attempt + 1 < int(config["query_policy"]["retries"]):
                time.sleep(float(attempt + 1))
    raise RuntimeError(f"SkyMapper TAP batch failed: {last_error}")


def optional_float(row: dict[str, str], key: str) -> float | None:
    text = row.get(key, "").strip()
    if not text:
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def optional_int(row: dict[str, str], key: str) -> int | None:
    text = row.get(key, "").strip()
    return None if not text else int(text)


def significance(
    value: float | None, error: float | None, *, positive_only: bool
) -> float | None:
    if value is None or error is None or error <= 0:
        return None
    if positive_only and value <= 0:
        return 0.0
    return (value if positive_only else abs(value)) / error


def format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def derive_foreground_diagnostic(
    row: dict[str, str], policy: dict[str, Any]
) -> dict[str, Any]:
    distance = optional_float(row, "gaia_dr3_dist1")
    parallax_snr = significance(
        optional_float(row, "parallax"),
        optional_float(row, "parallax_error"),
        positive_only=True,
    )
    pmra_snr = significance(
        optional_float(row, "pmra"),
        optional_float(row, "pmra_error"),
        positive_only=False,
    )
    pmdec_snr = significance(
        optional_float(row, "pmdec"),
        optional_float(row, "pmdec_error"),
        positive_only=False,
    )
    exact_match = distance is not None and distance <= float(
        policy["maximum_exact_match_distance_arcsec"]
    )
    reasons: list[str] = []
    if (
        exact_match
        and parallax_snr is not None
        and parallax_snr >= float(policy["minimum_positive_parallax_significance"])
    ):
        reasons.append("positive_parallax")
    for key, value in (("pmra", pmra_snr), ("pmdec", pmdec_snr)):
        if (
            exact_match
            and value is not None
            and value
            >= float(policy["minimum_absolute_component_proper_motion_significance"])
        ):
            reasons.append(key)
    evidence = bool(reasons)
    ruwe = optional_float(row, "ruwe")
    solution = optional_int(row, "astrometric_params_solved")
    quality_controlled = (
        evidence
        and ruwe is not None
        and ruwe <= float(policy["maximum_quality_controlled_ruwe"])
        and solution
        in {int(value) for value in policy["quality_controlled_astrometric_solutions"]}
    )
    return {
        "object_id": row["object_id"],
        "gaia_source_id": row.get("matched_gaia_source_id", ""),
        "gaia_match_distance_arcsec": row.get("gaia_dr3_dist1", ""),
        "exact_gaia_match": str(exact_match).lower(),
        "parallax_significance": format_optional(parallax_snr),
        "absolute_pmra_significance": format_optional(pmra_snr),
        "absolute_pmdec_significance": format_optional(pmdec_snr),
        "foreground_astrometric_evidence": str(evidence).lower(),
        "quality_controlled_foreground_contamination": str(quality_controlled).lower(),
        "evidence_channels": ";".join(reasons),
        "interpretation": (
            "foreground_astrometric_source_overlaps_detection"
            if evidence
            else "no_five_sigma_foreground_astrometry_at_exact_match"
        ),
    }


def acquire(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    honesty = config["honesty_boundary"]
    if not honesty["three_candidate_rows_piloted_before_freeze"]:
        raise RuntimeError("V19CA must disclose the source-only pilot")
    if honesty["complete_candidate_population_queried_before_freeze"]:
        raise RuntimeError("V19CA freeze boundary is inconsistent")
    if honesty["gravity_kinematic_or_lensing_target_inspected"]:
        raise RuntimeError("V19CA claims forbidden target access")

    parent_path = ROOT / config["parent"]["path"]
    input_path = ROOT / config["input"]["path"]
    if sha256(parent_path) != config["parent"]["sha256"]:
        raise RuntimeError("V19CA parent hash mismatch")
    if sha256(input_path) != config["input"]["sha256"]:
        raise RuntimeError("V19CA input hash mismatch")
    _, candidates = load_csv(input_path)
    if len(candidates) != int(config["input"]["rows"]):
        raise RuntimeError("V19CA candidate occurrence count mismatch")
    object_ids_text = {row["object_id"] for row in candidates}
    if any(not value.isdigit() for value in object_ids_text):
        raise RuntimeError("non-numeric SkyMapper object_id")
    object_ids = sorted(int(value) for value in object_ids_text)
    if len(object_ids) != int(config["input"]["unique_object_ids"]):
        raise RuntimeError("V19CA unique candidate count mismatch")

    batch_size = int(config["query_policy"]["batch_size"])
    batches = [
        object_ids[index : index + batch_size]
        for index in range(0, len(object_ids), batch_size)
    ]
    query_hashes: list[str] = []
    raw_rows: list[dict[str, str]] = []
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=int(config["query_policy"]["workers"])) as pool:
        futures = {
            pool.submit(request_batch, config, batch): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                query, rows = future.result()
                query_hashes.append(hashlib.sha256(query.encode("utf-8")).hexdigest())
                raw_rows.extend(rows)
            except Exception as error:  # pragma: no cover - live network branch
                failures[index] = str(error)
    if failures:
        raise RuntimeError(f"V19CA query failures: {failures}")
    if not raw_rows:
        raise RuntimeError("V19CA returned no rows")
    raw_rows.sort(key=lambda row: int(row["object_id"]))
    returned_ids = [int(row["object_id"]) for row in raw_rows]
    duplicates = [value for value, count in Counter(returned_ids).items() if count != 1]
    missing = sorted(set(object_ids) - set(returned_ids))
    unexpected = sorted(set(returned_ids) - set(object_ids))
    if duplicates or missing or unexpected or len(raw_rows) != len(object_ids):
        raise RuntimeError(
            f"V19CA object coverage failure duplicates={duplicates[:5]} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    raw_fields = list(raw_rows[0])
    returned_field_text = " ".join(raw_fields).lower()
    for token in config["query_policy"]["forbidden_projection_tokens"]:
        if token.lower() in returned_field_text:
            raise RuntimeError(f"forbidden returned field token: {token}")
    diagnostics = [
        derive_foreground_diagnostic(row, config["diagnostic_policy"])
        for row in raw_rows
    ]
    diagnostic_fields = list(diagnostics[0])
    raw_path = ROOT / config["outputs"]["raw_crossmatch"]
    derived_path = ROOT / config["outputs"]["derived_diagnostics"]
    report_path = ROOT / config["outputs"]["report"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(csv_bytes(raw_rows, raw_fields))
    derived_path.write_bytes(csv_bytes(diagnostics, diagnostic_fields))

    diagnostic_by_id = {row["object_id"]: row for row in diagnostics}
    field_summary: dict[str, Any] = {}
    for field in sorted({row["release_field"] for row in candidates}):
        occurrences = [row for row in candidates if row["release_field"] == field]
        unique = {row["object_id"] for row in occurrences}
        field_summary[field] = {
            "candidate_occurrences": len(occurrences),
            "unique_objects": len(unique),
            "foreground_astrometric_evidence_objects": sum(
                diagnostic_by_id[value]["foreground_astrometric_evidence"] == "true"
                for value in unique
            ),
            "quality_controlled_foreground_contamination_objects": sum(
                diagnostic_by_id[value]["quality_controlled_foreground_contamination"]
                == "true"
                for value in unique
            ),
        }

    boundary = dict(config["access_boundary"])
    gates = {
        "parent_and_input_hashes_exact": True,
        "exact_source_only_projection_used": len(raw_fields)
        == len(config["query_policy"]["projection"]),
        "all_17034_unique_objects_returned_once": len(raw_rows) == 17034
        and len(set(returned_ids)) == 17034,
        "all_17094_candidate_occurrences_represented": len(candidates) == 17094
        and set(diagnostic_by_id) == object_ids_text,
        "no_query_failure_or_duplicate_object": not failures and not duplicates,
        "raw_and_derived_outputs_reproduce": len(diagnostics) == len(raw_rows),
        "no_radial_velocity_candidate_removal_or_counterpart_assignment": not boundary[
            "gaia_radial_velocity_read"
        ]
        and not boundary["hard_star_mask_applied"]
        and not boundary["candidate_removed_or_weighted"]
        and not boundary["optical_counterpart_selected"],
        "kinematic_gravity_lensing_and_solar_targets_remain_sealed": not any(
            boundary[key]
            for key in (
                "wallaby_kinematic_table_row_read",
                "rotation_speed_or_velocity_field_read",
                "gravity_formula_residual_or_halo_result_read",
                "development_validation_holdout_split_selected",
                "gravity_action_or_constant_changed",
                "lensing_payload_opened",
                "solar_system_optimization_performed",
            )
        ),
        "pilot_and_freeze_state_reported_honestly": honesty[
            "three_candidate_rows_piloted_before_freeze"
        ]
        and not honesty["complete_candidate_population_queried_before_freeze"],
    }
    evidence_count = sum(
        row["foreground_astrometric_evidence"] == "true" for row in diagnostics
    )
    quality_count = sum(
        row["quality_controlled_foreground_contamination"] == "true"
        for row in diagnostics
    )
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_skymapper_gaia_source_only_foreground_diagnostics",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "honesty_boundary": honesty,
        "input_audit": {
            "candidate_occurrences": len(candidates),
            "unique_object_ids": len(object_ids),
            "query_batches": len(batches),
            "query_hashes_sha256": hashlib.sha256(
                "\n".join(sorted(query_hashes)).encode("utf-8")
            ).hexdigest(),
            "returned_rows": len(raw_rows),
            "missing": missing,
            "unexpected": unexpected,
            "duplicates": duplicates,
        },
        "diagnostic_policy": config["diagnostic_policy"],
        "diagnostic_summary": {
            "exact_gaia_matches": sum(
                row["exact_gaia_match"] == "true" for row in diagnostics
            ),
            "foreground_astrometric_evidence": evidence_count,
            "quality_controlled_foreground_contamination": quality_count,
            "unflagged_or_unresolved": len(diagnostics) - quality_count,
            "field_summary": field_summary,
            "evidence_channel_counts": dict(
                sorted(
                    Counter(
                        channel
                        for row in diagnostics
                        for channel in row["evidence_channels"].split(";")
                        if channel
                    ).items()
                )
            ),
        },
        "outputs": {
            "raw_crossmatch": {
                "path": config["outputs"]["raw_crossmatch"],
                "rows": len(raw_rows),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256(raw_path),
            },
            "derived_diagnostics": {
                "path": config["outputs"]["derived_diagnostics"],
                "rows": len(diagnostics),
                "bytes": derived_path.stat().st_size,
                "sha256": sha256(derived_path),
            },
        },
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "decision": (
            "foreground_diagnostics_acquired_without_candidate_assignment"
            if all(gates.values())
            else "failed_closed"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["decision"] == "failed_closed":
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = acquire(args.config)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "input_audit": report["input_audit"],
                "diagnostic_summary": report["diagnostic_summary"],
                "outputs": report["outputs"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
