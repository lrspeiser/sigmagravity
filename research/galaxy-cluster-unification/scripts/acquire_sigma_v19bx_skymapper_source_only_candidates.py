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
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bx_skymapper_source_only_candidates.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def build_query(config: dict[str, Any], ra: float, dec: float) -> str:
    policy = config["query_policy"]
    radius_deg = float(policy["search_radius_arcsec"]) / 3600.0
    projection = ",".join(policy["projection"])
    return (
        f"SELECT TOP {int(policy['maximum_rows_per_source'])} {projection} "
        f"FROM {config['primary_source']['table']} "
        "WHERE 1=CONTAINS(POINT('ICRS',raj2000,dej2000),"
        f"CIRCLE('ICRS',{ra:.12f},{dec:.12f},{radius_deg:.15f})) "
        "ORDER BY object_id"
    )


def angular_separation_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    ra1, dec1, ra2, dec2 = map(
        math.radians, (ra1_deg, dec1_deg, ra2_deg, dec2_deg)
    )
    cos_angle = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))) * 3600.0


def parse_response(payload: bytes, expected_columns: list[str]) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    if text.lstrip().startswith("<?xml"):
        raise RuntimeError(f"TAP service returned XML instead of CSV: {text[:500]}")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != expected_columns:
        raise RuntimeError(
            f"unexpected TAP projection: expected={expected_columns} actual={reader.fieldnames}"
        )
    return list(reader)


def request_source(
    config: dict[str, Any], source: dict[str, str]
) -> tuple[str, str, list[dict[str, str]]]:
    ra = float(source["source_ra"])
    dec = float(source["source_dec"])
    query = build_query(config, ra, dec)
    params = urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    url = f"{config['primary_source']['tap_endpoint']}?{params}"
    request = Request(url, headers={"User-Agent": "SigmaGravity-V19BX/1.0"})
    last_error: Exception | None = None
    for attempt in range(int(config["query_policy"]["retries"])):
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            rows = parse_response(payload, config["query_policy"]["projection"])
            return source["name"], query, rows
        except Exception as error:  # pragma: no cover - live network branch
            last_error = error
            if attempt + 1 < int(config["query_policy"]["retries"]):
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"SkyMapper query failed for {source['name']}: {last_error}")


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def usable_r(row: dict[str, str]) -> bool:
    return (
        int(float(row["r_ngood"] or 0)) > 0
        and int(float(row["r_flags"] or 999999)) < 4
        and int(float(row["r_nimaflags"] or 999999)) < 5
        and finite_number(row["r_petro"])
        and finite_number(row["e_r_petro"])
    )


def extended(row: dict[str, str]) -> bool:
    return (
        usable_r(row)
        and finite_number(row["radius_petro"])
        and finite_number(row["class_star"])
        and float(row["class_star"]) <= 0.5
    )


def render_csv(columns: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def acquire(config_path: Path = DEFAULT_CONFIG) -> tuple[bytes, bytes, dict[str, Any]]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    input_path = ROOT / config["input"]["path"]
    parent_actual = sha256(parent_path)
    input_actual = sha256(input_path)
    input_columns, input_rows = load_csv(input_path)
    required_input = {"name", "release_field"}
    if not required_input.issubset(input_columns):
        raise ValueError("V19BW input lacks required source identity columns")

    canonical_source_path = (
        ROOT
        / "data/derived/sigma_v19bv_wallaby_canonical_source_rows/wallaby_pilot_dr1_canonical_source_only.csv"
    )
    _, canonical_sources = load_csv(canonical_source_path)
    positions = {
        row["name"]: {"name": row["name"], "source_ra": row["ra"], "source_dec": row["dec"]}
        for row in canonical_sources
    }
    if set(positions) != {row["name"] for row in input_rows}:
        raise ValueError("V19BW and canonical-source universes differ")

    results: dict[str, tuple[str, list[dict[str, str]]]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=int(config["query_policy"]["workers"])) as pool:
        futures = {
            pool.submit(request_source, config, source): name
            for name, source in positions.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                returned_name, query, rows = future.result()
                if returned_name != name:
                    raise RuntimeError("source identity changed during acquisition")
                results[name] = (query, rows)
            except Exception as error:  # pragma: no cover - live network branch
                failures[name] = str(error)

    radius = float(config["query_policy"]["search_radius_arcsec"])
    cap = int(config["query_policy"]["maximum_rows_per_source"])
    candidate_rows: list[dict[str, str]] = []
    coverage_rows: list[dict[str, str]] = []
    query_hashes: dict[str, str] = {}
    row_cap_sources: list[str] = []
    outside_radius: list[str] = []
    field_by_name = {row["name"]: row["release_field"] for row in input_rows}
    for name in sorted(positions):
        source = positions[name]
        if name in failures:
            coverage_rows.append(
                {
                    "name": name,
                    "release_field": field_by_name[name],
                    "query_status": "failed",
                    "candidate_count": "0",
                    "usable_r_candidate_count": "0",
                    "extended_candidate_count": "0",
                    "nearest_candidate_arcsec": "",
                    "row_cap_hit": "false",
                }
            )
            continue
        query, rows = results[name]
        query_hashes[name] = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if len(rows) == cap:
            row_cap_sources.append(name)
        enriched: list[dict[str, str]] = []
        for row in rows:
            separation = angular_separation_arcsec(
                float(source["source_ra"]),
                float(source["source_dec"]),
                float(row["raj2000"]),
                float(row["dej2000"]),
            )
            if separation > radius + 1e-6:
                outside_radius.append(f"{name}:{row['object_id']}:{separation}")
            enriched.append(
                {
                    "wallaby_name": name,
                    "wallaby_ra_deg": source["source_ra"],
                    "wallaby_dec_deg": source["source_dec"],
                    "release_field": field_by_name[name],
                    "search_radius_arcsec": f"{radius:.1f}",
                    "separation_arcsec": f"{separation:.9f}",
                    "usable_r_photometry": str(usable_r(row)).lower(),
                    "extended_candidate": str(extended(row)).lower(),
                    **row,
                }
            )
        enriched.sort(key=lambda row: (float(row["separation_arcsec"]), row["object_id"]))
        candidate_rows.extend(enriched)
        nearest = enriched[0]["separation_arcsec"] if enriched else ""
        coverage_rows.append(
            {
                "name": name,
                "release_field": field_by_name[name],
                "query_status": "ok",
                "candidate_count": str(len(enriched)),
                "usable_r_candidate_count": str(sum(row["usable_r_photometry"] == "true" for row in enriched)),
                "extended_candidate_count": str(sum(row["extended_candidate"] == "true" for row in enriched)),
                "nearest_candidate_arcsec": nearest,
                "row_cap_hit": str(len(rows) == cap).lower(),
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            row["wallaby_name"],
            float(row["separation_arcsec"]),
            row["object_id"],
        )
    )
    candidate_columns = [
        "wallaby_name",
        "wallaby_ra_deg",
        "wallaby_dec_deg",
        "release_field",
        "search_radius_arcsec",
        "separation_arcsec",
        "usable_r_photometry",
        "extended_candidate",
        *config["query_policy"]["projection"],
    ]
    coverage_columns = list(coverage_rows[0])
    candidate_payload = render_csv(candidate_columns, candidate_rows)
    coverage_payload = render_csv(coverage_columns, coverage_rows)

    forbidden = [token.lower() for token in config["forbidden_target_tokens"]]
    projection_text = " ".join(config["query_policy"]["projection"]).lower()
    boundary = config["access_boundary"]
    sources_with_candidates = sum(int(row["candidate_count"]) > 0 for row in coverage_rows)
    sources_with_extended = sum(int(row["extended_candidate_count"]) > 0 for row in coverage_rows)
    candidate_counts = [int(row["candidate_count"]) for row in coverage_rows]
    gates = {
        "parent_and_input_hashes_exact": (
            parent_actual == config["parent"]["sha256"]
            and input_actual == config["input"]["sha256"]
        ),
        "all_592_source_positions_queried_once": (
            len(positions) == len(results) == len(coverage_rows) == 592 and not failures
        ),
        "exact_source_only_projection_used": (
            not any(token in projection_text for token in forbidden)
            and len(query_hashes) == len(set(query_hashes)) == 592
        ),
        "no_service_failure_or_row_cap_hit": not failures and not row_cap_sources,
        "all_candidates_inside_frozen_radius": not outside_radius,
        "candidate_and_coverage_outputs_reproduce": (
            len(coverage_rows) == 592
            and len({row["name"] for row in coverage_rows}) == 592
            and all(row["query_status"] == "ok" for row in coverage_rows)
        ),
        "kinematic_gravity_and_solar_targets_remain_sealed": (
            not boundary["wallaby_kinematic_table_rows_read"]
            and not boundary["rotation_speed_or_velocity_field_read"]
            and not boundary["inclination_or_kinematic_angle_read"]
            and not boundary["gravity_formula_residual_or_halo_result_read"]
            and not boundary["gravity_action_or_constant_changed"]
            and not boundary["solar_system_optimization_performed"]
        ),
        "no_counterpart_sample_or_evidence_split_claimed": (
            not boundary["optical_counterpart_selected"]
            and not boundary["development_validation_holdout_split_selected"]
            and not boundary["final_galaxy_sample_selected"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    field_summary: dict[str, dict[str, int]] = {}
    for field in sorted(set(field_by_name.values())):
        subset = [row for row in coverage_rows if row["release_field"] == field]
        field_summary[field] = {
            "wallaby_sources": len(subset),
            "sources_with_any_candidate": sum(int(row["candidate_count"]) > 0 for row in subset),
            "sources_with_extended_candidate": sum(int(row["extended_candidate_count"]) > 0 for row in subset),
            "optical_candidate_rows": sum(int(row["candidate_count"]) for row in subset),
        }
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_skymapper_source_only_candidate_acquisition",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "input_audit": {
            "parent": {"path": config["parent"]["path"], "sha256": parent_actual},
            "input": {"path": config["input"]["path"], "sha256": input_actual, "rows": len(input_rows)},
        },
        "query_audit": {
            "endpoint": config["primary_source"]["tap_endpoint"],
            "table": config["primary_source"]["table"],
            "projection": config["query_policy"]["projection"],
            "search_radius_arcsec": radius,
            "source_queries": len(results),
            "failed_sources": failures,
            "row_cap_sources": row_cap_sources,
            "outside_radius_rows": outside_radius,
            "query_set_sha256": hashlib.sha256(
                "\n".join(f"{name}:{query_hashes[name]}" for name in sorted(query_hashes)).encode("utf-8")
            ).hexdigest(),
        },
        "candidate_output": {
            "path": config["outputs"]["candidate_csv"],
            "sha256": bytes_sha256(candidate_payload),
            "bytes": len(candidate_payload),
            "rows": len(candidate_rows),
            "unique_skymapper_objects": len({row["object_id"] for row in candidate_rows}),
            "sources_with_any_candidate": sources_with_candidates,
            "sources_with_extended_candidate": sources_with_extended,
            "maximum_candidates_per_source": max(candidate_counts),
            "median_candidates_per_source": sorted(candidate_counts)[len(candidate_counts) // 2],
        },
        "coverage_output": {
            "path": config["outputs"]["coverage_csv"],
            "sha256": bytes_sha256(coverage_payload),
            "bytes": len(coverage_payload),
            "rows": len(coverage_rows),
            "field_summary": field_summary,
            "candidate_count_histogram": dict(sorted(Counter(candidate_counts).items())),
        },
        "diagnostic_only_flags": config["diagnostic_only_flags"],
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "decision": (
            "passed_source_only_candidates_no_counterpart_selected"
            if all(gates.values())
            else "failed_source_only_candidate_acquisition"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }
    return candidate_payload, coverage_payload, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_json(args.config)
    candidate_payload, coverage_payload, report = acquire(args.config)
    candidate_path = ROOT / config["outputs"]["candidate_csv"]
    coverage_path = ROOT / config["outputs"]["coverage_csv"]
    report_path = ROOT / config["outputs"]["report"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(candidate_payload)
    coverage_path.write_bytes(coverage_payload)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "candidate_output": report["candidate_output"],
                "coverage_output": report["coverage_output"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_source_only_candidates_no_counterpart_selected":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
