from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19by_wallaby_moment0_source_maps.json"


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return file_hash(path, "sha256")


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


def inventory_query(config: dict[str, Any]) -> str:
    product_ids = list(config["release_product_ids"].values())
    product_clause = " OR ".join(f"p.productID='{value}'" for value in product_ids)
    suffix = config["artifact_policy"]["required_uri_suffix"]
    return (
        "SELECT o.observationID,o.target_name,p.productID,a.uri,a.productType,"
        "a.releaseType,a.contentType,a.contentLength,a.contentChecksum "
        "FROM caom2.Observation o JOIN caom2.Plane p ON o.obsID=p.obsID "
        "JOIN caom2.Artifact a ON p.planeID=a.planeID "
        f"WHERE o.collection='{config['primary_source']['collection']}' "
        f"AND ({product_clause}) AND a.uri LIKE '%{suffix}'"
    )


def request_csv(url: str, query: str) -> tuple[list[str], list[dict[str, str]]]:
    params = urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    request = Request(f"{url}?{params}", headers={"User-Agent": "SigmaGravity-V19BY/1.0"})
    with urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8-sig")
    if text.lstrip().startswith("<?xml"):
        raise RuntimeError(f"CADC TAP returned XML instead of CSV: {text[:500]}")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise RuntimeError("CADC TAP response has no header")
    return list(reader.fieldnames), list(reader)


def artifact_key(row: dict[str, str]) -> tuple[str, str]:
    return row["target_name"], row["productID"]


def expected_product_id(config: dict[str, Any], source_row: dict[str, str]) -> str:
    try:
        return config["release_product_ids"][source_row["team_release"]]
    except KeyError as error:
        raise ValueError(f"unsupported team release: {source_row['team_release']}") from error


def filename_from_uri(uri: str) -> str:
    prefix = "cadc:WALLABY/"
    if not uri.startswith(prefix):
        raise ValueError(f"unexpected CADC artifact URI: {uri}")
    name = uri[len(prefix) :]
    if "/" in name or not name:
        raise ValueError(f"unsafe CADC artifact filename: {uri}")
    return name


def validate_artifact_policy(config: dict[str, Any], row: dict[str, str]) -> None:
    policy = config["artifact_policy"]
    uri_lower = row["uri"].lower()
    if not uri_lower.endswith(policy["required_uri_suffix"].lower()):
        raise ValueError(f"artifact is not a moment-zero FITS map: {row['uri']}")
    if any(token.lower() in uri_lower for token in policy["forbidden_uri_tokens"]):
        raise ValueError(f"forbidden artifact token in {row['uri']}")
    expected = {
        "contentType": policy["required_content_type"],
        "productType": policy["required_product_type"],
        "releaseType": policy["required_release_type"],
    }
    for field, value in expected.items():
        if row[field] != value:
            raise ValueError(f"unexpected {field} for {row['uri']}: {row[field]}")
    if not row["contentChecksum"].startswith("md5:"):
        raise ValueError(f"artifact has no archive MD5: {row['uri']}")


def download_one(
    config: dict[str, Any], artifact: dict[str, str], output_path: Path
) -> dict[str, Any]:
    expected_length = int(artifact["contentLength"])
    expected_md5 = artifact["contentChecksum"].split(":", 1)[1]
    if (
        output_path.is_file()
        and output_path.stat().st_size == expected_length
        and file_hash(output_path, "md5") == expected_md5
    ):
        reused = True
    else:
        reused = False
        url = config["primary_source"]["data_base_url"] + filename_from_uri(artifact["uri"])
        request = Request(url, headers={"User-Agent": "SigmaGravity-V19BY/1.0"})
        last_error: Exception | None = None
        for attempt in range(int(config["download_policy"]["retries"])):
            part = output_path.with_suffix(output_path.suffix + ".part")
            try:
                with urlopen(request, timeout=120) as response, part.open("wb") as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                if part.stat().st_size != expected_length:
                    raise RuntimeError("download length does not match CAOM metadata")
                if file_hash(part, "md5") != expected_md5:
                    raise RuntimeError("download MD5 does not match CAOM metadata")
                part.replace(output_path)
                last_error = None
                break
            except Exception as error:  # pragma: no cover - live network branch
                last_error = error
                if part.exists():
                    part.unlink()
                if attempt + 1 < int(config["download_policy"]["retries"]):
                    time.sleep(1.0 * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(f"download failed for {artifact['uri']}: {last_error}")

    with fits.open(output_path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
        header = hdul[0].header
        naxis = int(header.get("NAXIS", -1))
        naxis1 = int(header.get("NAXIS1", -1))
        naxis2 = int(header.get("NAXIS2", -1))
        has_spectral_axis = any(key in header for key in ("CTYPE3", "CRVAL3", "CDELT3"))
        ctype1 = str(header.get("CTYPE1", ""))
        ctype2 = str(header.get("CTYPE2", ""))
        bunit = str(header.get("BUNIT", ""))
    return {
        "reused": reused,
        "bytes": output_path.stat().st_size,
        "md5": file_hash(output_path, "md5"),
        "sha256": sha256(output_path),
        "naxis": naxis,
        "naxis1": naxis1,
        "naxis2": naxis2,
        "has_spectral_axis": has_spectral_axis,
        "ctype1": ctype1,
        "ctype2": ctype2,
        "bunit": bunit,
    }


def render_csv(columns: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def acquire(config_path: Path = DEFAULT_CONFIG) -> tuple[bytes, dict[str, Any]]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    input_path = ROOT / config["input"]["path"]
    parent_actual = sha256(parent_path)
    input_actual = sha256(input_path)
    _, source_rows = load_csv(input_path)
    query = inventory_query(config)
    inventory_columns, inventory_rows = request_csv(
        config["primary_source"]["tap_endpoint"], query
    )
    expected_columns = [
        "observationID",
        "target_name",
        "productID",
        "uri",
        "productType",
        "releaseType",
        "contentType",
        "contentLength",
        "contentChecksum",
    ]
    if inventory_columns != expected_columns:
        raise RuntimeError(
            f"CAOM inventory projection changed: {inventory_columns} != {expected_columns}"
        )
    inventory: dict[tuple[str, str], list[dict[str, str]]] = {}
    for artifact in inventory_rows:
        validate_artifact_policy(config, artifact)
        inventory.setdefault(artifact_key(artifact), []).append(artifact)

    matched: list[tuple[dict[str, str], dict[str, str]]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for source in source_rows:
        key = (source["name"], expected_product_id(config, source))
        candidates = inventory.get(key, [])
        if len(candidates) == 1:
            matched.append((source, candidates[0]))
        elif not candidates:
            missing.append(f"{source['id']}:{key[0]}:{key[1]}")
        else:
            ambiguous.append(f"{source['id']}:{key[0]}:{key[1]}:{len(candidates)}")

    map_dir = ROOT / config["outputs"]["map_directory"]
    map_dir.mkdir(parents=True, exist_ok=True)
    download_results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=int(config["download_policy"]["workers"])) as pool:
        futures = {}
        for source, artifact in matched:
            filename = filename_from_uri(artifact["uri"])
            futures[pool.submit(download_one, config, artifact, map_dir / filename)] = (
                source,
                artifact,
                filename,
            )
        for future in as_completed(futures):
            source, artifact, filename = futures[future]
            try:
                download_results[source["id"]] = {
                    "source": source,
                    "artifact": artifact,
                    "filename": filename,
                    "audit": future.result(),
                }
            except Exception as error:  # pragma: no cover - live network branch
                failures[source["id"]] = str(error)

    manifest_rows: list[dict[str, str]] = []
    for source in sorted(source_rows, key=lambda row: int(row["id"])):
        if source["id"] not in download_results:
            continue
        item = download_results[source["id"]]
        artifact = item["artifact"]
        audit = item["audit"]
        manifest_rows.append(
            {
                "source_row_id": source["id"],
                "name": source["name"],
                "team_release": source["team_release"],
                "product_id": artifact["productID"],
                "artifact_uri": artifact["uri"],
                "archive_content_length": artifact["contentLength"],
                "archive_md5": artifact["contentChecksum"].split(":", 1)[1],
                "local_path": (Path(config["outputs"]["map_directory"]) / item["filename"]).as_posix(),
                "local_sha256": audit["sha256"],
                "naxis": str(audit["naxis"]),
                "naxis1": str(audit["naxis1"]),
                "naxis2": str(audit["naxis2"]),
                "has_spectral_axis": str(audit["has_spectral_axis"]).lower(),
                "ctype1": audit["ctype1"],
                "ctype2": audit["ctype2"],
                "bunit": audit["bunit"],
            }
        )
    manifest_columns = list(manifest_rows[0]) if manifest_rows else []
    manifest_payload = render_csv(manifest_columns, manifest_rows) if manifest_rows else b""

    policy = config["artifact_policy"]
    length_md5_exact = all(
        item["audit"]["bytes"] == int(item["artifact"]["contentLength"])
        and item["audit"]["md5"] == item["artifact"]["contentChecksum"].split(":", 1)[1]
        for item in download_results.values()
    )
    fits_2d = all(
        item["audit"]["naxis"] == int(policy["required_fits_naxis"])
        and item["audit"]["naxis1"] > 0
        and item["audit"]["naxis2"] > 0
        and not item["audit"]["has_spectral_axis"]
        for item in download_results.values()
    )
    boundary = config["access_boundary"]
    gates = {
        "parent_and_input_hashes_exact": (
            parent_actual == config["parent"]["sha256"]
            and input_actual == config["input"]["sha256"]
        ),
        "exact_moment0_artifact_policy_used": (
            query.endswith(f"a.uri LIKE '%{policy['required_uri_suffix']}'")
            and all(not any(token in row["uri"].lower() for token in policy["forbidden_uri_tokens"]) for row in inventory_rows)
        ),
        "one_map_per_all_711_release_rows": (
            len(source_rows) == len(matched) == len(download_results) == 711
            and not missing
            and not ambiguous
            and not failures
        ),
        "all_archive_lengths_and_md5s_exact": length_md5_exact,
        "all_files_are_two_dimensional_fits": fits_2d,
        "manifest_reproduces_downloads": (
            len(manifest_rows) == 711
            and len({row["source_row_id"] for row in manifest_rows}) == 711
            and all((ROOT / row["local_path"]).is_file() for row in manifest_rows)
        ),
        "kinematic_gravity_and_solar_targets_remain_sealed": (
            not boundary["cube_or_spectral_mask_downloaded"]
            and not boundary["moment1_or_moment2_downloaded"]
            and not boundary["spectrum_downloaded"]
            and not boundary["kinematic_plane_or_table_read"]
            and not boundary["rotation_speed_or_velocity_field_read"]
            and not boundary["inclination_or_kinematic_angle_read"]
            and not boundary["gravity_formula_residual_or_halo_result_read"]
            and not boundary["gravity_action_or_constant_changed"]
            and not boundary["solar_system_optimization_performed"]
        ),
        "no_counterpart_or_evidence_split_claimed": (
            not boundary["optical_counterpart_selected"]
            and not boundary["development_validation_holdout_split_selected"]
        ),
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    release_counts = Counter(row["team_release"] for row in manifest_rows)
    total_bytes = sum(int(row["archive_content_length"]) for row in manifest_rows)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_wallaby_moment0_source_map_acquisition",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "input_audit": {
            "parent": {"path": config["parent"]["path"], "sha256": parent_actual},
            "input": {"path": config["input"]["path"], "sha256": input_actual, "rows": len(source_rows)},
        },
        "inventory_audit": {
            "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "inventory_rows": len(inventory_rows),
            "matched_release_rows": len(matched),
            "missing": missing,
            "ambiguous": ambiguous,
        },
        "map_output": {
            "directory": config["outputs"]["map_directory"],
            "files": len(download_results),
            "bytes": total_bytes,
            "release_counts": dict(sorted(release_counts.items())),
            "reused_files": sum(item["audit"]["reused"] for item in download_results.values()),
            "new_files": sum(not item["audit"]["reused"] for item in download_results.values()),
            "failures": failures,
            "shape_counts": dict(sorted(Counter(f"{row['naxis1']}x{row['naxis2']}" for row in manifest_rows).items())),
        },
        "manifest_output": {
            "path": config["outputs"]["manifest_csv"],
            "sha256": bytes_sha256(manifest_payload),
            "bytes": len(manifest_payload),
            "rows": len(manifest_rows),
        },
        "access_boundary_audit": boundary,
        "gate_results": gates,
        "decision": (
            "passed_moment0_source_maps_kinematics_sealed"
            if all(gates.values())
            else "failed_moment0_source_map_acquisition"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }
    return manifest_payload, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_json(args.config)
    manifest_payload, report = acquire(args.config)
    manifest_path = ROOT / config["outputs"]["manifest_csv"]
    report_path = ROOT / config["outputs"]["report"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "map_output": report["map_output"],
                "manifest_output": report["manifest_output"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_moment0_source_maps_kinematics_sealed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
