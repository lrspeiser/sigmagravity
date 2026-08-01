#!/usr/bin/env python3
"""Build the residual-blind, archive-level feasibility ledger for 14 SLACS-KCWI lenses."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from pyvo.dal import tap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_slacs_kcwi_sample_feasibility_protocol.json"
SOURCE_ROOT = ROOT / "data/raw/r1_slacs_kcwi_primary_source"
SOURCE_PATH = SOURCE_ROOT / "2409.10631/manuscript.tex"
SAMPLE_TABLE_PATH = SOURCE_ROOT / "2409.10631/paper_table_031725.tex"
DINOS_MAIN_PATH = SOURCE_ROOT / "2311.09307/main.tex"
DINOS_TABLE_PATH = SOURCE_ROOT / "2311.09307/tables.tex"
DINOS_APPENDIX_PATH = SOURCE_ROOT / "2311.09307/appendix.tex"
PROVENANCE_PATH = SOURCE_ROOT / "provenance.json"
REFERENCE_PROVENANCE_PATH = SOURCE_ROOT / "reference_provenance.json"
LEDGER_PATH = ROOT / "data/derived/r1_slacs_kcwi_candidate_ledger.csv"
INVENTORY_PATH = ROOT / "data/derived/r1_slacs_kcwi_archive_inventory.csv"
REPORT_PATH = ROOT / "results/r1_slacs_kcwi_sample_feasibility/report.json"

KOA_TAP_URL = "https://koa.ipac.caltech.edu/TAP"
DINOS_REPO = "Project-Dinos/dinos-i"
DINOS_REPO_URL = f"https://api.github.com/repos/{DINOS_REPO}"
DINOS_TREE_URL = f"{DINOS_REPO_URL}/git/trees/main?recursive=1"
DINOS_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1vh9lVTmE_ilxiCjkRYCsWdUk_XjDm7tt?usp=sharing"
)
DINOS_PAGE_URL = "https://www.projectdinos.com/dinos-i"
KCWI_DRP_CUBE_SOURCE_URL = (
    "https://raw.githubusercontent.com/Keck-DataReductionPipelines/"
    "KCWI_DRP/v1.1.0/kcwidrp/primitives/MakeCube.py"
)

SAMPLE_NAME_RE = re.compile(
    r"^(SDSSJ(?P<hh>\d{2})(?P<mm>\d{2})(?P<ss>\d{2}\.\d+)"
    r"(?P<sign>[+-])(?P<dd>\d{2})(?P<dm>\d{2})(?P<ds>\d{2}\.\d+))$"
)
SHORT_NAME_RE = re.compile(
    r"^SDSSJ(?P<hh>\d{2})(?P<mm>\d{2})\d{2}\.\d+"
    r"(?P<sign>[+-])(?P<dd>\d{2})(?P<dm>\d{2})\d{2}\.\d+$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bytes(url: str, timeout: int = 90) -> bytes:
    request = Request(url, headers={"User-Agent": "sigmagravity-observable-audit/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: int = 90) -> dict:
    return json.loads(fetch_bytes(url, timeout=timeout))


def parse_sample_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in SAMPLE_TABLE_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("SDSSJ"):
            continue
        fields = [field.strip() for field in line.removesuffix("\\\\").split("&")]
        if len(fields) < 9:
            raise ValueError(f"Unexpected sample-table row: {line}")
        full_name = fields[0]
        name_match = SAMPLE_NAME_RE.match(full_name)
        short_match = SHORT_NAME_RE.match(full_name)
        if name_match is None or short_match is None:
            raise ValueError(f"Unparseable SLACS coordinate name: {full_name}")
        parts = name_match.groupdict()
        coordinate = SkyCoord(
            f"{parts['hh']}h{parts['mm']}m{parts['ss']}s "
            f"{parts['sign']}{parts['dd']}d{parts['dm']}m{parts['ds']}s"
        )
        short = short_match.groupdict()
        alias = (
            f"SDSSJ{short['hh']}{short['mm']}"
            f"{short['sign']}{short['dd']}{short['dm']}"
        )
        rows.append(
            {
                "system": full_name,
                "alias": alias,
                "ra_deg": float(coordinate.ra.deg),
                "dec_deg": float(coordinate.dec.deg),
                "lens_redshift": float(fields[1]),
                "effective_radius_arcsec": float(fields[8]),
            }
        )
    if len(rows) != 14:
        raise ValueError(f"Expected 14 SLACS-KCWI rows, found {len(rows)}")
    return rows


def parse_dinos_products() -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    radii: dict[str, float] = {}
    for line in DINOS_TABLE_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(SDSSJ\d{4}[+-]\d{4})\s*&\s*\$([0-9.]+)_", line.strip())
        if match:
            radii[match.group(1)] = float(match.group(2))

    settings: dict[str, dict[str, object]] = {}
    pattern = re.compile(
        r"^(SDSSJ\d{4}[+-]\d{4})\s*&\s*SLACS\s*&\s*\[([^]]+)\]"
        r"\s*&\s*\[([^]]+)\]"
    )
    for line in DINOS_APPENDIX_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        filters = [value.strip() for value in match.group(2).split(",")]
        shapelet_orders = [int(value.strip()) for value in match.group(3).split(",")]
        settings[match.group(1)] = {
            "filters": filters,
            "shapelet_orders": shapelet_orders,
        }
    return radii, settings


def verify_source_archives(config: dict) -> tuple[dict[str, bool], dict[str, object]]:
    source_provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8-sig"))
    source_archive = SOURCE_ROOT / source_provenance["archive_path"]
    source_hash_pass = bool(
        source_archive.exists()
        and source_archive.stat().st_size == int(source_provenance["archive_bytes"])
        and sha256(source_archive) == source_provenance["archive_sha256"]
    )

    reference_provenance = json.loads(
        REFERENCE_PROVENANCE_PATH.read_text(encoding="utf-8-sig")
    )
    reference_checks: dict[str, bool] = {}
    reference_metadata: list[dict[str, object]] = []
    for item in reference_provenance["references"]:
        archive = SOURCE_ROOT / item["archive_path"]
        passed = bool(
            archive.exists()
            and archive.stat().st_size == int(item["archive_bytes"])
            and sha256(archive) == item["archive_sha256"]
        )
        reference_checks[item["arxiv_id"]] = passed
        reference_metadata.append(
            {
                "arxiv_id": item["arxiv_id"],
                "path": str(archive.relative_to(ROOT)).replace("\\", "/"),
                "bytes": int(item["archive_bytes"]),
                "sha256": item["archive_sha256"],
                "hash_pass": passed,
            }
        )

    manuscript = SOURCE_PATH.read_text(encoding="utf-8", errors="replace")
    dinos_main = DINOS_MAIN_PATH.read_text(encoding="utf-8", errors="replace")
    source_checks = {
        "protocol_frozen_before_candidate_archive_query": config["status"].startswith(
            "frozen_before"
        ),
        "knabel_archive_hash_pass": source_hash_pass,
        "reference_archive_hashes_pass": all(reference_checks.values()),
        "fourteen_lens_sample": "selection of 14 strong-lensing ETGs" in manuscript,
        "four_to_five_kcwi_exposures": "resulting 4-5 exposures for each object" in manuscript,
        "fifty_to_one_hundred_spatial_bins": "roughly 50-100 spatial bins" in manuscript,
        "hst_available_for_every_object": "HST} imaging is available for each of these objects" in manuscript,
        "numerical_dynamical_models_deferred": "will be presented in a follow-up paper (Paper II)" in manuscript,
        "uniform_dinos_models_stated": "uniformly modeled by \\cite{tan23}" in manuscript,
        "dinos_full_posteriors_used": "lens model posteriors" in dinos_main,
    }
    return source_checks, {
        "knabel": {
            "path": str(source_archive.relative_to(ROOT)).replace("\\", "/"),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256(source_archive),
        },
        "references": reference_metadata,
    }


def query_koa_candidate(candidate: dict[str, object]) -> tuple[pd.DataFrame, str]:
    service = tap.TAPService(KOA_TAP_URL)
    query = f"""
        SELECT TOP 2000
            koaid, object, koaimtyp, ra, dec, date_obs, elaptime,
            waveblue, wavered, camera, bgratnam, bfiltnam, ifunam,
            progid, progpi, progtitl, semester, ofname, filehand
        FROM koa_kcwi
        WHERE 1 = CONTAINS(
            POINT('J2000', ra, dec),
            CIRCLE('J2000', {candidate['ra_deg']:.8f}, {candidate['dec_deg']:.8f}, 0.015)
        )
    """
    try:
        table = service.run_sync(query, maxrec=2000).to_table().to_pandas()
        return table, ""
    except Exception as error:  # pragma: no cover - archive failures are reported in output
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


def query_mast_candidate(candidate: dict[str, object]) -> tuple[pd.DataFrame, str]:
    coordinate = SkyCoord(candidate["ra_deg"] * u.deg, candidate["dec_deg"] * u.deg)
    try:
        table = Observations.query_region(coordinate, radius=5 * u.arcsec).to_pandas()
        if table.empty:
            return table, ""
        table = table.loc[table["obs_collection"].astype(str) == "HST"].copy()
        return table, ""
    except Exception as error:  # pragma: no cover - archive failures are reported in output
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


def public_project_products(aliases: list[str]) -> dict[str, object]:
    repo = fetch_json(DINOS_REPO_URL)
    commit = fetch_json(f"{DINOS_REPO_URL}/commits/{repo['default_branch']}")
    tree_response = fetch_json(DINOS_TREE_URL)
    tree = tree_response["tree"]
    paths = [item["path"] for item in tree]
    path_set = set(paths)
    drive_html = fetch_bytes(DINOS_DRIVE_URL, timeout=120).decode("utf-8", errors="replace")
    page_html = fetch_bytes(DINOS_PAGE_URL).decode("utf-8", errors="replace")
    drp_html = fetch_bytes(KCWI_DRP_CUBE_SOURCE_URL).decode("utf-8", errors="replace")

    candidate_products: dict[str, dict[str, object]] = {}
    for alias in aliases:
        data_prefix = f"2_dolphin_modelling/data/{alias}/"
        image_paths = [
            path
            for path in paths
            if path.startswith(data_prefix) and "/image_" in path and path.endswith(".h5")
        ]
        psf_paths = [
            path
            for path in paths
            if path.startswith(data_prefix) and "/psf_" in path and path.endswith(".h5")
        ]
        config_path = f"2_dolphin_modelling/settings/{alias}_config.yml"
        candidate_products[alias] = {
            "github_image_h5_count": len(image_paths),
            "github_psf_h5_count": len(psf_paths),
            "github_config_public": config_path in path_set,
            "drive_alias_visible_in_public_folder_html": alias in drive_html,
        }
    return {
        "repo": {
            "url": repo["html_url"],
            "private": bool(repo["private"]),
            "default_branch": repo["default_branch"],
            "commit_sha": commit["sha"],
            "tree_truncated": bool(tree_response.get("truncated", False)),
            "tree_entries": len(tree),
        },
        "project_page_public": "including full chains" in page_html,
        "drive_folder_public": len(drive_html) > 100_000 and ".h5" in drive_html,
        "drive_html_bytes": len(drive_html.encode("utf-8")),
        "official_kcwi_drp_uncertainty_cube": all(
            token in drp_html
            for token in ("uncertainty", "mask", "flag", "noskysub")
        ),
        "candidate_products": candidate_products,
    }


def normalize_koa(
    candidate: dict[str, object], raw: pd.DataFrame, error: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "candidate": candidate["alias"],
                    "archive": "KOA_KCWI",
                    "product_id": "",
                    "target_name": "",
                    "product_type": "",
                    "date_obs": "",
                    "exposure_seconds": 0.0,
                    "program_id": "",
                    "program_pi": "",
                    "program_title": "",
                    "filter": "",
                    "instrument": "KCWI",
                    "public": False,
                    "metadata_match": False,
                    "file_handle": "",
                    "query_error": error,
                }
            ]
        )
    frame = raw.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    for column in ("object", "koaimtyp", "date_obs", "progid", "progpi", "progtitl", "ifunam", "filehand"):
        if column not in frame:
            frame[column] = ""
    frame["elaptime"] = pd.to_numeric(frame.get("elaptime", 0), errors="coerce").fillna(0.0)
    dates = pd.to_datetime(frame["date_obs"], errors="coerce", utc=True)
    science = frame.loc[
        frame["koaimtyp"].astype(str).str.lower().isin(["object", "science"])
        & frame["elaptime"].ge(1200.0)
        & dates.dt.year.isin([2021, 2022])
    ].copy()
    inventory = pd.DataFrame(
        {
            "candidate": candidate["alias"],
            "archive": "KOA_KCWI",
            "product_id": frame.get("koaid", "").astype(str),
            "target_name": frame["object"].astype(str),
            "product_type": frame["koaimtyp"].astype(str),
            "date_obs": frame["date_obs"].astype(str),
            "exposure_seconds": frame["elaptime"].astype(float),
            "program_id": frame["progid"].astype(str),
            "program_pi": frame["progpi"].astype(str),
            "program_title": frame["progtitl"].astype(str),
            "filter": frame["ifunam"].astype(str),
            "instrument": "KCWI",
            "public": True,
            "metadata_match": frame.index.isin(science.index),
            "file_handle": frame["filehand"].astype(str),
            "query_error": error,
        }
    )
    return science, inventory


def normalize_mast(
    candidate: dict[str, object], raw: pd.DataFrame, error: str, expected_filters: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "candidate": candidate["alias"],
                    "archive": "MAST_HST",
                    "product_id": "",
                    "target_name": "",
                    "product_type": "",
                    "date_obs": "",
                    "exposure_seconds": 0.0,
                    "program_id": "",
                    "program_pi": "",
                    "program_title": "",
                    "filter": "",
                    "instrument": "HST",
                    "public": False,
                    "metadata_match": False,
                    "file_handle": "",
                    "query_error": error,
                }
            ]
        )
    frame = raw.copy()
    for column in ("filters", "dataRights", "proposal_id", "obsid", "target_name", "instrument_name", "t_exptime", "obs_id"):
        if column not in frame:
            frame[column] = ""
    frame["t_exptime"] = pd.to_numeric(frame["t_exptime"], errors="coerce").fillna(0.0)
    public = frame["dataRights"].astype(str).str.upper().eq("PUBLIC")
    filter_match = frame["filters"].astype(str).isin(expected_filters)
    matched = frame.loc[public & filter_match & frame["t_exptime"].gt(0)].copy()
    inventory = pd.DataFrame(
        {
            "candidate": candidate["alias"],
            "archive": "MAST_HST",
            "product_id": frame["obsid"].astype(str),
            "target_name": frame["target_name"].astype(str),
            "product_type": frame.get("dataproduct_type", "").astype(str),
            "date_obs": frame.get("t_min", "").astype(str),
            "exposure_seconds": frame["t_exptime"].astype(float),
            "program_id": frame["proposal_id"].astype(str),
            "program_pi": "",
            "program_title": "",
            "filter": frame["filters"].astype(str),
            "instrument": frame["instrument_name"].astype(str),
            "public": public,
            "metadata_match": public & filter_match,
            "file_handle": frame["obs_id"].astype(str),
            "query_error": error,
        }
    )
    return matched, inventory


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = parse_sample_table()
    dinos_radii, dinos_settings = parse_dinos_products()
    source_checks, source_metadata = verify_source_archives(config)
    public_products = public_project_products([str(item["alias"]) for item in candidates])

    ledger_rows: list[dict[str, object]] = []
    inventory_frames: list[pd.DataFrame] = []
    query_errors: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        alias = str(candidate["alias"])
        settings = dinos_settings.get(alias, {"filters": [], "shapelet_orders": []})
        expected_filters = list(settings["filters"])
        koa_raw, koa_error = query_koa_candidate(candidate)
        mast_raw, mast_error = query_mast_candidate(candidate)
        koa_science, koa_inventory = normalize_koa(candidate, koa_raw, koa_error)
        mast_public, mast_inventory = normalize_mast(
            candidate, mast_raw, mast_error, expected_filters
        )
        inventory_frames.extend([koa_inventory, mast_inventory])
        query_errors[alias] = {"koa": koa_error, "mast": mast_error}

        product = public_products["candidate_products"][alias]
        einstein_radius = dinos_radii.get(alias, np.nan)
        accepted_dynamics_radius = min(float(candidate["effective_radius_arcsec"]), 3.1175)
        common_support = bool(
            np.isfinite(einstein_radius) and einstein_radius <= accepted_dynamics_radius
        )
        koa_programs = sorted(set(koa_science.get("progid", pd.Series(dtype=str)).astype(str)))
        hst_programs = sorted(set(mast_public.get("proposal_id", pd.Series(dtype=str)).astype(str)))
        hst_filters = sorted(set(mast_public.get("filters", pd.Series(dtype=str)).astype(str)))
        raw_kcwi_gate = bool(
            len(koa_science) >= 4
            and float(koa_science.get("elaptime", pd.Series(dtype=float)).sum()) >= 7200.0
        )
        hst_gate = bool(len(mast_public) >= 1 and set(expected_filters).issubset(hst_filters))
        published_model = alias in dinos_radii
        model_replay_gate = bool(
            published_model
            and product["github_config_public"]
            and product["github_image_h5_count"] >= len(expected_filters) >= 1
            and product["github_psf_h5_count"] >= len(expected_filters)
            and public_products["project_page_public"]
            and public_products["drive_folder_public"]
        )
        raw_dynamics_reconstructible = bool(
            raw_kcwi_gate and public_products["official_kcwi_drp_uncertainty_cube"]
        )
        pre_pixel_gate = bool(
            all(source_checks.values())
            and hst_gate
            and raw_dynamics_reconstructible
            and common_support
            and model_replay_gate
        )
        support_ratio = (
            accepted_dynamics_radius / einstein_radius
            if np.isfinite(einstein_radius) and einstein_radius > 0
            else np.nan
        )
        ledger_rows.append(
            {
                **candidate,
                "dinos_model_published": published_model,
                "dinos_einstein_radius_arcsec": einstein_radius,
                "dinos_filters": ";".join(expected_filters),
                "dinos_max_shapelet_order": max(settings["shapelet_orders"], default=0),
                "accepted_dynamics_outer_radius_arcsec": accepted_dynamics_radius,
                "dynamics_to_einstein_support_ratio": support_ratio,
                "common_radial_support_gate_pass": common_support,
                "published_spatial_bin_count": "roughly_50_to_100_samplewise",
                "numerical_kinematic_map_public": False,
                "koa_public_science_frame_count": len(koa_science),
                "koa_public_science_exposure_seconds": float(
                    koa_science.get("elaptime", pd.Series(dtype=float)).sum()
                ),
                "koa_program_ids": ";".join(koa_programs),
                "raw_kcwi_and_uncertainty_reconstructible": raw_dynamics_reconstructible,
                "mast_public_matching_observation_count": len(mast_public),
                "mast_program_ids": ";".join(hst_programs),
                "mast_public_matching_filters": ";".join(hst_filters),
                "public_hst_gate_pass": hst_gate,
                "github_image_h5_count": product["github_image_h5_count"],
                "github_psf_h5_count": product["github_psf_h5_count"],
                "github_config_public": product["github_config_public"],
                "public_full_chain_folder_declared": public_products["project_page_public"],
                "rerunnable_lens_model_gate_pass": model_replay_gate,
                "pre_fit_ring_scale_rank_upper_bound": 1 if published_model else 0,
                "pre_fit_extended_arc_rank_upper_bound": (
                    "potentially_at_least_3_pending_existing_model_replay"
                    if model_replay_gate
                    else "not_established"
                ),
                "pre_pixel_candidate_gate_pass": pre_pixel_gate,
                "counts_toward_ten_system_target": False,
                "disposition": (
                    "authorize_exact_acquisition_and_existing_model_replay_protocol"
                    if pre_pixel_gate
                    else "metadata_shortfall_do_not_download_science_arrays"
                ),
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    ledger = ledger.sort_values(
        ["pre_pixel_candidate_gate_pass", "dynamics_to_einstein_support_ratio", "mast_public_matching_observation_count"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ledger.insert(0, "residual_blind_priority", np.arange(1, len(ledger) + 1))
    inventory = pd.concat(inventory_frames, ignore_index=True, sort=False)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False, lineterminator="\n")
    inventory.to_csv(INVENTORY_PATH, index=False, lineterminator="\n")

    passed = ledger.loc[ledger["pre_pixel_candidate_gate_pass"]]
    next_candidate = str(passed.iloc[0]["alias"]) if len(passed) else "RXJ1131-1231"
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol_version"],
        "selection_blind": True,
        "science_arrays_downloaded": False,
        "source_checks": source_checks,
        "source_metadata": source_metadata,
        "public_product_audit": {
            key: value for key, value in public_products.items() if key != "candidate_products"
        },
        "sample_summary": {
            "candidates_audited": len(ledger),
            "koa_raw_science_gate_pass": int(
                ledger["raw_kcwi_and_uncertainty_reconstructible"].sum()
            ),
            "public_hst_gate_pass": int(ledger["public_hst_gate_pass"].sum()),
            "rerunnable_lens_model_gate_pass": int(
                ledger["rerunnable_lens_model_gate_pass"].sum()
            ),
            "common_radial_support_gate_pass": int(
                ledger["common_radial_support_gate_pass"].sum()
            ),
            "all_pre_pixel_gates_pass": int(ledger["pre_pixel_candidate_gate_pass"].sum()),
            "numerical_kinematic_maps_public": int(
                ledger["numerical_kinematic_map_public"].sum()
            ),
            "strict_rank_three_promotions": 0,
            "structural_ceiling_after_screen": 3,
            "strict_ready_systems_after_screen": 0,
        },
        "query_errors": query_errors,
        "decision": (
            "freeze_one_exact_acquisition_and_existing_model_replay_protocol"
            if len(passed)
            else "hard_shortfall_move_to_RXJ1131_without_weakening_gate"
        ),
        "next_candidate": next_candidate,
        "next_candidate_selection_basis": (
            "largest accepted-dynamics-to-Einstein-radius support ratio among candidates "
            "passing every frozen archive and replay gate; no gravity residual used"
        ),
        "authorization": {
            "download_science_arrays_now": False,
            "freeze_exact_candidate_protocol": bool(len(passed)),
            "count_candidate_as_rank_three": False,
            "fit_gravity_response": False,
            "authorize_R2": False,
        },
        "outputs": {
            "candidate_ledger": str(LEDGER_PATH.relative_to(ROOT)).replace("\\", "/"),
            "archive_inventory": str(INVENTORY_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
