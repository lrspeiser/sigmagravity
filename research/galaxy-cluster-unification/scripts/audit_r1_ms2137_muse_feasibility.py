#!/usr/bin/env python3
"""Run the frozen, metadata-only MS2137 MUSE feasibility gate."""

from __future__ import annotations

import io
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_ms2137_muse_feasibility_protocol.json"
LENS_PATH = ROOT / "data/derived/r1_strong_lens_radial_support.csv"
RANK_PATH = ROOT / "data/derived/r1_lensing_geometric_rank.csv"
SOURCE_PATH = ROOT / "data/raw/replacement_sample_audit/kaleidoscope2025_main.tex"
REPORT_PATH = ROOT / "results/r1_ms2137_muse_feasibility/report.json"

VOTABLE_NS = {"v": "http://www.ivoa.net/xml/VOTable/v1.3"}


def read_url(url: str, timeout: int = 60) -> bytes:
    with urlopen(url, timeout=timeout) as response:
        return response.read()


def tap_metadata(config: dict) -> dict:
    fields = [
        "dp_id", "obs_collection", "dataproduct_type", "calib_level",
        "target_name", "s_ra", "s_dec", "s_fov", "t_exptime", "em_min",
        "em_max", "proposal_id", "instrument_name", "access_url",
        "access_estsize",
    ]
    dp_id = config["archive_product"]["dp_id"]
    query = f"SELECT {','.join(fields)} FROM ivoa.ObsCore WHERE dp_id='{dp_id}'"
    url = config["archive_product"]["tap_url"] + "?" + urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    table = pd.read_csv(io.BytesIO(read_url(url)))
    if len(table) != 1:
        raise RuntimeError(f"expected one ESO ObsCore row for {dp_id}, found {len(table)}")
    row = table.iloc[0]
    return {
        key: (int(row[key]) if key == "calib_level" else
              float(row[key]) if key in {"s_ra", "s_dec", "s_fov", "t_exptime", "em_min", "em_max", "access_estsize"}
              else str(row[key]))
        for key in fields
    }


def datalink_metadata(config: dict) -> dict:
    xml_bytes = read_url(config["archive_product"]["datalink_url"])
    root = ET.fromstring(xml_bytes)
    fields = [field.attrib["name"] for field in root.findall(".//v:RESOURCE[@type='results']/v:TABLE/v:FIELD", VOTABLE_NS)]
    rows = []
    for tr in root.findall(".//v:RESOURCE[@type='results']/v:TABLE/v:DATA/v:TABLEDATA/v:TR", VOTABLE_NS):
        values = [(td.text or "").strip() for td in tr.findall("v:TD", VOTABLE_NS)]
        rows.append(dict(zip(fields, values)))
    this_rows = [row for row in rows if row.get("semantics") == "#this"]
    cutout_rows = [row for row in rows if row.get("semantics") == "#cutout"]
    service_id = cutout_rows[0]["service_def"] if len(cutout_rows) == 1 else ""
    service = root.find(f".//v:RESOURCE[@ID='{service_id}']", VOTABLE_NS) if service_id else None
    access_url_param = service.find("v:PARAM[@name='accessURL']", VOTABLE_NS) if service is not None else None
    circle_param = service.find(".//v:PARAM[@name='CIRCLE']/v:VALUES/v:MAX", VOTABLE_NS) if service is not None else None
    band_param = service.find(".//v:PARAM[@name='BAND']/v:VALUES/v:MAX", VOTABLE_NS) if service is not None else None
    circle_max = [float(value) for value in circle_param.attrib["value"].split()] if circle_param is not None else []
    band_max = [float(value) for value in band_param.attrib["value"].split()] if band_param is not None else []
    this_row = this_rows[0] if len(this_rows) == 1 else {}
    return {
        "query_status_ok": root.find(".//v:INFO[@name='QUERY_STATUS']", VOTABLE_NS).attrib.get("value") == "OK",
        "this_row_count": len(this_rows),
        "cutout_row_count": len(cutout_rows),
        "content_type": this_row.get("content_type"),
        "content_length_bytes": int(this_row["content_length"]) if this_row.get("content_length") else None,
        "original_filename": this_row.get("eso_origfile"),
        "eso_category": this_row.get("eso_category"),
        "soda_service_id": service_id,
        "soda_access_url": access_url_param.attrib.get("value") if access_url_param is not None else None,
        "circle_max": circle_max,
        "band_max_m": band_max,
    }


def build_report() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    archive = config["archive_product"]
    cutout = config["frozen_cutout_request"]
    overlap = config["pre_pixel_overlap_target"]

    source = SOURCE_PATH.read_text(encoding="utf-8")
    center_pattern = re.compile(
        r"tab\.ms2137model.*?located at R\.A\.=21:40:15\.16 and Decl\.=-23:39:40\.09",
        re.DOTALL,
    )
    center_source_pass = bool(center_pattern.search(source))

    product = tap_metadata(config)
    datalink = datalink_metadata(config)

    bcg = SkyCoord(
        config["published_bcg_center"]["ra_deg"] * u.deg,
        config["published_bcg_center"]["dec_deg"] * u.deg,
    )
    cube_center = SkyCoord(product["s_ra"] * u.deg, product["s_dec"] * u.deg)
    center_separation_arcsec = float(bcg.separation(cube_center).arcsec)

    lens = pd.read_csv(LENS_PATH)
    selected = lens.loc[
        (lens["system"] == config["system"])
        & lens["alternative_metric_likelihood_ready"].astype(bool)
        & (lens["bcg_centric_radius_arcsec"] <= overlap["frozen_outer_radial_edge_arcsec"])
    ].sort_values("bcg_centric_radius_arcsec")
    expected_images = overlap["preidentified_images_inside_frozen_support_sorted_by_radius"]
    image_match = len(selected) == len(expected_images) and all(
        str(row.image_id) == expected["image_id"]
        and str(row.source_family) == expected["family_id"]
        and math.isclose(float(row.bcg_centric_radius_arcsec), expected["radius_arcsec"], abs_tol=1.0e-10)
        for (_, row), expected in zip(selected.iterrows(), expected_images)
    )
    overlap_pass = bool(
        image_match
        and len(selected) >= overlap["minimum_images_inside_accepted_dynamics_support"]
        and selected["source_family"].nunique() >= overlap["minimum_independent_families_inside_accepted_dynamics_support"]
    )

    rank = pd.read_csv(RANK_PATH)
    rank_row = rank.loc[rank["system"] == config["system"]].iloc[0]
    rank_pass = bool(
        float(rank_row["pilot_priority"]) == 2.0
        and str(rank_row["pilot_role"]) == "second non-disturbed pilot"
        and str(rank_row["disturbed_control"]).lower() == "false"
    )

    exact_product_pass = bool(
        product["dp_id"] == archive["dp_id"]
        and product["proposal_id"] == archive["proposal_id"]
        and product["target_name"] == archive["target_name"]
    )
    level2_cube_pass = bool(
        product["obs_collection"] == archive["obs_collection"]
        and product["instrument_name"] == archive["instrument_name"]
        and product["dataproduct_type"] == archive["dataproduct_type"]
        and product["calib_level"] == archive["calib_level"]
        and math.isclose(product["t_exptime"], archive["archive_exposure_seconds"], abs_tol=1.0e-6)
    )
    archive_metadata_match = all(
        math.isclose(product[key], archive[expected], rel_tol=0.0, abs_tol=tolerance)
        for key, expected, tolerance in (
            ("s_ra", "cube_center_ra_deg", 1.0e-9),
            ("s_dec", "cube_center_dec_deg", 1.0e-9),
            ("s_fov", "archive_spatial_fov_deg", 1.0e-12),
            ("em_min", "archive_wavelength_min_m", 1.0e-14),
            ("em_max", "archive_wavelength_max_m", 1.0e-14),
            ("access_estsize", "archive_access_estsize_kib", 0.5),
        )
    )
    center_pass = center_separation_arcsec <= archive["maximum_bcg_to_cube_center_arcsec"]
    soda_pass = bool(
        datalink["query_status_ok"]
        and datalink["this_row_count"] == 1
        and datalink["cutout_row_count"] == 1
        and datalink["soda_access_url"] == archive["soda_url"]
        and datalink["content_type"] == "application/fits"
        and datalink["eso_category"] == "SCIENCE.CUBE.IFS"
        and datalink["content_length_bytes"] == archive["full_product_content_length_bytes"]
        and datalink["original_filename"] == archive["original_filename"]
    )
    circle_pass = bool(
        len(datalink["circle_max"]) == 3
        and cutout["radius_deg"] + center_separation_arcsec / 3600.0 <= datalink["circle_max"][2]
    )
    band_pass = bool(
        len(datalink["band_max_m"]) == 2
        and product["em_min"] <= cutout["wavelength_min_m"] < cutout["wavelength_max_m"] <= product["em_max"]
        and datalink["band_max_m"][0] <= cutout["wavelength_min_m"]
        and cutout["wavelength_max_m"] <= datalink["band_max_m"][1]
    )

    gates = {
        "published_bcg_center_source_traceable": center_source_pass,
        "exact_archive_product_and_proposal_passed": exact_product_pass,
        "level2_muse_cube_metadata_passed": level2_cube_pass,
        "archive_numerical_metadata_match_frozen_values": archive_metadata_match,
        "bcg_to_cube_center_offset_passed": center_pass,
        "datalink_and_soda_service_passed": soda_pass,
        "requested_circle_within_soda_limit": circle_pass,
        "requested_band_within_product_and_soda_limits": band_pass,
        "pre_pixel_lens_overlap_target_passed": overlap_pass,
        "residual_blind_selection_rank_passed": rank_pass,
    }
    metadata_gate = all(gates.values())
    report = {
        "report_version": config["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "science_pixels_downloaded_or_inspected": False,
        "system": config["system"],
        "archive_product": product,
        "datalink": datalink,
        "published_bcg_center": config["published_bcg_center"],
        "bcg_to_cube_center_arcsec": center_separation_arcsec,
        "frozen_cutout_request": cutout,
        "pre_pixel_overlap_target": overlap,
        "matched_lens_images": [
            {
                "image_id": str(row.image_id),
                "family_id": str(row.source_family),
                "radius_arcsec": float(row.bcg_centric_radius_arcsec),
            }
            for _, row in selected.iterrows()
        ],
        "matched_lens_image_count": int(len(selected)),
        "matched_lens_family_count": int(selected["source_family"].nunique()),
        "gates": {**gates, "metadata_feasibility_gate_passed": metadata_gate},
        "decision": "authorize_frozen_soda_cutout_acquisition" if metadata_gate else "stop_MS2137_hard_public_data_shortfall",
        "next_action": (
            "Download only the frozen 18-arcsec, 4860-7160-Angstrom SODA cutout and verify SHA-256/FITS/WCS metadata; do not inspect science arrays until a separate numerical reduction/covariance protocol is frozen."
            if metadata_gate else
            "Record the exact failed metadata gate and select the next candidate without changing the support or archive thresholds."
        ),
        "authorization": {
            "download_frozen_cutout": metadata_gate,
            "inspect_science_pixels": False,
            "extract_stellar_kinematics": False,
            "infer_dynamical_or_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
