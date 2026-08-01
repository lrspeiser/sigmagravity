#!/usr/bin/env python3
"""Exhaust the 32-BCG Loubser et al. sample for the residual-blind host ledger."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/raw/r1_replacement_search_sources/loubser2018_bcg_kinematics"
TEX = SOURCE_ROOT / "source/BCG_datapaper_mnrasv3.tex"
PROVENANCE = SOURCE_ROOT / "provenance.json"
CURRENT_QUEUE = ROOT / "data/derived/r1_replacement_acquisition_queue.csv"
CURRENT_RANK = ROOT / "data/derived/r1_lensing_geometric_rank.csv"

PREVIOUSLY_SCREENED = {
    "Abell 2390",
    "Abell 2537",
    "Abell 2667",
    "Abell 383",
    "Abell 611",
    "Abell 963",
    "MS 2137",
    "MACS J0326",
    "MACS J0417",
    "MACS J0949",
    "MACS J1427",
    "MACS J1206",
    "Abell S1063",
    "MACS J0416",
    "RXJ 1133",
    "Abell 1201",
    "SDSS J0100+1818",
    "RX J2129",
}

QUEUE_NAME = {
    "A2390": "Abell 2390",
    "A2537": "Abell 2537",
    "A2667": "Abell 2667",
    "A383": "Abell 383",
    "A611": "Abell 611",
    "A963": "Abell 963",
    "MS2137": "MS 2137",
    "MACS J0326": "MACS J0326",
    "MACS J0417": "MACS J0417",
    "MACS J0949": "MACS J0949",
    "MACS J1427": "MACS J1427",
}

DISTURBANCE = {
    "Abell 2104": (
        "disturbed",
        "Loubser et al. explicitly report significant central X-ray substructure and an overall elliptical appearance, and state that the cluster has not reached dynamical equilibrium.",
    ),
    "Abell 586": (
        "central_substructure",
        "Loubser et al. flag multiple nuclei/central substructure that may affect the kinematic profile.",
    ),
    "MS 0440+02": (
        "central_substructure",
        "Loubser et al. flag multiple nuclei/central substructure and a possibly affected central dispersion profile.",
    ),
    "MS 0906+11": (
        "central_substructure",
        "Loubser et al. flag multiple nuclei/central substructure that may affect the kinematic profile.",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def clean_name(value: str) -> str:
    value = re.sub(r"\$\^\{\\star\}\$", "", value)
    return " ".join(value.strip().split())


def number(value: str) -> float:
    match = re.search(r"--?[0-9]+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?", value)
    if match is None:
        raise ValueError(f"No number in {value!r}")
    return float(match.group().replace("--", "-"))


def value_error(value: str) -> tuple[float, float]:
    values = re.findall(r"--?[0-9]+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?", value)
    if len(values) < 2:
        raise ValueError(f"No value/error pair in {value!r}")
    return tuple(float(item.replace("--", "-")) for item in values[:2])


def parse_sample(text: str) -> pd.DataFrame:
    start = text.index("\\label{objects_BCGs}")
    end = text.index("\\end{table*}", start)
    lines = text[start:end].splitlines()
    sample = None
    rows = []
    for line in lines:
        if "{MENeaCS}" in line:
            sample = "MENeaCS"
        elif "{CCCP}" in line:
            sample = "CCCP"
        if not re.match(r"^(Abell|MS )", line.strip()):
            continue
        fields = [item.strip() for item in line.split("&")]
        if len(fields) < 7:
            continue
        rows.append(
            {
                "system": clean_name(fields[0]),
                "sample": sample,
                "redshift": number(fields[1]),
                "bcg_ra_hms": fields[2],
                "bcg_dec_dms": fields[3].replace("--", "-"),
                "spectroscopic_exposure_seconds": int(number(fields[4])),
                "slit_position_angle_deg": int(number(fields[5])),
                "gemini_semester": fields[6],
                "reported_spatial_bins": "11/13/15" if sample == "MENeaCS" else "9",
                "typical_one_sided_support_kpc": 15.0,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 32 or frame["system"].nunique() != 32:
        raise RuntimeError(f"Expected 32 unique BCG rows; parsed {len(frame)}")
    return frame


def parse_kinematics(text: str) -> pd.DataFrame:
    start = text.index("\\label{properties}")
    end = text.index("\\end{table}", start)
    rows = []
    for line in text[start:end].splitlines():
        if not re.match(r"^(Abell|MS )", line.strip()):
            continue
        fields = [item.strip() for item in line.split("&")]
        sigma, sigma_error = value_error(fields[2])
        slope, slope_error = value_error(fields[4])
        rows.append(
            {
                "system": clean_name(fields[0]),
                "central_sigma_km_s": sigma,
                "central_sigma_error_km_s": sigma_error,
                "dispersion_slope_eta": slope,
                "dispersion_slope_error": slope_error,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 32:
        raise RuntimeError(f"Expected 32 kinematic summary rows; parsed {len(frame)}")
    return frame


def profile_path(system: str) -> Path:
    if system.startswith("Abell "):
        code = "A" + system.split()[1]
    elif system == "MS 0906+11":
        code = "MS09"
    elif system == "MS 0440+02":
        code = "MS04"
    elif system == "MS 1455+22":
        code = "MS14"
    else:
        raise ValueError(system)
    return SOURCE_ROOT / "source" / f"{code}S.pdf"


def canonical_local_lens() -> tuple[dict, dict]:
    queue = pd.read_csv(CURRENT_QUEUE)
    queue["canonical"] = queue["system"].map(QUEUE_NAME)
    queue_lookup = {
        row["canonical"]: row.to_dict()
        for _, row in queue.dropna(subset=["canonical"]).iterrows()
    }
    rank = pd.read_csv(CURRENT_RANK)
    rank["canonical"] = rank["system"].map(QUEUE_NAME)
    rank_lookup = {
        row["canonical"]: row.to_dict()
        for _, row in rank.dropna(subset=["canonical"]).iterrows()
    }
    return queue_lookup, rank_lookup


def build_audit(ledger_path: Path, report_path: Path) -> dict:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8-sig"))
    for record in provenance["records"]:
        path = ROOT / record["local_path"]
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Provenance mismatch for {path}")

    text = TEX.read_text(encoding="utf-8")
    sample = parse_sample(text).merge(parse_kinematics(text), on="system", validate="one_to_one")
    queue_lookup, rank_lookup = canonical_local_lens()
    records = []
    for row in sample.to_dict(orient="records"):
        system = row["system"]
        profile = profile_path(system)
        if not profile.exists():
            raise RuntimeError(f"Missing profile plot for {system}: {profile}")
        queue = queue_lookup.get(system)
        rank = rank_lookup.get(system)
        local_lens = bool(queue and queue["local_observable_level_lens_positions"])
        strict_inputs = bool(queue and queue["local_position_redshift_likelihood_inputs"])
        disturbance_state, disturbance_note = DISTURBANCE.get(
            system,
            (
                "not_adjudicated_by_cycle3_source",
                "The Loubser source provides central/X-ray selection context but no uniform relaxed/disturbed classification for this system.",
            ),
        )
        strict_inner = int(rank["strict_inner_image_positions"]) if rank else None
        rank_bound = int(rank["structural_radial_rank_upper_bound"]) if rank else None
        overlap = system in PREVIOUSLY_SCREENED
        if overlap:
            exclusion = (
                "previously screened; Cycle 3 adds an independent Gemini radial-kinematics source but does not increment the unique-host count"
            )
        elif not local_lens:
            exclusion = (
                "new dynamics host, but no observable image-position/source-redshift likelihood is present in the Loubser package or current normalized lens ledger; complete baryonic profiles and covariance are also absent"
            )
        else:
            exclusion = (
                "observable lens inputs exist locally, but the figure-only radial kinematic profile, complete baryonic components, and covariance are not strict-ready"
            )
        records.append(
            {
                **row,
                "profile_plot": str(profile.relative_to(ROOT)).replace("\\", "/"),
                "profile_plot_sha256": sha256(profile),
                "radial_profile_machine_readable": False,
                "central_sigma_and_power_law_slope_machine_readable": True,
                "measurement_covariance_published": False,
                "bcg_light_available": "2MASS K-band magnitude only; promised r-band profile is not in this paper",
                "hot_gas_profile_available": False,
                "satellite_baryonic_profile_available": False,
                "disturbance_state": disturbance_state,
                "disturbance_note": disturbance_note,
                "local_observable_lens_positions": local_lens,
                "local_position_redshift_likelihood_inputs": strict_inputs,
                "strict_inner_image_positions": strict_inner,
                "structural_radial_rank_upper_bound": rank_bound,
                "new_unique_host_in_cycle3": not overlap,
                "structural_promotion_pass": bool(rank_bound is not None and rank_bound >= 3),
                "full_r1_ready": False,
                "exclusion_reason": exclusion,
            }
        )
    ledger = pd.DataFrame(records).sort_values(["redshift", "system"]).reset_index(drop=True)
    new_hosts = int(ledger["new_unique_host_in_cycle3"].sum())
    cumulative = 18 + new_hosts
    if new_hosts != 27 or cumulative != 45:
        raise RuntimeError(f"Expected 27 new and 45 cumulative hosts; got {new_hosts}, {cumulative}")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)
    report = {
        "audit_version": "R1A2-replacement-cycle3-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "The complete 32-BCG Loubser et al. spatial-kinematics sample was ingested and screened without any gravity residual, inferred dark-matter slope, or alternative-gravity result.",
        "source": {
            "citation": "Loubser et al. 2018, MNRAS 477, 335",
            "doi": "10.1093/mnras/sty498",
            "arxiv": "1802.07745",
            "provenance": str(PROVENANCE.relative_to(ROOT)).replace("\\", "/"),
            "source_tex_sha256": sha256(TEX),
        },
        "summary": {
            "source_bcg_hosts": len(ledger),
            "previously_screened_overlaps": int((~ledger["new_unique_host_in_cycle3"]).sum()),
            "cycle3_new_unique_hosts": new_hosts,
            "cumulative_unique_hosts_source_screened": cumulative,
            "inventory_target": 30,
            "inventory_boundary_reached": cumulative >= 30,
            "new_hosts_with_local_observable_lens_positions": int(
                ledger.loc[ledger["new_unique_host_in_cycle3"], "local_observable_lens_positions"].sum()
            ),
            "new_structural_promotions": int(
                ledger.loc[ledger["new_unique_host_in_cycle3"], "structural_promotion_pass"].sum()
            ),
            "strict_r1_ready_systems": int(ledger["full_r1_ready"].sum()),
        },
        "data_findings": {
            "dynamics": "All 32 hosts have spatially resolved Gemini long-slit profiles: CCCP uses nine bins and MENeaCS generally 11, 13, or 15, typically reaching 15 kpc on each side. The source archive contains per-object profile plots, while only central sigma and a power-law slope/error are tabulated numerically.",
            "baryons": "The paper tabulates 2MASS K-band magnitudes but not the promised r-band surface-brightness profiles, hot-gas radial profiles, satellite baryonic profiles, or their covariance.",
            "lensing": "Five overlaps already have local lens-observable workstreams. None of the 27 new hosts has an image-position/source-redshift likelihood in the Loubser package or current normalized project ledger; this is a concrete acquisition queue, not proof that no public lens catalog exists elsewhere.",
        },
        "decision": {
            "cycle3_status": "completed_named_32_bcg_sample_exhausted_inventory_boundary_reached",
            "host_count_gate": "passed",
            "strict_readiness_gate": "failed_zero_new_strict_ready",
            "next_action": "Stop expanding host count for its own sake. Prioritize lens-catalog searches for the new Loubser strong-lens candidates and complete raw/likelihood products per host; preserve figure-only dynamics and missing covariance as exact blockers.",
        },
        "outputs": {
            "candidate_ledger": str(ledger_path.relative_to(ROOT)).replace("\\", "/")
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ledger_path = ROOT / "data/derived/r1_replacement_cycle3_candidate_ledger.csv"
    report_path = ROOT / "results/r1_replacement_search_cycle3/report.json"
    print(json.dumps(build_audit(ledger_path, report_path), indent=2))


if __name__ == "__main__":
    main()
