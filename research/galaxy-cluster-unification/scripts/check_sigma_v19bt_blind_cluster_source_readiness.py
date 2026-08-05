from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bt_blind_cluster_source_readiness.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_url_is_safe(url: str, allowed_fragment: str, forbidden: list[str]) -> bool:
    parsed = urlparse(url)
    lowered = url.lower()
    return (
        parsed.scheme == "https"
        and parsed.netloc == "archive.stsci.edu"
        and parsed.path.startswith("/hlsps/sgas/")
        and allowed_fragment in parsed.path
        and parsed.path.endswith(".fits")
        and not any(fragment.lower() in lowered for fragment in forbidden)
    )


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    parent_path = ROOT / config["parent"]["path"]
    parent_actual = sha256(parent_path)
    parent = load_json(parent_path)
    parent_exact = parent_actual == config["parent"]["sha256"]

    candidates = config["candidate_source_metadata"]
    by_id = {row["id"]: row for row in candidates}
    parent_ids = {row["id"] for row in parent["metadata_shortlist"]}
    candidate_ids = set(by_id)

    mast = config["primary_sources"]["mast_sgas_hlsp"]
    all_urls = [url for row in candidates for url in row["hst"]["science_image_urls"]]
    urls_safe = bool(all_urls) and all(
        source_url_is_safe(
            url,
            mast["allowed_product_path_fragment"],
            mast["forbidden_product_path_fragments"],
        )
        for url in all_urls
    )
    head_audit = mast["source_image_HEAD_audit"]
    urls_live_at_freeze = (
        head_audit["urls_checked"] == len(all_urls)
        and head_audit["http_200_or_206"] == len(all_urls)
        and head_audit["total_content_length_bytes"] > 0
    )

    direct = [
        row
        for row in candidates
        if row["source_imaging_preflight"] == "passed_not_admitted"
    ]
    direct_ready = (
        len(direct) == config["source_imaging_preflight_gate"][
            "minimum_direct_HST_F160W_and_Chandra_systems"
        ]
        and all(row["hst"]["direct_source_only_hlsp"] for row in direct)
        and all(row["hst"]["has_F160W"] for row in direct)
        and all(
            any("f160w" in url.lower() for url in row["hst"]["science_image_urls"])
            for row in direct
        )
        and all(row["chandra"]["obsids"] for row in direct)
        and all(
            row["chandra"]["counts_within_R500"]
            >= config["source_imaging_preflight_gate"][
                "minimum_Chandra_counts_within_R500"
            ]
            for row in direct
        )
    )

    direct_states = [row["state_side"] for row in direct]
    direct_masses = [row["M500_1e14_solar_nominal"] for row in direct]
    broad_range = (
        direct_states.count("relaxed")
        >= config["source_imaging_preflight_gate"]["minimum_relaxed_side"]
        and direct_states.count("disturbed")
        >= config["source_imaging_preflight_gate"]["minimum_disturbed_side"]
        and max(direct_masses) / min(direct_masses)
        >= config["source_imaging_preflight_gate"]["minimum_mass_span_ratio"]
    )

    reserves = [
        row
        for row in candidates
        if row["source_imaging_preflight"] == "reserve_not_admitted"
    ]
    reserve_blockers = (
        len(reserves) == 2
        and set(row["id"] for row in reserves)
        == {"SDSS_J1002+2031", "SDSS_J1226+2149"}
        and all(row["remaining_source_blockers"] for row in reserves)
        and any(
            "not well constrained" in blocker
            for blocker in by_id["SDSS_J1002+2031"]["remaining_source_blockers"]
        )
        and not by_id["SDSS_J1226+2149"]["hst"]["has_F160W"]
    )

    no_completeness_overclaim = (
        not any(row["complete_baryon_model_ready"] for row in candidates)
        and not config["source_imaging_preflight_gate"]["complete_baryon_model_claimed"]
        and not config["source_imaging_preflight_gate"]["final_holdout_admission_claimed"]
        and all(row["source_imaging_preflight"] != "admitted" for row in candidates)
    )

    boundary = config["access_boundary"]
    authorization = config["authorization"]
    target_safe = (
        boundary["temporary_mixed_manuscript_container_removed"]
        and not boundary["raw_lens_coordinate_values_ingested"]
        and not boundary["lens_map_downloaded"]
        and not boundary["gravity_formula_scored"]
        and not boundary["final_six_selected"]
        and not authorization["download_source_images_or_events_now"]
        and not authorization["open_raw_lensing_coordinates"]
        and not authorization["download_lens_maps"]
    )

    no_selection = (
        not authorization["select_final_six"]
        and not authorization["fit_action_or_constant"]
        and not authorization["change_gravity_formula"]
        and not authorization["perform_detailed_solar_optimization"]
    )

    gates = {
        "parent_hash_exact": parent_exact,
        "candidate_set_matches_v19bh": candidate_ids == parent_ids and len(candidates) == len(by_id),
        "source_urls_are_strictly_whitelisted": urls_safe,
        "source_urls_were_live_at_freeze": urls_live_at_freeze,
        "six_balanced_direct_source_imaging_systems_exist": direct_ready,
        "mass_and_state_range_is_broad": broad_range,
        "reserve_blockers_are_explicit": reserve_blockers,
        "complete_baryons_and_holdout_admission_not_claimed": no_completeness_overclaim,
        "raw_lensing_and_gravity_payload_remain_unused": target_safe,
        "no_theory_constant_holdout_or_solar_selection": no_selection,
    }
    if set(gates) != set(config["required_gates"]):
        raise ValueError("implemented and declared gate names differ")
    if not all(config["required_gates"].values()):
        raise ValueError("every declared gate must be mandatory")

    report = {
        "protocol_version": config["protocol_version"],
        "status": "completed_blind_cluster_source_readiness_checkpoint",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "parent_audit": {
            "path": config["parent"]["path"],
            "expected_sha256": config["parent"]["sha256"],
            "actual_sha256": parent_actual,
            "exact": parent_exact,
        },
        "source_imaging_preflight": {
            "shortlist_systems": len(candidates),
            "direct_HST_F160W_plus_Chandra_systems": len(direct),
            "direct_relaxed_side": direct_states.count("relaxed"),
            "direct_disturbed_side": direct_states.count("disturbed"),
            "direct_mass_range_1e14_solar": [min(direct_masses), max(direct_masses)],
            "direct_mass_span_ratio": max(direct_masses) / min(direct_masses),
            "reserve_systems": [row["id"] for row in reserves],
            "complete_baryon_models": 0,
            "admitted_holdouts": 0,
            "final_six_selected": False,
        },
        "systems": [
            {
                "id": row["id"],
                "state_side": row["state_side"],
                "published_classification": row["published_classification"],
                "M500_1e14_solar_nominal": row["M500_1e14_solar_nominal"],
                "chandra_obsids": row["chandra"]["obsids"],
                "chandra_counts_within_R500": row["chandra"]["counts_within_R500"],
                "HST_filters": row["hst"]["filters"],
                "direct_source_only_HLSP": row["hst"]["direct_source_only_hlsp"],
                "source_imaging_preflight": row["source_imaging_preflight"],
                "complete_baryon_model_ready": row["complete_baryon_model_ready"],
                "remaining_source_blockers": row["remaining_source_blockers"],
            }
            for row in candidates
        ],
        "access_boundary_audit": boundary,
        "source_image_HEAD_audit": head_audit,
        "authorization_audit": authorization,
        "next_source_only_work": config["next_source_only_work"],
        "gate_results": gates,
        "decision": (
            "passed_source_imaging_preflight_not_holdout_admission"
            if all(gates.values())
            else "failed_source_imaging_preflight"
        ),
        "claim_boundary": config["claim_boundary"],
        "implementation": config["implementation"],
    }
    return report


def main() -> None:
    report = build_report()
    output = ROOT / load_json(DEFAULT_CONFIG)["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "output": output.relative_to(ROOT).as_posix(),
                "source_imaging_preflight": report["source_imaging_preflight"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["decision"] != "passed_source_imaging_preflight_not_holdout_admission":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
