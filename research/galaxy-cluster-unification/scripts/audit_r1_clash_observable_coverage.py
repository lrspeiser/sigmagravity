#!/usr/bin/env python3
"""Consolidate local observable-level lens coverage for the 20 CLASH targets."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_clash_observable_acquisition_targets.json"
OUTPUT_PATH = ROOT / "data/derived/r1_clash_observable_acquisition_ledger.csv"
REPORT_PATH = ROOT / "results/r1_clash_observable_coverage/report.json"


def blank_records(targets: list[str]) -> dict[str, dict]:
    return {
        target: {
            "system": target,
            "raw_or_likelihood_catalog_acquired": False,
            "normalized_position_likelihood_ready": False,
            "observable_position_rows": 0,
            "strict_position_redshift_rows": 0,
            "independent_source_families": 0,
            "declared_position_error_or_covariance": False,
            "rerunnable_model_chain_local": False,
            "metric_neutral_weyl_posterior_local": False,
            "primary_local_evidence": "",
            "evidence_tier": "not_acquired",
            "shortfall_or_note": "No local raw/likelihood-level source has yet been normalized in the consolidated ledger.",
            "gravity_target_used": False,
        }
        for target in targets
    }


def set_record(records: dict[str, dict], target: str, **values) -> None:
    if target not in records:
        raise KeyError(target)
    records[target].update(values)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_audit(
    output_path: Path = OUTPUT_PATH, report_path: Path = REPORT_PATH
) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = blank_records(config["target_systems"])

    legacy = pd.read_csv(ROOT / "data/derived/r1_strong_lens_image_observables.csv")
    for source_name, target in {"A383": "A383", "A611": "A611", "MS2137": "MS2137"}.items():
        rows = legacy.loc[legacy["system"] == source_name]
        strict = rows.loc[rows["alternative_metric_likelihood_ready"].astype(bool)]
        set_record(
            records,
            target,
            raw_or_likelihood_catalog_acquired=len(rows) > 0,
            normalized_position_likelihood_ready=len(strict) > 0,
            observable_position_rows=int(rows["observable_level_image_position"].astype(bool).sum()),
            strict_position_redshift_rows=int(len(strict)),
            independent_source_families=int(strict["source_family"].nunique()),
            declared_position_error_or_covariance=bool(strict["image_position_sigma_arcsec"].notna().all()),
            primary_local_evidence="data/derived/r1_strong_lens_image_observables.csv",
            evidence_tier="published_image_plane_likelihood",
            shortfall_or_note="Published independent diagonal image-plane likelihood; no cross-image systematic covariance.",
        )

    cycle1 = pd.read_csv(ROOT / "data/derived/r1_replacement_cycle1_image_support.csv")
    for source_name, target in {"MACS J1206": "MACS1206", "Abell S1063": "RXJ2248"}.items():
        rows = cycle1.loc[cycle1["system"] == source_name]
        observable = rows.loc[rows["observable_position"].astype(bool)]
        strict = rows.loc[rows["strict_position_redshift_input"].astype(bool)]
        set_record(
            records,
            target,
            raw_or_likelihood_catalog_acquired=len(observable) > 0,
            normalized_position_likelihood_ready=len(strict) > 0,
            observable_position_rows=int(len(observable)),
            strict_position_redshift_rows=int(len(strict)),
            independent_source_families=int(strict["family_id"].nunique()),
            declared_position_error_or_covariance=True,
            primary_local_evidence="data/derived/r1_replacement_cycle1_image_support.csv",
            evidence_tier="published_spectroscopic_image_likelihood",
            shortfall_or_note="Spectroscopic image positions are normalized; the complete nuisance posterior is not local.",
        )

    extension = pd.read_csv(ROOT / "data/derived/r1_replacement_cycle1_extension_images.csv")
    macs0416 = extension.loc[extension["system"] == "MACS J0416"]
    set_record(
        records,
        "MACS0416",
        raw_or_likelihood_catalog_acquired=len(macs0416) > 0,
        normalized_position_likelihood_ready=False,
        observable_position_rows=int(macs0416["observable_position"].astype(bool).sum()),
        strict_position_redshift_rows=0,
        independent_source_families=0,
        declared_position_error_or_covariance=False,
        primary_local_evidence="data/derived/r1_replacement_cycle1_extension_images.csv",
        evidence_tier="spectroscopic_image_catalog_position_error_not_consolidated",
        shortfall_or_note="All 237 positions are spectroscopic, but a declared coordinate-error model has not yet been consolidated into this ledger.",
    )

    a2261 = pd.read_csv(ROOT / "data/derived/r1_a2261_lens_observables.csv")
    set_record(
        records,
        "A2261",
        raw_or_likelihood_catalog_acquired=len(a2261) == 30,
        normalized_position_likelihood_ready=False,
        observable_position_rows=int(len(a2261)),
        strict_position_redshift_rows=0,
        independent_source_families=int(a2261["family_id"].nunique()),
        declared_position_error_or_covariance=False,
        primary_local_evidence="data/derived/r1_a2261_lens_observables.csv",
        evidence_tier="measured_image_catalog_without_position_covariance",
        shortfall_or_note="Thirty measured positions and independent family redshift summaries are local; published position errors/covariance are absent.",
    )

    camin = pd.read_csv(ROOT / "data/derived/r1_clash_caminha2019_image_observables.csv")
    camin_summary = pd.read_csv(ROOT / "data/derived/r1_clash_caminha2019_system_summary.csv")
    camin_map = {
        "MACS J0329.7-0211": "MACS0329",
        "MACS J0429.6-0253": "MACS0429",
        "MACS J1115.9+0129": "MACS1115",
        "MACS J1931.8-2635": "MACS1931",
        "RX J1347.5-1145": "RXJ1347",
        "RX J2129.7+0005": "RXJ2129",
    }
    for source_name, target in camin_map.items():
        rows = camin.loc[camin["system"] == source_name]
        strict = rows.loc[rows["metric_neutral_likelihood_row"].astype(bool)]
        summary = camin_summary.loc[camin_summary["system"] == source_name].iloc[0]
        set_record(
            records,
            target,
            raw_or_likelihood_catalog_acquired=len(rows) > 0,
            normalized_position_likelihood_ready=len(strict) > 0,
            observable_position_rows=int(len(rows)),
            strict_position_redshift_rows=int(len(strict)),
            independent_source_families=int(strict["family_id"].nunique()),
            declared_position_error_or_covariance=bool(summary["declared_position_errors"]),
            rerunnable_model_chain_local=bool(summary["rerunnable_lenstool_package_acquired"]),
            metric_neutral_weyl_posterior_local=False,
            primary_local_evidence="data/derived/r1_clash_caminha2019_image_observables.csv",
            evidence_tier="spectroscopic_image_likelihood_plus_metric_dependent_model_chain",
            shortfall_or_note="Observable coordinate likelihood and Lenstool chain are local; the chain is metric-dependent and is not a Weyl-response posterior.",
        )

    zitrin = pd.read_csv(ROOT / "data/derived/r1_clash_zitrin2015_system_summary.csv")
    for _, row in zitrin.iterrows():
        target = str(row["system"])
        acquired = bool(row["raw_observable_catalog_acquired"])
        set_record(
            records,
            target,
            raw_or_likelihood_catalog_acquired=acquired,
            normalized_position_likelihood_ready=False,
            observable_position_rows=int(row["measured_position_rows"]),
            strict_position_redshift_rows=int(row["metric_neutral_observable_rows"]),
            independent_source_families=int(row["independently_redshift_anchored_families"]),
            declared_position_error_or_covariance=False,
            rerunnable_model_chain_local=False,
            metric_neutral_weyl_posterior_local=False,
            primary_local_evidence="data/derived/r1_clash_zitrin2015_image_observables.csv",
            evidence_tier=(
                "published_measured_image_catalog_without_metric_neutral_covariance"
                if acquired
                else "published_candidate_only_catalog_primary_source_hard_shortfall"
            ),
            shortfall_or_note=(
                "Measured coordinates and independent family-redshift summaries are local; archived sigma_pos is a model-dependent LTM optimization scale, not astrometric covariance."
                if acquired
                else "All three published RXJ1532 entries are candidate/ambiguous; the official CLASH release independently says the sole multiple-image family is unconfirmed."
            ),
        )

    ledger = pd.DataFrame(records.values())
    ledger["in_frozen_next_queue"] = ledger["system"].isin(config["frozen_next_queue"])
    ledger = ledger.sort_values("system").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_path, index=False)
    acquired = int(ledger["raw_or_likelihood_catalog_acquired"].sum())
    likelihood = int(ledger["normalized_position_likelihood_ready"].sum())
    missing = ledger.loc[~ledger["raw_or_likelihood_catalog_acquired"], "system"].tolist()
    report = {
        "report_version": "R1-CLASH-observable-coverage-0.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blind": True,
        "target_systems": int(len(ledger)),
        "raw_or_likelihood_catalogs_acquired": acquired,
        "normalized_position_likelihoods_ready": likelihood,
        "rerunnable_model_chains_local": int(ledger["rerunnable_model_chain_local"].sum()),
        "metric_neutral_weyl_posteriors_local": int(ledger["metric_neutral_weyl_posterior_local"].sum()),
        "remaining_systems": len(missing),
        "frozen_next_queue": missing,
        "coverage_gate_passed": acquired == len(ledger),
        "primary_source_hard_shortfall_systems": ["RXJ1532"],
        "resolved_catalog_or_shortfall_dispositions": acquired + 1,
        "coverage_or_hard_shortfall_gate_passed": acquired + 1 == len(ledger),
        "completed_cycle_outcome": config["numeric_outcomes"]["completed_cycle_outcome"],
        "next_stage_success": config["numeric_outcomes"]["next_stage_success"],
        "next_stage_rethink_trigger": config["numeric_outcomes"]["next_stage_rethink_trigger"],
        "authorization": {
            "continue_observable_acquisition": True,
            "infer_weyl_response": False,
            "fit_gravity_response": False,
            "fit_new_force_or_action": False,
        },
        "output": display_path(output_path),
    }
    if acquired != config["numeric_outcomes"]["current_raw_or_likelihood_catalogs"]:
        raise RuntimeError(f"Frozen coverage count mismatch: expected 19, measured {acquired}")
    if missing != config["frozen_next_queue"]:
        raise RuntimeError(f"Frozen next queue mismatch: {missing}")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    arguments = parser.parse_args()
    print(json.dumps(build_audit(arguments.output, arguments.report), indent=2))
