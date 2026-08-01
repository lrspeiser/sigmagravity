#!/usr/bin/env python3
"""Repeat selected hybrid raw-lensing fits with stronger geometry optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import (  # noqa: E402
    expand_variants,
    json_safe,
    run_raw_lensing,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/reopened_hybrid_raw_robustness_protocol.json",
        help="Protocol path relative to the research root",
    )
    arguments = parser.parse_args()
    config_path = ROOT / arguments.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_raw_robustness_scores":
        raise RuntimeError("raw robustness protocol was not frozen")
    source_protocol_path = ROOT / protocol["source_protocol"]
    source_report_path = ROOT / protocol["source_report"]
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    variant_lookup = {
        row["name"]: row for row in expand_variants(source_protocol)
    }
    requested = protocol["selected_variants"]
    if requested == "all":
        selected_names = list(variant_lookup)
    elif isinstance(requested, list):
        selected_names = requested
    else:
        raise ValueError("selected_variants must be a list or the string 'all'")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("selected_variants must not contain duplicates")
    missing = [name for name in selected_names if name not in variant_lookup]
    if missing:
        raise ValueError(f"selected variants are absent from source protocol: {missing}")
    selected = [variant_lookup[name] for name in selected_names]
    parameters = {
        name: [
            source_report["results"][name]["full_fit_parameters"][parameter]
            for parameter in source_protocol["universal_parameters"]["names"]
        ]
        for name in selected_names
    }
    source_protocol["raw_lensing"]["geometry_multi_starts"] = int(
        protocol["geometry_multi_starts"]
    )
    source_protocol["raw_lensing"]["maximum_function_evaluations"] = int(
        protocol["maximum_function_evaluations"]
    )
    bridge = pd.read_csv(
        ROOT / source_protocol["inputs"]["bridge_sample"]
    )
    raw, predictions, geometry, profiles = run_raw_lensing(
        selected, parameters, bridge, source_protocol
    )
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(ROOT / protocol["outputs"]["predictions"], index=False)
    geometry.to_csv(ROOT / protocol["outputs"]["geometry"], index=False)
    profiles.to_csv(ROOT / protocol["outputs"]["profiles"], index=False)
    comparisons = {}
    for name in selected_names:
        earlier = source_report["results"][name]["raw_lensing"]
        robust = raw["aggregate"][name]
        prior_rms = earlier["equal_system_radial_RMS_arcsec"]
        robust_rms = robust["equal_system_radial_RMS_arcsec"]
        comparisons[name] = {
            "two_start": earlier,
            "eight_start": robust,
            "RMS_change_arcsec": (
                robust_rms - prior_rms
                if robust_rms is not None and prior_rms is not None
                else None
            ),
            "RMS_fractional_change": (
                robust_rms / prior_rms - 1.0
                if robust_rms is not None
                and prior_rms is not None
                and prior_rms != 0.0
                else None
            ),
        }
    ranked = sorted(
        selected_names,
        key=lambda name: (
            not raw["aggregate"][name]["all_roots_converged"],
            raw["aggregate"][name]["equal_system_radial_RMS_arcsec"]
            if raw["aggregate"][name]["equal_system_radial_RMS_arcsec"]
            is not None
            else float("inf"),
        ),
    )
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed eight-start raw-lensing robustness replay",
        "protocol_sha256": sha256(config_path),
        "source_protocol_sha256": sha256(source_protocol_path),
        "source_report_sha256": sha256(source_report_path),
        "selected_variants": selected_names,
        "gravity_parameters_fit_to_raw_lensing": 0,
        "ranking": ranked,
        "comparisons": comparisons,
        "per_system": raw["per_system"],
        "claim_boundary": protocol["claim_boundary"],
    }
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Reopened hybrid raw-lensing robustness",
        "",
        "Universal gravity constants were kept fixed. Only image geometry was "
        "reoptimized with eight starts.",
        "",
        "| rank | variant | 2-start RMS | 8-start RMS | change | roots |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for rank, name in enumerate(ranked, 1):
        row = comparisons[name]
        robust = row["eight_start"]
        lines.append(
            f"| {rank} | {name} | "
            f"{row['two_start']['equal_system_radial_RMS_arcsec']:.3f} | "
            f"{robust['equal_system_radial_RMS_arcsec']:.3f} | "
            f"{row['RMS_change_arcsec']:+.3f} | "
            f"{robust['all_roots_converged']} |"
        )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ranking": ranked, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
