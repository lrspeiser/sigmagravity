#!/usr/bin/env python3
"""Raw RXJ2129 image-position check for locked sensitivity variants."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sigma_field_exploration import run_diagnostic_lensing  # noqa: E402
from run_unbounded_running_full_test import json_safe  # noqa: E402
from run_vector_completion_full_test import raw_lensing_profile  # noqa: E402


def main() -> None:
    config_path = ROOT / "configs/unbounded_running_sensitivity_raw_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_raw_scores":
        raise RuntimeError("raw protocol was not frozen")
    sensitivity = json.loads((ROOT / protocol["inputs"]["sensitivity_report"]).read_text())
    raw_protocol = json.loads((ROOT / protocol["inputs"]["raw_solver_protocol"]).read_text())
    reference = json.loads((ROOT / protocol["inputs"]["raw_reference_report"]).read_text())
    variants = {row["variant"]: row for row in sensitivity["variants"]}
    predictions = []
    scores = {}
    for name in protocol["variants"]:
        row = variants[name]
        parameters = list(row["parameters"].values())
        print(f"raw RXJ2129 locked variant={name}", flush=True)
        profile = raw_lensing_profile(row["model"], parameters, raw_protocol)
        table, summary = run_diagnostic_lensing(pd.Series(row["parameters"]), raw_protocol, profile)
        table.insert(0, "variant", name)
        predictions.append(table)
        scores[name] = summary
    compact = reference["raw_lensing"]["compact_halo_reference_heldout_RMS_arcsec"]
    ranking = sorted(scores, key=lambda name: scores[name]["heldout"]["exact_radial_RMS_arcsec"])
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed post-transfer exploratory raw lensing check",
        "selection_warning": protocol["selection_warning"],
        "protocol": {"path": str(config_path.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "cluster": "RXJ2129",
        "reference_compact_halo_heldout_RMS_arcsec": compact,
        "scores": scores,
        "ranking": ranking,
    }
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(ROOT / protocol["outputs"]["predictions"], index=False)
    lines = ["# Raw RXJ2129 sensitivity check", "", protocol["selection_warning"], "", "| rank | variant | training RMS (arcsec) | held-out RMS (arcsec) | reduced chi2 | vs compact halo |", "|---:|---|---:|---:|---:|---:|"]
    for index, name in enumerate(ranking, 1):
        result = scores[name]
        heldout = result["heldout"]["exact_radial_RMS_arcsec"]
        lines.append(f"| {index} | {name} | {result['training']['exact_radial_RMS_arcsec']:.3f} | {heldout:.3f} | {result['heldout']['reduced_chi2']:.2f} | {heldout / compact:.3f}x |")
    lines += ["", f"Compact-halo held-out reference: **{compact:.3f} arcsec**."]
    (ROOT / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((ROOT / protocol["outputs"]["summary"]).read_text())


if __name__ == "__main__":
    main()
