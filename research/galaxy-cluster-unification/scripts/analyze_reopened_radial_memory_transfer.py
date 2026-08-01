#!/usr/bin/env python3
"""Audit radial-memory leverage before refitting or raw-lensing optimization."""

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

from run_cpr0_accept_clash_bridge import domain_metrics  # noqa: E402
from run_reopened_hybrid_sensitivity import (  # noqa: E402
    expand_variants,
    json_safe,
    predict_log_acceleration,
    sparc_scores,
)
from voidscreen.reopened_hybrids import solar_system_diagnostics  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/reopened_hybrid_radial_memory_protocol.json"
OUTPUT = ROOT / "results/reopened_radial_memory_transfer_audit"
BASES = {
    "unsaturated": {
        "report": "results/reopened_hybrid_channel_saturation/report.json",
        "variant": "baseline_unsaturated:screen_power=1.5",
    },
    "dual": {
        "report": "results/reopened_hybrid_channel_saturation_fine/report.json",
        "variant": "rg_2_sigma_fine:sigma_saturation_ceiling=1.5",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    bridge = pd.read_csv(ROOT / protocol["inputs"]["bridge_sample"])
    sparc = pd.read_csv(ROOT / protocol["inputs"]["sparc_outer_sample"])
    parameters = {}
    source_hashes = {}
    for base, source in BASES.items():
        path = ROOT / source["report"]
        report = json.loads(path.read_text(encoding="utf-8"))
        row = report["results"][source["variant"]]
        parameters[base] = [
            row["full_fit_parameters"][name]
            for name in protocol["universal_parameters"]["names"]
        ]
        source_hashes[base] = sha256(path)

    rows = []
    for variant in expand_variants(protocol):
        base = (
            "unsaturated"
            if variant["family"].startswith("unsaturated_")
            else "dual"
        )
        values = parameters[base]
        sparc_metric, _ = sparc_scores(
            sparc,
            values,
            variant,
            protocol["shared_constants"],
        )
        bridge_prediction = predict_log_acceleration(
            bridge,
            values,
            variant,
            protocol["shared_constants"],
        )
        bridge_metric = domain_metrics(bridge, bridge_prediction)
        solar = solar_system_diagnostics(
            values,
            variant["settings"],
            cassini_fractional_limit=float(
                protocol["solar_tests"]["cassini_fractional_force_proxy_limit"]
            ),
            interplanetary_density_g_cm3=float(
                protocol["shared_constants"]["interplanetary_density_g_cm3"]
            ),
            acceleration_screen_m_s2=float(
                protocol["shared_constants"]["acceleration_screen_m_s2"]
            ),
        )
        rows.append(
            {
                "base": base,
                "variant": variant["name"],
                "family": variant["family"],
                "fixed_name": variant["fixed_name"],
                "fixed_value": variant["fixed_value"],
                "SPARC_outer_RMSE_km_s": sparc_metric["RMSE_km_s"],
                "bridge_RMSE_dex": bridge_metric["equal_domain_RMSE_dex"],
                "solar_maximum_fractional_change": solar[
                    "maximum_fractional_change_limb_to_Saturn"
                ],
                "Cassini_proxy_pass": solar["Cassini_proxy_pass"],
            }
        )
    scores = pd.DataFrame(rows)
    best = {
        base: (
            block.sort_values(
                ["SPARC_outer_RMSE_km_s", "bridge_RMSE_dex"]
            )
            .iloc[0]
            .to_dict()
        )
        for base, block in scores.groupby("base", sort=False)
    }
    report = {
        "report_version": "REOPENED-RADIAL-MEMORY-TRANSFER-AUDIT-0.1.0",
        "status": "completed fixed-parameter transfer audit",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_report_hashes": source_hashes,
        "rows": len(scores),
        "best_fixed_parameter_galaxy_setting_by_base": best,
        "claim_boundary": [
            "Universal parameters are copied from preceding local-law fits and are not refitted in this audit.",
            "Raw lensing is deliberately absent; the frozen full sweep performs that test.",
            "The audit only selects useful memory ranges and does not constitute a cross-domain score.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Radial-memory fixed-parameter transfer audit",
        "",
        "This development audit holds the prior gravity parameters fixed and "
        "changes only the radial-memory structure.",
        "",
        "| base | best setting | galaxy RMSE | bridge RMSE | Solar max |",
        "|---|---|---:|---:|---:|",
    ]
    for base, row in best.items():
        lines.append(
            f"| {base} | {row['variant']} | "
            f"{row['SPARC_outer_RMSE_km_s']:.3f} km/s | "
            f"{row['bridge_RMSE_dex']:.3f} dex | "
            f"{row['solar_maximum_fractional_change']:.3e} |"
        )
    lines += [
        "",
        "These are range-selection results. The full sweep refits the four "
        "universal parameters and performs raw cluster lensing.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
