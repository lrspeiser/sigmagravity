#!/usr/bin/env python3
"""Screen generalized radial-memory carriers before full raw-lensing fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cpr0_accept_clash_bridge import domain_metrics  # noqa: E402
from run_reopened_hybrid_sensitivity import (  # noqa: E402
    json_safe,
    predict_log_acceleration,
    sparc_scores,
)
from voidscreen.reopened_hybrids import (  # noqa: E402
    mercury_precession_mas_per_century,
    solar_system_diagnostics,
)


PROTOCOL_PATH = (
    ROOT / "configs/reopened_hybrid_memory_carrier_audit_protocol.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_audit(protocol: dict) -> list[dict]:
    rows = []
    common = protocol["common_memory"]
    for base_name, base in protocol["base_sources"].items():
        for design_name, design in protocol["audit_designs"].items():
            fixed_name = design["fixed_name"]
            for value in design["values"]:
                settings = dict(base["settings"])
                settings.update(common)
                settings.update(design.get("overrides", {}))
                settings[fixed_name] = float(value)
                rows.append(
                    {
                        "name": (
                            f"{base_name}_{design_name}:"
                            f"{fixed_name}={float(value):g}"
                        ),
                        "base": base_name,
                        "family": f"{base_name}_{design_name}",
                        "fixed_name": fixed_name,
                        "fixed_value": float(value),
                        "settings": settings,
                    }
                )
    if len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError("audit variant names must be unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROTOCOL_PATH.relative_to(ROOT)),
        help="Protocol path relative to the research root",
    )
    arguments = parser.parse_args()
    protocol_path = ROOT / arguments.config
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_fixed_parameter_scores":
        raise RuntimeError("memory-carrier audit protocol is not frozen")
    bridge = pd.read_csv(ROOT / protocol["inputs"]["bridge_sample"])
    sparc = pd.read_csv(ROOT / protocol["inputs"]["sparc_outer_sample"])
    parameters = {}
    source_hashes = {}
    for base_name, source in protocol["base_sources"].items():
        path = ROOT / source["report"]
        report = json.loads(path.read_text(encoding="utf-8"))
        fitted = report["results"][source["variant"]]["full_fit_parameters"]
        parameters[base_name] = [
            fitted[name] for name in protocol["universal_parameters"]["names"]
        ]
        source_hashes[base_name] = sha256(path)

    rows = []
    for index, variant in enumerate(expand_audit(protocol), 1):
        print(
            f"{index:03d} fixed transfer {variant['name']}",
            flush=True,
        )
        values = parameters[variant["base"]]
        try:
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
                    protocol["solar_tests"][
                        "cassini_fractional_force_proxy_limit"
                    ]
                ),
                interplanetary_density_g_cm3=float(
                    protocol["shared_constants"][
                        "interplanetary_density_g_cm3"
                    ]
                ),
                acceleration_screen_m_s2=float(
                    protocol["shared_constants"][
                        "acceleration_screen_m_s2"
                    ]
                ),
            )
            mercury = mercury_precession_mas_per_century(
                values,
                variant["settings"],
                interplanetary_density_g_cm3=float(
                    protocol["shared_constants"][
                        "interplanetary_density_g_cm3"
                    ]
                ),
                acceleration_screen_m_s2=float(
                    protocol["shared_constants"][
                        "acceleration_screen_m_s2"
                    ]
                ),
                quadrature_points=int(
                    protocol["solar_tests"][
                        "audit_mercury_quadrature_points"
                    ]
                ),
            )
            earth_pass = (
                abs(solar["Earth_orbit_fractional_change"])
                <= float(
                    protocol["solar_tests"][
                        "earth_orbit_fractional_change_max"
                    ]
                )
            )
            mercury_pass = (
                abs(mercury)
                <= float(
                    protocol["solar_tests"][
                        "mercury_supplementary_precession_absolute_max_mas_per_century"
                    ]
                )
            )
            galaxy_error = sparc_metric["RMSE_km_s"]
            bridge_error = bridge_metric["equal_domain_RMSE_dex"]
            row = {
                **variant,
                "settings": json.dumps(variant["settings"], sort_keys=True),
                "valid": True,
                "error": "",
                "SPARC_outer_RMSE_km_s": galaxy_error,
                "bridge_RMSE_dex": bridge_error,
                "solar_maximum_fractional_change": solar[
                    "maximum_fractional_change_limb_to_Saturn"
                ],
                "Earth_orbit_fractional_change": solar[
                    "Earth_orbit_fractional_change"
                ],
                "Mercury_precession_mas_per_century": mercury,
                "Cassini_proxy_pass": solar["Cassini_proxy_pass"],
                "Earth_pass": earth_pass,
                "Mercury_pass": mercury_pass,
                "solar_all_pass": (
                    solar["Cassini_proxy_pass"] and earth_pass and mercury_pass
                ),
                "galaxy_ratio_to_RAR": (
                    galaxy_error
                    / protocol["references"][
                        "SPARC_fixed_RAR_outer_RMSE_km_s"
                    ]
                ),
                "bridge_ratio_to_target": (
                    bridge_error
                    / protocol["references"]["bridge_target_RMSE_dex"]
                ),
                "audit_worst_reference_ratio": max(
                    galaxy_error
                    / protocol["references"][
                        "SPARC_fixed_RAR_outer_RMSE_km_s"
                    ],
                    bridge_error
                    / protocol["references"]["bridge_target_RMSE_dex"],
                ),
            }
        except (FloatingPointError, OverflowError, ValueError) as error:
            row = {
                **variant,
                "settings": json.dumps(variant["settings"], sort_keys=True),
                "valid": False,
                "error": str(error),
                "SPARC_outer_RMSE_km_s": math.nan,
                "bridge_RMSE_dex": math.nan,
                "solar_maximum_fractional_change": math.nan,
                "Earth_orbit_fractional_change": math.nan,
                "Mercury_precession_mas_per_century": math.nan,
                "Cassini_proxy_pass": False,
                "Earth_pass": False,
                "Mercury_pass": False,
                "solar_all_pass": False,
                "galaxy_ratio_to_RAR": math.nan,
                "bridge_ratio_to_target": math.nan,
                "audit_worst_reference_ratio": math.nan,
            }
        rows.append(row)

    scores = pd.DataFrame(rows)
    impacts = []
    for family, block in scores[scores.valid].groupby("family", sort=False):
        for metric in [
            "SPARC_outer_RMSE_km_s",
            "bridge_RMSE_dex",
            "solar_maximum_fractional_change",
            "Mercury_precession_mas_per_century",
            "audit_worst_reference_ratio",
        ]:
            values = block[metric].to_numpy(float)
            impacts.append(
                {
                    "family": family,
                    "fixed_name": block.fixed_name.iloc[0],
                    "metric": metric,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "absolute_span": float(np.max(values) - np.min(values)),
                }
            )
    impacts = pd.DataFrame(impacts)
    eligible = scores[scores.valid & scores.solar_all_pass].copy()
    best_by_base = {
        base: (
            block.sort_values(
                ["audit_worst_reference_ratio", "bridge_RMSE_dex"]
            )
            .iloc[0]
            .to_dict()
        )
        for base, block in eligible.groupby("base", sort=False)
    }
    best_by_carrier = {}
    for label, mask in {
        "fractional": (
            eligible.settings.str.contains(
                '"radial_memory_gbar_power": 0.0'
            )
            & eligible.settings.str.contains(
                '"radial_memory_radius_power": 0.0'
            )
        ),
        "acceleration": (
            eligible.settings.str.contains(
                '"radial_memory_gbar_power": 1.0'
            )
            & eligible.settings.str.contains(
                '"radial_memory_radius_power": 0.0'
            )
        ),
        "speed_squared": (
            eligible.settings.str.contains(
                '"radial_memory_gbar_power": 1.0'
            )
            & eligible.settings.str.contains(
                '"radial_memory_radius_power": 1.0'
            )
        ),
    }.items():
        block = eligible[mask]
        if len(block):
            best_by_carrier[label] = (
                block.sort_values(
                    ["audit_worst_reference_ratio", "bridge_RMSE_dex"]
                )
                .iloc[0]
                .to_dict()
            )

    output = ROOT / protocol["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(ROOT / protocol["outputs"]["scores"], index=False)
    impacts.to_csv(ROOT / protocol["outputs"]["family_impacts"], index=False)
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed fixed-parameter memory-carrier transfer audit",
        "protocol_sha256": sha256(protocol_path),
        "source_report_hashes": source_hashes,
        "rows": len(scores),
        "valid_rows": int(scores.valid.sum()),
        "solar_valid_rows": int((scores.valid & scores.solar_all_pass).sum()),
        "best_by_base": best_by_base,
        "best_by_named_carrier": best_by_carrier,
        "claim_boundary": protocol["claim_boundary"],
    }
    output.write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Generalized radial-memory carrier audit",
        "",
        f"- Scored variants: **{len(scores)}**",
        f"- Valid variants: **{int(scores.valid.sum())}**",
        f"- Solar-valid variants: **{int((scores.valid & scores.solar_all_pass).sum())}**",
        "",
        "| base | best setting | galaxy RMSE | bridge RMSE | audit ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for base, row in best_by_base.items():
        lines.append(
            f"| {base} | {row['name']} | "
            f"{row['SPARC_outer_RMSE_km_s']:.3f} km/s | "
            f"{row['bridge_RMSE_dex']:.3f} dex | "
            f"{row['audit_worst_reference_ratio']:.3f} |"
        )
    lines += [
        "",
        "These scores hold the preceding gravity parameters fixed and exclude raw "
        "lensing. They select formula ranges for the frozen full test.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
