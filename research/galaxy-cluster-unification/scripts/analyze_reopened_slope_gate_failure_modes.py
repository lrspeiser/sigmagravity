#!/usr/bin/env python3
"""Identify which measured galaxy properties amplify a local-slope gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_reopened_hybrid_sensitivity import sparc_scores  # noqa: E402


AUDIT_CONFIG = ROOT / "configs/reopened_hybrid_slope_adaptive_carrier_audit_protocol.json"
AUDIT_SCORES = ROOT / "results/reopened_hybrid_slope_adaptive_carrier_audit/scores.csv"
SOURCE_REPORT = ROOT / "results/reopened_hybrid_channel_saturation_fine/report.json"
SPARC_PATH = ROOT / "results/sparc_density_transfer/primary_predictions.csv"
GEOMETRY_PATH = ROOT / "results/reopened_slope_gate_geometry_audit/system_gate_geometry.csv"
MORPHOLOGY_PATH = ROOT / "data/derived/nbp0_sparc_morphology.csv"
OUTPUT = ROOT / "results/reopened_slope_gate_failure_modes"

BASELINE = "dual_slope_adaptive_gate_strength:radial_memory_slope_gate_strength=0"
SELECTED = [
    BASELINE,
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=0.25",
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=0.5",
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=1",
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=2",
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=4",
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=8",
    "dual_slope_adaptive_slope_sharpness:radial_memory_slope_gate_sharpness=16",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short_name(name: str) -> str:
    if name == BASELINE:
        return "baseline"
    return "sharpness_" + name.rsplit("=", 1)[1].replace(".", "p")


def main() -> None:
    protocol = json.loads(AUDIT_CONFIG.read_text(encoding="utf-8"))
    score_rows = pd.read_csv(AUDIT_SCORES).set_index("name")
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    parameter_names = protocol["universal_parameters"]["names"]
    fitted = source["results"][
        "rg_2_sigma_fine:sigma_saturation_ceiling=1.5"
    ]["full_fit_parameters"]
    parameters = [fitted[name] for name in parameter_names]
    sparc = pd.read_csv(SPARC_PATH)

    per_galaxy = None
    aggregate = {}
    for name in SELECTED:
        settings = json.loads(score_rows.loc[name, "settings"])
        metrics, predictions = sparc_scores(
            sparc,
            parameters,
            {"name": name, "settings": settings},
            protocol["shared_constants"],
        )
        key = short_name(name)
        galaxy = (
            predictions.assign(
                squared_residual=np.square(predictions.residual_km_s)
            )
            .groupby("galaxy", sort=False)
            .squared_residual.mean()
            .pow(0.5)
            .rename(f"RMSE_{key}_km_s")
        )
        per_galaxy = (
            galaxy.to_frame()
            if per_galaxy is None
            else per_galaxy.join(galaxy, how="outer")
        )
        aggregate[key] = metrics["RMSE_km_s"]

    per_galaxy = per_galaxy.reset_index()
    baseline_column = "RMSE_baseline_km_s"
    for name in SELECTED[1:]:
        key = short_name(name)
        per_galaxy[f"delta_RMSE_{key}_vs_baseline_km_s"] = (
            per_galaxy[f"RMSE_{key}_km_s"] - per_galaxy[baseline_column]
        )

    geometry = pd.read_csv(GEOMETRY_PATH)
    geometry = geometry[geometry.domain.eq("SPARC")].drop(columns="domain")
    geometry = geometry.rename(columns={"system": "galaxy"})
    morphology = pd.read_csv(MORPHOLOGY_PATH)
    table = per_galaxy.merge(geometry, on="galaxy", how="left", validate="one_to_one")
    table = table.merge(morphology, on="galaxy", how="left", validate="one_to_one")

    excluded = {
        "galaxy",
        "fold",
        "quality",
        "morphology_input_pass",
    }
    feature_columns = [
        name
        for name in table.columns
        if name not in excluded
        and not name.startswith("RMSE_")
        and not name.startswith("delta_RMSE_")
        and pd.api.types.is_numeric_dtype(table[name])
    ]
    correlation_rows = []
    for name in SELECTED[1:]:
        key = short_name(name)
        target = f"delta_RMSE_{key}_vs_baseline_km_s"
        for feature in feature_columns:
            block = table[[target, feature]].dropna()
            if len(block) < 20 or block[feature].nunique() < 3:
                continue
            correlation = block[target].corr(block[feature], method="spearman")
            correlation_rows.append(
                {
                    "variant": key,
                    "feature": feature,
                    "systems": len(block),
                    "spearman_delta_RMSE": float(correlation),
                    "absolute_spearman": float(abs(correlation)),
                }
            )
    correlations = pd.DataFrame(correlation_rows).sort_values(
        ["variant", "absolute_spearman"], ascending=[True, False]
    )
    strongest = {}
    for variant, block in correlations.groupby("variant", sort=False):
        strongest[variant] = block.head(10).to_dict(orient="records")

    report = {
        "status": "completed slope-gate galaxy failure-mode audit",
        "SPARC_galaxies": int(table.galaxy.nunique()),
        "fixed_parameter_aggregate_RMSE_km_s": aggregate,
        "strongest_per_galaxy_correlations": strongest,
        "interpretation": [
            "Positive correlation means the measured feature is associated with a larger per-galaxy RMSE penalty relative to the ungated p=-1 q=-0.5 carrier.",
            "Gate-geometry features and ordinary morphology features are ranked together; this tests whether failure follows radial derivative structure more directly than catalog morphology.",
            "Correlations diagnose the implemented gate and do not reject local-slope-dependent gravity as a broader idea.",
        ],
        "claim_boundary": [
            "The gravity parameters are held at the preceding local dual-cap fit, so this isolates transfer behavior before refit compensation.",
            "The same SPARC data were used to discover the aggregate sharpness trend; correlations are exploratory, not independent significances.",
            "Galaxy properties are correlated with one another and no causal interpretation is assigned.",
        ],
        "input_hashes": {
            "audit_config": sha256(AUDIT_CONFIG),
            "audit_scores": sha256(AUDIT_SCORES),
            "source_report": sha256(SOURCE_REPORT),
            "SPARC": sha256(SPARC_PATH),
            "gate_geometry": sha256(GEOMETRY_PATH),
            "morphology": sha256(MORPHOLOGY_PATH),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "per_galaxy_failure_modes.csv", index=False)
    correlations.to_csv(OUTPUT / "correlations.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Local-slope gate failure modes",
        "",
        f"- SPARC galaxies: **{len(table)}**",
        "",
        "The tables rank which measured profile and morphology properties accompany the per-galaxy error change as the local-slope switch becomes sharper. Positive correlations mean a larger penalty relative to the ungated balanced carrier.",
    ]
    (OUTPUT / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
