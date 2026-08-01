#!/usr/bin/env python3
"""Summarize the frozen conservative profile-diffusion experiment."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/reopened_hybrid_profile_diffusion"
ROBUST = ROOT / "results/reopened_hybrid_profile_diffusion_raw_robustness"
OUTPUT = ROOT / "results/reopened_hybrid_profile_diffusion_analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orthogonal_effects(
    frame: pd.DataFrame,
    metric: str,
    factors: list[str],
) -> dict[str, object]:
    values = frame[metric].astype(float)
    grand = float(values.mean())
    total = float(np.square(values - grand).sum())
    levels = {factor: sorted(frame[factor].unique().tolist()) for factor in factors}
    sums: dict[str, float] = {}
    for factor in factors:
        means = frame.groupby(factor, observed=True)[metric].mean()
        multiplier = len(frame) / len(levels[factor])
        sums[factor] = float(multiplier * np.square(means - grand).sum())
    for left, right in combinations(factors, 2):
        left_means = frame.groupby(left, observed=True)[metric].mean()
        right_means = frame.groupby(right, observed=True)[metric].mean()
        joint = frame.groupby([left, right], observed=True)[metric].mean()
        multiplier = len(frame) / (len(levels[left]) * len(levels[right]))
        value = 0.0
        for (left_value, right_value), mean in joint.items():
            residual = (
                mean
                - left_means.loc[left_value]
                - right_means.loc[right_value]
                + grand
            )
            value += multiplier * residual**2
        sums[f"{left}_x_{right}"] = float(value)
    remainder_name = "higher_order" if len(factors) > 2 else "interaction"
    sums[remainder_name] = max(0.0, total - sum(sums.values()))
    fractions = {
        name: (100.0 * value / total if total > 0.0 else 0.0)
        for name, value in sums.items()
    }
    return {
        "metric": metric,
        "rows": int(len(frame)),
        "span": float(values.max() - values.min()),
        "variance_percent": fractions,
    }


def row_record(row: pd.Series) -> dict[str, object]:
    names = [
        "variant",
        "family",
        "SPARC_outer_RMSE_km_s",
        "bridge_RMSE_dex",
        "raw_eight_start_RMS_arcsec",
        "raw_eight_start_all_roots_converged",
        "raw_eight_start_pooled_reduced_chi2",
        "cross_domain_reference_ratio_eight_start",
        "solar_maximum_fractional_change",
        "Mercury_precession_mas_per_century",
        "any_universal_parameter_at_boundary",
    ]
    record: dict[str, object] = {}
    for name in names:
        value = row[name]
        if isinstance(value, (np.bool_, bool)):
            record[name] = bool(value)
        elif isinstance(value, (np.integer, int)):
            record[name] = int(value)
        elif isinstance(value, (np.floating, float)):
            record[name] = float(value)
        else:
            record[name] = value
    return record


def main() -> None:
    protocol_path = ROOT / "configs/reopened_hybrid_profile_diffusion_protocol.json"
    robust_protocol_path = (
        ROOT / "configs/reopened_hybrid_profile_diffusion_raw_robustness_protocol.json"
    )
    main_report_path = MAIN / "report.json"
    scores_path = MAIN / "scores.csv"
    robust_report_path = ROBUST / "report.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    main_report = json.loads(main_report_path.read_text(encoding="utf-8"))
    robust_report = json.loads(robust_report_path.read_text(encoding="utf-8"))
    scores = pd.read_csv(scores_path)

    robust_rows = []
    for variant, comparison in robust_report["comparisons"].items():
        eight = comparison["eight_start"]
        robust_rows.append(
            {
                "variant": variant,
                "raw_eight_start_RMS_arcsec": eight[
                    "equal_system_radial_RMS_arcsec"
                ],
                "raw_eight_start_all_roots_converged": eight[
                    "all_roots_converged"
                ],
                "raw_eight_start_pooled_reduced_chi2": eight[
                    "pooled_reduced_chi2"
                ],
            }
        )
    joined = scores.merge(pd.DataFrame(robust_rows), on="variant", validate="one_to_one")
    references = main_report["references"]
    galaxy_reference = float(references["SPARC_fixed_RAR_outer_RMSE_km_s"])
    raw_reference = float(references["raw_compact_halo_RMS_arcsec"])
    joined["cross_domain_reference_ratio_eight_start"] = np.maximum(
        joined["SPARC_outer_RMSE_km_s"] / galaxy_reference,
        joined["raw_eight_start_RMS_arcsec"] / raw_reference,
    )

    diffusion = joined[
        joined["family"].str.match(
            r"diff_(fractional|added_acceleration|speed_squared)_"
        )
    ].copy()
    diffusion["carrier"] = diffusion["family"].str.extract(
        r"diff_(fractional|added_acceleration|speed_squared)_"
    )[0]
    diffusion["scale"] = (
        diffusion["family"]
        .str.extract(r"_l(0p15|0p35|0p7)$")[0]
        .map({"0p15": 0.15, "0p35": 0.35, "0p7": 0.7})
    )
    diffusion["strength"] = (
        diffusion["variant"].str.extract(r"=([0-9.]+)$")[0].astype(float)
    )
    memory = joined[joined["family"].str.startswith("best_memory_plus")].copy()
    memory["scale"] = (
        memory["family"]
        .str.extract(r"_l(0p15|0p35|0p7)$")[0]
        .map({"0p15": 0.15, "0p35": 0.35, "0p7": 0.7})
    )
    memory["strength"] = (
        memory["variant"].str.extract(r"=([0-9.]+)$")[0].astype(float)
    )

    metrics = [
        "SPARC_outer_RMSE_km_s",
        "bridge_RMSE_dex",
        "raw_eight_start_RMS_arcsec",
        "solar_maximum_fractional_change",
        "Mercury_precession_mas_per_century",
    ]
    diffusion_effects = [
        orthogonal_effects(diffusion, metric, ["carrier", "scale", "strength"])
        for metric in metrics
    ]
    memory_effects = [
        orthogonal_effects(memory, metric, ["scale", "strength"])
        for metric in metrics
    ]

    complete = joined[joined["raw_eight_start_all_roots_converged"]].copy()
    best_complete = complete.sort_values(
        "cross_domain_reference_ratio_eight_start"
    ).iloc[0]
    best_raw = complete.sort_values("raw_eight_start_RMS_arcsec").iloc[0]
    local_control = joined.loc[
        joined["variant"] == "local_control:radial_diffusion_strength=0"
    ].iloc[0]
    memory_control = joined.loc[
        joined["variant"] == "best_memory_control:radial_diffusion_strength=0"
    ].iloc[0]
    memory_best_galaxy = memory.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
    nearly_free = diffusion[
        (diffusion["raw_eight_start_all_roots_converged"])
        & (diffusion["SPARC_outer_RMSE_km_s"] <= local_control["SPARC_outer_RMSE_km_s"] + 1.0)
        & (diffusion["raw_eight_start_RMS_arcsec"] < local_control["raw_eight_start_RMS_arcsec"])
    ].sort_values("raw_eight_start_RMS_arcsec").iloc[0]

    def deltas(row: pd.Series, control: pd.Series) -> dict[str, float]:
        return {
            "SPARC_outer_RMSE_change_km_s": float(
                row["SPARC_outer_RMSE_km_s"] - control["SPARC_outer_RMSE_km_s"]
            ),
            "bridge_RMSE_change_dex": float(
                row["bridge_RMSE_dex"] - control["bridge_RMSE_dex"]
            ),
            "raw_eight_start_RMS_change_arcsec": float(
                row["raw_eight_start_RMS_arcsec"]
                - control["raw_eight_start_RMS_arcsec"]
            ),
        }

    report = {
        "report_version": "REOPENED-HYBRID-PROFILE-DIFFUSION-ANALYSIS-0.1.0",
        "status": "completed",
        "inputs": {
            "protocol_sha256": sha256(protocol_path),
            "robustness_protocol_sha256": sha256(robust_protocol_path),
            "main_report_sha256": sha256(main_report_path),
            "scores_sha256": sha256(scores_path),
            "robustness_report_sha256": sha256(robust_report_path),
        },
        "coverage": {
            "rows": int(len(joined)),
            "diffusion_factorial_rows": int(len(diffusion)),
            "memory_plus_diffusion_rows": int(len(memory)),
            "eight_start_complete_root_rows": int(
                joined["raw_eight_start_all_roots_converged"].sum()
            ),
            "universal_parameter_boundary_rows": int(
                joined["any_universal_parameter_at_boundary"].sum()
            ),
        },
        "references": references,
        "controls": {
            "local": row_record(local_control),
            "best_one_sided_memory": row_record(memory_control),
        },
        "best_complete_cross_domain": row_record(best_complete),
        "best_complete_raw_diffusion": {
            **row_record(best_raw),
            "change_from_local_control": deltas(best_raw, local_control),
        },
        "smallest_galaxy_cost_raw_improvement": {
            **row_record(nearly_free),
            "change_from_local_control": deltas(nearly_free, local_control),
        },
        "best_memory_plus_diffusion_galaxy_row": {
            **row_record(memory_best_galaxy),
            "change_from_memory_control": deltas(memory_best_galaxy, memory_control),
        },
        "diffusion_factorial_effects": diffusion_effects,
        "memory_plus_diffusion_effects": memory_effects,
        "carrier_means": diffusion.groupby("carrier", observed=True)[
            [
                "SPARC_outer_RMSE_km_s",
                "bridge_RMSE_dex",
                "raw_eight_start_RMS_arcsec",
            ]
        ].mean().to_dict(orient="index"),
        "scale_means": diffusion.groupby("scale", observed=True)[
            [
                "SPARC_outer_RMSE_km_s",
                "bridge_RMSE_dex",
                "raw_eight_start_RMS_arcsec",
            ]
        ].mean().to_dict(orient="index"),
        "strength_means": diffusion.groupby("strength", observed=True)[
            [
                "SPARC_outer_RMSE_km_s",
                "bridge_RMSE_dex",
                "raw_eight_start_RMS_arcsec",
            ]
        ].mean().to_dict(orient="index"),
        "conclusions": [
            "Symmetric conservative diffusion is a real parameter lever but does not beat either exact control on the complete-root cross-domain objective.",
            "Added-acceleration transport produces the strongest raw-lensing improvement and simultaneously the largest galaxy penalty.",
            "Short-scale circular-speed-squared transport changes galaxies very little and yields only a small raw-lensing improvement.",
            "All nine diffusion additions to the preceding best one-sided-memory row lose at least one held-out lens root, even though they improve galaxy RMSE.",
            "Every bridge refit touches at least one universal-parameter boundary, so the experiment identifies response placement more strongly than microscopic constants.",
            "All Solar proxies pass; the limiting tension remains galaxy dynamics versus cluster lensing.",
        ],
        "claim_boundary": protocol["claim_boundary"],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUTPUT / "joined_scores.csv", index=False)
    effect_rows = []
    for stage, effects in (
        ("diffusion", diffusion_effects),
        ("memory_plus_diffusion", memory_effects),
    ):
        for effect in effects:
            for name, percent in effect["variance_percent"].items():
                effect_rows.append(
                    {
                        "stage": stage,
                        "metric": effect["metric"],
                        "span": effect["span"],
                        "effect": name,
                        "variance_percent": percent,
                    }
                )
    pd.DataFrame(effect_rows).to_csv(OUTPUT / "factorial_effects.csv", index=False)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    best_raw_delta = report["best_complete_raw_diffusion"]["change_from_local_control"]
    nearly_delta = report["smallest_galaxy_cost_raw_improvement"][
        "change_from_local_control"
    ]
    memory_delta = report["best_memory_plus_diffusion_galaxy_row"][
        "change_from_memory_control"
    ]
    summary = f"""# Conservative profile-diffusion analysis

## Outcome

The 38-row frozen experiment completed, including eight-start raw-lens replays for every row. Symmetric no-flux diffusion is measurable, but it does not improve the complete-root galaxy/lensing compromise over either exact control.

- Complete-root rows: {report['coverage']['eight_start_complete_root_rows']} / {report['coverage']['rows']}.
- Universal-parameter boundary rows: {report['coverage']['universal_parameter_boundary_rows']} / {report['coverage']['rows']}.
- Best complete raw diffusion: `{best_raw['variant']}`, {best_raw['raw_eight_start_RMS_arcsec']:.3f} arcsec and {best_raw['SPARC_outer_RMSE_km_s']:.2f} km/s.
- Relative to the local control, that changes raw RMS by {best_raw_delta['raw_eight_start_RMS_change_arcsec']:+.3f} arcsec and galaxy RMSE by {best_raw_delta['SPARC_outer_RMSE_change_km_s']:+.2f} km/s.
- The low-cost short-scale speed-squared row changes raw RMS by {nearly_delta['raw_eight_start_RMS_change_arcsec']:+.3f} arcsec and galaxy RMSE by only {nearly_delta['SPARC_outer_RMSE_change_km_s']:+.3f} km/s.
- The strongest memory-plus-diffusion galaxy row improves its one-sided-memory control by {-memory_delta['SPARC_outer_RMSE_change_km_s']:.2f} km/s, but every one of the nine such rows loses a held-out lens root.

## Interpretation

The transported quantity is a larger control than a tiny strength change. Added acceleration is the lensing-favored carrier and the strongest galaxy penalty. Circular-speed-squared is nearly galaxy-neutral at short range but supplies only a small lensing gain. Symmetric diffusion therefore confirms the established anti-galaxy/lensing direction instead of resolving it.

The exact one-sided-memory control remains the best complete-root cross-domain row in this stage at {memory_control['SPARC_outer_RMSE_km_s']:.2f} km/s and {memory_control['raw_eight_start_RMS_arcsec']:.2f} arcsec. No conclusion here applies to a full two- or three-dimensional tensor response.
"""
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "rows": report["coverage"]["rows"],
        "complete_roots": report["coverage"]["eight_start_complete_root_rows"],
        "best_complete": best_complete["variant"],
        "best_raw": best_raw["variant"],
    }, indent=2))


if __name__ == "__main__":
    main()
