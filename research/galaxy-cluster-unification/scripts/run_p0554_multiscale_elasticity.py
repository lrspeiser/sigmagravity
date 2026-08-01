#!/usr/bin/env python3
"""Measure multi-scale central sensitivities around the frozen P0554 law."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_apogee_boundary_refinement import morphology_scores  # noqa: E402
from run_arc_apogee_cross_domain import score_predictions, velocity_prediction  # noqa: E402
from run_arc_invariant_absolute_lensing import (  # noqa: E402
    cluster_score,
    prepare_clusters,
    prepare_galaxies,
    response_for_frame,
    response_parameters,
)
from run_p0554_local_cross_domain_sensitivity import (  # noqa: E402
    A0,
    Variant,
    evaluate_raw_context,
    json_safe,
    raw_contexts,
    rms,
    sha256,
)
from voidscreen.arc_invariants import generalized_solar_diagnostics  # noqa: E402


def step_label(u: float) -> str:
    return f"u{int(round(100 * float(u))):03d}"


def build_variants(protocol: dict) -> list[Variant]:
    baseline = dict(protocol["baseline"])
    q0 = float(baseline.pop("universal_q"))
    baseline["candidate_id"] = "baseline"
    variants = [
        Variant("baseline", "none", "baseline", "P0554 parent", baseline, q0, "baseline")
    ]
    for parameter, settings in protocol["parameter_coordinates"].items():
        for u_abs in protocol["coordinate_steps"]:
            for sign, direction in ((-1.0, "minus"), (1.0, "plus")):
                u = sign * float(u_abs)
                spec = dict(baseline)
                q = q0
                base = q0 if parameter == "universal_q" else float(spec[parameter])
                if settings["mode"] == "fractional":
                    value = base * (1.0 + u * float(settings["reference_change"]))
                elif settings["mode"] == "additive":
                    value = base + u * float(settings["reference_change"])
                else:
                    raise ValueError(settings["mode"])
                if parameter == "universal_q":
                    q = value
                else:
                    spec[parameter] = value
                variant_id = f"{parameter}_{direction}_{step_label(u_abs)}"
                spec["candidate_id"] = variant_id
                variants.append(
                    Variant(
                        variant_id,
                        parameter,
                        direction,
                        str(settings["concept"]),
                        spec,
                        q,
                        value,
                    )
                )
    return variants


def evaluate_scalar_domains(protocol: dict, variants: list[Variant]):
    parent = json.loads(
        (ROOT / protocol["inputs"]["parent_protocol"]).read_text(encoding="utf-8")
    )
    galaxy, properties = prepare_galaxies(parent, A0)
    clusters, _ = prepare_clusters(parent)
    score_rows, morphology_rows = [], []
    for index, variant in enumerate(variants):
        print(f"scalar {variant.variant_id} ({index + 1}/{len(variants)})", flush=True)
        local = galaxy.copy()
        response = response_for_frame(
            local,
            variant.spec,
            q=variant.q,
            a0=A0,
            radius_column="radius_adjusted_kpc",
            gbar_column="g_bar_m_s2",
        )
        local["arc_coordinate"] = (
            response["fractional_dynamical_response"] / float(variant.q)
        )
        local["velocity_arc_km_s"] = velocity_prediction(local, variant.q)
        local["candidate_id"] = variant.variant_id
        outer = local[local.split.eq("outer_holdout")]
        galaxy_score = score_predictions(outer, outer.velocity_arc_km_s.to_numpy(float))
        response = response_for_frame(
            clusters,
            variant.spec,
            q=variant.q,
            a0=A0,
            radius_column="radius_kpc",
            gbar_column="gbar_m_s2",
        )
        cluster_metrics = cluster_score(
            clusters,
            clusters.gbar_m_s2.to_numpy(float) * response["lensing_enhancement"],
        )
        solar = generalized_solar_diagnostics(
            **response_parameters(variant.spec, q=variant.q, a0=A0)
        )
        score_rows.append(
            {
                "variant_id": variant.variant_id,
                "parameter": variant.parameter,
                "direction": variant.direction,
                "concept": variant.concept,
                "changed_value": variant.changed_value,
                "universal_q": variant.q,
                "galaxy_outer_RMSE_km_s": galaxy_score["RMSE_km_s"],
                "galaxy_equal_RMSE_km_s": galaxy_score["equal_galaxy_RMSE_km_s"],
                **cluster_metrics,
                **solar,
                "all_solar_proxies_pass": bool(
                    solar["Cassini_proxy_pass"]
                    and solar["Earth_proxy_pass"]
                    and solar["Mercury_proxy_pass"]
                ),
            }
        )
        morphology_rows.extend(morphology_scores(local, properties, variant.variant_id))
    return pd.DataFrame(score_rows), pd.DataFrame(morphology_rows), galaxy, clusters


def complete_three_way_raw(
    raw: pd.DataFrame,
    minus_id: str,
    plus_id: str,
    group: str,
) -> dict:
    block = raw[raw.raw_group.eq(group)]
    frames = {
        name: block[block.variant_id.eq(variant)].set_index("system_label")
        for name, variant in {
            "minus": minus_id,
            "baseline": "baseline",
            "plus": plus_id,
        }.items()
    }
    labels = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    common = [
        label
        for label in labels
        if all(
            bool(frame.loc[label, "heldout_all_roots"])
            and np.isfinite(float(frame.loc[label, "heldout_RMS_arcsec"]))
            for frame in frames.values()
        )
    ]
    output = {
        "requested_systems": len(labels),
        "common_complete_systems": len(common),
        "common_labels": "+".join(common),
    }
    for name, frame in frames.items():
        output[f"{name}_RMS_arcsec"] = (
            rms(frame.loc[common, "heldout_RMS_arcsec"]) if common else np.nan
        )
        output[f"{name}_roots"] = int(frame.heldout_roots_converged.sum())
        output[f"{name}_complete_systems"] = int(
            (
                frame.heldout_all_roots.astype(bool)
                & np.isfinite(pd.to_numeric(frame.heldout_RMS_arcsec, errors="coerce"))
            ).sum()
        )
    return output


def normalized_difference(low: float, base: float, high: float, u: float, normalizer: float):
    if not all(np.isfinite(value) for value in (low, base, high, normalizer)):
        return np.nan, np.nan
    slope = (float(high) - float(low)) / (2.0 * float(u) * float(normalizer))
    curvature = (
        float(high) + float(low) - 2.0 * float(base)
    ) / (float(u) ** 2 * float(normalizer))
    return slope, curvature


def build_central_differences(protocol: dict, scores: pd.DataFrame, raw: pd.DataFrame):
    score = scores.set_index("variant_id")
    baseline = score.loc["baseline"]
    rows = []
    for parameter, settings in protocol["parameter_coordinates"].items():
        for u in map(float, protocol["coordinate_steps"]):
            minus_id = f"{parameter}_minus_{step_label(u)}"
            plus_id = f"{parameter}_plus_{step_label(u)}"
            minus, plus = score.loc[minus_id], score.loc[plus_id]
            galaxy_slope, galaxy_curve = normalized_difference(
                minus.galaxy_outer_RMSE_km_s,
                baseline.galaxy_outer_RMSE_km_s,
                plus.galaxy_outer_RMSE_km_s,
                u,
                baseline.galaxy_outer_RMSE_km_s,
            )
            cluster_slope, cluster_curve = normalized_difference(
                minus.cluster_RMSE_dex,
                baseline.cluster_RMSE_dex,
                plus.cluster_RMSE_dex,
                u,
                baseline.cluster_RMSE_dex,
            )
            mercury_slope, mercury_curve = normalized_difference(
                abs(minus.Mercury_precession_mas_per_century),
                abs(baseline.Mercury_precession_mas_per_century),
                abs(plus.Mercury_precession_mas_per_century),
                u,
                float(protocol["analysis_rules"]["solar_mercury_normalizer_mas_per_century"]),
            )
            row = {
                "parameter": parameter,
                "concept": settings["concept"],
                "coordinate_u": u,
                "minus_value": minus.changed_value,
                "plus_value": plus.changed_value,
                "galaxy_normalized_slope": galaxy_slope,
                "galaxy_normalized_curvature": galaxy_curve,
                "cluster_normalized_slope": cluster_slope,
                "cluster_normalized_curvature": cluster_curve,
                "Mercury_margin_slope": mercury_slope,
                "Mercury_margin_curvature": mercury_curve,
                "minus_solar_pass": bool(minus.all_solar_proxies_pass),
                "plus_solar_pass": bool(plus.all_solar_proxies_pass),
            }
            for prefix, group in (("RXJ2129", "RXJ2129"), ("four_cluster", "four_cluster")):
                comparison = complete_three_way_raw(raw, minus_id, plus_id, group)
                raw_slope, raw_curve = normalized_difference(
                    comparison["minus_RMS_arcsec"],
                    comparison["baseline_RMS_arcsec"],
                    comparison["plus_RMS_arcsec"],
                    u,
                    comparison["baseline_RMS_arcsec"],
                )
                row.update(
                    {
                        f"{prefix}_normalized_slope": raw_slope,
                        f"{prefix}_normalized_curvature": raw_curve,
                        f"{prefix}_common_complete_systems": comparison[
                            "common_complete_systems"
                        ],
                        f"{prefix}_common_labels": comparison["common_labels"],
                        f"{prefix}_minus_roots": comparison["minus_roots"],
                        f"{prefix}_baseline_roots": comparison["baseline_roots"],
                        f"{prefix}_plus_roots": comparison["plus_roots"],
                        f"{prefix}_minus_complete_systems": comparison[
                            "minus_complete_systems"
                        ],
                        f"{prefix}_plus_complete_systems": comparison[
                            "plus_complete_systems"
                        ],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_parameters(protocol: dict, central: pd.DataFrame) -> pd.DataFrame:
    domains = {
        "galaxy": "galaxy_normalized_slope",
        "cluster": "cluster_normalized_slope",
        "RXJ2129": "RXJ2129_normalized_slope",
        "four_cluster": "four_cluster_normalized_slope",
        "Mercury": "Mercury_margin_slope",
    }
    stable_rule = float(protocol["analysis_rules"]["material_median_slope"])
    rows = []
    for parameter, block in central.groupby("parameter", sort=False):
        row = {
            "parameter": parameter,
            "concept": block.concept.iloc[0],
        }
        material_directions = []
        for domain, column in domains.items():
            values = pd.to_numeric(block[column], errors="coerce").dropna().to_numpy(float)
            nonzero = values[np.abs(values) > 1.0e-12]
            median = float(np.median(values)) if len(values) else np.nan
            median_abs = float(np.median(np.abs(values))) if len(values) else np.nan
            if len(nonzero):
                plus_fraction = float(np.mean(nonzero < 0.0))
                minus_fraction = float(np.mean(nonzero > 0.0))
                consistency = max(plus_fraction, minus_fraction)
                better = "plus" if plus_fraction >= minus_fraction else "minus"
            else:
                consistency, better = np.nan, "neutral"
            stable = bool(
                len(values) >= 3
                and np.isfinite(consistency)
                and consistency >= 0.75
                and np.isfinite(median_abs)
                and median_abs >= stable_rule
            )
            row[f"{domain}_median_slope"] = median
            row[f"{domain}_median_abs_slope"] = median_abs
            row[f"{domain}_direction_consistency"] = consistency
            row[f"{domain}_better_direction"] = better
            row[f"{domain}_stable_material"] = stable
            row[f"{domain}_available_steps"] = len(values)
            if domain != "Mercury" and stable:
                material_directions.append(better)
        row["stable_material_nonSolar_domains"] = len(material_directions)
        row["stable_nonSolar_directions_agree"] = bool(
            len(material_directions) >= 2 and len(set(material_directions)) == 1
        )
        baseline_roots = int(
            block.RXJ2129_baseline_roots.iloc[0]
            + block.four_cluster_baseline_roots.iloc[0]
        )
        root_changes = []
        for item in block.itertuples(index=False):
            for direction in ("minus", "plus"):
                roots = int(
                    getattr(item, f"RXJ2129_{direction}_roots")
                    + getattr(item, f"four_cluster_{direction}_roots")
                )
                if roots != baseline_roots:
                    root_changes.append(float(item.coordinate_u))
        row["smallest_root_bifurcation_u"] = min(root_changes) if root_changes else np.nan
        row["root_bifurcation_at_smallest_step"] = bool(
            root_changes and min(root_changes) == min(protocol["coordinate_steps"])
        )
        solar_fail = block[~(block.minus_solar_pass & block.plus_solar_pass)]
        row["smallest_solar_boundary_crossing_u"] = (
            float(solar_fail.coordinate_u.min()) if not solar_fail.empty else np.nan
        )
        curve_columns = [column.replace("slope", "curvature") for column in domains.values()]
        row["maximum_abs_normalized_curvature"] = float(
            np.nanmax(np.abs(block[curve_columns].to_numpy(float)))
        )
        row["largest_median_abs_slope"] = float(
            np.nanmax([row[f"{domain}_median_abs_slope"] for domain in domains])
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["largest_median_abs_slope", "maximum_abs_normalized_curvature"],
        ascending=False,
    )


def make_figure(central: pd.DataFrame, summary: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    domains = ["galaxy", "cluster", "RXJ2129", "four_cluster", "Mercury"]
    matrix = summary.set_index("parameter")[[f"{d}_median_abs_slope" for d in domains]]
    shown = np.clip(matrix.to_numpy(float), 0.0, 2.0)
    image = axes[0, 0].imshow(shown, aspect="auto", cmap="magma")
    axes[0, 0].set(
        yticks=np.arange(len(matrix)),
        yticklabels=matrix.index,
        xticks=np.arange(len(domains)),
        xticklabels=domains,
        title="Median absolute central slope across scales",
    )
    axes[0, 0].tick_params(axis="x", rotation=25)
    fig.colorbar(image, ax=axes[0, 0], label="normalized impact per reference move")

    top = summary.head(5).parameter.tolist()
    for parameter in top:
        block = central[central.parameter.eq(parameter)]
        axes[0, 1].plot(
            block.coordinate_u,
            np.abs(block.cluster_normalized_slope),
            "o-",
            label=parameter,
        )
    axes[0, 1].set(
        xscale="log",
        yscale="log",
        xlabel="fraction of declared reference move",
        ylabel="absolute CLASH central slope",
        title="Does cluster sensitivity survive tiny steps?",
    )
    axes[0, 1].legend(fontsize=7)

    roots = []
    for row in central.itertuples(index=False):
        for direction in ("minus", "plus"):
            roots.append(
                {
                    "parameter": row.parameter,
                    "u": row.coordinate_u * (-1 if direction == "minus" else 1),
                    "roots": getattr(row, f"RXJ2129_{direction}_roots")
                    + getattr(row, f"four_cluster_{direction}_roots"),
                }
            )
    roots = pd.DataFrame(roots)
    for parameter, block in roots.groupby("parameter", sort=False):
        axes[1, 0].plot(block.u, block.roots, "o-", ms=3, alpha=0.7, label=parameter)
    axes[1, 0].axhline(17, color="black", ls="--", label="P0554 parent")
    axes[1, 0].set(
        xlabel="signed perturbation coordinate",
        ylabel="converged roots of 18",
        title="Strong-lens topology bifurcations",
    )
    axes[1, 0].legend(fontsize=6, ncol=2)

    ordered = summary.sort_values("maximum_abs_normalized_curvature")
    axes[1, 1].barh(ordered.parameter, ordered.maximum_abs_normalized_curvature)
    axes[1, 1].set(
        xlabel="largest absolute normalized curvature",
        title="Nonlinearity ranking across domains",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0554_multiscale_elasticity_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    variants = build_variants(protocol)

    scores, morphology, galaxy, clusters = evaluate_scalar_domains(protocol, variants)
    scores.to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    morphology.to_csv(output / protocol["outputs"]["galaxy_morphology"], index=False)

    raw_rows, raw_predictions = [], []
    contexts = raw_contexts(protocol)
    for context in contexts:
        rows, predictions, _ = evaluate_raw_context(context, variants)
        raw_rows.extend(rows)
        raw_predictions.extend(predictions)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output / protocol["outputs"]["raw_scores"], index=False)
    pd.concat(raw_predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["raw_predictions"], index=False
    )

    central = build_central_differences(protocol, scores, raw)
    summary = summarize_parameters(protocol, central)
    central.to_csv(output / protocol["outputs"]["central_differences"], index=False)
    summary.to_csv(output / protocol["outputs"]["parameter_summary"], index=False)

    top_by_domain = {}
    for domain in ("galaxy", "cluster", "RXJ2129", "four_cluster", "Mercury"):
        column = f"{domain}_median_abs_slope"
        available = summary[
            np.isfinite(pd.to_numeric(summary[column], errors="coerce"))
            & summary[f"{domain}_stable_material"].astype(bool)
        ]
        if available.empty:
            available = summary[
                np.isfinite(pd.to_numeric(summary[column], errors="coerce"))
            ]
        row = available.sort_values(column, ascending=False).iloc[0]
        top_by_domain[domain] = {
            "parameter": row.parameter,
            "median_abs_normalized_slope": float(row[column]),
            "direction_consistency": float(row[f"{domain}_direction_consistency"]),
            "better_direction": row[f"{domain}_better_direction"],
        }
    raw_root_totals = raw.groupby("variant_id").heldout_roots_converged.sum()
    report = {
        "report_version": "P0554-MULTISCALE-ELASTICITY-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "coverage": {
            "variants": len(variants),
            "parameters": len(protocol["parameter_coordinates"]),
            "coordinate_steps": len(protocol["coordinate_steps"]),
            "SPARC_galaxies": int(galaxy.galaxy.nunique()),
            "SPARC_outer_points": int(galaxy.split.eq("outer_holdout").sum()),
            "CLASH_systems": int(clusters.system.nunique()),
            "CLASH_points": len(clusters),
            "raw_clusters": len(contexts),
            "raw_heldout_images": int(raw[raw.variant_id.eq("baseline")].heldout_images.sum()),
        },
        "baseline": scores[scores.variant_id.eq("baseline")].iloc[0].to_dict(),
        "top_stable_parameter_by_median_multiscale_slope": top_by_domain,
        "stable_same_direction_nonSolar": summary[
            summary.stable_nonSolar_directions_agree.astype(bool)
        ].parameter.tolist(),
        "root_topology": {
            "baseline_roots": int(raw_root_totals.loc["baseline"]),
            "minimum_roots": int(raw_root_totals.min()),
            "maximum_roots": int(raw_root_totals.max()),
            "parameters_bifurcating_at_smallest_step": summary[
                summary.root_bifurcation_at_smallest_step.astype(bool)
            ].parameter.tolist(),
        },
        "solar_boundary_crossings": summary[
            summary.smallest_solar_boundary_crossing_u.notna()
        ][["parameter", "smallest_solar_boundary_crossing_u"]].to_dict("records"),
        "parameter_summary": summary.to_dict("records"),
        "claim_limits": protocol["claim_limits"],
        "verdict": {
            "candidate_selected": False,
            "purpose": "parameter sensitivity and structural diagnostics only",
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(central, summary, output / protocol["outputs"]["figure"])
    summary_text = f"""# P0554 multi-scale elasticity

The frozen experiment evaluated {len(variants)} variants: the P0554 parent and
central perturbations of 11 parameters at four declared scales. No gravity or
ordinary lens-geometry parameter was fit.

The dominant median multi-scale levers are `{top_by_domain['galaxy']['parameter']}`
for SPARC, `{top_by_domain['cluster']['parameter']}` for derived CLASH,
`{top_by_domain['RXJ2129']['parameter']}` for RX J2129,
`{top_by_domain['four_cluster']['parameter']}` for the four other raw clusters,
and `{top_by_domain['Mercury']['parameter']}` for Mercury. Raw topology spans
{report['root_topology']['minimum_roots']}--{report['root_topology']['maximum_roots']}
of 18 roots around the 17-root parent.

This stage selects no candidate. It distinguishes stable sensitivities from
nonlinearity and root bifurcations on spent exploratory systems.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary_text, encoding="utf-8")
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "top_stable_parameter_by_median_multiscale_slope": top_by_domain,
                    "stable_same_direction_nonSolar": report[
                        "stable_same_direction_nonSolar"
                    ],
                    "root_topology": report["root_topology"],
                    "solar_boundary_crossings": report["solar_boundary_crossings"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
