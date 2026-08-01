#!/usr/bin/env python3
"""Cross bounded endpoint width, strength, and saturation on real observables."""

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

from run_adaptive_route_multicluster_raw import (  # noqa: E402
    MODEL,
    build_contexts,
    decorate_predictions,
    json_safe,
    make_lens,
)
from run_p0581_locked_endpoint_exact_root import endpoint_field  # noqa: E402
from run_p0582_smooth_endpoint_saturation import source_positions  # noqa: E402
from run_rxj2129_raw_theory_lensing import FIXED_LABELS, score  # noqa: E402


PARAMETERS = ["width_over_R80", "route_fraction_multiplier", "contrast_cap"]


def variant_id(width: float, fraction: float, cap: float) -> str:
    def token(value: float) -> str:
        return str(float(value)).replace(".", "p")

    return f"eta{token(width)}_q{token(fraction)}_A{token(cap)}"


def summarize_variants(cluster: pd.DataFrame, galaxy_lookup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, block in cluster.groupby("variant", sort=False):
        complete = block[block.heldout_all_roots.astype(bool)]
        roots = int(block.heldout_converged_roots.sum())
        finite = complete.heldout_RMS_arcsec.to_numpy(float)
        cluster_rms = float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else float("inf")
        first = block.iloc[0]
        galaxy = galaxy_lookup[
            np.isclose(galaxy_lookup.width_over_R80, float(first.width_over_R80))
            & np.isclose(
                galaxy_lookup.route_fraction_multiplier,
                float(first.route_fraction_multiplier),
            )
        ]
        if len(galaxy) != 1:
            raise RuntimeError(f"ambiguous P0580 galaxy match for {variant}")
        galaxy = galaxy.iloc[0]
        rows.append(
            {
                "variant": variant,
                "width_over_R80": float(first.width_over_R80),
                "route_fraction_multiplier": float(first.route_fraction_multiplier),
                "contrast_cap": float(first.contrast_cap),
                "complete_systems": int(len(complete)),
                "heldout_converged_roots": roots,
                "all_four_complete": bool(len(complete) == len(block)),
                "cluster_equal_complete_RMS_arcsec": cluster_rms,
                "cluster_median_complete_RMS_arcsec": (
                    float(np.median(finite)) if len(finite) else float("inf")
                ),
                "complete_labels": "+".join(sorted(complete.system_label)),
                "SPARC_outer_RMSE_km_s": float(galaxy.outer_RMSE_km_s),
                "SPARC_equal_galaxy_RMSE_km_s": float(
                    galaxy.outer_equal_galaxy_RMSE_km_s
                ),
                "SPARC_galaxies_improved_vs_Newtonian": int(
                    galaxy.galaxies_improved_vs_Newtonian
                ),
            }
        )
    return pd.DataFrame(rows)


def interaction_residual_rms(
    frame: pd.DataFrame, left: str, right: str, value: str
) -> float:
    table = frame.pivot_table(index=left, columns=right, values=value, aggfunc="mean")
    residual = (
        table
        - table.mean(axis=1).to_numpy()[:, None]
        - table.mean(axis=0).to_numpy()[None, :]
        + float(table.to_numpy(float).mean())
    )
    return float(np.sqrt(np.mean(np.square(residual.to_numpy(float)))))


def effect_tables(
    variants: pd.DataFrame, cluster: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    level_rows = []
    impact_rows = []
    for parameter in PARAMETERS:
        for level, block in variants.groupby(parameter):
            safe = block[block.all_four_complete.astype(bool)]
            level_rows.append(
                {
                    "parameter": parameter,
                    "level": float(level),
                    "variants": len(block),
                    "mean_heldout_roots": float(block.heldout_converged_roots.mean()),
                    "safe_variant_fraction": float(block.all_four_complete.mean()),
                    "median_safe_cluster_RMS_arcsec": (
                        float(safe.cluster_equal_complete_RMS_arcsec.median())
                        if len(safe)
                        else float("nan")
                    ),
                    "mean_SPARC_outer_RMSE_km_s": float(
                        block.SPARC_outer_RMSE_km_s.mean()
                    ),
                }
            )
        levels = pd.DataFrame(level_rows)
        levels = levels[levels.parameter.eq(parameter)]
        finite_cluster = levels.median_safe_cluster_RMS_arcsec.dropna()
        system_level = cluster.groupby(
            ["system_label", parameter]
        ).heldout_converged_roots.mean().unstack(parameter)
        system_spans = system_level.max(axis=1) - system_level.min(axis=1)
        impact_rows.append(
            {
                "parameter": parameter,
                "mean_root_count_span": float(
                    levels.mean_heldout_roots.max() - levels.mean_heldout_roots.min()
                ),
                "safe_variant_fraction_span": float(
                    levels.safe_variant_fraction.max() - levels.safe_variant_fraction.min()
                ),
                "mean_system_root_pattern_span": float(system_spans.mean()),
                "maximum_system_root_pattern_span": float(system_spans.max()),
                "safe_cluster_RMS_span_arcsec": (
                    float(finite_cluster.max() - finite_cluster.min())
                    if len(finite_cluster)
                    else float("nan")
                ),
                "SPARC_RMSE_span_km_s": float(
                    levels.mean_SPARC_outer_RMSE_km_s.max()
                    - levels.mean_SPARC_outer_RMSE_km_s.min()
                ),
            }
        )
    interaction_rows = []
    for index, left in enumerate(PARAMETERS):
        for right in PARAMETERS[index + 1 :]:
            interaction_rows.append(
                {
                    "left_parameter": left,
                    "right_parameter": right,
                    "root_count_interaction_RMS": interaction_residual_rms(
                        variants, left, right, "heldout_converged_roots"
                    ),
                    "root_safe_interaction_RMS": interaction_residual_rms(
                        variants, left, right, "all_four_complete"
                    ),
                    "SPARC_RMSE_interaction_RMS_km_s": interaction_residual_rms(
                        variants, left, right, "SPARC_outer_RMSE_km_s"
                    ),
                }
            )
    return (
        pd.DataFrame(level_rows),
        pd.DataFrame(impact_rows),
        pd.DataFrame(interaction_rows),
    )


def matched_scalar_comparison(
    cluster: pd.DataFrame, winner: pd.Series, scalar: pd.DataFrame
) -> dict:
    candidate = cluster[cluster.variant.eq(winner.variant)].set_index("system_label")
    reference = scalar.set_index("system_label")
    common = sorted(
        label
        for label in candidate.index.intersection(reference.index)
        if bool(candidate.loc[label, "heldout_all_roots"])
        and bool(reference.loc[label, "heldout_all_roots"])
        and np.isfinite(float(candidate.loc[label, "heldout_RMS_arcsec"]))
        and np.isfinite(float(reference.loc[label, "heldout_RMS_arcsec"]))
    )
    ref = np.asarray([reference.loc[label, "heldout_RMS_arcsec"] for label in common], float)
    cand = np.asarray([candidate.loc[label, "heldout_RMS_arcsec"] for label in common], float)
    ref_rms = float(np.sqrt(np.mean(np.square(ref))))
    cand_rms = float(np.sqrt(np.mean(np.square(cand))))
    return {
        "matched_systems": len(common),
        "matched_labels": common,
        "scalar_RMS_arcsec": ref_rms,
        "candidate_RMS_arcsec": cand_rms,
        "fractional_improvement": 1.0 - cand_rms / ref_rms,
    }


def main() -> None:
    protocol_path = ROOT / "configs/p0613_bounded_endpoint_cross_domain_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0612_before_factorial_scores":
        raise RuntimeError("P0613 protocol is not frozen")
    inputs = {key: ROOT / value for key, value in protocol["inputs"].items()}
    p0581 = json.loads(inputs["P0581_protocol"].read_text(encoding="utf-8"))
    base = json.loads(inputs["base_exact_protocol"].read_text(encoding="utf-8"))
    contexts, _, _ = build_contexts(base)
    geometry = pd.read_csv(inputs["P0581_geometry"])
    prior_predictions = pd.read_csv(inputs["P0581_predictions"])
    scalar = pd.read_csv(inputs["base_exact_scores"])
    scalar = scalar[scalar.variant.eq("scalar_baseline")].copy()
    galaxy_all = pd.read_csv(inputs["P0580_candidate_scores"])
    galaxy_lookup = galaxy_all[
        galaxy_all.gate_mode.eq(protocol["locked"]["gate_mode"])
        & np.isclose(
            galaxy_all.return_length_over_R80,
            float(protocol["locked"]["return_length_over_R80"]),
        )
        & galaxy_all.route_mode.eq(protocol["locked"]["route_mode"])
        & galaxy_all.width_over_R80.isin(protocol["factorial"]["width_over_R80_eta"])
        & galaxy_all.route_fraction_multiplier.isin(
            protocol["factorial"]["route_fraction_multiplier_q"]
        )
    ].copy()
    if len(galaxy_lookup) != 9:
        raise RuntimeError("P0580 factorial coverage changed")

    cluster_rows = []
    prediction_frames = []
    audit_rows = []
    for context in contexts:
        label = context.system["label"]
        geometry_row = geometry[
            geometry.system_label.eq(label) & geometry.variant.eq("K0338_primary")
        ].iloc[0]
        parameters = np.asarray([float(geometry_row[name]) for name in FIXED_LABELS])
        sources = source_positions(prior_predictions, label)
        for width in protocol["factorial"]["width_over_R80_eta"]:
            for fraction in protocol["factorial"]["route_fraction_multiplier_q"]:
                for cap in protocol["factorial"]["smooth_saturation_A"]:
                    variant = variant_id(width, fraction, cap)
                    spec = {
                        "route_fraction_multiplier": float(fraction),
                        "return_length_over_R80": float(
                            protocol["locked"]["return_length_over_R80"]
                        ),
                        "width_over_R80": float(width),
                        "gate_mode": protocol["locked"]["gate_mode"],
                        "contrast_cap": float(cap),
                        "contrast_mode": protocol["locked"]["contrast_mode"],
                        "variant": variant,
                    }
                    field, audit = endpoint_field(p0581, context, spec)
                    lens = make_lens(context, field)
                    heldout = lens.exact_predictions(
                        MODEL,
                        parameters,
                        sources,
                        context.heldout,
                        stage="heldout",
                    )
                    heldout_score = score(heldout, lens.sigma)
                    cluster_rows.append(
                        {
                            "system_label": label,
                            "variant": variant,
                            "width_over_R80": float(width),
                            "route_fraction_multiplier": float(fraction),
                            "contrast_cap": float(cap),
                            "heldout_images": len(context.heldout),
                            "heldout_converged_roots": heldout_score[
                                "converged_roots"
                            ],
                            "heldout_all_roots": heldout_score[
                                "all_roots_converged"
                            ],
                            "heldout_RMS_arcsec": heldout_score[
                                "exact_radial_RMS_arcsec"
                            ],
                        }
                    )
                    prediction_frames.append(
                        decorate_predictions(heldout, context, variant)
                    )
                    audit_rows.append(
                        {
                            "system_label": label,
                            "variant": variant,
                            "width_over_R80": float(width),
                            "route_fraction_multiplier": float(fraction),
                            "contrast_cap": float(cap),
                            **audit,
                        }
                    )
        print(f"{label}: scored {protocol['factorial']['variants']} bounded endpoint fields", flush=True)

    cluster = pd.DataFrame(cluster_rows)
    variants = summarize_variants(cluster, galaxy_lookup)
    levels, impacts, interactions = effect_tables(variants, cluster)
    safe = variants[variants.all_four_complete.astype(bool)].sort_values(
        ["cluster_equal_complete_RMS_arcsec", "SPARC_outer_RMSE_km_s"]
    )
    if not len(safe):
        raise RuntimeError("no root-safe P0613 variant")
    winner = safe.iloc[0]
    matched = matched_scalar_comparison(cluster, winner, scalar)
    p0580_report = json.loads(inputs["P0580_report"].read_text(encoding="utf-8"))
    newtonian = float(
        p0580_report["references"]["Newtonian_same_nuisance"]["outer_RMSE_km_s"]
    )
    rar = float(
        p0580_report["references"]["fixed_RAR_same_nuisance"]["outer_RMSE_km_s"]
    )
    audits = pd.DataFrame(audit_rows)
    gates = {
        "all_heldout_roots_pass": bool(
            int(winner.heldout_converged_roots)
            == int(protocol["gates"]["heldout_roots_required"])
        ),
        "all_cluster_systems_complete_pass": bool(
            int(winner.complete_systems)
            == int(protocol["gates"]["complete_systems_required"])
        ),
        "matched_cluster_improvement_pass": bool(matched["fractional_improvement"] > 0.0),
        "galaxy_improves_Newtonian_pass": bool(
            float(winner.SPARC_outer_RMSE_km_s) < newtonian
        ),
        "galaxy_near_RAR_pass": bool(
            float(winner.SPARC_outer_RMSE_km_s)
            <= float(protocol["gates"]["galaxy_RMSE_to_fixed_RAR_max"]) * rar
        ),
        "route_normalization_pass": bool(
            float(audits.route_map_normalization_error.max())
            <= float(protocol["gates"]["maximum_route_map_normalization_error"])
        ),
        "annular_monopole_pass": bool(
            float(audits.maximum_annular_convergence_mean_fraction.max())
            <= float(
                protocol["gates"]["maximum_annular_convergence_mean_fraction"]
            )
        ),
        "curl_free_pass": bool(
            float(audits.normalized_curl_RMS.max())
            <= float(protocol["gates"]["maximum_normalized_curl_RMS"])
        ),
        "solar_fractional_change_pass": bool(
            float(protocol["gates"]["solar_maximum_fractional_change"]) == 0.0
        ),
        "Mercury_precession_pass": bool(
            float(protocol["gates"]["Mercury_precession_mas_per_century"]) == 0.0
        ),
        "universal_parameter_count_pass": True,
    }
    gates["cross_domain_advance_pass"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    cluster.to_csv(output / protocol["outputs"]["cluster_scores"], index=False)
    variants.sort_values(
        ["all_four_complete", "heldout_converged_roots", "cluster_equal_complete_RMS_arcsec"],
        ascending=[False, False, True],
    ).to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    audits.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    levels.to_csv(output / protocol["outputs"]["main_effect_levels"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    interactions.to_csv(
        output / protocol["outputs"]["interaction_effects"], index=False
    )

    report = {
        "report_version": "P0613-BOUNDED-ENDPOINT-CROSS-DOMAIN-RESULTS-0.1.0",
        "status": "complete_spent_factorial_response_test",
        "coverage": {
            "universal_variants": len(variants),
            "raw_clusters": int(cluster.system_label.nunique()),
            "heldout_images": int(cluster.groupby("system_label").heldout_images.first().sum()),
            "SPARC_galaxies": int(p0580_report["coverage"]["galaxies"]),
            "SPARC_outer_points": int(p0580_report["coverage"]["outer_points"]),
            "root_safe_variants": int(variants.all_four_complete.sum()),
        },
        "formula": protocol["formula"],
        "locked": protocol["locked"],
        "diagnostic_winner": winner.to_dict(),
        "matched_winner_vs_scalar": matched,
        "comparators": {
            "Newtonian_SPARC_outer_RMSE_km_s": newtonian,
            "fixed_RAR_SPARC_outer_RMSE_km_s": rar,
            "winner_to_Newtonian_ratio": float(winner.SPARC_outer_RMSE_km_s / newtonian),
            "winner_to_fixed_RAR_ratio": float(winner.SPARC_outer_RMSE_km_s / rar),
        },
        "parameter_impacts": impacts.to_dict("records"),
        "interaction_effects": interactions.to_dict("records"),
        "solar": {
            "maximum_fractional_change": float(
                protocol["gates"]["solar_maximum_fractional_change"]
            ),
            "Mercury_precession_mas_per_century": float(
                protocol["gates"]["Mercury_precession_mas_per_century"]
            ),
            "reason": "The routing layer collapses to its local point-source null; this does not test the P0554 scalar parent independently.",
        },
        "gates": gates,
        "interpretation": {
            "formula_promoted": False,
            "smooth_bounded_endpoint_is_root_safe_somewhere": bool(len(safe) > 0),
            "same_formula_near_RAR_for_galaxies": bool(gates["galaxy_near_RAR_pass"]),
            "parameter_or_formula_changes_most_impactful": [
                "Read mean_root_count_span for topology sensitivity.",
                "Read mean_system_root_pattern_span when total root counts cancel across clusters.",
                "Read safe_cluster_RMS_span_arcsec only among all-root variants.",
                "Read SPARC_RMSE_span_km_s for galaxy sensitivity; contrast cap is an exact galaxy null.",
            ],
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    root_grid = variants.pivot_table(
        index="width_over_R80",
        columns="route_fraction_multiplier",
        values="heldout_converged_roots",
        aggfunc="mean",
    )
    safe_grid = variants.pivot_table(
        index="width_over_R80",
        columns="route_fraction_multiplier",
        values="all_four_complete",
        aggfunc="mean",
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    image0 = axes[0].imshow(root_grid.to_numpy(float), aspect="auto", vmin=8, vmax=11, cmap="viridis")
    axes[0].set_xticks(np.arange(len(root_grid.columns)), root_grid.columns)
    axes[0].set_yticks(np.arange(len(root_grid.index)), root_grid.index)
    axes[0].set(xlabel="universal strength q", ylabel="width eta R80", title="mean roots across caps")
    figure.colorbar(image0, ax=axes[0])
    image1 = axes[1].imshow(safe_grid.to_numpy(float), aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[1].set_xticks(np.arange(len(safe_grid.columns)), safe_grid.columns)
    axes[1].set_yticks(np.arange(len(safe_grid.index)), safe_grid.index)
    axes[1].set(xlabel="universal strength q", ylabel="width eta R80", title="fraction of caps root-safe")
    figure.colorbar(image1, ax=axes[1])
    display = impacts.set_index("parameter")
    x = np.arange(len(display))
    axes[2].bar(x - 0.18, display.mean_root_count_span, 0.36, label="root-count span")
    axes[2].bar(x + 0.18, display.safe_variant_fraction_span, 0.36, label="safe-fraction span")
    axes[2].set_xticks(x, display.index, rotation=25, ha="right")
    axes[2].set(title="topology impact", ylabel="within-grid span")
    axes[2].legend(fontsize=8)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    (output / protocol["outputs"]["summary"]).write_text(
        "# P0613 bounded endpoint cross-domain response\n\n"
        f"Root-safe variants: **{int(variants.all_four_complete.sum())}/{len(variants)}**.\n\n"
        f"Diagnostic winner: **{winner.variant}**, cluster RMS "
        f"**{winner.cluster_equal_complete_RMS_arcsec:.3f} arcsec**, SPARC outer RMSE "
        f"**{winner.SPARC_outer_RMSE_km_s:.3f} km/s** versus fixed RAR **{rar:.3f} km/s**.\n\n"
        f"All advance gates pass: **{gates['cross_domain_advance_pass']}**.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "coverage": report["coverage"],
                    "diagnostic_winner": report["diagnostic_winner"],
                    "comparators": report["comparators"],
                    "parameter_impacts": report["parameter_impacts"],
                    "interaction_effects": report["interaction_effects"],
                    "gates": report["gates"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
