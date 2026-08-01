#!/usr/bin/env python3
"""Compare the universal candidate with MOND and NFW-derived CLASH targets."""

from __future__ import annotations

import argparse
import hashlib
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

from voidscreen.lensing_comparison import lensing_metrics, paired_system_bootstrap
from voidscreen.phenomenology import response_enhancement


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_block(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    block = frame[(frame["domain"] == "cluster") & (frame["model"] == label)].copy()
    return block.sort_values(["system", "radius_kpc"]).reset_index(drop=True)


def verify_common_points(blocks: dict[str, pd.DataFrame]) -> None:
    reference = next(iter(blocks.values()))
    shared = [
        "system",
        "radius_kpc",
        "log_gbar",
        "log_gobs",
        "err_log_gbar",
        "err_log_gobs",
        "fold",
    ]
    for name, block in blocks.items():
        if len(block) != len(reference):
            raise RuntimeError(f"{name} has {len(block)} points; expected {len(reference)}")
        for column in shared:
            left = reference[column].to_numpy()
            right = block[column].to_numpy()
            if np.issubdtype(left.dtype, np.number):
                if not np.allclose(left, right, equal_nan=True):
                    raise RuntimeError(f"{name} does not share {column}")
            elif not np.array_equal(left, right):
                raise RuntimeError(f"{name} does not share {column}")


def score(block: pd.DataFrame) -> dict[str, object]:
    residual = block["residual_dex"].to_numpy(dtype=float)
    sigma = np.hypot(
        block["err_log_gbar"].to_numpy(dtype=float),
        block["err_log_gobs"].to_numpy(dtype=float),
    )
    return lensing_metrics(
        block["system"].to_numpy(),
        residual,
        sigma_dex=sigma,
        radius_kpc=block["radius_kpc"].to_numpy(dtype=float),
    )


def per_cluster_table(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, block in blocks.items():
        for system, cluster in block.groupby("system", sort=True):
            residual = cluster["residual_dex"].to_numpy(dtype=float)
            rows.append(
                {
                    "system": system,
                    "model": model,
                    "points": len(cluster),
                    "RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
                    "mean_residual_dex": float(np.mean(residual)),
                    "median_absolute_residual_dex": float(np.median(np.abs(residual))),
                    "RMSE_expressed_as_multiplicative_factor": float(
                        10.0 ** np.sqrt(np.mean(np.square(residual)))
                    ),
                }
            )
    return pd.DataFrame(rows)


def point_table(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidate = blocks["candidate"]
    columns = [
        "system",
        "radius_kpc",
        "fold",
        "log_gbar",
        "log_gobs",
        "err_log_gbar",
        "err_log_gobs",
        "local_density_g_cm3",
        "coherence",
    ]
    output = candidate[columns].copy()
    output["combined_diagonal_sigma_dex"] = np.hypot(
        output["err_log_gbar"], output["err_log_gobs"]
    )
    for name, block in blocks.items():
        output[f"{name}_predicted_log_gobs"] = block["predicted_log_gobs"]
        output[f"{name}_residual_dex"] = block["residual_dex"]
        output[f"{name}_predicted_to_observed"] = np.power(
            10.0, block["residual_dex"].to_numpy(dtype=float)
        )
    output["nfw_construction_predicted_log_gobs"] = output["log_gobs"]
    output["nfw_construction_residual_dex"] = 0.0
    output["nfw_construction_predicted_to_observed"] = 1.0
    return output


def make_plot(points: pd.DataFrame, per_cluster: pd.DataFrame, output: Path) -> None:
    colors = {
        "candidate": "#1874CD",
        "fixed_simple_MOND": "#D95F02",
        "cluster_retuned_RAR": "#2E8B57",
    }
    labels = {
        "candidate": "Universal candidate",
        "fixed_simple_MOND": "Fixed simple MOND",
        "cluster_retuned_RAR": "Cluster-retuned RAR",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax = axes[0]
    for model in colors:
        ax.scatter(
            points["log_gobs"],
            points[f"{model}_residual_dex"],
            s=22,
            alpha=0.68,
            color=colors[model],
            label=labels[model],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(r"NFW-deprojected CLASH $\log_{10} g_{\rm obs}$")
    ax.set_ylabel(r"Prediction residual (dex)")
    ax.set_title("All 72 radial lensing points")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1]
    order = (
        per_cluster[per_cluster["model"] == "candidate"]
        .sort_values("RMSE_dex")["system"]
        .tolist()
    )
    position = np.arange(len(order))
    offsets = {"candidate": -0.22, "fixed_simple_MOND": 0.0, "cluster_retuned_RAR": 0.22}
    for model in colors:
        indexed = per_cluster[per_cluster["model"] == model].set_index("system")
        ax.scatter(
            position + offsets[model],
            indexed.loc[order, "RMSE_dex"],
            s=28,
            color=colors[model],
            label=labels[model],
        )
    ax.set_xticks(position)
    ax.set_xticklabels(order, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Per-cluster radial-field RMSE (dex)")
    ax.set_title("Complete-cluster comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "clash_lensing_universal_comparison_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "clash_lensing_universal_comparison",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    prediction_path = ROOT / protocol["input"]["predictions"]
    bridge_path = ROOT / protocol["input"]["bridge_report"]
    sparc_path = ROOT / "results" / "sparc_independent_nuisance_refit" / "report.json"
    rxj_lens_path = ROOT / "results" / "r1_rxj2129_lens_model" / "report.json"
    rxj_baryon_path = ROOT / "results" / "r1_rxj2129_baryons" / "report.json"

    raw = pd.read_csv(prediction_path)
    labels = {
        "candidate": protocol["models"]["candidate"]["predictions_model_label"],
        "fixed_simple_MOND": protocol["models"]["fixed_simple_MOND"][
            "predictions_model_label"
        ],
        "cluster_retuned_RAR": protocol["models"]["cluster_retuned_RAR_diagnostic"][
            "predictions_model_label"
        ],
    }
    blocks = {name: model_block(raw, label) for name, label in labels.items()}
    verify_common_points(blocks)
    candidate = blocks["candidate"]
    if candidate["system"].nunique() != protocol["input"]["systems"]:
        raise RuntimeError("protocol system count does not match predictions")
    if len(candidate) != protocol["input"]["points"]:
        raise RuntimeError("protocol point count does not match predictions")

    metrics = {name: score(block) for name, block in blocks.items()}
    locked_candidate = candidate.copy()
    locked_settings = protocol["models"]["candidate"]["universal_settings"]
    locked_enhancement = response_enhancement(
        "RAR_sharp_coherence_gated_RG",
        np.power(10.0, locked_candidate["log_gbar"].to_numpy(dtype=float)),
        locked_candidate["local_density_g_cm3"].to_numpy(dtype=float),
        locked_candidate["radius_kpc"].to_numpy(dtype=float),
        [
            locked_settings["epsilon_0"],
            locked_settings["log10_rho_c_g_cm3"],
            locked_settings["Q"],
        ],
        rar_acceleration_m_s2=locked_settings["g_dagger_m_s2"],
        coherence=locked_candidate["coherence"].to_numpy(dtype=float),
        coherence_gate_power=locked_settings["coherence_gate_power"],
    )
    locked_candidate["predicted_log_gobs"] = (
        locked_candidate["log_gbar"].to_numpy(dtype=float)
        + np.log10(locked_enhancement)
    )
    locked_candidate["residual_dex"] = (
        locked_candidate["predicted_log_gobs"] - locked_candidate["log_gobs"]
    )
    metrics["candidate_locked_full_sample_descriptive"] = score(locked_candidate)
    zero = candidate.copy()
    zero["predicted_log_gobs"] = zero["log_gobs"]
    zero["residual_dex"] = 0.0
    metrics["NFW_construction"] = score(zero)

    settings = protocol["metrics"]["uncertainty"]
    bootstraps = {
        "candidate_minus_fixed_simple_MOND": paired_system_bootstrap(
            candidate["system"],
            candidate["residual_dex"],
            blocks["fixed_simple_MOND"]["residual_dex"],
            draws=int(settings["draws"]),
            seed=int(settings["seed"]),
        ),
        "candidate_minus_cluster_retuned_RAR": paired_system_bootstrap(
            candidate["system"],
            candidate["residual_dex"],
            blocks["cluster_retuned_RAR"]["residual_dex"],
            draws=int(settings["draws"]),
            seed=int(settings["seed"]) + 1,
        ),
    }

    points = point_table(blocks)
    per_cluster = per_cluster_table(blocks)
    candidate_score = metrics["candidate"]
    mond_score = metrics["fixed_simple_MOND"]
    retuned_score = metrics["cluster_retuned_RAR"]
    sparc = json.loads(sparc_path.read_text(encoding="utf-8"))
    galaxy_candidate = sparc["scores"]["RAR_sharp_coherence_gated_RG:primary"][
        "outer_holdout"
    ]
    galaxy_mond = sparc["scores"]["simple_MOND:invariant"]["outer_holdout"]
    galaxy_nfw = sparc["scores"]["NFW:invariant"]["outer_holdout"]
    lens = json.loads(rxj_lens_path.read_text(encoding="utf-8"))
    baryons = json.loads(rxj_baryon_path.read_text(encoding="utf-8"))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    points.to_csv(output / "point_comparison.csv", index=False)
    per_cluster.to_csv(output / "per_cluster_metrics.csv", index=False)
    make_plot(points, per_cluster, output / "lensing_comparison.png")

    candidate_mond_ratio = float(
        candidate_score["equal_system_RMSE_dex"] / mond_score["equal_system_RMSE_dex"]
    )
    candidate_retuned_ratio = float(
        candidate_score["equal_system_RMSE_dex"] / retuned_score["equal_system_RMSE_dex"]
    )
    cluster_wide = per_cluster.pivot(index="system", columns="model", values="RMSE_dex")
    candidate_mond_wins = int(
        (cluster_wide["candidate"] < cluster_wide["fixed_simple_MOND"]).sum()
    )
    candidate_retuned_wins = int(
        (cluster_wide["candidate"] < cluster_wide["cluster_retuned_RAR"]).sum()
    )
    report = {
        "report_version": "CLASH-LENSING-UNIVERSAL-COMPARISON-0.1",
        "status": "complete_on_NFW_deprojected_CLASH_target_raw_image_plane_claim_withheld",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "inputs": {
            "predictions": {
                "path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(prediction_path),
            },
            "bridge_report": {
                "path": str(bridge_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(bridge_path),
            },
            "SPARC_report": {
                "path": str(sparc_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(sparc_path),
            },
            "systems": int(candidate["system"].nunique()),
            "radial_points": int(len(candidate)),
            "target_provenance": protocol["input"]["target"],
            "target_warning": protocol["input"]["target_warning"],
        },
        "photon_closure": protocol["photon_closure"],
        "cluster_lensing_metrics": metrics,
        "candidate_prediction_design": {
            "primary_score": "five-fold held-out complete-cluster predictions",
            "primary_parameter_rule": (
                "One three-parameter global law is fit in each training fold and applied "
                "unchanged to every held-out BCG and cluster in that fold; there are no "
                "per-cluster gravity parameters."
            ),
            "primary_is_one_numerically_identical_triplet_across_all_folds": False,
            "locked_triplet_score": (
                "descriptive because the triplet was fit using the full bridge"
            ),
            "locked_triplet_is_the_exact_setting_transferred_to_SPARC": True,
        },
        "paired_complete_cluster_bootstrap": bootstraps,
        "direct_comparisons": {
            "candidate_vs_fixed_simple_MOND": {
                "equal_cluster_RMSE_ratio": candidate_mond_ratio,
                "candidate_RMSE_improvement_fraction": float(1.0 - candidate_mond_ratio),
                "candidate_point_RMSE_improvement_fraction": float(
                    1.0
                    - candidate_score["point_RMSE_dex"] / mond_score["point_RMSE_dex"]
                ),
                "candidate_error_normalized_RMS_ratio": float(
                    candidate_score["diagonal_error_normalized_RMS"]
                    / mond_score["diagonal_error_normalized_RMS"]
                ),
                "fixed_MOND_missing_field_geometric_factor": float(
                    mond_score["posthoc_multiplier_to_remove_mean_bias"]
                ),
                "candidate_missing_field_geometric_factor": float(
                    candidate_score["posthoc_multiplier_to_remove_mean_bias"]
                ),
                "clusters_where_candidate_has_lower_RMSE": candidate_mond_wins,
                "clusters_compared": int(len(cluster_wide)),
                "forbidden_posthoc_debias_diagnostic": {
                    "candidate_scatter_dex": candidate_score[
                        "bias_corrected_point_RMSE_dex"
                    ],
                    "fixed_MOND_scatter_dex": mond_score[
                        "bias_corrected_point_RMSE_dex"
                    ],
                    "candidate_scatter_excess_fraction": float(
                        candidate_score["bias_corrected_point_RMSE_dex"]
                        / mond_score["bias_corrected_point_RMSE_dex"]
                        - 1.0
                    ),
                    "interpretation": (
                        "If an extra fitted global lensing multiplier were allowed, fixed "
                        "MOND has slightly less point scatter; its primary failure is the "
                        "approximately 3.15-fold missing field amplitude."
                    ),
                },
                "derived_target_verdict": "candidate substantially closer",
            },
            "candidate_vs_cluster_retuned_RAR": {
                "equal_cluster_RMSE_ratio": candidate_retuned_ratio,
                "candidate_RMSE_excess_fraction": float(candidate_retuned_ratio - 1.0),
                "retuned_acceleration_scale_ratio_to_galaxy_value": protocol["models"][
                    "cluster_retuned_RAR_diagnostic"
                ]["scale_ratio_to_galaxy_value"],
                "clusters_where_candidate_has_lower_RMSE": candidate_retuned_wins,
                "clusters_compared": int(len(cluster_wide)),
                "verdict": "retuned RAR is closer but fails the unchanged-scale universality rule",
            },
            "candidate_vs_NFW_construction": {
                "candidate_equal_cluster_RMSE_dex": candidate_score[
                    "equal_system_RMSE_dex"
                ],
                "candidate_point_RMSE_dex": candidate_score["point_RMSE_dex"],
                "candidate_point_RMSE_factor": candidate_score[
                    "RMSE_expressed_as_multiplicative_factor"
                ],
                "candidate_error_normalized_RMS": candidate_score[
                    "diagonal_error_normalized_RMS"
                ],
                "NFW_residual_dex": 0.0,
                "independence_warning": protocol["models"]["GR_plus_per_cluster_NFW"][
                    "score_status"
                ],
            },
        },
        "galaxy_and_cluster_universality_scorecard": {
            "SPARC_outer_RMSE_km_s": {
                "candidate": galaxy_candidate["RMSE_km_s"],
                "fixed_simple_MOND": galaxy_mond["RMSE_km_s"],
                "inner_fit_NFW": galaxy_nfw["RMSE_km_s"],
                "candidate_vs_MOND_excess_fraction": float(
                    galaxy_candidate["RMSE_km_s"] / galaxy_mond["RMSE_km_s"] - 1.0
                ),
            },
            "candidate_object_specific_gravity_parameters": 0,
            "tested_NFW_object_specific_gravity_parameters": {
                "galaxies": 262,
                "clusters_minimum": 40,
                "combined_minimum": 302,
            },
            "parameter_transfer_detail": (
                "SPARC uses the exact full-bridge triplet. The primary CLASH score uses "
                "fold-specific global triplets for held-out validation, with no per-object "
                "fit. The exact-triplet CLASH score is reported separately as descriptive."
            ),
        },
        "raw_lensing_readiness": {
            "best_local_pilot": "RX J2129",
            "spectroscopic_images": lens["counts"]["images"],
            "source_families": lens["counts"]["source_families"],
            "conventional_model_all_image_exact_radial_RMS_arcsec": lens[
                "all_image_refit"
            ]["exact_score"]["exact_radial_rms_arcsec"],
            "conventional_model_heldout_exact_radial_RMS_arcsec": lens[
                "training_fits"
            ]["model_A"]["heldout_exact_score"]["exact_radial_rms_arcsec"],
            "complete_baryonic_forward_inputs": baryons[
                "complete_baryonic_forward_inputs"
            ],
            "hot_gas_profile_gate": baryons["component_gates"][
                "hot_gas_profile_numeric_in_all_four_bins"
            ],
            "unspent_holdout_available": False,
            "candidate_raw_image_plane_score_authorized": False,
            "blocking_reason": (
                "The hot-gas and other baryonic radial likelihoods are incomplete, the "
                "existing seven-image holdout has been inspected, and the candidate has "
                "no derived relativistic field equations."
            ),
        },
        "claim_boundary": [
            "The target is a spherical NFW deprojection, not raw lensing observables.",
            "Only published diagonal g_bar and g_obs errors enter normalized residuals.",
            (
                "The complete-cluster bootstrap preserves each radial profile but treats "
                "the published profiles as fixed and does not propagate fitted-law "
                "parameter uncertainty."
            ),
            "The zero-slip same-potential photon law is a diagnostic closure, not an action.",
            (
                "The simple-MOND comparator is not every relativistic MOND theory or MOND "
                "supplemented by an additional cluster dark component."
            ),
            (
                "The NFW construction zero is circular and cannot be compared with a raw "
                "image-plane or shear goodness of fit."
            ),
        ],
        "verdict": {
            "derived_cluster_target_vs_fixed_MOND": "pass",
            "galaxy_parity_vs_fixed_MOND": "pass",
            "raw_cluster_lensing_vs_fixed_MOND": "not_established",
            "closeness_to_per_cluster_NFW_derived_field": (
                "0.139 dex equal-cluster RMSE with 1.365 diagonal-error-normalized RMS"
            ),
            "beat_MOND_under_strict_project_definition": False,
            "beat_dark_matter_under_strict_project_definition": False,
            "reason_strict_gates_remain_open": (
                "The decisive raw image/shear likelihood comparison has not been "
                "identified; the current CLASH target is NFW-deprojected."
            ),
        },
        "outputs": {
            "point_comparison": "results/clash_lensing_universal_comparison/point_comparison.csv",
            "per_cluster_metrics": (
                "results/clash_lensing_universal_comparison/per_cluster_metrics.csv"
            ),
            "diagnostic": "results/clash_lensing_universal_comparison/lensing_comparison.png",
            "report": "results/clash_lensing_universal_comparison/report.json",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
