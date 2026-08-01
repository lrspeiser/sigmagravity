#!/usr/bin/env python3
"""Apply the frozen P0599 law to intermediate-potential BCG dynamics."""

from __future__ import annotations

import itertools
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

from voidscreen.conservative_diffusion import low_acceleration_activation, radial_shape_activation
from voidscreen.unified import rar_acceleration


def metrics(frame: pd.DataFrame, predicted_log: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted_log, dtype=float) - frame.log_gobs.to_numpy(float)
    sigma = frame.sigma_residual_dex.to_numpy(float)
    return {
        "systems": int(len(frame)),
        "RMS_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_residual_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
        "chi2_per_point": float(np.mean(np.square(residual / sigma))),
    }


def variant_id(potential: str, shape: str, screen: str) -> str:
    return f"{potential}__{shape}__{screen}"


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0600_bcg_potential_gap_protocol.json").read_text(encoding="utf-8")
    )
    formula = protocol["frozen_formula"]
    raw = pd.read_csv(ROOT / protocol["data"]["predictions"])
    frame = raw[raw.model == "fixed_galaxy_rar"].copy().reset_index(drop=True)
    if len(frame) != protocol["data"]["systems"] or frame.plateifu.nunique() != len(frame):
        raise RuntimeError("P0600 BCG sample coverage changed")
    comparison_models = {
        name: raw[raw.model == name].sort_values("plateifu").reset_index(drop=True)
        for name in protocol["comparators"]
    }
    frame = frame.sort_values("plateifu").reset_index(drop=True)
    potential_values = {
        "BCG_only": frame.bcg_baryonic_chi.to_numpy(float),
        "BCG_plus_eRASS_median_gas": (
            frame.bcg_baryonic_chi + frame.host_erass_median_gas_chi_r200
        ).to_numpy(float),
        "BCG_plus_eRASS_p90_gas": (
            frame.bcg_baryonic_chi + frame.host_erass_p90_gas_chi_r200
        ).to_numpy(float),
        "BCG_plus_cosmic_baryon_host": frame.combined_bcg_host_chi.to_numpy(float),
    }
    hernquist_c = (np.sqrt(0.5) / (1.0 - np.sqrt(0.5))) / (
        np.sqrt(0.8) / (1.0 - np.sqrt(0.8))
    )
    p0599_clusters = pd.read_csv(
        ROOT / "results/p0599_bounded_potential_amplitude/cluster_oof_predictions.csv"
    )
    clash_median_c = float(
        p0599_clusters.drop_duplicates("system")
        .force_equivalent_concentration_r50_over_r80.median()
    )
    shape_values = {
        "Hernquist_BCG": np.full(
            len(frame),
            radial_shape_activation(
                hernquist_c,
                midpoint=formula["shape_midpoint"],
                width=formula["shape_width"],
            ),
        ),
        "CLASH_median_C": np.full(
            len(frame),
            radial_shape_activation(
                clash_median_c,
                midpoint=formula["shape_midpoint"],
                width=formula["shape_width"],
            ),
        ),
        "neutral_H1": np.ones(len(frame)),
    }
    screen_values = {
        "local_BCG_gbar": np.asarray(
            [
                low_acceleration_activation(
                    value,
                    a0_m_s2=formula["a0_m_s2"],
                    power=formula["source_acceleration_gate_power"],
                )
                for value in frame.gbar_m_s2.to_numpy(float)
            ]
        ),
        "weak_host_S1": np.ones(len(frame)),
    }
    gbar = frame.gbar_m_s2.to_numpy(float)
    base = rar_acceleration(gbar, formula["a0_m_s2"])
    score_rows, prediction_rows = [], []
    for potential_name, shape_name, screen_name in itertools.product(
        protocol["brackets"]["potential_source"],
        protocol["brackets"]["radial_shape"],
        protocol["brackets"]["source_screen"],
    ):
        chi = potential_values[potential_name]
        potential_gate = 1.0 / (
            1.0
            + np.power(
                formula["potential_threshold_chi"] / chi,
                formula["potential_power"],
            )
        )
        multiplier = 1.0 + (
            formula["amplitude_A"]
            * screen_values[screen_name]
            * shape_values[shape_name]
            * potential_gate
        )
        predicted_log = np.log10(base * multiplier)
        identifier = variant_id(potential_name, shape_name, screen_name)
        all_metrics = metrics(frame, predicted_log)
        direct_mask = frame.measurement_source == "Tian2024_direct"
        direct_metrics = metrics(frame[direct_mask], predicted_log[direct_mask])
        proxy_metrics = metrics(frame[~direct_mask], predicted_log[~direct_mask])
        score_rows.append(
            {
                "variant_id": identifier,
                "potential_source": potential_name,
                "radial_shape": shape_name,
                "source_screen": screen_name,
                "median_potential_gate": float(np.median(potential_gate)),
                "median_shape_gate": float(np.median(shape_values[shape_name])),
                "median_source_screen": float(np.median(screen_values[screen_name])),
                **{f"all_{key}": value for key, value in all_metrics.items()},
                **{f"direct_{key}": value for key, value in direct_metrics.items()},
                **{f"proxy_{key}": value for key, value in proxy_metrics.items()},
            }
        )
        for index, row in frame.iterrows():
            prediction_rows.append(
                {
                    "variant_id": identifier,
                    "plateifu": row.plateifu,
                    "measurement_source": row.measurement_source,
                    "potential_chi": chi[index],
                    "potential_gate": potential_gate[index],
                    "shape_gate": shape_values[shape_name][index],
                    "source_screen": screen_values[screen_name][index],
                    "predicted_log_gobs": predicted_log[index],
                    "observed_log_gobs": row.log_gobs,
                    "residual_dex": predicted_log[index] - row.log_gobs,
                }
            )
    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    if len(scores) != protocol["brackets"]["candidate_count"]:
        raise RuntimeError("P0600 bracket count changed")
    primary_cfg = protocol["primary"]
    primary_id = variant_id(
        primary_cfg["potential_source"],
        primary_cfg["radial_shape"],
        primary_cfg["source_screen"],
    )
    primary = scores.set_index("variant_id").loc[primary_id]
    comparator_scores = {
        name: metrics(block, block.predicted_log_gobs.to_numpy(float))
        for name, block in comparison_models.items()
    }
    parameter_rows = []
    for parameter in ("potential_source", "radial_shape", "source_screen"):
        grouped = scores.groupby(parameter).all_RMS_dex.median().sort_values()
        parameter_rows.append(
            {
                "parameter": parameter,
                "best_level": str(grouped.index[0]),
                "worst_level": str(grouped.index[-1]),
                "median_RMS_span_dex": float(grouped.iloc[-1] - grouped.iloc[0]),
            }
        )
    impacts = pd.DataFrame(parameter_rows).sort_values("median_RMS_span_dex", ascending=False)
    primary_predictions = predictions[predictions.variant_id == primary_id].sort_values("plateifu")
    reference = comparison_models["fixed_galaxy_rar"]
    primary_abs = np.abs(primary_predictions.residual_dex.to_numpy(float))
    reference_abs = np.abs(reference.residual_dex.to_numpy(float))
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    samples = rng.integers(0, len(frame), size=(protocol["bootstrap"]["draws"], len(frame)))
    probability_better = float(
        np.mean(
            np.sqrt(np.mean(np.square(primary_abs[samples]), axis=1))
            < np.sqrt(np.mean(np.square(reference_abs[samples]), axis=1))
        )
    )
    cfg = protocol["interpretation_gates"]
    gates = {
        "primary_RMS_pass": bool(primary.all_RMS_dex <= cfg["primary_RMS_dex_max"]),
        "primary_bias_pass": bool(abs(primary.all_mean_residual_dex) <= cfg["primary_absolute_mean_bias_dex_max"]),
        "direct_RMS_pass": bool(primary.direct_RMS_dex <= cfg["direct_Tian_RMS_dex_max"]),
        "bootstrap_pass": bool(probability_better >= cfg["probability_better_than_fixed_RAR_min"]),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["variant_scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    labels = ["fixed RAR", "cluster RAR", "prior host", "P0599 primary"]
    values = [
        comparator_scores["fixed_galaxy_rar"]["RMS_dex"],
        comparator_scores["cluster_scale_rar"]["RMS_dex"],
        comparator_scores["H7s_standard_mu_erass_median_gas_host_r200"]["RMS_dex"],
        primary.all_RMS_dex,
    ]
    axes[0].bar(labels, values)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="BCG RMS (dex)", title="34 intermediate-potential BCGs")
    display = impacts.sort_values("median_RMS_span_dex")
    axes[1].barh(display.parameter, display.median_RMS_span_dex)
    axes[1].set(xlabel="median RMS span (dex)", title="incomplete-host assumption impact")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0600-BCG-POTENTIAL-GAP-RESULTS-0.1.0",
        "status": "complete_external_BCG_bridge",
        "coverage": {"systems": len(frame), "direct_Tian2024": int((frame.measurement_source == 'Tian2024_direct').sum()), "calibrated_DynPop_proxy": int((frame.measurement_source != 'Tian2024_direct').sum()), "variants": len(scores)},
        "frozen_formula": protocol["frozen_formula"],
        "derived_shape_brackets": {"Hernquist_C_R50_over_R80": hernquist_c, "CLASH_median_C_R50_over_R80": clash_median_c, "H_at_Hernquist_C": float(shape_values['Hernquist_BCG'][0]), "H_at_CLASH_median_C": float(shape_values['CLASH_median_C'][0])},
        "primary": {"variant_id": primary_id, **primary.to_dict()},
        "best_posthoc_variant": scores.sort_values("all_RMS_dex").iloc[0].to_dict(),
        "comparators": comparator_scores,
        "probability_primary_better_than_fixed_RAR": probability_better,
        "parameter_impacts": impacts.to_dict("records"),
        "gates": gates,
        "all_interpretation_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0600 BCG potential-gap test\n\n"
        f"The frozen physically motivated bracket `{primary_id}` scores {primary.all_RMS_dex:.3f} dex on 34 BCGs "
        f"and {primary.direct_RMS_dex:.3f} dex on the 11 direct Tian measurements, versus "
        f"{comparator_scores['fixed_galaxy_rar']['RMS_dex']:.3f} dex for fixed RAR and "
        f"{comparator_scores['cluster_scale_rar']['RMS_dex']:.3f} dex for cluster-scale RAR. Its mean bias is "
        f"{primary.all_mean_residual_dex:+.3f} dex and bootstrap probability of beating fixed RAR is "
        f"{100.0 * probability_better:.1f}%. The strongest missing-input lever is {impacts.iloc[0].parameter}, "
        f"spanning {impacts.iloc[0].median_RMS_span_dex:.3f} dex across declared brackets.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
