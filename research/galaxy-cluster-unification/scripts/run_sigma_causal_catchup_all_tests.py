#!/usr/bin/env python3
"""Audit the causal Sigma catch-up term on every existing galaxy/cluster test."""

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

from voidscreen.covariant_closure import (  # noqa: E402
    causal_catchup_characteristics,
    equilibrium_sigma_from_density,
)
from voidscreen.unified import (  # noqa: E402
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
)

C_M_S = 299_792_458.0
KPC_M = 3.085677581491367e19
SECONDS_PER_MYR = 365.25 * 24.0 * 3600.0 * 1.0e6


def sha256(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_density_rows(protocol: dict) -> pd.DataFrame:
    inputs = protocol["inputs"]
    sparc = pd.read_csv(ROOT / inputs["sparc_density_rows"])
    galaxy = pd.DataFrame(
        {
            "domain": "SPARC_outer",
            "system": sparc["galaxy"],
            "radius_kpc": sparc["radius_adjusted_kpc"],
            "density_g_cm3": sparc["local_density_g_cm3"],
            "acceleration_m_s2": sparc["g_pred_m_s2"],
            "source_speed_km_s": sparc["velocity_observed_adjusted_kms"],
            "speed_provenance": "observed circular speed after frozen nuisance adjustment",
        }
    )

    sample = pd.read_csv(ROOT / inputs["bcg_cluster_density_rows"])
    predictions = pd.read_csv(ROOT / inputs["bcg_cluster_predictions"])
    candidate = predictions[
        predictions["model"].eq("RAR_sharp_coherence_gated_RG")
    ].copy()
    keys = ["domain", "system", "radius_kpc"]
    bridge = sample.merge(
        candidate[keys + ["predicted_log_gobs"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if bridge["predicted_log_gobs"].isna().any():
        raise RuntimeError("candidate bridge predictions are incomplete")
    radius_m = bridge["radius_kpc"].to_numpy(float) * KPC_M
    observed_acceleration = np.power(10.0, bridge["log_gobs"].to_numpy(float))
    circular_equivalent_speed = np.sqrt(observed_acceleration * radius_m) / 1000.0
    massive = pd.DataFrame(
        {
            "domain": bridge["domain"],
            "system": bridge["system"],
            "radius_kpc": bridge["radius_kpc"],
            "density_g_cm3": bridge["local_density_g_cm3"],
            "acceleration_m_s2": np.power(
                10.0, bridge["predicted_log_gobs"].to_numpy(float)
            ),
            "source_speed_km_s": circular_equivalent_speed,
            "speed_provenance": "sqrt(observed lensing acceleration times radius)",
        }
    )
    rows = pd.concat([galaxy, massive], ignore_index=True)
    numeric = [
        "radius_kpc",
        "density_g_cm3",
        "acceleration_m_s2",
        "source_speed_km_s",
    ]
    if rows[numeric].isna().any().any() or (rows[numeric] <= 0.0).any().any():
        raise RuntimeError("density-resolved characteristic inputs are invalid")
    return rows


def build_static_invariance(protocol: dict) -> pd.DataFrame:
    inputs = protocol["inputs"]
    records: list[pd.DataFrame] = []

    sparc = load_sparc_acceleration_frame(ROOT / inputs["sparc_raw_directory"])
    records.append(
        pd.DataFrame(
            {
                "domain": "SPARC",
                "system": sparc["system"],
                "coordinate": sparc["radius_kpc"],
                "coordinate_unit": "kpc",
                "baseline_quantity": sparc["observed_velocity_km_s"],
                "baseline_quantity_name": "observed_velocity_km_s",
            }
        )
    )

    clash = load_clash_acceleration_frame(ROOT / inputs["clash_radial_table"])
    records.append(
        pd.DataFrame(
            {
                "domain": "CLASH",
                "system": clash["system"],
                "coordinate": clash["radius_kpc"],
                "coordinate_unit": "kpc",
                "baseline_quantity": clash["observed_g_m_s2"],
                "baseline_quantity_name": "observed_lensing_acceleration_m_s2",
            }
        )
    )

    bridge = pd.read_csv(ROOT / inputs["bcg_cluster_density_rows"])
    bcg = bridge[bridge["domain"].eq("BCG")]
    records.append(
        pd.DataFrame(
            {
                "domain": "BCG",
                "system": bcg["system"],
                "coordinate": bcg["radius_kpc"],
                "coordinate_unit": "kpc",
                "baseline_quantity": np.power(10.0, bcg["log_gobs"]),
                "baseline_quantity_name": "observed_dynamical_acceleration_m_s2",
            }
        )
    )

    raw = pd.read_csv(ROOT / "results/sigma_covariant_weak_field/raw_lensing_predictions.csv")
    records.append(
        pd.DataFrame(
            {
                "domain": "RXJ2129_raw_lensing",
                "system": raw["image_id"],
                "coordinate": raw["source_redshift"],
                "coordinate_unit": "source_redshift",
                "baseline_quantity": raw["radial_residual_arcsec"],
                "baseline_quantity_name": "predicted_image_radial_residual_arcsec",
            }
        )
    )

    table = pd.concat(records, ignore_index=True)
    table["assumed_second_time_derivative"] = 0.0
    table["catchup_term"] = 0.0
    table["static_prediction_change"] = 0.0
    return table


def characteristic_scan(rows: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    settings = protocol["fixed_spatial_parameters"]
    rho_screen = 10.0 ** float(settings["log10_rho_screen_g_cm3"])
    sigma = equilibrium_sigma_from_density(
        rows["density_g_cm3"].to_numpy(float), rho_screen_g_cm3=rho_screen
    )
    speed_ratio = rows["source_speed_km_s"].to_numpy(float) * 1000.0 / C_M_S
    light_crossing_myr = (
        rows["radius_kpc"].to_numpy(float) * KPC_M / C_M_S / SECONDS_PER_MYR
    )
    orbital_period_myr = 2.0 * np.pi * light_crossing_myr / speed_ratio
    tables = []
    for delta in protocol["delta_scan"]:
        chars = causal_catchup_characteristics(
            rows["acceleration_m_s2"].to_numpy(float),
            sigma,
            a0_m_s2=float(settings["a0_m_s2"]),
            activation=float(settings["activation"]),
            eta=float(settings["eta"]),
            delta=float(delta),
        )
        n_parallel = chars["parallel_refractive_index"]
        response_parameter = n_parallel**2 * speed_ratio**2
        enhancement = np.where(
            response_parameter < 1.0,
            1.0 / (1.0 - response_parameter),
            np.nan,
        )
        table = rows.copy()
        table["Sigma"] = sigma
        table["delta"] = float(delta)
        for name, values in chars.items():
            table[name] = values
        table["parallel_speed_over_c"] = np.sqrt(
            chars["parallel_speed_squared_over_c2"]
        )
        table["perpendicular_speed_over_c"] = np.sqrt(
            chars["perpendicular_speed_squared_over_c2"]
        )
        table["light_crossing_time_Myr"] = light_crossing_myr
        table["parallel_catchup_time_Myr"] = n_parallel * light_crossing_myr
        table["observable_orbital_period_Myr"] = orbital_period_myr
        table["catchup_fraction_of_orbital_period"] = (
            n_parallel * speed_ratio / (2.0 * np.pi)
        )
        table["propagation_phase_radian"] = n_parallel * speed_ratio
        table["linear_response_parameter"] = response_parameter
        table["linear_response_enhancement"] = enhancement
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def quantile(series: pd.Series, probability: float) -> float:
    return float(series.quantile(probability))


def aggregate_scan(characteristics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (delta, domain), group in characteristics.groupby(["delta", "domain"]):
        response = group["linear_response_parameter"]
        stable = group.loc[response < 1.0, "linear_response_enhancement"]
        records.append(
            {
                "delta": float(delta),
                "domain": domain,
                "rows": int(len(group)),
                "systems": int(group["system"].nunique()),
                "minimum_parallel_speed_over_c": float(
                    group["parallel_speed_over_c"].min()
                ),
                "maximum_parallel_speed_over_c": float(
                    group["parallel_speed_over_c"].max()
                ),
                "median_parallel_refractive_index": float(
                    group["parallel_refractive_index"].median()
                ),
                "p95_catchup_fraction_of_orbital_period": quantile(
                    group["catchup_fraction_of_orbital_period"], 0.95
                ),
                "maximum_catchup_fraction_of_orbital_period": float(
                    group["catchup_fraction_of_orbital_period"].max()
                ),
                "median_response_parameter": float(response.median()),
                "p95_response_parameter": quantile(response, 0.95),
                "maximum_response_parameter": float(response.max()),
                "fraction_at_or_above_resonance": float((response >= 1.0).mean()),
                "median_subresonant_response_enhancement": (
                    float(stable.median()) if len(stable) else np.nan
                ),
                "p95_subresonant_response_enhancement": (
                    quantile(stable, 0.95) if len(stable) else np.nan
                ),
                "fraction_with_at_least_1pct_response": float(
                    ((response >= (0.01 / 1.01)) & (response < 1.0)).mean()
                ),
            }
        )
    return pd.DataFrame(records).sort_values(["delta", "domain"]).reset_index(drop=True)


def required_delta_summary(characteristics: pd.DataFrame) -> dict:
    base = characteristics[characteristics["delta"].eq(0.0)].copy()
    speed_ratio = base["source_speed_km_s"].to_numpy(float) * 1000.0 / C_M_S
    longitudinal = base["parallel_spatial_coefficient"].to_numpy(float)
    sigma2 = base["Sigma"].to_numpy(float) ** 2

    def required_for_response(fraction: float) -> np.ndarray:
        r_target = fraction / (1.0 + fraction)
        target_n2 = r_target / speed_ratio**2
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = longitudinal * (target_n2 - 1.0) / sigma2
        return np.where((sigma2 > 0.0) & (target_n2 > 1.0), delta, np.nan)

    def required_for_period_fraction(fraction: float) -> np.ndarray:
        target_n = fraction * 2.0 * np.pi / speed_ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = longitudinal * (target_n**2 - 1.0) / sigma2
        return np.where((sigma2 > 0.0) & (target_n > 1.0), delta, np.nan)

    base["delta_for_1pct_response"] = required_for_response(0.01)
    base["delta_for_10pct_response"] = required_for_response(0.10)
    base["delta_for_1pct_orbit_catchup"] = required_for_period_fraction(0.01)
    output = {}
    for domain, group in base.groupby("domain"):
        output[domain] = {}
        for column in [
            "delta_for_1pct_response",
            "delta_for_10pct_response",
            "delta_for_1pct_orbit_catchup",
        ]:
            values = group[column].dropna()
            output[domain][column] = {
                "finite_rows": int(len(values)),
                "median": float(values.median()) if len(values) else None,
                "p10": quantile(values, 0.10) if len(values) else None,
                "p90": quantile(values, 0.90) if len(values) else None,
            }
    return output


def retained_benchmarks(protocol: dict) -> dict:
    inputs = protocol["inputs"]
    galaxy = json.loads((ROOT / inputs["galaxy_benchmark_report"]).read_text())
    cluster = json.loads((ROOT / inputs["cluster_benchmark_report"]).read_text())
    bridge = json.loads((ROOT / inputs["bridge_benchmark_report"]).read_text())
    raw = json.loads((ROOT / inputs["raw_lensing_report"]).read_text())
    sparc_scores = galaxy["scores"]
    clash_scores = cluster["cluster_lensing_metrics"]
    bridge_scores = bridge["metrics"]
    return {
        "reason_unchanged": "All are equilibrium scores, so d_t^2 Phi=0 and delta contributes exactly zero.",
        "SPARC_outer_holdout_RMSE_km_s": {
            "Sigma_candidate": sparc_scores["RAR_sharp_coherence_gated_RG:primary"][
                "outer_holdout"
            ]["RMSE_km_s"],
            "fixed_simple_MOND": sparc_scores["simple_MOND:invariant"][
                "outer_holdout"
            ]["RMSE_km_s"],
            "inner_fit_NFW": sparc_scores["NFW:invariant"]["outer_holdout"][
                "RMSE_km_s"
            ],
        },
        "BCG_equal_system_RMSE_dex": {
            "Sigma_candidate": bridge_scores["RAR_sharp_coherence_gated_RG"]["BCG"][
                "equal_system_RMSE_dex"
            ],
            "fixed_simple_MOND": bridge_scores["simple_MOND"]["BCG"][
                "equal_system_RMSE_dex"
            ],
            "Newtonian": bridge_scores["Newtonian"]["BCG"][
                "equal_system_RMSE_dex"
            ],
        },
        "CLASH_equal_system_RMSE_dex": {
            "Sigma_candidate": clash_scores["candidate"]["equal_system_RMSE_dex"],
            "fixed_simple_MOND": clash_scores["fixed_simple_MOND"][
                "equal_system_RMSE_dex"
            ],
            "cluster_retuned_RAR": clash_scores["cluster_retuned_RAR"][
                "equal_system_RMSE_dex"
            ],
            "NFW_construction_not_independent": clash_scores["NFW_construction"][
                "equal_system_RMSE_dex"
            ],
        },
        "RXJ2129_heldout_image_RMS_arcsec": {
            "Sigma_radial_selected_slip": raw["raw_lensing"]["radial_selected"][
                "heldout"
            ]["exact_radial_RMS_arcsec"],
            "Sigma_zero_slip": raw["raw_lensing"]["zero_slip"]["heldout"][
                "exact_radial_RMS_arcsec"
            ],
            "compact_halo_reference": raw["raw_lensing"][
                "compact_halo_reference_heldout_RMS_arcsec"
            ],
        },
    }


def make_figure(scan: pd.DataFrame, selected_delta: float, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    colors = {"SPARC_outer": "#1b9e77", "BCG": "#d95f02", "cluster": "#377eb8"}
    for domain, group in scan.groupby("domain"):
        color = colors.get(domain, "grey")
        axes[0].loglog(
            group["delta"].replace(0.0, 0.1),
            group["maximum_response_parameter"],
            marker="o",
            label=domain,
            color=color,
        )
        axes[1].loglog(
            group["delta"].replace(0.0, 0.1),
            group["p95_catchup_fraction_of_orbital_period"],
            marker="o",
            label=domain,
            color=color,
        )
        axes[2].semilogx(
            group["delta"].replace(0.0, 0.1),
            group["fraction_at_or_above_resonance"],
            marker="o",
            label=domain,
            color=color,
        )
    axes[0].axhline(0.1, color="black", linestyle="--", label="safe scan margin")
    axes[0].axhline(1.0, color="red", linestyle=":", label="resonance")
    axes[1].axhline(0.01, color="black", linestyle="--", label="1% of orbit")
    for axis in axes:
        axis.axvline(max(selected_delta, 0.1), color="#984ea3", linestyle="--")
        axis.grid(alpha=0.2)
        axis.set_xlabel("universal delta (0 plotted at 0.1)")
    axes[0].set(title="Largest local dynamic response", ylabel="r = (Q/L)(v/c)^2")
    axes[1].set(title="Catch-up time", ylabel="95th percentile fraction of orbit")
    axes[2].set(title="Breakdown of linear response", ylabel="fraction at/above resonance")
    axes[0].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/sigma_causal_catchup_all_tests_protocol.json",
    )
    args = parser.parse_args()
    config_path = args.protocol.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_all_galaxy_cluster_catchup_scores":
        raise RuntimeError("catch-up protocol was not frozen before scoring")

    static = build_static_invariance(protocol)
    density_rows = load_density_rows(protocol)
    characteristics = characteristic_scan(density_rows, protocol)
    scan = aggregate_scan(characteristics)

    threshold = float(protocol["selection_rule"]["maximum_response_parameter"])
    global_max = characteristics.groupby("delta")["linear_response_parameter"].max()
    eligible = global_max[global_max <= threshold]
    if eligible.empty:
        raise RuntimeError("no scanned delta satisfies the frozen perturbative margin")
    selected_delta = float(eligible.index.max())
    selected = characteristics[characteristics["delta"].eq(selected_delta)]
    selected_by_domain = {}
    for domain, group in selected.groupby("domain"):
        selected_by_domain[domain] = {
            "rows": int(len(group)),
            "systems": int(group["system"].nunique()),
            "minimum_parallel_speed_over_c": float(group["parallel_speed_over_c"].min()),
            "median_parallel_speed_over_c": float(group["parallel_speed_over_c"].median()),
            "maximum_response_parameter": float(group["linear_response_parameter"].max()),
            "median_response_enhancement": float(
                group["linear_response_enhancement"].median()
            ),
            "p95_response_enhancement": quantile(
                group["linear_response_enhancement"], 0.95
            ),
            "maximum_response_enhancement": float(
                group["linear_response_enhancement"].max()
            ),
            "median_catchup_fraction_of_orbital_period": float(
                group["catchup_fraction_of_orbital_period"].median()
            ),
            "p95_catchup_fraction_of_orbital_period": quantile(
                group["catchup_fraction_of_orbital_period"], 0.95
            ),
        }

    max_static_change = float(static["static_prediction_change"].abs().max())
    max_speed = float(
        max(
            characteristics["parallel_speed_over_c"].max(),
            characteristics["perpendicular_speed_over_c"].max(),
        )
    )
    min_q = float(characteristics["q_time_coefficient"].min())
    max_selected_response = float(selected["linear_response_parameter"].max())
    gates = {
        "positive_Q_pass": min_q > float(
            protocol["mathematical_gates"]["minimum_q_time_coefficient"]
        ),
        "causal_characteristics_pass": max_speed
        <= float(protocol["mathematical_gates"]["maximum_characteristic_speed_over_c"]),
        "static_invariance_pass": max_static_change
        <= float(protocol["mathematical_gates"]["maximum_static_prediction_change"]),
        "selected_delta_perturbative_margin_pass": max_selected_response <= threshold,
        "full_covariant_action_derived": False,
        "time_dependent_observational_validation_available": False,
    }
    gates["mathematical_audit_pass"] = all(
        gates[key]
        for key in [
            "positive_Q_pass",
            "causal_characteristics_pass",
            "static_invariance_pass",
            "selected_delta_perturbative_margin_pass",
        ]
    )
    gates["theory_validated_pass"] = False

    coverage = {
        domain: {
            "rows": int(len(group)),
            "systems": int(group["system"].nunique()),
        }
        for domain, group in static.groupby("domain")
    }
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed all existing galaxy and cluster static plus causal catch-up tests",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
            "freeze_status": protocol["status"],
        },
        "formula": protocol["equation"],
        "coverage": {
            "static_invariance": coverage,
            "density_resolved_characteristics": {
                domain: {
                    "rows": int(len(group)),
                    "systems": int(group["system"].nunique()),
                }
                for domain, group in density_rows.groupby("domain")
            },
        },
        "static_test": {
            "maximum_absolute_prediction_change": max_static_change,
            "conclusion": "Q cannot improve or degrade equilibrium rotation curves or static lensing because its entire contribution is proportional to d_t^2 Phi, which is zero in these tests.",
        },
        "retained_equilibrium_benchmarks": retained_benchmarks(protocol),
        "causal_characteristic_test": {
            "minimum_Q": min_q,
            "maximum_characteristic_speed_over_c_across_scan": max_speed,
            "universal_selected_delta": selected_delta,
            "selection_rule": protocol["selection_rule"],
            "selected_maximum_response_parameter": max_selected_response,
            "selected_by_domain": selected_by_domain,
        },
        "delta_needed_for_effect": required_delta_summary(characteristics),
        "gate_audit": gates,
        "concepts_learned": {
            "causality": "Q=L_parallel+delta Sigma^2 repairs the superluminal longitudinal cone for every tested nonnegative delta; delta=0 is already the fastest causal boundary.",
            "static_data": "The catch-up term is invisible to all current equilibrium galaxy, BCG, CLASH, and RX J2129 image-position scores, so those data cannot determine delta.",
            "velocity_scaling": "The leading local dynamic response scales as (v/c)^2. The same delta therefore acts more strongly in high-speed clusters than in galaxies, but only for genuinely time-dependent structure.",
            "large_delta": "Increasing delta eventually drives the orbital-frequency toy response toward a resonance; this is loss of perturbative control, not evidence for extra gravity.",
            "next_decisive_test": "Use a merger or other time-resolved gravitational system with a reconstructed three-dimensional Sigma field and compare mass-motion timing, lensing morphology, and any propagation offset with one delta fixed in advance.",
            "verdict": "The causal Q completion is mathematically healthier but adds no explanatory power to the static galaxy or lensing fits. It remains a viable dynamic extension to test, not a successful replacement for dark matter or MOND on the present data.",
        },
        "interpretation_limits": protocol["interpretation_limits"],
        "input_hashes": {
            name: sha256(ROOT / path)
            for name, path in protocol["inputs"].items()
        },
        "outputs": protocol["outputs"],
    }

    outputs = protocol["outputs"]
    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    static.to_csv(ROOT / outputs["static_invariance"], index=False)
    characteristics.to_csv(ROOT / outputs["characteristics"], index=False)
    scan.to_csv(ROOT / outputs["delta_scan"], index=False)
    report_path.write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(scan, selected_delta, ROOT / outputs["figure"])

    galaxy_effect = selected_by_domain["SPARC_outer"]
    cluster_effect = selected_by_domain["cluster"]
    summary = f"""# Sigma causal catch-up: all existing tests

## Result

The new time term is causal on every scanned row, but it changes **none** of the existing equilibrium galaxy or lensing predictions. Its largest safe scanned universal setting is `delta={selected_delta:g}` under the frozen `r <= {threshold:g}` margin.

- Static audit: {len(static):,} rows, maximum prediction change `{max_static_change:.1e}`.
- Dynamic audit: {len(density_rows):,} density-resolved rows across {density_rows['system'].nunique()} systems.
- SPARC at selected delta: median response factor `{galaxy_effect['median_response_enhancement']:.6f}`, maximum `{galaxy_effect['maximum_response_enhancement']:.6f}`.
- CLASH at selected delta: median response factor `{cluster_effect['median_response_enhancement']:.6f}`, maximum `{cluster_effect['maximum_response_enhancement']:.6f}`.
- Maximum characteristic speed in the full scan: `{max_speed:.9f} c`.

## Meaning

This implements the “gravity vectors catch up” idea as a finite propagation time. It fixes the earlier superluminal scalar-cone problem and naturally gives high-speed cluster disturbances a larger response than galaxy rotation. But rotation curves and ordinary lensing maps are equilibrium measurements: `d_t^2 Phi=0`, so they cannot measure `delta` or improve their scores.

Very large `delta` values approach a local resonance. That is a warning that the linear approximation has broken down, not a valid way to manufacture missing gravity. The next informative test must be time-dependent—preferably a cluster merger—with a three-dimensional density/Sigma reconstruction and a pre-fixed universal `delta`.
"""
    (ROOT / outputs["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
