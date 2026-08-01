from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.constitutive import (
    required_response,
    simple_mu_acceleration,
    standard_mu_acceleration,
)
from voidscreen.unified import (
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
    predict_acceleration,
    rar_acceleration,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {}
    return {
        "p05": float(np.quantile(finite, 0.05)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _relative_max(predicted, expected) -> float:
    predicted_values = np.asarray(predicted, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    return float(np.max(np.abs(predicted_values / expected_values - 1.0)))


def _add_targets(frame: pd.DataFrame, u0_vector: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    target = required_response(output["gbar_m_s2"], output["observed_g_m_s2"])
    for name, values in target.items():
        output[name] = values
    output["log10_chi"] = np.log10(output["chi"])
    output["log10_gbar"] = np.log10(output["gbar_m_s2"])
    output["log10_gobs"] = np.log10(output["observed_g_m_s2"])
    output["u0_predicted_g_m_s2"] = predict_acceleration(
        "U0_emond_like",
        output["gbar_m_s2"],
        output["chi"],
        output["ell_bar_kpc"],
        u0_vector,
        domain=str(output["domain"].iloc[0]),
    )
    output["u0_residual_dex"] = np.log10(
        output["u0_predicted_g_m_s2"] / output["observed_g_m_s2"]
    )
    return output


def _domain_summary(frame: pd.DataFrame) -> dict:
    valid = frame["inverse_valid"].to_numpy(dtype=bool)
    return {
        "systems": int(frame["system"].nunique()),
        "points": len(frame),
        "pointwise_inverse_valid_points": int(valid.sum()),
        "pointwise_inverse_valid_fraction": float(valid.mean()),
        "log10_chi": _quantiles(frame["log10_chi"]),
        "log10_gbar_m_s2": _quantiles(frame["log10_gbar"]),
        "log10_gobs_m_s2": _quantiles(frame["log10_gobs"]),
        "nu_required": _quantiles(frame["nu_required"]),
        "mu_required": _quantiles(frame["mu_required"]),
        "log10_rar_a_eff_m_s2_valid": _quantiles(
            np.log10(frame.loc[valid, "rar_a_eff_m_s2"])
        ),
        "log10_simple_a_x_m_s2_valid": _quantiles(
            np.log10(frame.loc[valid, "simple_a_x_m_s2"])
        ),
        "log10_standard_a_x_m_s2_valid": _quantiles(
            np.log10(frame.loc[valid, "standard_a_x_m_s2"])
        ),
        "frozen_u0_full_data_rms_dex": float(
            np.sqrt(np.mean(np.square(frame["u0_residual_dex"])))
        ),
        "frozen_u0_full_data_median_abs_dex": float(
            np.median(np.abs(frame["u0_residual_dex"]))
        ),
        "frozen_u0_full_data_mean_residual_dex": float(
            frame["u0_residual_dex"].mean()
        ),
    }


def _transition_support(
    frame: pd.DataFrame, *, chi_t: float, w_dex: float
) -> dict[str, float | int]:
    logit_10_to_90 = float(np.log(9.0))
    low = float(chi_t * 10.0 ** (-logit_10_to_90 * w_dex))
    high = float(chi_t * 10.0 ** (logit_10_to_90 * w_dex))
    selected = frame[frame["chi"].between(low, high, inclusive="both")]
    return {
        "activation_interval": "frozen U0 activation from 0.1 to 0.9",
        "chi_low": low,
        "chi_high": high,
        "systems": int(selected["system"].nunique()),
        "points": len(selected),
    }


def _support_overlap(galaxy: pd.DataFrame, cluster: pd.DataFrame) -> dict[str, float]:
    galaxy_values = galaxy["log10_chi"].to_numpy(dtype=float)
    cluster_values = cluster["log10_chi"].to_numpy(dtype=float)
    full_low = max(float(galaxy_values.min()), float(cluster_values.min()))
    full_high = min(float(galaxy_values.max()), float(cluster_values.max()))
    central_low = max(
        float(np.quantile(galaxy_values, 0.1)), float(np.quantile(cluster_values, 0.1))
    )
    central_high = min(
        float(np.quantile(galaxy_values, 0.9)), float(np.quantile(cluster_values, 0.9))
    )
    return {
        "full_log10_chi_overlap_dex": max(0.0, full_high - full_low),
        "central_10_to_90_percent_log10_chi_overlap_dex": max(
            0.0, central_high - central_low
        ),
        "galaxy_p90_to_cluster_p10_gap_dex": float(
            np.quantile(cluster_values, 0.1) - np.quantile(galaxy_values, 0.9)
        ),
    }


def _round_trip_report(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame["inverse_valid"].to_numpy(dtype=bool)
    gbar = frame.loc[valid, "gbar_m_s2"].to_numpy(dtype=float)
    observed = frame.loc[valid, "observed_g_m_s2"].to_numpy(dtype=float)
    return {
        "rar_max_relative_error": _relative_max(
            rar_acceleration(gbar, frame.loc[valid, "rar_a_eff_m_s2"]), observed
        ),
        "simple_mu_max_relative_error": _relative_max(
            simple_mu_acceleration(gbar, frame.loc[valid, "simple_a_x_m_s2"]),
            observed,
        ),
        "standard_mu_max_relative_error": _relative_max(
            standard_mu_acceleration(gbar, frame.loc[valid, "standard_a_x_m_s2"]),
            observed,
        ),
    }


def _plot_target(frame: pd.DataFrame, u0_parameters: dict, destination: Path) -> None:
    valid = frame[frame["inverse_valid"]].copy()
    colors = {"galaxy": "#4c78a8", "cluster": "#f58518"}
    labels = {"galaxy": "SPARC dynamics", "cluster": "CLASH lensing"}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for domain in ("galaxy", "cluster"):
        selected = valid[valid["domain"] == domain]
        axes[0].scatter(
            selected["log10_chi"],
            np.log10(selected["rar_a_eff_m_s2"]),
            s=8 if domain == "galaxy" else 24,
            alpha=0.12 if domain == "galaxy" else 0.55,
            color=colors[domain],
            label=labels[domain],
        )
        axes[1].scatter(
            selected["log10_gobs"],
            selected["mu_required"],
            s=8 if domain == "galaxy" else 24,
            alpha=0.12 if domain == "galaxy" else 0.55,
            color=colors[domain],
            label=labels[domain],
        )

    log_chi = np.linspace(frame["log10_chi"].min(), frame["log10_chi"].max(), 500)
    activation = expit(
        (log_chi - np.log10(u0_parameters["chi_t"])) / u0_parameters["w_dex"]
    )
    log_a_eff = np.log10(1.2e-10) + np.log10(u0_parameters["F"]) * activation
    axes[0].plot(log_chi, log_a_eff, color="black", linewidth=2, label="frozen U0 target")
    axes[0].set(
        xlabel=r"$\log_{10}(|\Phi_{\rm bar}|/c^2)$",
        ylabel=r"inferred $\log_{10}a_{\rm eff}$ (m s$^{-2}$)",
        title="Pointwise RAR-scale inverse",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set(
        xlabel=r"$\log_{10}g_{\rm obs}$ (m s$^{-2}$)",
        ylabel=r"required $\mu=g_{\rm bar}/g_{\rm obs}$",
        title="Constitutive response target",
        ylim=(0.0, min(2.0, float(frame["mu_required"].quantile(0.995)))),
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct the nonlinear constitutive response required by SPARC and CLASH."
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--u0-report",
        type=Path,
        default=ROOT / "results" / "external_bcg" / "report.json",
    )
    parser.add_argument(
        "--gates", type=Path, default=ROOT / "configs" / "theory_stage_gates.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "constitutive_target"
    )
    args = parser.parse_args()

    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    u0_report = json.loads(args.u0_report.read_text(encoding="utf-8"))
    u0_parameters = u0_report["development_fit"]["parameters"]
    u0_vector = np.asarray(
        [
            np.log10(u0_parameters["F"]),
            np.log10(u0_parameters["chi_t"]),
            u0_parameters["w_dex"],
        ]
    )

    galaxy = _add_targets(load_sparc_acceleration_frame(args.sparc), u0_vector)
    cluster = _add_targets(load_clash_acceleration_frame(args.clash), u0_vector)
    combined = pd.concat([galaxy, cluster], ignore_index=True)
    expected = gates["stage_0_controls"]["expected_data"]
    data_gate = {
        "sparc_systems": int(galaxy["system"].nunique()) == expected["sparc_systems"],
        "sparc_points": len(galaxy) == expected["sparc_points"],
        "clash_systems": int(cluster["system"].nunique()) == expected["clash_systems"],
        "clash_points": len(cluster) == expected["clash_points"],
    }
    inverse_minimum = gates["stage_1_constitutive_target"][
        "minimum_fraction_with_gobs_above_gbar_for_pointwise_inverse"
    ]
    round_trip_maximum = gates["stage_1_constitutive_target"][
        "round_trip_max_relative_error"
    ]
    transition_minimum = gates["stage_1_constitutive_target"][
        "required_transition_support_systems_per_domain"
    ]
    summaries = {
        "galaxy": _domain_summary(galaxy),
        "cluster": _domain_summary(cluster),
    }
    transition = {
        "galaxy": _transition_support(galaxy, **{
            "chi_t": u0_parameters["chi_t"], "w_dex": u0_parameters["w_dex"]
        }),
        "cluster": _transition_support(cluster, **{
            "chi_t": u0_parameters["chi_t"], "w_dex": u0_parameters["w_dex"]
        }),
    }
    round_trip = _round_trip_report(combined)
    inverse_gate = {
        domain: summaries[domain]["pointwise_inverse_valid_fraction"] >= inverse_minimum
        for domain in ("galaxy", "cluster")
    }
    transition_gate = {
        domain: transition[domain]["systems"] >= transition_minimum
        for domain in ("galaxy", "cluster")
    }
    numerical_gate = {
        name: value <= round_trip_maximum for name, value in round_trip.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    columns = [
        "domain",
        "system",
        "radius_kpc",
        "gbar_m_s2",
        "observed_g_m_s2",
        "chi",
        "ell_bar_kpc",
        "nu_required",
        "mu_required",
        "extra_g_m_s2",
        "inverse_valid",
        "rar_a_eff_m_s2",
        "simple_a_x_m_s2",
        "standard_a_x_m_s2",
        "u0_predicted_g_m_s2",
        "u0_residual_dex",
    ]
    combined[columns].to_csv(args.output / "targets.csv", index=False)
    report = {
        "status": "completed Stage 1 constitutive target reconstruction",
        "interpretation": (
            "Pointwise inverse values are diagnostics, not fitted measurements. "
            "Rows with gobs<=gbar are retained and marked invalid for the inverse."
        ),
        "inputs": {
            "clash_sha256": _sha256(args.clash),
            "u0_report_sha256": _sha256(args.u0_report),
            "stage_gate_registry_sha256": _sha256(args.gates),
        },
        "frozen_u0_parameters": u0_parameters,
        "domain_summaries": summaries,
        "support_overlap": _support_overlap(galaxy, cluster),
        "frozen_u0_transition_support": transition,
        "round_trip": round_trip,
        "gate_audit": {
            "data_counts": data_gate,
            "inverse_fraction": inverse_gate,
            "analytic_round_trip": numerical_gate,
            "transition_system_support": transition_gate,
            "pointwise_inverse_usable": all(inverse_gate.values())
            and all(numerical_gate.values()),
            "continuous_transition_independently_supported": all(transition_gate.values()),
        },
        "next_action": (
            "Use the forward likelihood for all points while deriving the minimal H7 closure. "
            "Treat insufficient cross-domain transition support as requiring independent BCG "
            "host potentials, not as permission to tune the logistic transition."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _plot_target(combined, u0_parameters, args.output / "constitutive_target.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
