from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import KPC_M
from voidscreen.unified import (
    A0_M_S2,
    C_M_S,
    fit_unified_model,
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
    predict_acceleration,
    rar_acceleration,
)


def _predict_bcg(name: str, frame: pd.DataFrame, u0_vector) -> np.ndarray:
    gbar = frame["gbar_m_s2"].to_numpy(dtype=float)
    if name == "fixed_galaxy_rar":
        return rar_acceleration(gbar, A0_M_S2)
    if name == "cluster_scale_rar":
        return rar_acceleration(gbar, 2.0e-9)
    if name == "U0_emond_like":
        return predict_acceleration(
            name,
            gbar,
            frame["chi"],
            frame["ell_bar_kpc"],
            u0_vector,
            domain="galaxy",
        )
    raise ValueError(f"unknown BCG model: {name}")


def _model_slope(name: str, frame: pd.DataFrame, u0_vector) -> np.ndarray:
    epsilon = 1e-5
    radius_m = frame["radius_kpc"].to_numpy(dtype=float) * KPC_M
    base_gbar = frame["gbar_m_s2"].to_numpy(dtype=float)

    def shifted(sign: float) -> np.ndarray:
        shifted_frame = frame.copy()
        shifted_gbar = base_gbar * np.exp(sign * epsilon)
        shifted_frame["gbar_m_s2"] = shifted_gbar
        shifted_frame["chi"] = shifted_gbar * radius_m / (C_M_S**2)
        return _predict_bcg(name, shifted_frame, u0_vector)

    return (np.log(shifted(1.0)) - np.log(shifted(-1.0))) / (2.0 * epsilon)


def _score(name: str, frame: pd.DataFrame, u0_vector) -> pd.DataFrame:
    output = frame.copy()
    output["model"] = name
    output["predicted_g_m_s2"] = _predict_bcg(name, output, u0_vector)
    output["predicted_log_gobs"] = np.log10(output["predicted_g_m_s2"])
    slope = _model_slope(name, output, u0_vector)
    output["sigma_residual_dex"] = np.sqrt(
        output["err_log_gobs"].to_numpy() ** 2
        + (slope * output["err_log_gbar"].to_numpy()) ** 2
    )
    output["residual_dex"] = output["predicted_log_gobs"] - output["log_gobs"]
    output["chi2_term"] = (output["residual_dex"] / output["sigma_residual_dex"]) ** 2
    return output


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    residual = frame["residual_dex"].to_numpy(dtype=float)
    return {
        "bcgs": len(frame),
        "chi2_per_point": float(frame["chi2_term"].mean()),
        "rms_dex": float(np.sqrt(np.mean(residual**2))),
        "median_abs_dex": float(np.median(np.abs(residual))),
        "mean_residual_dex": float(np.mean(residual)),
    }


def _bootstrap_comparison(
    fixed: pd.DataFrame, candidate: pd.DataFrame, *, draws: int, seed: int
) -> dict[str, float | int]:
    merged = fixed[["plateifu", "chi2_term"]].merge(
        candidate[["plateifu", "chi2_term"]],
        on="plateifu",
        suffixes=("_fixed", "_candidate"),
    )
    delta = (merged["chi2_term_candidate"] - merged["chi2_term_fixed"]).to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 10_000):
        chunk = min(10_000, draws - start)
        indices = rng.integers(0, len(delta), size=(chunk, len(delta)))
        samples.append(delta[indices].mean(axis=1))
    distribution = np.concatenate(samples)
    return {
        "bcgs": len(delta),
        "candidate_minus_fixed_chi2_per_point": float(delta.mean()),
        "bootstrap_ci95_low": float(np.quantile(distribution, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_probability_candidate_improves": float(np.mean(distribution < 0.0)),
    }


def _figure(predictions: pd.DataFrame, destination: Path) -> None:
    colors = {
        "fixed_galaxy_rar": "#7f7f7f",
        "cluster_scale_rar": "#4c78a8",
        "U0_emond_like": "#f58518",
    }
    labels = {
        "fixed_galaxy_rar": "galaxy RAR",
        "cluster_scale_rar": "cluster-scale RAR",
        "U0_emond_like": "U0 prediction",
    }
    figure, axis = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    observed = predictions[predictions["model"] == "fixed_galaxy_rar"]
    axis.errorbar(
        observed["log_gbar"],
        observed["log_gobs"],
        xerr=observed["err_log_gbar"],
        yerr=observed["err_log_gobs"],
        fmt="o",
        color="black",
        alpha=0.45,
        markersize=3,
        label="MaNGA BCG dynamics",
    )
    order = np.argsort(observed["log_gbar"].to_numpy())
    for name, color in colors.items():
        frame = predictions[predictions["model"] == name].iloc[order]
        axis.plot(
            frame["log_gbar"],
            frame["predicted_log_gobs"],
            color=color,
            linewidth=1.8,
            label=labels[name],
        )
    axis.set(
        xlabel=r"$\log_{10} g_{\rm bar}$ (m s$^{-2}$)",
        ylabel=r"$\log_{10} g_{\rm dyn}$ (m s$^{-2}$)",
        title="Untuned intermediate-scale test on 50 MaNGA BCGs",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="External MaNGA BCG check of frozen U0.")
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--bcg",
        type=Path,
        default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "external_bcg")
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    galaxy = load_sparc_acceleration_frame(args.sparc)
    cluster = load_clash_acceleration_frame(args.clash)
    fit = fit_unified_model(
        "U0_emond_like", galaxy, cluster, starts=args.starts, seed=args.seed
    )

    bcg = pd.read_csv(args.bcg)
    bcg["gbar_m_s2"] = np.power(10.0, bcg["log_gbar"])
    radius_m = bcg["radius_kpc"].to_numpy(dtype=float) * KPC_M
    bcg["phi_bar_m2_s2"] = bcg["gbar_m_s2"].to_numpy() * radius_m
    bcg["chi"] = bcg["phi_bar_m2_s2"] / (C_M_S**2)
    bcg["ell_bar_kpc"] = bcg["radius_kpc"]

    u0_parameters = fit.parameters
    activation = expit(
        (np.log10(bcg["chi"]) - np.log10(u0_parameters["chi_t"]))
        / u0_parameters["w_dex"]
    )
    bcg["U0_activation"] = activation
    bcg["U0_a_eff_m_s2"] = A0_M_S2 * np.exp(
        np.log(u0_parameters["F"]) * activation
    )

    scored = [
        _score(name, bcg, fit.vector)
        for name in ("fixed_galaxy_rar", "cluster_scale_rar", "U0_emond_like")
    ]
    predictions = pd.concat(scored, ignore_index=True)
    metrics = {name: _metrics(frame) for name, frame in predictions.groupby("model")}
    fixed = predictions[predictions["model"] == "fixed_galaxy_rar"]
    u0 = predictions[predictions["model"] == "U0_emond_like"]
    report = {
        "status": "completed post-discovery external BCG check",
        "bcg_values_used_in_fit": False,
        "development_fit": {
            "SPARC_galaxies": int(galaxy["system"].nunique()),
            "CLASH_clusters": int(cluster["system"].nunique()),
            "train_chi2": fit.chi2,
            "parameters": fit.parameters,
            "optimizer_success": fit.success,
            "starts": fit.starts,
        },
        "bcg_geometry": {
            "potential": "gbar*r_last point-mass tail; BCG baryons only",
            "chi_min": float(bcg["chi"].min()),
            "chi_median": float(bcg["chi"].median()),
            "chi_max": float(bcg["chi"].max()),
            "U0_activation_median": float(bcg["U0_activation"].median()),
            "U0_a_eff_median_m_s2": float(bcg["U0_a_eff_m_s2"].median()),
        },
        "metrics": metrics,
        "U0_vs_fixed_galaxy_rar": _bootstrap_comparison(
            fixed, u0, draws=args.bootstrap_draws, seed=args.seed
        ),
        "guardrail": (
            "This is an external prediction from the frozen U0 formula but not a blind test; "
            "the paper's aggregate BCG result was known. No host-cluster potential was added."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "bcg_predictions.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _figure(predictions, args.output / "external_bcg_prediction.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
