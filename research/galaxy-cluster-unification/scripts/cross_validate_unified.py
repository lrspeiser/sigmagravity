from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.unified import (
    MODEL_NAMES,
    assign_system_folds,
    fit_unified_model,
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
    prediction_frame,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(predictions: pd.DataFrame) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for model, model_frame in predictions.groupby("model", sort=False):
        domains = {}
        for domain, frame in model_frame.groupby("domain", sort=False):
            residual = frame["residual"].to_numpy(dtype=float)
            record = {
                "systems": int(frame["system"].nunique()),
                "points": len(frame),
                "chi2": float(frame["chi2_term"].sum()),
                "chi2_per_point": float(frame["chi2_term"].mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "median_abs_residual": float(np.median(np.abs(residual))),
                "residual_unit": "km/s" if domain == "galaxy" else "dex",
            }
            if domain == "cluster":
                intrinsic_sigma = np.sqrt(frame["sigma"].to_numpy() ** 2 + 0.063**2)
                record["chi2_per_point_with_0p063_dex_intrinsic_scatter"] = float(
                    np.mean((residual / intrinsic_sigma) ** 2)
                )
            domains[domain] = record
        domains["equal_domain_macro_chi2_per_point"] = 0.5 * (
            domains["galaxy"]["chi2_per_point"] + domains["cluster"]["chi2_per_point"]
        )
        output[str(model)] = domains
    return output


def _system_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["squared_residual"] = frame["residual"] ** 2
    return (
        frame.groupby(["model", "domain", "system"], sort=True)
        .agg(n=("chi2_term", "size"), chi2=("chi2_term", "sum"), sse=("squared_residual", "sum"))
        .reset_index()
    )


def _paired_arrays(
    scores: pd.DataFrame, baseline: str, candidate: str, domain: str
) -> tuple[np.ndarray, np.ndarray]:
    left = scores[(scores["model"] == baseline) & (scores["domain"] == domain)]
    right = scores[(scores["model"] == candidate) & (scores["domain"] == domain)]
    merged = left.merge(right, on=["domain", "system"], suffixes=("_baseline", "_candidate"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("paired system scores do not align")
    if not np.array_equal(merged["n_baseline"], merged["n_candidate"]):
        raise ValueError("paired models have different point counts")
    n = merged["n_baseline"].to_numpy(dtype=float)
    delta = (merged["chi2_candidate"] - merged["chi2_baseline"]).to_numpy(dtype=float)
    return n, delta


def _domain_bootstrap(
    n: np.ndarray, delta: np.ndarray, *, draws: int, seed: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 5000):
        chunk = min(5000, draws - start)
        indices = rng.integers(0, len(n), size=(chunk, len(n)))
        samples.append(delta[indices].sum(axis=1) / n[indices].sum(axis=1))
    distribution = np.concatenate(samples)
    return {
        "systems": len(n),
        "points": int(n.sum()),
        "candidate_minus_fixed_rar_chi2_per_point": float(delta.sum() / n.sum()),
        "bootstrap_ci95_low": float(np.quantile(distribution, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_probability_candidate_improves": float(np.mean(distribution < 0.0)),
    }


def _macro_bootstrap(
    galaxy: tuple[np.ndarray, np.ndarray],
    cluster: tuple[np.ndarray, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    galaxy_n, galaxy_delta = galaxy
    cluster_n, cluster_delta = cluster
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 5000):
        chunk = min(5000, draws - start)
        galaxy_indices = rng.integers(
            0, len(galaxy_n), size=(chunk, len(galaxy_n))
        )
        cluster_indices = rng.integers(
            0, len(cluster_n), size=(chunk, len(cluster_n))
        )
        galaxy_score = (
            galaxy_delta[galaxy_indices].sum(axis=1) / galaxy_n[galaxy_indices].sum(axis=1)
        )
        cluster_score = (
            cluster_delta[cluster_indices].sum(axis=1) / cluster_n[cluster_indices].sum(axis=1)
        )
        samples.append(0.5 * (galaxy_score + cluster_score))
    distribution = np.concatenate(samples)
    observed = 0.5 * (
        galaxy_delta.sum() / galaxy_n.sum() + cluster_delta.sum() / cluster_n.sum()
    )
    return {
        "galaxy_systems": len(galaxy_n),
        "cluster_systems": len(cluster_n),
        "candidate_minus_fixed_rar_macro_chi2": float(observed),
        "bootstrap_ci95_low": float(np.quantile(distribution, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_probability_candidate_improves": float(np.mean(distribution < 0.0)),
    }


def _comparisons(
    predictions: pd.DataFrame, metrics: dict[str, dict], *, draws: int, seed: int
) -> dict[str, dict]:
    scores = _system_scores(predictions)
    baseline = metrics["fixed_rar"]
    output = {}
    for index, candidate in enumerate(MODEL_NAMES[1:]):
        galaxy = _paired_arrays(scores, "fixed_rar", candidate, "galaxy")
        cluster = _paired_arrays(scores, "fixed_rar", candidate, "cluster")
        macro = _macro_bootstrap(
            galaxy, cluster, draws=draws, seed=seed + 100 * index + 2
        )
        candidate_metrics = metrics[candidate]
        gate = {
            "cluster_improves": candidate_metrics["cluster"]["chi2_per_point"]
            < baseline["cluster"]["chi2_per_point"],
            "galaxy_within_5_percent": candidate_metrics["galaxy"]["chi2_per_point"]
            <= 1.05 * baseline["galaxy"]["chi2_per_point"],
            "macro_improves": candidate_metrics["equal_domain_macro_chi2_per_point"]
            < baseline["equal_domain_macro_chi2_per_point"],
            "macro_bootstrap_ci_excludes_zero": macro["bootstrap_ci95_high"] < 0.0,
        }
        gate["passes_all"] = all(gate.values())
        output[candidate] = {
            "galaxy": _domain_bootstrap(
                *galaxy, draws=draws, seed=seed + 100 * index
            ),
            "cluster": _domain_bootstrap(
                *cluster, draws=draws, seed=seed + 100 * index + 1
            ),
            "equal_domain_macro": macro,
            "advancement_gate": gate,
        }
    return output


def _score_figure(metrics: dict[str, dict], destination: Path) -> None:
    labels = list(MODEL_NAMES)
    values = {
        "Galaxy": [metrics[name]["galaxy"]["chi2_per_point"] for name in labels],
        "Cluster lensing": [metrics[name]["cluster"]["chi2_per_point"] for name in labels],
        "Equal-domain macro": [
            metrics[name]["equal_domain_macro_chi2_per_point"] for name in labels
        ],
    }
    short = ["RAR", "J0", "U0", "U1", "oracle"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    colors = ["#7f7f7f", "#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for axis, (title, scores) in zip(axes, values.items(), strict=True):
        axis.bar(short, scores, color=colors)
        axis.set_title(title)
        axis.set_ylabel(r"held-out $\chi^2$ / point")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("One law tested on SPARC speeds and CLASH lensing")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _prediction_figure(predictions: pd.DataFrame, destination: Path) -> None:
    selected_models = ["fixed_rar", "U0_emond_like", "U1_coherence_length"]
    colors = {"fixed_rar": "#7f7f7f", "U0_emond_like": "#f58518", "U1_coherence_length": "#54a24b"}
    labels = {"fixed_rar": "fixed RAR", "U0_emond_like": "U0 potential", "U1_coherence_length": "U1 length"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)

    galaxy = predictions[predictions["domain"] == "galaxy"]
    for model in selected_models:
        frame = galaxy[galaxy["model"] == model]
        axes[0].scatter(
            frame["observed_velocity_km_s"],
            frame["predicted_velocity_km_s"],
            s=5,
            alpha=0.18,
            color=colors[model],
            label=labels[model],
        )
    limit = float(
        max(galaxy["observed_velocity_km_s"].max(), galaxy["predicted_velocity_km_s"].max())
    )
    axes[0].plot([0, limit], [0, limit], color="black", linewidth=1)
    axes[0].set(
        xlabel="observed circular speed (km/s)",
        ylabel="held-out predicted speed (km/s)",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    axes[0].legend(markerscale=3)

    cluster = predictions[predictions["domain"] == "cluster"]
    observed = cluster[cluster["model"] == "fixed_rar"]
    axes[1].errorbar(
        observed["log_gbar"],
        observed["log_gtot"],
        xerr=observed["err_log_gbar"],
        yerr=observed["err_log_gtot"],
        fmt="o",
        markersize=3,
        alpha=0.35,
        color="black",
        label="CLASH lensing",
    )
    for model in selected_models:
        frame = cluster[cluster["model"] == model]
        axes[1].scatter(
            frame["log_gbar"],
            frame["predicted_log_gtot"],
            s=11,
            alpha=0.65,
            color=colors[model],
            label=labels[model],
        )
    axes[1].set(
        xlabel=r"$\log_{10} g_{\rm bar}$ (m s$^{-2}$)",
        ylabel=r"$\log_{10} g_{\rm lens}$ (m s$^{-2}$)",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint whole-system validation on SPARC dynamics and CLASH lensing."
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "configs" / "unified_model_registry.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "unified_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if registry["status"] != "frozen_before_fit":
        raise ValueError("unified model registry is not frozen")
    if args.seed != registry["random_seed"] or args.folds != registry["validation"]["folds"]:
        raise ValueError("seed and fold count must match the preregistered registry")

    galaxy = assign_system_folds(
        load_sparc_acceleration_frame(args.sparc), folds=args.folds, seed=args.seed
    )
    cluster = assign_system_folds(
        load_clash_acceleration_frame(args.clash), folds=args.folds, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    assignments = pd.concat(
        [
            galaxy[["domain", "system", "fold"]].drop_duplicates(),
            cluster[["domain", "system", "fold"]].drop_duplicates(),
        ],
        ignore_index=True,
    ).sort_values(["domain", "system"])
    assignments.to_csv(args.output / "fold_assignments.csv", index=False)

    prediction_pieces = []
    fit_records = []
    for fold in range(args.folds):
        galaxy_train = galaxy[galaxy["fold"] != fold]
        galaxy_test = galaxy[galaxy["fold"] == fold]
        cluster_train = cluster[cluster["fold"] != fold]
        cluster_test = cluster[cluster["fold"] == fold]
        for model_index, model in enumerate(MODEL_NAMES):
            print(f"fold={fold} model={model}", flush=True)
            fit = fit_unified_model(
                model,
                galaxy_train,
                cluster_train,
                starts=args.starts,
                seed=args.seed + 1000 * fold + model_index,
            )
            fit_records.append(
                {
                    "fold": fold,
                    "model": model,
                    "train_chi2": fit.chi2,
                    "optimizer_success": fit.success,
                    **fit.parameters,
                }
            )
            for test in (galaxy_test, cluster_test):
                predicted = prediction_frame(model, fit.vector, test)
                prediction_pieces.append(predicted)

    predictions = pd.concat(prediction_pieces, ignore_index=True)
    predictions.to_csv(args.output / "heldout_predictions.csv", index=False)
    fit_frame = pd.DataFrame(fit_records)
    fit_frame.to_csv(args.output / "fold_parameters.csv", index=False)
    system_scores = _system_scores(predictions)
    system_scores.to_csv(args.output / "heldout_system_scores.csv", index=False)
    metrics = _metrics(predictions)
    comparisons = _comparisons(
        predictions, metrics, draws=args.bootstrap_draws, seed=args.seed
    )
    report = {
        "status": "completed preregistered joint validation",
        "design": {
            "galaxy_holdout_unit": "whole galaxy",
            "cluster_holdout_unit": "whole cluster",
            "folds": args.folds,
            "optimizer_starts": args.starts,
            "bootstrap_draws": args.bootstrap_draws,
            "likelihood_weighting": "ordinary summed standardized residuals; no domain reweighting",
            "metric_slip": "fixed Phi=Psi; no fitted lensing multiplier",
        },
        "data": {
            "galaxy_systems": int(galaxy["system"].nunique()),
            "galaxy_points": len(galaxy),
            "cluster_systems": int(cluster["system"].nunique()),
            "cluster_points": len(cluster),
            "clash_sha256": _sha256(args.clash),
            "registry_sha256": _sha256(args.registry),
        },
        "heldout_metrics": metrics,
        "paired_comparisons_vs_fixed_rar": comparisons,
        "interpretation_guardrail": (
            "U0 is an EMOND-like prior-art control. U1 is a phenomenological closure, "
            "not a covariant theory. Passing predicts these observables only under the "
            "declared symmetry and no-slip assumptions."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _score_figure(metrics, args.output / "unified_score_summary.png")
    _prediction_figure(predictions, args.output / "unified_prediction_summary.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
