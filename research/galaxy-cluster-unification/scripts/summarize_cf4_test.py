from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _galaxy_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[frame["split"] == "outer_holdout"].copy()
    residual = (
        selected["velocity_predicted_kms"] - selected["velocity_observed_adjusted_kms"]
    )
    selected["chi2"] = (residual / selected["velocity_error_total_kms"]) ** 2
    selected["squared_error"] = residual**2
    return (
        selected.groupby("galaxy", sort=True)
        .agg(n=("chi2", "size"), chi2=("chi2", "sum"), squared_error=("squared_error", "sum"))
        .reset_index()
    )


def _paired_bootstrap(
    baseline_path: Path,
    candidate_path: Path,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    baseline = _galaxy_metrics(baseline_path)
    candidate = _galaxy_metrics(candidate_path)
    merged = baseline.merge(candidate, on="galaxy", suffixes=("_baseline", "_candidate"))
    n = merged["n_baseline"].to_numpy(dtype=float)
    delta_chi2 = (
        merged["chi2_candidate"] - merged["chi2_baseline"]
    ).to_numpy(dtype=float)
    sse_baseline = merged["squared_error_baseline"].to_numpy(dtype=float)
    sse_candidate = merged["squared_error_candidate"].to_numpy(dtype=float)
    observed_chi2 = float(delta_chi2.sum() / n.sum())
    observed_rmse = float(
        math.sqrt(sse_candidate.sum() / n.sum()) - math.sqrt(sse_baseline.sum() / n.sum())
    )

    rng = np.random.default_rng(seed)
    boot_chi2 = []
    boot_rmse = []
    for start in range(0, draws, 10_000):
        chunk = min(10_000, draws - start)
        indices = rng.integers(0, len(merged), size=(chunk, len(merged)))
        points = n[indices].sum(axis=1)
        boot_chi2.append(delta_chi2[indices].sum(axis=1) / points)
        base = np.sqrt(sse_baseline[indices].sum(axis=1) / points)
        candidate_values = np.sqrt(sse_candidate[indices].sum(axis=1) / points)
        boot_rmse.append(candidate_values - base)
    chi2_values = np.concatenate(boot_chi2)
    rmse_values = np.concatenate(boot_rmse)
    return {
        "galaxies": len(merged),
        "points": int(n.sum()),
        "candidate_minus_baseline_chi2_per_point": observed_chi2,
        "chi2_ci95_low": float(np.quantile(chi2_values, 0.025)),
        "chi2_ci95_high": float(np.quantile(chi2_values, 0.975)),
        "bootstrap_probability_candidate_improves": float(np.mean(chi2_values < 0.0)),
        "candidate_minus_baseline_rmse_kms": observed_rmse,
        "rmse_ci95_low_kms": float(np.quantile(rmse_values, 0.025)),
        "rmse_ci95_high_kms": float(np.quantile(rmse_values, 0.975)),
        "galaxy_win_fraction": float(
            np.mean(merged["chi2_candidate"] < merged["chi2_baseline"])
        ),
    }


def _full_fit_inputs(primary: Path, sensitivity: Path) -> list[dict[str, Any]]:
    specs = [
        ("Newtonian", primary / "newtonian"),
        ("RAR", primary / "rar"),
        ("NFW", primary / "nfw"),
        ("Void free p", primary / "void"),
        ("Void p=0.5", primary / "void_p05"),
        ("Void env grouped 64", primary / "void_env"),
        ("Void p=0.5 env grouped 64", sensitivity / "void_p05_env_grouped64"),
        ("Void env ungrouped 64", sensitivity / "void_env_ungrouped64"),
        ("Void env ungrouped 128", sensitivity / "void_env_ungrouped128"),
    ]
    rows = []
    for label, directory in specs:
        summary = _read_json(directory / "summary.json")
        parameters = summary["global_parameters"]
        rows.append(
            {
                "label": label,
                "directory": directory.relative_to(ROOT).as_posix(),
                "train_chi2_per_point": summary["train"]["chi2_per_point"],
                "outer_chi2_per_point": summary["outer_holdout"]["chi2_per_point"],
                "outer_rmse_kms": summary["outer_holdout"]["rmse_kms"],
                "outer_mae_kms": summary["outer_holdout"]["mae_kms"],
                "power_p": parameters.get("power_p"),
                "environment_beta": parameters.get("environment_beta"),
                "transition_width_dex": parameters.get("transition_width_dex"),
            }
        )
    return rows


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(str(value) if isinstance(value, str) else _format_float(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _save_plot(
    output: Path,
    full_frame: pd.DataFrame,
    cv_report: dict[str, Any],
    fold_summary: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)

    selected = full_frame.loc[
        full_frame["label"].isin(
            [
                "RAR",
                "Void free p",
                "Void p=0.5",
                "Void env grouped 64",
                "Void p=0.5 env grouped 64",
                "Void env ungrouped 64",
                "Void env ungrouped 128",
            ]
        )
    ].sort_values("outer_chi2_per_point")
    axes[0].barh(selected["label"], selected["outer_chi2_per_point"], color="#4C78A8")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Outer radial holdout χ² / point", title="Same-galaxy outer prediction")

    cv_rows = pd.DataFrame(
        [
            {"variant": key, **value}
            for key, value in cv_report["variant_metrics"].items()
            if key != "newtonian_reference"
        ]
    ).sort_values("chi2_per_point")
    readable = {
        "rar_reference": "RAR",
        "void_no_environment": "Void, no environment",
        "void_env_grouped_64": "Void + grouped 64",
        "void_env_ungrouped_64": "Void + ungrouped 64",
        "void_env_ungrouped_128": "Void + ungrouped 128",
    }
    axes[1].barh(
        cv_rows["variant"].map(readable), cv_rows["chi2_per_point"], color="#F58518"
    )
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Strict held-out-galaxy χ² / point", title="Five-fold galaxy prediction")

    beta_variants = [
        "void_env_grouped_64",
        "void_env_ungrouped_64",
        "void_env_ungrouped_128",
    ]
    colors = ["#54A24B", "#E45756", "#B279A2"]
    beta_labels = ["Grouped 64", "Ungrouped 64", "Ungrouped 128"]
    for index, (variant, color, label) in enumerate(
        zip(beta_variants, colors, beta_labels, strict=True)
    ):
        values = fold_summary.loc[fold_summary["variant"] == variant, "environment_beta"]
        x = np.full(len(values), index, dtype=float) + np.linspace(-0.08, 0.08, len(values))
        axes[2].scatter(x, values, color=color, s=45, label=label)
        axes[2].plot([index - 0.18, index + 0.18], [values.mean()] * 2, color=color, linewidth=3)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set(
        xticks=range(3),
        xticklabels=beta_labels,
        ylabel="Environment coefficient β",
        title="β across held-out-galaxy folds",
    )
    axes[2].tick_params(axis="x", rotation=20)
    figure.suptitle("CF4 test of the smoothly screened void hypothesis", fontsize=15)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the completed CF4 theory test.")
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "results" / "cf4_primary_comparison_5000",
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=ROOT / "results" / "cf4_full_sensitivity_5000",
    )
    parser.add_argument(
        "--cross-validation",
        type=Path,
        default=ROOT / "results" / "cf4_galaxy_cv_5000",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "cf4_theory_test"
    )
    parser.add_argument(
        "--markdown", type=Path, default=ROOT / "docs" / "CF4_THEORY_TEST.md"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    full_frame = pd.DataFrame(_full_fit_inputs(args.primary, args.sensitivity))
    full_frame.to_csv(args.output / "full_fit_comparison.csv", index=False)
    cv_report = _read_json(args.cross_validation / "cross_validation_report.json")
    fold_summary = pd.read_csv(args.cross_validation / "fold_summary.csv")

    radial_pairs = {
        "free_grouped64_vs_no_environment": (
            args.primary / "void" / "predictions.csv",
            args.primary / "void_env" / "predictions.csv",
        ),
        "free_ungrouped64_vs_no_environment": (
            args.primary / "void" / "predictions.csv",
            args.sensitivity / "void_env_ungrouped64" / "predictions.csv",
        ),
        "free_ungrouped128_vs_no_environment": (
            args.primary / "void" / "predictions.csv",
            args.sensitivity / "void_env_ungrouped128" / "predictions.csv",
        ),
        "fixed_p_grouped64_vs_no_environment": (
            args.primary / "void_p05" / "predictions.csv",
            args.sensitivity / "void_p05_env_grouped64" / "predictions.csv",
        ),
    }
    radial_bootstrap = {
        label: _paired_bootstrap(
            baseline,
            candidate,
            draws=args.bootstrap_draws,
            seed=5090 + index,
        )
        for index, (label, (baseline, candidate)) in enumerate(radial_pairs.items())
    }

    void_folds = fold_summary.loc[fold_summary["variant"].str.startswith("void")]
    fold_parameters = (
        void_folds.groupby("variant")[["environment_beta", "power_p", "transition_width_dex"]]
        .agg(["mean", "min", "max"])
        .to_dict()
    )
    result = {
        "verdict": (
            "The current data do not support the specific claim that a stronger CF4 void "
            "environment increases the additional galactic acceleration."
        ),
        "full_radial_holdout": full_frame.to_dict(orient="records"),
        "paired_radial_bootstrap": radial_bootstrap,
        "strict_galaxy_cross_validation": cv_report,
        "fold_parameter_summary": {
            str(key): {str(inner_key): value for inner_key, value in values.items()}
            for key, values in fold_parameters.items()
        },
    }
    (args.output / "theory_test_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _save_plot(args.output / "cf4_test_summary.png", full_frame, cv_report, fold_summary)

    display_full = full_frame[
        [
            "label",
            "train_chi2_per_point",
            "outer_chi2_per_point",
            "outer_rmse_kms",
            "power_p",
            "environment_beta",
        ]
    ]
    full_table = _markdown_table(
        display_full,
        [
            ("label", "Model"),
            ("train_chi2_per_point", "Train χ²/pt"),
            ("outer_chi2_per_point", "Outer χ²/pt"),
            ("outer_rmse_kms", "Outer RMSE km/s"),
            ("power_p", "p"),
            ("environment_beta", "β"),
        ],
    )
    cv_frame = pd.DataFrame(
        [
            {"variant": key, **value}
            for key, value in cv_report["variant_metrics"].items()
        ]
    )
    cv_frame["variant"] = cv_frame["variant"].replace(
        {
            "newtonian_reference": "Newtonian",
            "rar_reference": "RAR",
            "void_no_environment": "Void, no environment",
            "void_env_grouped_64": "Void + grouped 64",
            "void_env_ungrouped_64": "Void + ungrouped 64",
            "void_env_ungrouped_128": "Void + ungrouped 128",
        }
    )
    cv_table = _markdown_table(
        cv_frame.sort_values("chi2_per_point"),
        [
            ("variant", "Model"),
            ("chi2_per_point", "Held-out χ²/pt"),
            ("rmse_kms", "Held-out RMSE km/s"),
        ],
    )

    cv_pairs = cv_report["paired_environment_vs_no_environment"]
    grouped = cv_pairs["void_env_grouped_64"]
    ungrouped = cv_pairs["void_env_ungrouped_64"]
    high_resolution = cv_pairs["void_env_ungrouped_128"]
    noenv_p = void_folds.loc[
        void_folds["variant"] == "void_no_environment", "power_p"
    ]
    markdown = f"""# CF4 test of the smoothly screened void hypothesis

Status: completed confirmatory engineering/statistical test, 2026-07-26.

## Verdict

**The current SPARC + Cosmicflows-4 test does not support the specific claim
that stronger surrounding void density produces a larger anomalous galactic
acceleration.**

![CF4 test summary](../results/cf4_theory_test/cf4_test_summary.png)

The smooth low-acceleration law is decisively better than Newtonian baryons
alone, so it remains a useful phenomenological rotation-curve formula. The
void-specific prediction fails the preregistered robustness conditions:

- The primary grouped score gives positive β in all five galaxy folds, but it
  *worsens* strict held-out-galaxy χ² by {grouped['candidate_minus_baseline_chi2_per_point']:.3f}
  per point (95% galaxy-bootstrap interval
  {grouped['chi2_per_point_bootstrap_ci95_low']:.3f} to
  {grouped['chi2_per_point_bootstrap_ci95_high']:.3f}).
- Ungrouped 64^3 also worsens held-out prediction by
  {ungrouped['candidate_minus_baseline_chi2_per_point']:.3f} χ²/point.
- The 128^3 reconstruction improves the point estimate by
  {-high_resolution['candidate_minus_baseline_chi2_per_point']:.3f} χ²/point, but
  its 95% interval crosses zero and its mean β is
  {high_resolution['mean_fold_beta']:.3f}, opposite the predicted sign.
- The free exponent is stable across folds at {noenv_p.mean():.3f}
  (range {noenv_p.min():.3f}–{noenv_p.max():.3f}), below the flat-curve value
  p = 0.5.
- Strict galaxy CV favors fixed empirical RAR over every tested free-p void
  model ({cv_report['variant_metrics']['rar_reference']['chi2_per_point']:.3f}
  versus {cv_report['variant_metrics']['void_no_environment']['chi2_per_point']:.3f}
  χ²/point without environment).

Positive β in a training fit is therefore not sufficient evidence: in the
primary reconstruction it fails to predict new galaxies, and its sign is not
stable across CF4 releases.

## Data and locked design

- 175 SPARC galaxies received independent CF4 scores; the preregistered cuts
  retain 131 galaxies and 3,034 radial measurements.
- Primary environment: negative grouped 64^3 CF4 density contrast. Ungrouped
  64^3 and the official 128^3 release are frozen sensitivities.
- Radial test: optimize on each galaxy's inner 70%, predict its outer 30%.
- Galaxy test: five folds balanced across the primary environment score. Global
  parameters train on four folds using all their radii. No velocity from the
  held-out galaxies enters optimization; held-out nuisance parameters remain at
  their prior centers.
- Each fitted model receives 5,000 Adam steps in float64 on the RTX 5090.
- Paired uncertainty intervals use 100,000 galaxy-level bootstrap resamples.

The CF4 grids and axis convention come from the [official Cosmicflows
release](https://projets.ip2i.in2p3.fr/cosmicflows/); the reconstruction method
is described by [Courtois et al. (2023)](https://doi.org/10.1051/0004-6361/202245331),
and the 128^3 sensitivity is the [official Zenodo
release](https://doi.org/10.5281/zenodo.20653238).

## Full radial-holdout results

{full_table}

The fixed-p model is the best radial extrapolator in this run, slightly ahead of
RAR, but adding grouped environment to that model worsens its outer score. The
free-p environmental improvements seen in one reconstruction do not reproduce
across the other grids.

## Strict held-out-galaxy results

{cv_table}

These absolute χ² values are high because the strict test does not calibrate any
nuisance parameter with held-out velocities. The paired environment/no-
environment comparison remains fair because both models receive exactly the
same information.

## Decision-rule audit

| Preregistered requirement | Outcome |
|---|---|
| Better than Newtonian baryons | Pass |
| Free p robustly approaches 0.5 | Fail; fold range {noenv_p.min():.3f}–{noenv_p.max():.3f} |
| Competitive with RAR on new galaxies | Fail |
| Positive β from independent environment | Mixed by reconstruction |
| Environment improves held-out galaxies | Fail for grouped and ungrouped 64^3 |
| Positive β survives catalog sensitivity | Fail; 128^3 mean β is {high_resolution['mean_fold_beta']:.3f} |

Overall: **the universal low-acceleration phenomenology remains viable, but the
specific CF4 void-enhancement interpretation is not supported by this test.**

## Limitations

- This is MAP optimization, not a full posterior analysis.
- CF4's published 2-D error products cannot be propagated as voxel-wise 3-D
  uncertainties with the available metadata.
- SPARC rotation curves primarily trace H I/H-alpha gas, not individual outer
  stars.
- The strict galaxy test fixes held-out nuisance parameters at their priors. A
  second hierarchical test could calibrate nuisance parameters from only the
  inner radii, but it must be declared as a distinct design.
- Testing a new potential-screened or differently smoothed model would be a new
  hypothesis, not a rescue of this preregistered equation.
"""
    args.markdown.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.output / 'theory_test_report.json'}")


if __name__ == "__main__":
    main()
