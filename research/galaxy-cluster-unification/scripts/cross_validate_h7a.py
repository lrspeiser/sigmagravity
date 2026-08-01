from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.theory import (
    H7A_MODEL_NAME,
    H7S_MODEL_NAME,
    fit_h7a,
    fit_h7s,
    h7a_bounds,
    h7a_prediction_frame,
    h7s_prediction_frame,
)
from voidscreen.unified import (
    assign_system_folds,
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
)


def _metrics(predictions: pd.DataFrame) -> dict[str, dict]:
    output = {}
    for model, model_frame in predictions.groupby("model", sort=False):
        domains = {}
        for domain, frame in model_frame.groupby("domain", sort=False):
            residual = frame["residual"].to_numpy(dtype=float)
            record = {
                "systems": int(frame["system"].nunique()),
                "points": len(frame),
                "chi2_per_point": float(frame["chi2_term"].mean()),
                "rms": float(np.sqrt(np.mean(np.square(residual)))),
                "median_abs_residual": float(np.median(np.abs(residual))),
                "residual_unit": "km/s" if domain == "galaxy" else "dex",
            }
            if domain == "cluster":
                intrinsic_sigma = np.sqrt(np.square(frame["sigma"]) + 0.063**2)
                record["chi2_per_point_with_0p063_dex_intrinsic_scatter"] = float(
                    np.mean(np.square(residual / intrinsic_sigma))
                )
            domains[domain] = record
        domains["equal_domain_macro_chi2_per_point"] = 0.5 * (
            domains["galaxy"]["chi2_per_point"]
            + domains["cluster"]["chi2_per_point"]
        )
        output[str(model)] = domains
    return output


def _system_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["model", "domain", "system"], sort=True)
        .agg(n=("chi2_term", "size"), chi2=("chi2_term", "sum"))
        .reset_index()
    )


def _paired_values(
    scores: pd.DataFrame, *, candidate: str, comparator: str, domain: str
) -> tuple[np.ndarray, np.ndarray]:
    candidate_frame = scores[
        (scores["model"] == candidate) & (scores["domain"] == domain)
    ]
    comparator_frame = scores[
        (scores["model"] == comparator) & (scores["domain"] == domain)
    ]
    paired = comparator_frame.merge(
        candidate_frame, on=["domain", "system"], suffixes=("_comparator", "_candidate")
    )
    if len(paired) != len(candidate_frame) or len(paired) != len(comparator_frame):
        raise ValueError("candidate and comparator systems do not align")
    if not np.array_equal(paired["n_comparator"], paired["n_candidate"]):
        raise ValueError("candidate and comparator point counts do not align")
    return (
        paired["n_candidate"].to_numpy(dtype=float),
        (paired["chi2_candidate"] - paired["chi2_comparator"]).to_numpy(dtype=float),
    )


def _comparison(
    scores: pd.DataFrame,
    *,
    candidate: str,
    comparator: str,
    draws: int,
    seed: int,
) -> dict:
    arrays = {
        domain: _paired_values(
            scores, candidate=candidate, comparator=comparator, domain=domain
        )
        for domain in ("galaxy", "cluster")
    }
    rng = np.random.default_rng(seed)
    distributions = {"galaxy": [], "cluster": [], "macro": []}
    for start in range(0, draws, 5000):
        chunk = min(5000, draws - start)
        domain_draws = {}
        for domain in ("galaxy", "cluster"):
            n, delta = arrays[domain]
            indices = rng.integers(0, len(n), size=(chunk, len(n)))
            values = delta[indices].sum(axis=1) / n[indices].sum(axis=1)
            distributions[domain].append(values)
            domain_draws[domain] = values
        distributions["macro"].append(
            0.5 * (domain_draws["galaxy"] + domain_draws["cluster"])
        )

    output = {}
    for name, pieces in distributions.items():
        distribution = np.concatenate(pieces)
        if name == "macro":
            observed = 0.5 * sum(
                arrays[domain][1].sum() / arrays[domain][0].sum()
                for domain in ("galaxy", "cluster")
            )
        else:
            n, delta = arrays[name]
            observed = delta.sum() / n.sum()
        output[name] = {
            "candidate_minus_comparator_chi2_per_point": float(observed),
            "bootstrap_ci95_low": float(np.quantile(distribution, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(distribution, 0.975)),
            "bootstrap_probability_candidate_improves": float(
                np.mean(distribution < 0.0)
            ),
        }
    return output


def _at_bound(vector: np.ndarray) -> bool:
    lower, upper = h7a_bounds()
    tolerance = 1e-5 * (upper - lower)
    return bool(np.any(vector - lower <= tolerance) or np.any(upper - vector <= tolerance))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-fold whole-system validation of the H7a simple-mu closure."
    )
    parser.add_argument("--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--unified-predictions",
        type=Path,
        default=ROOT / "results" / "unified_cv" / "heldout_predictions.csv",
    )
    parser.add_argument(
        "--gates", type=Path, default=ROOT / "configs" / "theory_stage_gates.json"
    )
    parser.add_argument("--family", choices=("simple", "standard"), default="simple")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    if args.folds != 5:
        raise ValueError("H7a must use the existing five-fold design")
    galaxy = assign_system_folds(
        load_sparc_acceleration_frame(args.sparc), folds=args.folds, seed=args.seed
    )
    cluster = assign_system_folds(
        load_clash_acceleration_frame(args.clash), folds=args.folds, seed=args.seed
    )
    if args.family == "simple":
        model_name = H7A_MODEL_NAME
        fit_function = fit_h7a
        prediction_function = h7a_prediction_frame
        default_output = ROOT / "results" / "h7a_cv"
        derivation = "docs/H7A_WEAK_FIELD_DERIVATION.md"
        cycle_status = "completed first H7 weak-field cycle"
    else:
        model_name = H7S_MODEL_NAME
        fit_function = fit_h7s
        prediction_function = h7s_prediction_frame
        default_output = ROOT / "results" / "h7s_cv"
        derivation = "standard-mu second constitutive cycle"
        cycle_status = "completed second H7 weak-field cycle"
    output_path = args.output or default_output
    fit_records = []
    prediction_pieces = []
    raw_vectors = []
    for fold in range(args.folds):
        print(f"fold={fold} model={model_name}", flush=True)
        fit = fit_function(
            galaxy[galaxy["fold"] != fold],
            cluster[cluster["fold"] != fold],
            starts=args.starts,
            seed=args.seed + 1000 * fold,
        )
        raw_vectors.append(fit.vector)
        fit_records.append(
            {
                "fold": fold,
                "train_chi2": fit.chi2,
                "optimizer_success": fit.success,
                "at_hard_bound": _at_bound(fit.vector),
                "log10_F": fit.vector[0],
                "log10_chi_t": fit.vector[1],
                **fit.parameters,
            }
        )
        for test in (
            galaxy[galaxy["fold"] == fold],
            cluster[cluster["fold"] == fold],
        ):
            prediction_pieces.append(prediction_function(fit.vector, test))

    candidate_predictions = pd.concat(prediction_pieces, ignore_index=True)
    comparator_predictions = pd.read_csv(args.unified_predictions)
    comparator_predictions = comparator_predictions[
        comparator_predictions["model"].isin(["fixed_rar", "U0_emond_like"])
    ]
    predictions = pd.concat(
        [comparator_predictions, candidate_predictions], ignore_index=True, sort=False
    )
    metrics = _metrics(predictions)
    scores = _system_scores(predictions)
    comparisons = {
        comparator: _comparison(
            scores,
            candidate=model_name,
            comparator=comparator,
            draws=args.bootstrap_draws,
            seed=args.seed + index,
        )
        for index, comparator in enumerate(("fixed_rar", "U0_emond_like"))
    }
    candidate_metrics = metrics[model_name]
    continue_thresholds = gates["stage_3_whole_system_validation"]["continue_gate"]
    scientific_thresholds = gates["stage_3_whole_system_validation"][
        "scientific_success_gate"
    ]
    fit_frame = pd.DataFrame(fit_records)
    continue_gate = {
        "sparc": candidate_metrics["galaxy"]["chi2_per_point"]
        <= continue_thresholds["sparc_chi2_per_point_max"],
        "clash_raw": candidate_metrics["cluster"]["chi2_per_point"]
        <= continue_thresholds["clash_raw_chi2_per_point_max"],
        "equal_domain_macro": candidate_metrics["equal_domain_macro_chi2_per_point"]
        <= continue_thresholds["equal_domain_macro_chi2_per_point_max"],
        "environment_sign_all_folds": bool((fit_frame["F"] > 1.0 + 1e-8).all()),
        "no_parameter_at_hard_bound": not bool(fit_frame["at_hard_bound"].any()),
    }
    continue_gate["passes_all"] = all(continue_gate.values())
    scientific_gate = {
        "sparc": candidate_metrics["galaxy"]["chi2_per_point"]
        <= scientific_thresholds["sparc_chi2_per_point_max"],
        "clash_with_intrinsic_scatter": candidate_metrics["cluster"][
            "chi2_per_point_with_0p063_dex_intrinsic_scatter"
        ]
        <= scientific_thresholds["clash_chi2_per_point_with_0p063_dex_scatter_max"],
        "clash_rms": candidate_metrics["cluster"]["rms"]
        <= scientific_thresholds["clash_rms_dex_max"],
        "macro_beats_fixed_rar": comparisons["fixed_rar"]["macro"][
            "bootstrap_ci95_high"
        ]
        <= scientific_thresholds["paired_macro_bootstrap_ci95_high_max"],
    }
    scientific_gate["passes_all"] = all(scientific_gate.values())

    output_path.mkdir(parents=True, exist_ok=True)
    candidate_predictions.to_csv(output_path / "heldout_predictions.csv", index=False)
    fit_frame.to_csv(output_path / "fold_parameters.csv", index=False)
    report = {
        "status": cycle_status,
        "model": model_name,
        "derivation": derivation,
        "design": {
            "folds": args.folds,
            "starts": args.starts,
            "bootstrap_draws": args.bootstrap_draws,
            "global_parameters": 3,
            "per_object_force_parameters": 0,
            "lensing_only_parameters": 0,
            "preliminary_metric_assumption": "Phi=Psi; H6 relativistic completion not yet derived",
        },
        "heldout_metrics": metrics,
        "paired_comparisons": comparisons,
        "fold_parameter_summary": {
            "F_min": float(fit_frame["F"].min()),
            "F_max": float(fit_frame["F"].max()),
            "chi_t_min": float(fit_frame["chi_t"].min()),
            "chi_t_max": float(fit_frame["chi_t"].max()),
            "w_dex_min": float(fit_frame["w_dex"].min()),
            "w_dex_max": float(fit_frame["w_dex"].max()),
            "folds_at_hard_bound": int(fit_frame["at_hard_bound"].sum()),
        },
        "gate_audit": {
            "continue": continue_gate,
            "scientific_success": scientific_gate,
        },
        "decision_rule": (
            "Advance to H7b only if the continue gate passes. If the constitutive shape "
            "fails while U0 passes, test one other action-derived mu family without "
            "adding an environment parameter."
        ),
    }
    (output_path / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
