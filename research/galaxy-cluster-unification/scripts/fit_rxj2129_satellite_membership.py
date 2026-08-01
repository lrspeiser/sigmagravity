"""Fit the frozen RX J2129 satellite-membership probability model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit_rxj2129_satellite_training import _read_molino


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_satellite_classifier_protocol.json"
FEATURES = [
    "photoz_abs_offset",
    "photoz_interval_width",
    "photoz_odds",
    "log1p_photoz_chi2",
    "f814w_mass_mag",
    "f606w_minus_f814w_mass_color",
    "photoz_template_type",
    "log10_stellar_mass",
]


def _resolve(path: str) -> Path:
    return ROOT / path


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    values[(values >= 90.0) | ~np.isfinite(values)] = np.nan
    return values


def _features(frame: pd.DataFrame, prefix: str, cluster_redshift: float) -> pd.DataFrame:
    photoz = _numeric(frame, f"{prefix}zb_1")
    low = _numeric(frame, f"{prefix}zb_Min_1")
    high = _numeric(frame, f"{prefix}zb_Max_1")
    chi2 = _numeric(frame, f"{prefix}Chi2")
    f606 = _numeric(frame, f"{prefix}F606W_ACS_MASS")
    f814 = _numeric(frame, f"{prefix}F814W_ACS_MASS")
    return pd.DataFrame(
        {
            "photoz_abs_offset": np.abs(photoz - cluster_redshift),
            "photoz_interval_width": high - low,
            "photoz_odds": _numeric(frame, f"{prefix}Odds_1"),
            "log1p_photoz_chi2": np.log1p(np.clip(chi2, 0.0, None)),
            "f814w_mass_mag": f814,
            "f606w_minus_f814w_mass_color": f606 - f814,
            "photoz_template_type": _numeric(frame, f"{prefix}Tb_1"),
            "log10_stellar_mass": _numeric(frame, f"{prefix}Stell_Mass"),
        },
        index=frame.index,
    )


def _pipeline(config: dict[str, Any]) -> Pipeline:
    processor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", processor),
            (
                "model",
                LogisticRegression(
                    C=config["model"]["regularization_C"],
                    class_weight=config["model"]["class_weight"],
                    max_iter=config["model"]["maximum_iterations"],
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _spatial_groups(frame: pd.DataFrame, center_ra: float, center_dec: float) -> np.ndarray:
    east = (
        (pd.to_numeric(frame["molino_RA"], errors="coerce").to_numpy() - center_ra)
        * np.cos(np.deg2rad(center_dec))
    )
    north = pd.to_numeric(frame["molino_Dec"], errors="coerce").to_numpy() - center_dec
    angle = np.mod(np.arctan2(north, east), 2.0 * np.pi)
    return np.floor(angle / (2.0 * np.pi) * 12.0).astype(int)


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index < bins - 1
            else probability <= edges[index + 1]
        )
        if not selected.any():
            continue
        value += selected.sum() / total * abs(y[selected].mean() - probability[selected].mean())
    return float(value)


def _plot(
    y: np.ndarray,
    probability: np.ndarray,
    candidates: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    edges = np.linspace(0.0, 1.0, 6)
    centers: list[float] = []
    observed: list[float] = []
    for index in range(5):
        mask = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index < 4
            else probability <= edges[index + 1]
        )
        if mask.any():
            centers.append(float(probability[mask].mean()))
            observed.append(float(y[mask].mean()))
    axes[0].plot([0, 1], [0, 1], color="0.5", linestyle="--")
    axes[0].plot(centers, observed, marker="o")
    axes[0].set(xlabel="held-out predicted membership", ylabel="observed member fraction")
    axes[1].errorbar(
        candidates["separation_arcsec"],
        candidates["membership_probability"],
        yerr=np.vstack(
            [
                candidates["membership_probability"]
                - candidates["membership_probability_p16"],
                candidates["membership_probability_p84"]
                - candidates["membership_probability"],
            ]
        ),
        fmt="o",
        markersize=3,
        alpha=0.7,
    )
    axes[1].set(xlabel="projected separation (arcsec)", ylabel="membership probability")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("RX J2129 frozen satellite-membership likelihood")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    if not authorization["membership_classifier_fit"]:
        raise ValueError("membership classifier is not authorized")
    if authorization["off_center_mass_acceleration_likelihood"]:
        raise ValueError("classifier protocol cannot authorize acceleration mapping")
    if authorization["lens_residual_read"] or authorization["gravity_response_fit"]:
        raise ValueError("classifier protocol cannot read a residual")
    crossmatch = json.loads(
        _resolve(config["inputs"]["crossmatch_report"]).read_text(encoding="utf-8")
    )
    if not crossmatch["training_viability_gate_pass"]:
        raise ValueError("spectroscopic training viability gate failed")
    training_all = pd.read_csv(_resolve(config["inputs"]["training_ledger"]))
    cluster_redshift = config["training_domain"]["cluster_redshift"]
    training = training_all[
        (training_all["radius_from_bcg_arcsec"]
         > config["training_domain"]["bcg_exclusion_radius_arcsec"])
        & (training_all["molino_zb_Min_1"] <= cluster_redshift)
        & (training_all["molino_zb_Max_1"] >= cluster_redshift)
    ].copy().reset_index(drop=True)
    x = _features(training, "molino_", cluster_redshift)
    y = training["is_cluster_member"].astype(bool).to_numpy(dtype=int)
    groups = _spatial_groups(training, 322.41651, 0.08923)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2129)
    oof = np.full(len(training), np.nan)
    baseline_oof = np.full(len(training), np.nan)
    fold_index = np.full(len(training), -1, dtype=int)
    fold_metrics: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(x, y, groups)):
        model = _pipeline(config)
        model.fit(x.iloc[train_indices], y[train_indices])
        prediction = model.predict_proba(x.iloc[test_indices])[:, 1]
        prevalence = float(y[train_indices].mean())
        oof[test_indices] = prediction
        baseline_oof[test_indices] = prevalence
        fold_index[test_indices] = fold
        fold_brier = brier_score_loss(y[test_indices], prediction)
        baseline_brier = brier_score_loss(
            y[test_indices], np.full(len(test_indices), prevalence)
        )
        fold_metrics.append(
            {
                "fold": fold,
                "test_count": int(len(test_indices)),
                "test_members": int(y[test_indices].sum()),
                "spatial_groups": sorted(np.unique(groups[test_indices]).tolist()),
                "brier": float(fold_brier),
                "prevalence_brier": float(baseline_brier),
                "beats_prevalence_brier": bool(fold_brier < baseline_brier),
            }
        )
    if not np.isfinite(oof).all() or (fold_index < 0).any():
        raise ValueError("incomplete out-of-fold predictions")
    brier = float(brier_score_loss(y, oof))
    baseline_brier = float(brier_score_loss(y, baseline_oof))
    heldout_log_loss = float(log_loss(y, oof, labels=[0, 1]))
    baseline_log_loss = float(log_loss(y, baseline_oof, labels=[0, 1]))
    metrics = {
        "training_rows": int(len(training)),
        "training_members": int(y.sum()),
        "training_nonmembers": int((1 - y).sum()),
        "oof_brier": brier,
        "prevalence_oof_brier": baseline_brier,
        "oof_brier_improvement_fraction": 1.0 - brier / baseline_brier,
        "oof_log_loss": heldout_log_loss,
        "prevalence_oof_log_loss": baseline_log_loss,
        "oof_log_loss_improvement_fraction": 1.0 - heldout_log_loss / baseline_log_loss,
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "five_bin_expected_calibration_error": _ece(y, oof),
        "spatial_folds_beating_prevalence_brier": int(
            sum(item["beats_prevalence_brier"] for item in fold_metrics)
        ),
    }
    thresholds = config["advance_thresholds"]
    checks = {
        "minimum_oof_brier_improvement_fraction_over_fold_prevalence": metrics[
            "oof_brier_improvement_fraction"
        ]
        >= thresholds["minimum_oof_brier_improvement_fraction_over_fold_prevalence"],
        "minimum_oof_log_loss_improvement_fraction_over_fold_prevalence": metrics[
            "oof_log_loss_improvement_fraction"
        ]
        >= thresholds["minimum_oof_log_loss_improvement_fraction_over_fold_prevalence"],
        "minimum_oof_roc_auc": metrics["oof_roc_auc"]
        >= thresholds["minimum_oof_roc_auc"],
        "maximum_five_bin_expected_calibration_error": metrics[
            "five_bin_expected_calibration_error"
        ]
        <= thresholds["maximum_five_bin_expected_calibration_error"],
        "minimum_spatial_folds_beating_prevalence_brier": metrics[
            "spatial_folds_beating_prevalence_brier"
        ]
        >= thresholds["minimum_spatial_folds_beating_prevalence_brier"],
    }
    molino = _read_molino(_resolve(config["inputs"]["molino_catalog"]))
    candidates = pd.read_csv(_resolve(config["inputs"]["candidate_ledger"]))
    candidates = candidates[
        candidates["separation_arcsec"] <= config["candidate_scope"]["maximum_radius_arcsec"]
    ].copy()
    catalog_columns = [
        "CLASHID", "zb_1", "zb_Min_1", "zb_Max_1", "Odds_1", "Chi2", "Tb_1",
        "F606W_ACS_MASS", "F814W_ACS_MASS", "Stell_Mass", "a", "b", "theta",
    ]
    candidates = candidates.merge(
        molino[catalog_columns], left_on="clash_id", right_on="CLASHID", how="left",
        validate="one_to_one",
    )
    candidate_x = _features(candidates, "", cluster_redshift)
    rng = np.random.default_rng(config["model"]["random_seed"])
    bootstrap_predictions = []
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    for _ in range(config["model"]["bootstrap_replicates"]):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        model = _pipeline(config)
        model.fit(x.iloc[sampled], y[sampled])
        bootstrap_predictions.append(model.predict_proba(candidate_x)[:, 1])
    bootstrap = np.asarray(bootstrap_predictions)
    candidates["membership_probability"] = np.mean(bootstrap, axis=0)
    candidates["membership_probability_p16"] = np.quantile(bootstrap, 0.16, axis=0)
    candidates["membership_probability_p84"] = np.quantile(bootstrap, 0.84, axis=0)
    label_map = training_all.set_index("molino_CLASHID")["is_cluster_member"].to_dict()
    known = candidates["clash_id"].isin(label_map)
    known_values = candidates.loc[known, "clash_id"].map(label_map).astype(float)
    for column in (
        "membership_probability",
        "membership_probability_p16",
        "membership_probability_p84",
    ):
        candidates.loc[known, column] = known_values.to_numpy()
    bootstrap[:, known.to_numpy()] = known_values.to_numpy()[None, :]
    candidates["membership_probability_source"] = np.where(
        known, "MUSE_spectroscopic_label", "bootstrap_logistic_classifier"
    )
    probability_columns = [
        "membership_probability",
        "membership_probability_p16",
        "membership_probability_p84",
    ]
    probabilities = candidates[probability_columns].to_numpy()
    probability_check = bool(
        np.isfinite(probabilities).all()
        and (probabilities >= 0.0).all()
        and (probabilities <= 1.0).all()
        and (candidates["membership_probability_p16"]
             <= candidates["membership_probability"]).all()
        and (candidates["membership_probability"]
             <= candidates["membership_probability_p84"]).all()
    )
    checks["candidate_probabilities_and_intervals_finite_and_bounded"] = probability_check
    gate_pass = all(checks.values())
    oof_frame = training[
        ["muse_id", "molino_CLASHID", "is_cluster_member", "radius_from_bcg_arcsec"]
    ].copy()
    oof_frame["spatial_group"] = groups
    oof_frame["fold"] = fold_index
    oof_frame["oof_membership_probability"] = oof
    oof_frame["fold_training_prevalence_probability"] = baseline_oof
    oof_path = _resolve(config["outputs"]["oof_predictions"])
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(oof_path, index=False)
    membership_path = _resolve(config["outputs"]["membership_likelihood"])
    candidates.to_csv(membership_path, index=False)
    bootstrap_path = _resolve(config["outputs"]["membership_probability_bootstrap"])
    np.savez_compressed(
        bootstrap_path,
        clash_ids=candidates["clash_id"].astype(str).to_numpy(dtype="U"),
        membership_probability=bootstrap,
    )
    _plot(y, oof, candidates, _resolve(config["outputs"]["diagnostic"]))
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "membership_probability_likelihood_complete_mass_mapping_pending"
            if gate_pass
            else "membership_classifier_heldout_gate_failed"
        ),
        "gravity_or_lens_residual_read": False,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "checks": checks,
        "membership_probability_gate_pass": gate_pass,
        "candidate_count_inside_30arcsec": int(len(candidates)),
        "spectroscopically_labeled_candidates_inside_30arcsec": int(known.sum()),
        "bootstrap_replicates": int(bootstrap.shape[0]),
        "off_center_mass_acceleration_likelihood_complete": False,
        "strict_r1_ready": False,
        "outputs": config["outputs"],
        "next_action": (
            "Freeze and propagate Bernoulli membership plus lognormal stellar-mass "
            "draws through the off-center force geometry in all four dynamics bins."
            if gate_pass
            else "Record normalized satellite membership as unavailable; do not tune features or thresholds after failure."
        ),
    }
    report_path = _resolve(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(fit(args.config), indent=2))


if __name__ == "__main__":
    main()
