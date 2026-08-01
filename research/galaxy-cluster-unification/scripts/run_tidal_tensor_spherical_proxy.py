#!/usr/bin/env python3
"""Cross-validate spherical tidal-response mappings on SPARC and CLASH."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.photon_joint_forward import stable_source_folds  # noqa: E402
from voidscreen.tidal_tensor_spherical import spherical_boost  # noqa: E402

KPC_M = 3.085677581491367e19


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frames(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = protocol["inputs"]
    sparc = pd.read_csv(ROOT / inputs["SPARC_predictions"])
    sparc = sparc[
        (sparc["model"] == "fixed_RAR")
        & (sparc["scenario"] == "invariant")
        & (sparc["split"] == "outer_holdout")
    ].copy()
    sparc = sparc.rename(columns={"galaxy": "system"})
    clash = pd.read_csv(ROOT / inputs["CLASH_points"])
    clash = clash.rename(columns={"log_gobs": "observed_log_g"})
    if len(sparc) != 968 or sparc["system"].nunique() != 131:
        raise RuntimeError("unexpected SPARC outer sample")
    if len(clash) != 72 or clash["system"].nunique() != 20:
        raise RuntimeError("unexpected scored CLASH sample")
    return sparc, clash


def parameter_grid(protocol: dict, family: str) -> np.ndarray:
    start, stop, count = protocol["parameter_grids"][family]
    return np.linspace(float(start), float(stop), int(count))


def predict_sparc(
    frame: pd.DataFrame,
    protocol: dict,
    family: str,
    power: float,
    kappa: float,
) -> np.ndarray:
    boost = spherical_boost(
        frame["g_bar_m_s2"].to_numpy(float),
        kappa=kappa,
        family=family,
        gate_power=power,
        a0_m_s2=float(protocol["inputs"]["a0_m_s2"]),
        radial_q=float(protocol["inputs"]["spherical_radial_Q_eigenvalue"]),
    )
    radius_m = frame["radius_adjusted_kpc"].to_numpy(float) * KPC_M
    return np.sqrt(
        frame["g_bar_m_s2"].to_numpy(float) * boost * radius_m
    ) / 1000.0


def predict_clash(
    frame: pd.DataFrame,
    protocol: dict,
    family: str,
    power: float,
    kappa: float,
) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    boost = spherical_boost(
        gbar,
        kappa=kappa,
        family=family,
        gate_power=power,
        a0_m_s2=float(protocol["inputs"]["a0_m_s2"]),
        radial_q=float(protocol["inputs"]["spherical_radial_Q_eigenvalue"]),
    )
    return np.log10(gbar * boost)


def equal_system_mse(frame: pd.DataFrame, residual: np.ndarray) -> float:
    temporary = pd.DataFrame(
        {"system": frame["system"].to_numpy(str), "square": np.square(residual)}
    )
    return float(temporary.groupby("system")["square"].mean().mean())


def objective(
    sparc: pd.DataFrame,
    clash: pd.DataFrame,
    protocol: dict,
    family: str,
    power: float,
    kappa: float,
) -> float:
    sparc_prediction = predict_sparc(sparc, protocol, family, power, kappa)
    sparc_log_residual = np.log10(sparc_prediction) - np.log10(
        sparc["velocity_observed_adjusted_km_s"].to_numpy(float)
    )
    clash_log_residual = (
        predict_clash(clash, protocol, family, power, kappa)
        - clash["observed_log_g"].to_numpy(float)
    )
    return 0.5 * (
        equal_system_mse(sparc, sparc_log_residual)
        + equal_system_mse(clash, clash_log_residual)
    )


def select_kappa(
    sparc: pd.DataFrame,
    clash: pd.DataFrame,
    protocol: dict,
    family: str,
    power: float,
) -> tuple[float, float]:
    grid = parameter_grid(protocol, family)
    values = np.asarray(
        [
            objective(sparc, clash, protocol, family, power, float(kappa))
            for kappa in grid
        ]
    )
    selected = int(np.argmin(values))
    return float(grid[selected]), float(values[selected])


def select_domain_kappa(
    frame: pd.DataFrame,
    protocol: dict,
    family: str,
    power: float,
    *,
    domain: str,
) -> tuple[float, float]:
    grid = parameter_grid(protocol, family)
    values = []
    for kappa in grid:
        if domain == "SPARC":
            prediction = predict_sparc(
                frame, protocol, family, power, float(kappa)
            )
            residual = np.log10(prediction) - np.log10(
                frame["velocity_observed_adjusted_km_s"].to_numpy(float)
            )
        elif domain == "CLASH":
            prediction = predict_clash(
                frame, protocol, family, power, float(kappa)
            )
            residual = prediction - frame["observed_log_g"].to_numpy(float)
        else:
            raise ValueError(f"unknown domain: {domain}")
        values.append(equal_system_mse(frame, residual))
    values = np.asarray(values)
    selected = int(np.argmin(values))
    return float(grid[selected]), float(values[selected])


def score_sparc(
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, float | int]:
    residual = prediction - frame["velocity_observed_adjusted_km_s"].to_numpy(float)
    system_rmse = (
        pd.DataFrame(
            {"system": frame["system"].to_numpy(str), "square": np.square(residual)}
        )
        .groupby("system")["square"]
        .mean()
        .pow(0.5)
    )
    return {
        "points": int(len(frame)),
        "systems": int(frame["system"].nunique()),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_system_RMSE_km_s": float(np.sqrt(np.mean(np.square(system_rmse)))),
    }


def score_clash(
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, float | int]:
    residual = prediction - frame["observed_log_g"].to_numpy(float)
    return {
        "points": int(len(frame)),
        "systems": int(frame["system"].nunique()),
        "point_RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_system_RMSE_dex": float(
            np.sqrt(equal_system_mse(frame, residual))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "tidal_tensor_spherical_proxy_protocol.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_spherical_proxy_scoring":
        raise RuntimeError("protocol was not frozen before proxy scoring")
    sparc, clash = load_frames(protocol)
    folds = int(protocol["validation"]["folds"])
    seed = int(protocol["validation"]["fold_seed"])
    sparc_fold = stable_source_folds(sparc["system"], folds, seed)
    clash_fold = stable_source_folds(clash["system"], folds, seed + 1)

    model_rows = []
    fold_rows = []
    prediction_rows = []
    for mapping in ("linear", "reciprocal", "exponential"):
        for power in protocol["formula_families"]["gate_powers"]:
            label = f"{mapping}_gate_n{power:g}"
            full_kappa, full_objective = select_kappa(
                sparc, clash, protocol, mapping, float(power)
            )
            sparc_only_kappa, sparc_only_objective = select_domain_kappa(
                sparc,
                protocol,
                mapping,
                float(power),
                domain="SPARC",
            )
            clash_only_kappa, clash_only_objective = select_domain_kappa(
                clash,
                protocol,
                mapping,
                float(power),
                domain="CLASH",
            )
            sparc_full_prediction = predict_sparc(
                sparc, protocol, mapping, float(power), full_kappa
            )
            clash_full_prediction = predict_clash(
                clash, protocol, mapping, float(power), full_kappa
            )
            sparc_score = score_sparc(sparc, sparc_full_prediction)
            clash_score = score_clash(clash, clash_full_prediction)

            heldout_sparc_predictions = np.empty(len(sparc))
            heldout_clash_predictions = np.empty(len(clash))
            fold_kappas = []
            for fold in range(folds):
                selected_kappa, train_objective = select_kappa(
                    sparc[sparc_fold != fold],
                    clash[clash_fold != fold],
                    protocol,
                    mapping,
                    float(power),
                )
                fold_kappas.append(selected_kappa)
                heldout_sparc_predictions[sparc_fold == fold] = predict_sparc(
                    sparc[sparc_fold == fold],
                    protocol,
                    mapping,
                    float(power),
                    selected_kappa,
                )
                heldout_clash_predictions[clash_fold == fold] = predict_clash(
                    clash[clash_fold == fold],
                    protocol,
                    mapping,
                    float(power),
                    selected_kappa,
                )
                fold_rows.append(
                    {
                        "model": label,
                        "fold": fold,
                        "kappa": selected_kappa,
                        "training_objective": train_objective,
                    }
                )
            sparc_cv = score_sparc(sparc, heldout_sparc_predictions)
            clash_cv = score_clash(clash, heldout_clash_predictions)
            relative_range = (
                (max(fold_kappas) - min(fold_kappas)) / np.median(fold_kappas)
                if np.median(fold_kappas) > 0.0
                else math.inf
            )
            gates = protocol["advance_gates"]
            survives = bool(
                sparc_cv["RMSE_km_s"]
                <= float(gates["SPARC_outer_RMSE_km_s_max"])
                and clash_cv["equal_system_RMSE_dex"]
                <= float(gates["CLASH_equal_cluster_RMSE_dex_max"])
                and relative_range
                <= float(gates["fold_kappa_relative_range_max"])
            )
            model_rows.append(
                {
                    "model": label,
                    "mapping": mapping,
                    "gate_power": power,
                    "full_kappa": full_kappa,
                    "full_objective": full_objective,
                    "SPARC_only_kappa": sparc_only_kappa,
                    "SPARC_only_objective": sparc_only_objective,
                    "CLASH_only_kappa": clash_only_kappa,
                    "CLASH_only_objective": clash_only_objective,
                    "CLASH_to_SPARC_kappa_ratio": (
                        clash_only_kappa / sparc_only_kappa
                        if sparc_only_kappa > 0.0
                        else math.inf
                    ),
                    "fold_kappa_min": min(fold_kappas),
                    "fold_kappa_median": float(np.median(fold_kappas)),
                    "fold_kappa_max": max(fold_kappas),
                    "fold_kappa_relative_range": relative_range,
                    "SPARC_full_RMSE_km_s": sparc_score["RMSE_km_s"],
                    "SPARC_heldout_RMSE_km_s": sparc_cv["RMSE_km_s"],
                    "SPARC_heldout_equal_system_RMSE_km_s": sparc_cv[
                        "equal_system_RMSE_km_s"
                    ],
                    "CLASH_full_equal_system_RMSE_dex": clash_score[
                        "equal_system_RMSE_dex"
                    ],
                    "CLASH_heldout_equal_system_RMSE_dex": clash_cv[
                        "equal_system_RMSE_dex"
                    ],
                    "CLASH_heldout_point_RMSE_dex": clash_cv["point_RMSE_dex"],
                    "survives": survives,
                }
            )
            prediction_rows.extend(
                {
                    "model": label,
                    "domain": "SPARC",
                    "system": system,
                    "fold": int(fold),
                    "observed": observed,
                    "prediction": prediction,
                    "residual": prediction - observed,
                }
                for system, fold, observed, prediction in zip(
                    sparc["system"],
                    sparc_fold,
                    sparc["velocity_observed_adjusted_km_s"],
                    heldout_sparc_predictions,
                    strict=True,
                )
            )
            prediction_rows.extend(
                {
                    "model": label,
                    "domain": "CLASH",
                    "system": system,
                    "fold": int(fold),
                    "observed": observed,
                    "prediction": prediction,
                    "residual": prediction - observed,
                }
                for system, fold, observed, prediction in zip(
                    clash["system"],
                    clash_fold,
                    clash["observed_log_g"],
                    heldout_clash_predictions,
                    strict=True,
                )
            )

    scores = pd.DataFrame(model_rows).sort_values(
        ["survives", "full_objective"],
        ascending=[False, True],
    )
    winner = scores.iloc[0]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed spherical tidal-response proxy sweep",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "samples": {
            "SPARC_systems": int(sparc["system"].nunique()),
            "SPARC_points": int(len(sparc)),
            "CLASH_systems": int(clash["system"].nunique()),
            "CLASH_points": int(len(clash)),
        },
        "winner": {
            key: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in winner.to_dict().items()
        },
        "families_surviving": int(scores["survives"].sum()),
        "linear_amplitude_ceiling": {
            "maximum_low_acceleration_boost": 1.0
            / (
                1.0
                - float(protocol["inputs"]["spherical_radial_Q_eigenvalue"])
                * 0.999
            ),
            "CLASH_median_required_boost_context": 7.861,
        },
        "advance_gates": protocol["advance_gates"],
        "claim_boundary": protocol["claim_boundary"],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(ROOT / protocol["outputs"]["model_scores"], index=False)
    pd.DataFrame(fold_rows).to_csv(
        ROOT / protocol["outputs"]["fold_scores"], index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        ROOT / protocol["outputs"]["predictions"], index=False
    )
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(scores))
    axes[0].bar(x, scores["SPARC_heldout_RMSE_km_s"])
    axes[0].axhline(
        float(protocol["advance_gates"]["SPARC_outer_RMSE_km_s_max"]),
        color="#2E8B57",
        linestyle=":",
        label="advance target",
    )
    axes[0].set(
        xticks=x,
        xticklabels=scores["model"],
        ylabel="held-out RMSE (km/s)",
        title="SPARC outer rotation",
    )
    axes[1].bar(x, scores["CLASH_heldout_equal_system_RMSE_dex"])
    axes[1].axhline(
        float(protocol["advance_gates"]["CLASH_equal_cluster_RMSE_dex_max"]),
        color="#2E8B57",
        linestyle=":",
        label="advance target",
    )
    axes[1].set(
        xticks=x,
        xticklabels=scores["model"],
        ylabel="held-out equal-cluster RMSE (dex)",
        title="CLASH radial lensing proxy",
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=28, labelsize=8)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(ROOT / protocol["outputs"]["figure"], dpi=190)
    plt.close(figure)

    lines = [
        "# Tidal-tensor spherical proxy",
        "",
        f"Best mapping: **{winner['model']}**, kappa={winner['full_kappa']:.4g}.",
        "",
        f"SPARC held-out RMSE: **{winner['SPARC_heldout_RMSE_km_s']:.3f} km/s**.",
        "",
        f"CLASH held-out equal-cluster RMSE: **{winner['CLASH_heldout_equal_system_RMSE_dex']:.3f} dex**.",
        "",
        f"Formula families surviving all gates: **{int(scores['survives'].sum())}**.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
