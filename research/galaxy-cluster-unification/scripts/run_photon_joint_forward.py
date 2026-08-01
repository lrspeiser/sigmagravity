#!/usr/bin/env python3
"""Fit photon-sector anomalies directly to measured maser velocity channels."""

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

from voidscreen.photon_joint_forward import (  # noqa: E402
    channel_design,
    fit_weighted_linear,
    stable_source_folds,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_build(frame: pd.DataFrame, protocol: dict, order: int, model: str) -> dict:
    parent_path = ROOT / protocol["input"]["parent_protocol"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    constants = parent["galactic_constants"]
    nuisance = protocol["ordinary_nuisance_model"]
    return channel_design(
        frame,
        rotation_order=order,
        photon_model=model,
        r0_kpc=float(constants["R0_kpc"]),
        theta_reference_km_s=float(constants["Theta0_km_s"]),
        a_star_m_s2=float(constants["a_star_m_s2"]),
        fixed_solar_z_km_s=float(nuisance["fixed_solar_cartesian_z_km_s"]),
        velocity_error_floor_km_s=float(
            protocol["uncertainty"]["velocity_error_floor_km_s"]
        ),
    )


def fit_model(frame: pd.DataFrame, protocol: dict, order: int, model: str) -> dict:
    built = model_build(frame, protocol, order, model)
    fit = fit_weighted_linear(
        built["design"], built["observed"], built["sigma"]
    )
    fit["parameter_names"] = built["parameter_names"]
    return fit


def predict_model(
    frame: pd.DataFrame,
    protocol: dict,
    order: int,
    model: str,
    values: np.ndarray,
) -> pd.DataFrame:
    built = model_build(frame, protocol, order, model)
    prediction_adjusted = built["design"] @ values
    prediction = prediction_adjusted - built["radial_solar_z_correction"]
    observed = built["observed"] - built["radial_solar_z_correction"]
    residual = observed - prediction
    return pd.DataFrame(
        {
            "source": built["source"],
            "channel": built["channel"],
            "observed_km_s": observed,
            "predicted_km_s": prediction,
            "residual_km_s": residual,
            "sigma_km_s": built["sigma"],
        }
    )


def metric(frame: pd.DataFrame) -> dict:
    residual = frame["residual_km_s"].to_numpy(float)
    sigma = frame["sigma_km_s"].to_numpy(float)
    return {
        "observations": int(len(frame)),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "weighted_RMS_sigma": float(np.sqrt(np.mean(np.square(residual / sigma)))),
        "mean_residual_km_s": float(np.mean(residual)),
    }


def parameter_rows(order: int, model: str, fit: dict, fold: int | None) -> list[dict]:
    errors = np.sqrt(np.diag(fit["covariance"]))
    return [
        {
            "rotation_order": order,
            "model": model,
            "fold": fold,
            "parameter": name,
            "value": float(value),
            "formal_standard_error": float(error),
            "chi2": float(fit["chi2"]),
            "condition_number": float(fit["condition_number"]),
        }
        for name, value, error in zip(
            fit["parameter_names"], fit["values"], errors, strict=True
        )
    ]


def cross_validate(
    sample: pd.DataFrame, protocol: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = int(protocol["uncertainty"]["folds"])
    assignments = stable_source_folds(
        sample["system"],
        folds,
        int(protocol["uncertainty"]["fold_seed"]),
    )
    sample = sample.copy()
    sample["fold"] = assignments
    predictions = []
    parameters = []
    for order in protocol["ordinary_nuisance_model"]["rotation_curve_orders"]:
        for model in protocol["photon_models"]:
            for fold in range(folds):
                training = sample[sample["fold"] != fold]
                heldout = sample[sample["fold"] == fold]
                fit = fit_model(training, protocol, int(order), model)
                block = predict_model(
                    heldout, protocol, int(order), model, fit["values"]
                )
                block["rotation_order"] = int(order)
                block["model"] = model
                block["fold"] = fold
                predictions.append(block)
                parameters.extend(
                    parameter_rows(int(order), model, fit, fold)
                )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters)


def bootstrap(
    sample: pd.DataFrame, protocol: dict, order: int, model: str
) -> pd.DataFrame:
    draws = int(protocol["uncertainty"]["complete_source_bootstrap_draws"])
    seed = int(protocol["uncertainty"]["bootstrap_seed"])
    rng = np.random.default_rng(seed + 100 * order + len(model))
    rows = []
    for draw in range(draws):
        selected = rng.integers(0, len(sample), len(sample))
        resampled = sample.iloc[selected].copy()
        fit = fit_model(resampled, protocol, order, model)
        values = dict(zip(fit["parameter_names"], fit["values"], strict=True))
        row = {"draw": draw, "rotation_order": order, "model": model}
        row.update({name: float(value) for name, value in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def information_criteria(fit: dict, observations: int) -> tuple[float, float]:
    parameters = len(fit["values"])
    return (
        float(fit["chi2"] + 2 * parameters),
        float(fit["chi2"] + math.log(observations) * parameters),
    )


def make_figure(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
    full_parameters: pd.DataFrame,
    protocol: dict,
    output: Path,
) -> None:
    primary_order = int(
        protocol["ordinary_nuisance_model"]["primary_rotation_curve_order"]
    )
    primary = predictions[predictions["rotation_order"] == primary_order]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    null = primary[primary["model"] == "null"]
    for channel, block in null.groupby("channel"):
        axes[0].scatter(
            block["predicted_km_s"],
            block["observed_km_s"],
            s=18,
            alpha=0.65,
            label=channel.replace("_", " "),
        )
    low = min(null["predicted_km_s"].min(), null["observed_km_s"].min())
    high = max(null["predicted_km_s"].max(), null["observed_km_s"].max())
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(
        xlabel="held-out prediction (km/s)",
        ylabel="measured velocity (km/s)",
        title="Raw-channel forward predictions",
    )
    axes[0].legend(frameon=False, fontsize=8)

    metric_rows = []
    for model, block in primary.groupby("model"):
        metric_rows.append((model, metric(block)["weighted_RMS_sigma"]))
    axes[1].bar(
        [name.replace("frequency_", "").replace("_", "\n") for name, _ in metric_rows],
        [value for _, value in metric_rows],
    )
    axes[1].set(
        ylabel="held-out weighted RMS (sigma)",
        title="Photon models vs shared rotation",
    )
    axes[1].tick_params(axis="x", labelsize=8)

    constant = full_parameters[
        (full_parameters["model"] == "frequency_constant")
        & (full_parameters["parameter"] == "photon_A_km_s")
    ].sort_values("rotation_order")
    axes[2].errorbar(
        constant["rotation_order"],
        constant["value"],
        yerr=constant["formal_standard_error"],
        marker="o",
        capsize=4,
        label="constant photon term",
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].axhline(
        float(protocol["advance_gate"]["minimum_required_positive_amplitude_km_s"]),
        color="#2E8B57",
        linestyle=":",
        label="minimum target",
    )
    axes[2].set(
        xlabel="ordinary rotation-curve polynomial order",
        ylabel="photon amplitude A (km/s)",
        title="Stability to conventional flexibility",
        xticks=protocol["ordinary_nuisance_model"]["rotation_curve_orders"],
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "photon_joint_forward_protocol.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_raw_channel_forward_scoring":
        raise RuntimeError("protocol was not frozen before forward scoring")
    source_path = ROOT / protocol["input"]["source_estimates"]
    source = pd.read_csv(source_path)
    sample = source[source[protocol["input"]["primary_sample_column"]]].copy()
    if len(sample) < 20:
        raise RuntimeError("too few primary sources")

    full_rows = []
    full_lookup = {}
    for order in protocol["ordinary_nuisance_model"]["rotation_curve_orders"]:
        for model in protocol["photon_models"]:
            fit = fit_model(sample, protocol, int(order), model)
            full_lookup[(int(order), model)] = fit
            rows = parameter_rows(int(order), model, fit, None)
            aic, bic = information_criteria(fit, 2 * len(sample))
            for row in rows:
                row["AIC"] = aic
                row["BIC"] = bic
            full_rows.extend(rows)
    full_parameters = pd.DataFrame(full_rows)
    predictions, fold_parameters = cross_validate(sample, protocol)
    cv_metrics = {
        f"order_{order}:{model}": metric(block)
        for (order, model), block in predictions.groupby(
            ["rotation_order", "model"], sort=True
        )
    }

    primary_order = int(
        protocol["ordinary_nuisance_model"]["primary_rotation_curve_order"]
    )
    bootstrap_frame = bootstrap(
        sample, protocol, primary_order, protocol["advance_gate"]["primary_model"]
    )
    amplitude_draw = bootstrap_frame["photon_A_km_s"].to_numpy(float)
    primary_fit = full_lookup[
        (primary_order, protocol["advance_gate"]["primary_model"])
    ]
    amplitude_index = primary_fit["parameter_names"].index("photon_A_km_s")
    amplitude = float(primary_fit["values"][amplitude_index])
    interval = [
        float(np.quantile(amplitude_draw, 0.025)),
        float(np.quantile(amplitude_draw, 0.975)),
    ]
    primary_model_key = (
        f"order_{primary_order}:{protocol['advance_gate']['primary_model']}"
    )
    null_key = f"order_{primary_order}:null"
    improves = (
        cv_metrics[primary_model_key]["weighted_RMS_sigma"]
        < cv_metrics[null_key]["weighted_RMS_sigma"]
    )
    reaches = (
        interval[1]
        >= float(protocol["advance_gate"]["minimum_required_positive_amplitude_km_s"])
    )
    positive = amplitude > 0.0
    order_amplitudes = full_parameters[
        (full_parameters["model"] == "frequency_constant")
        & (full_parameters["parameter"] == "photon_A_km_s")
    ].sort_values("rotation_order")
    stable = bool(
        (order_amplitudes["value"] > 0.0).all()
        and order_amplitudes["value"].min()
        >= 0.5 * order_amplitudes["value"].max()
    )
    survives = bool(improves and reaches and positive and stable)

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed raw-channel joint forward test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input": {
            "path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(source_path),
            "sources": int(len(sample)),
            "observations": int(2 * len(sample)),
        },
        "equations": protocol["equations"],
        "new_physics_interpretation": protocol["new_physics_interpretation"],
        "cross_validated_metrics": cv_metrics,
        "primary_constant_photon_amplitude": {
            "rotation_curve_order": primary_order,
            "amplitude_km_s": amplitude,
            "bootstrap_95_interval_km_s": interval,
            "bootstrap_probability_positive": float(np.mean(amplitude_draw > 0.0)),
            "draws": int(len(amplitude_draw)),
        },
        "rotation_order_stability": [
            {
                "rotation_order": int(row.rotation_order),
                "amplitude_km_s": float(row.value),
                "formal_standard_error_km_s": float(row.formal_standard_error),
            }
            for row in order_amplitudes.itertuples()
        ],
        "advance_gate": {
            "heldout_weighted_RMS_improves": bool(improves),
            "bootstrap_upper_interval_reaches_30_km_s": bool(reaches),
            "fitted_sign_positive": bool(positive),
            "stable_across_rotation_orders": bool(stable),
            "frequency_only_illusion_survives": survives,
        },
        "claim_boundary": protocol["claim_boundary"],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    full_parameters.to_csv(ROOT / protocol["outputs"]["full_fits"], index=False)
    predictions.to_csv(
        ROOT / protocol["outputs"]["cross_validated_predictions"], index=False
    )
    fold_parameters.to_csv(ROOT / protocol["outputs"]["fold_fits"], index=False)
    bootstrap_frame.to_csv(ROOT / protocol["outputs"]["bootstrap"], index=False)
    make_figure(
        sample,
        predictions,
        full_parameters,
        protocol,
        ROOT / protocol["outputs"]["figure"],
    )
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Photon joint forward test",
        "",
        f"Same-source masers: **{len(sample)}**; raw velocity observations: **{2 * len(sample)}**.",
        "",
        f"Primary constant photon amplitude: **{amplitude:.3f} km/s** "
        f"(95% bootstrap {interval[0]:.3f} to {interval[1]:.3f}).",
        "",
        "| model (linear ordinary curve) | held-out weighted RMS (sigma) |",
        "|---|---:|",
    ]
    for model in protocol["photon_models"]:
        key = f"order_{primary_order}:{model}"
        lines.append(f"| {model} | {cv_metrics[key]['weighted_RMS_sigma']:.4f} |")
    lines.extend(
        [
            "",
            f"Frequency-only illusion survives: **{survives}**.",
            "",
            "The anisotropic/time-dependent optical-metric branch remains open and requires epoch astrometry.",
        ]
    )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
