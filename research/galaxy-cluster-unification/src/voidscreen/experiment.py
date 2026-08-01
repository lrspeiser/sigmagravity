from __future__ import annotations

import copy
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import PackedDataset
from .models import Prediction, RotationModel, TensorDataset, build_model


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 5090
    disk_mass_to_light_prior: float = 0.5
    bulge_mass_to_light_prior: float = 0.7
    log_mass_to_light_prior_sigma: float = 0.25
    velocity_error_floor_kms: float = 2.0
    rar_acceleration_m_s2: float = 1.2e-10
    hubble_km_s_mpc: float = 70.0
    sigma_response_amplitude: float = 1.1725
    sigma_g_dagger_m_s2: float = 9.6e-11
    learning_rate: float = 0.03
    steps: int = 5000


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment has no CUDA-enabled PyTorch build. "
            f"Installed torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r}."
        )
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def device_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        report["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return report


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _objective(
    model: RotationModel,
    data: TensorDataset,
    error_floor_kms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = model(data)
    sigma = torch.sqrt(prediction.velocity_error_adjusted_kms.square() + error_floor_kms**2)
    residual = prediction.velocity_predicted_kms - prediction.velocity_observed_adjusted_kms
    mask = data.train_mask
    gaussian_nll = (
        0.5 * ((residual[mask] / sigma[mask]).square() + 2.0 * torch.log(sigma[mask])).sum()
    )
    prior = model.prior_penalty()
    return gaussian_nll + prior, gaussian_nll, prior


def fit_model(
    model: RotationModel,
    data: TensorDataset,
    *,
    steps: int,
    learning_rate: float,
    error_floor_kms: float,
    progress: bool = True,
) -> tuple[list[dict[str, float]], float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1), eta_min=learning_rate * 0.03
    )
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, gaussian_nll, prior = _objective(model, data, error_floor_kms)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite objective at optimization step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1_000.0)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())
        if step == 1 or step == steps or step % log_every == 0:
            row = {
                "step": float(step),
                "objective": loss_value,
                "gaussian_nll": float(gaussian_nll.detach().cpu()),
                "prior_penalty": float(prior.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if progress:
                print(
                    f"step={step:6d} objective={row['objective']:.3f} "
                    f"nll={row['gaussian_nll']:.3f} prior={row['prior_penalty']:.3f}"
                )
    if best_state is None:
        raise RuntimeError("Optimizer did not produce a finite state")
    model.load_state_dict(best_state)
    return history, best_loss


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def prediction_frame(
    packed: PackedDataset,
    tensor_data: TensorDataset,
    prediction: Prediction,
    error_floor_kms: float,
) -> pd.DataFrame:
    galaxy_index = packed.galaxy_index
    names = np.asarray(packed.galaxy_names, dtype=object)[galaxy_index]
    adjusted_error = _tensor_numpy(prediction.velocity_error_adjusted_kms)
    return pd.DataFrame(
        {
            "galaxy": names,
            "galaxy_index": galaxy_index,
            "split": np.where(packed.train_mask, "train", "outer_holdout"),
            "radius_catalog_kpc": packed.radius_kpc,
            "radius_adjusted_kpc": _tensor_numpy(prediction.radius_adjusted_kpc),
            "velocity_observed_catalog_kms": packed.velocity_observed_kms,
            "velocity_observed_adjusted_kms": _tensor_numpy(
                prediction.velocity_observed_adjusted_kms
            ),
            "velocity_error_adjusted_kms": adjusted_error,
            "velocity_error_total_kms": np.sqrt(adjusted_error**2 + error_floor_kms**2),
            "velocity_baryonic_kms": _tensor_numpy(prediction.velocity_baryonic_kms),
            "velocity_predicted_kms": _tensor_numpy(prediction.velocity_predicted_kms),
            "g_bar_m_s2": _tensor_numpy(prediction.baryonic_acceleration_m_s2),
            "g_pred_m_s2": _tensor_numpy(prediction.predicted_acceleration_m_s2),
            "environment_score_raw": packed.environment_raw[galaxy_index],
            "environment_score_standardized": packed.environment_standardized[galaxy_index],
        }
    )


def galaxy_parameter_frame(
    packed: PackedDataset, model: RotationModel, prediction: Prediction
) -> pd.DataFrame:
    first_point = np.zeros(packed.n_galaxies, dtype=np.int64)
    for index in range(packed.n_galaxies):
        first_point[index] = int(np.flatnonzero(packed.galaxy_index == index)[0])
    disk_ml = _tensor_numpy(prediction.disk_mass_to_light)[first_point]
    bulge_ml = _tensor_numpy(prediction.bulge_mass_to_light)[first_point]
    distance_scale = _tensor_numpy(prediction.distance_scale)[first_point]
    inclination = _tensor_numpy(prediction.inclination_adjusted_deg)[first_point]
    frame = pd.DataFrame(
        {
            "galaxy": packed.galaxy_names,
            "quality": packed.quality,
            "distance_scale": distance_scale,
            "inclination_adjusted_deg": inclination,
            "disk_mass_to_light": disk_ml,
            "bulge_mass_to_light": bulge_ml,
            "environment_score_raw": packed.environment_raw,
            "environment_score_standardized": packed.environment_standardized,
        }
    )
    if hasattr(model, "log_v200"):
        frame["nfw_v200_kms"] = _tensor_numpy(torch.exp(model.log_v200))
        frame["nfw_concentration"] = _tensor_numpy(torch.exp(model.log_concentration))
    return frame


def metrics(frame: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = frame.loc[frame["split"] == split]
    residual = (
        selected["velocity_predicted_kms"] - selected["velocity_observed_adjusted_kms"]
    ).to_numpy(dtype=float)
    sigma = selected["velocity_error_total_kms"].to_numpy(dtype=float)
    chi2 = float(np.sum((residual / sigma) ** 2))
    log_likelihood = float(-0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)))
    return {
        "n": len(selected),
        "chi2": chi2,
        "chi2_per_point": chi2 / max(len(selected), 1),
        "rmse_kms": float(np.sqrt(np.mean(residual**2))),
        "mae_kms": float(np.mean(np.abs(residual))),
        "mean_standardized_residual": float(np.mean(residual / sigma)),
        "log_likelihood": log_likelihood,
    }


def _save_diagnostic_plot(
    output: Path,
    frame: pd.DataFrame,
    history: list[dict[str, float]],
    model_label: str,
) -> None:
    train = frame.loc[frame["split"] == "train"]
    holdout = frame.loc[frame["split"] == "outer_holdout"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].scatter(
        train["velocity_observed_adjusted_kms"],
        train["velocity_predicted_kms"],
        s=8,
        alpha=0.20,
        label="train",
    )
    axes[0, 0].scatter(
        holdout["velocity_observed_adjusted_kms"],
        holdout["velocity_predicted_kms"],
        s=12,
        alpha=0.45,
        label="outer holdout",
    )
    maximum = float(
        max(frame["velocity_observed_adjusted_kms"].max(), frame["velocity_predicted_kms"].max())
    )
    axes[0, 0].plot([0, maximum], [0, maximum], color="black", linewidth=1)
    axes[0, 0].set(xlabel="Observed velocity (km/s)", ylabel="Predicted velocity (km/s)")
    axes[0, 0].legend()

    residual = holdout["velocity_predicted_kms"] - holdout["velocity_observed_adjusted_kms"]
    axes[0, 1].scatter(holdout["g_bar_m_s2"], residual, s=12, alpha=0.45)
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(xlabel="Baryonic acceleration (m/s²)", ylabel="Outer residual (km/s)")

    standardized = residual / holdout["velocity_error_total_kms"]
    axes[1, 0].hist(standardized, bins=40, color="#4C78A8", alpha=0.85)
    axes[1, 0].axvline(0.0, color="black", linewidth=1)
    axes[1, 0].set(xlabel="Outer standardized residual", ylabel="Count")

    history_frame = pd.DataFrame(history)
    axes[1, 1].plot(history_frame["step"], history_frame["objective"], label="objective")
    axes[1, 1].plot(history_frame["step"], history_frame["gaussian_nll"], label="data NLL")
    axes[1, 1].set(xlabel="Optimization step", ylabel="Loss")
    axes[1, 1].legend()
    figure.suptitle(f"{model_label}: radial holdout diagnostics")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_experiment(
    packed: PackedDataset,
    *,
    model_name: str,
    output_dir: Path,
    device_name: str,
    dtype_name: str,
    settings: ExperimentSettings,
    fixed_flat_power: bool = False,
    environment_enabled: bool = False,
    boundary_layer_enabled: bool = False,
    run_label: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    set_reproducible_seed(settings.seed)
    device = resolve_device(device_name)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    tensor_data = TensorDataset.from_packed(packed, device=device, dtype=dtype)
    model = build_model(
        model_name,
        packed,
        disk_ml_prior=settings.disk_mass_to_light_prior,
        bulge_ml_prior=settings.bulge_mass_to_light_prior,
        log_ml_prior_sigma=settings.log_mass_to_light_prior_sigma,
        rar_acceleration_m_s2=settings.rar_acceleration_m_s2,
        hubble_km_s_mpc=settings.hubble_km_s_mpc,
        sigma_response_amplitude=settings.sigma_response_amplitude,
        sigma_g_dagger_m_s2=settings.sigma_g_dagger_m_s2,
        fixed_flat_power=fixed_flat_power,
        environment_enabled=environment_enabled,
        boundary_layer_enabled=boundary_layer_enabled,
    ).to(device=device, dtype=dtype)

    started = time.perf_counter()
    history, best_objective = fit_model(
        model,
        tensor_data,
        steps=settings.steps,
        learning_rate=settings.learning_rate,
        error_floor_kms=settings.velocity_error_floor_kms,
        progress=progress,
    )
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_data)
    frame = prediction_frame(packed, tensor_data, prediction, settings.velocity_error_floor_kms)
    galaxy_frame = galaxy_parameter_frame(packed, model, prediction)
    train_metrics = metrics(frame, "train")
    holdout_metrics = metrics(frame, "outer_holdout")
    parameter_count = model.parameter_count
    n_train = int(train_metrics["n"])
    train_log_likelihood = float(train_metrics["log_likelihood"])
    label = run_label or (
        f"{model_name}_p05" if model_name == "void" and fixed_flat_power else model_name
    )
    summary: dict[str, Any] = {
        "run_label": label,
        "model": model_name,
        "fixed_flat_power": fixed_flat_power,
        "environment_enabled": environment_enabled,
        "boundary_layer_enabled": boundary_layer_enabled,
        "global_parameters": model.physical_parameters(),
        "parameter_count": parameter_count,
        "best_objective": best_objective,
        "elapsed_seconds": elapsed,
        "train": train_metrics,
        "outer_holdout": holdout_metrics,
        "aic_train": 2.0 * parameter_count - 2.0 * train_log_likelihood,
        "bic_train": parameter_count * math.log(max(n_train, 1)) - 2.0 * train_log_likelihood,
        "information_criteria_note": (
            "AIC/BIC are approximate because nuisance parameters are MAP-regularized by explicit priors. "
            "Held-out predictive scores are the primary comparison."
        ),
        "data": {
            "fingerprint_sha256": packed.data_fingerprint,
            "galaxies": packed.n_galaxies,
            "points": packed.n_points,
            "train_points": packed.n_train,
            "outer_holdout_points": packed.n_holdout,
            "environment_score_column": packed.environment_score_column,
            "environment_fingerprint_sha256": packed.environment_fingerprint,
        },
        "optimization": asdict(settings),
        "runtime": {
            "requested_device": device_name,
            "used_device": str(device),
            "dtype": dtype_name,
        },
        "machine": device_report(),
        "interpretation_guardrail": (
            "A rotation-curve fit tests the phenomenology only. A void origin additionally requires "
            "a positive, out-of-sample beta from an independent environment reconstruction."
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    galaxy_frame.to_csv(output_dir / "galaxy_parameters.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "optimization_history.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    _save_diagnostic_plot(output_dir / "diagnostics.png", frame, history, label)
    return summary
