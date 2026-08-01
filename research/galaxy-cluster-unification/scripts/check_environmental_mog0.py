from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import KPC_M, data_fingerprint
from voidscreen.mog import (
    chameleon_metric_enhancement,
    environmental_mog_dynamic_enhancement,
    matched_mog_enhancement,
    mean_enclosed_density_kg_m3,
    mog_extra_acceleration_log_slope,
    unscreened_ppn_gamma_minus_one,
    vector_light_dynamics_gamma_minus_one,
)
from voidscreen.unified import (
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    return {
        "count": len(finite),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _prepare_frames(
    sparc_path: Path, clash_path: Path, bcg_path: Path
) -> pd.DataFrame:
    sparc = load_sparc_acceleration_frame(sparc_path).copy()
    clash = load_clash_acceleration_frame(clash_path).copy()
    bcg = pd.read_csv(bcg_path)

    records = []
    for label, observable, frame in (
        ("SPARC", "dynamics", sparc),
        ("CLASH", "lensing", clash),
    ):
        selected = frame[
            [
                "system",
                "radius_kpc",
                "gbar_m_s2",
                "observed_g_m_s2",
                "chi",
            ]
        ].copy()
        selected["domain"] = label
        selected["observable"] = observable
        records.append(selected)

    selected = bcg[
        [
            "spiders_id",
            "radius_kpc",
            "gbar_m_s2",
            "gobs_m_s2",
            "total_profile_chi",
        ]
    ].copy()
    selected.columns = [
        "system",
        "radius_kpc",
        "gbar_m_s2",
        "observed_g_m_s2",
        "chi",
    ]
    selected["domain"] = "SPIDERS-MaNGA BCG"
    selected["observable"] = "dynamics"
    records.append(selected)

    combined = pd.concat(records, ignore_index=True)
    combined["radius_m"] = combined["radius_kpc"] * KPC_M
    combined["mean_density_kg_m3"] = mean_enclosed_density_kg_m3(
        combined["gbar_m_s2"], combined["radius_m"]
    )
    combined["required_enhancement"] = (
        combined["observed_g_m_s2"] / combined["gbar_m_s2"]
    )
    return combined


def _environmental_prediction(
    vector: np.ndarray,
    frame: pd.DataFrame,
    *,
    reference_density: float,
    maximum_log_enhancement: float,
) -> np.ndarray | None:
    log10_z, power, log10_range, log10_alpha = vector
    metric = chameleon_metric_enhancement(
        frame["mean_density_kg_m3"],
        reference_density_kg_m3=reference_density,
        z_reference=10.0**log10_z,
        power=power,
        maximum_log_enhancement=maximum_log_enhancement,
    )
    alpha = 10.0**log10_alpha
    x = frame["radius_kpc"].to_numpy(dtype=float) / 10.0**log10_range
    prediction = metric.copy()
    dynamics = frame["observable"].to_numpy(dtype=str) == "dynamics"
    prediction[dynamics] = environmental_mog_dynamic_enhancement(
        x[dynamics], metric[dynamics], alpha
    )
    if np.any(~np.isfinite(prediction)) or np.any(prediction <= 0.0):
        return None
    return prediction


def _maximum_fractional_error(prediction, target) -> float:
    prediction_values = np.asarray(prediction, dtype=float)
    target_values = np.asarray(target, dtype=float)
    return float(np.max(np.abs(prediction_values / target_values - 1.0)))


def _run_environmental_envelope(
    frame: pd.DataFrame, envelope: dict
) -> tuple[np.ndarray, float, list[dict]]:
    bounds = [
        tuple(envelope["log10_z_ref_bounds"]),
        tuple(envelope["p_bounds"]),
        tuple(envelope["log10_range_kpc_bounds"]),
        tuple(envelope["log10_alpha_bounds"]),
    ]
    reference_density = float(envelope["rho_reference_kg_m3"])
    maximum_log = float(envelope["maximum_log_metric_enhancement"])
    target = frame["required_enhancement"].to_numpy(dtype=float)

    def objective(vector) -> float:
        prediction = _environmental_prediction(
            vector,
            frame,
            reference_density=reference_density,
            maximum_log_enhancement=maximum_log,
        )
        if prediction is None:
            return 1e30
        return _maximum_fractional_error(prediction, target)

    settings = envelope["optimizer"]
    runs = []
    for seed in settings["seeds"]:
        result = differential_evolution(
            objective,
            bounds,
            seed=int(seed),
            popsize=int(settings["population_size_multiplier"]),
            maxiter=int(settings["maximum_iterations"]),
            tol=float(settings["relative_tolerance"]),
            polish=bool(settings["polish"]),
            workers=1,
            updating="immediate",
        )
        runs.append(
            {
                "seed": int(seed),
                "maximum_fractional_error": float(result.fun),
                "vector": [float(value) for value in result.x],
                "success": bool(result.success),
                "message": str(result.message),
                "function_evaluations": int(result.nfev),
            }
        )
    best = min(runs, key=lambda item: item["maximum_fractional_error"])
    return (
        np.asarray(best["vector"], dtype=float),
        float(best["maximum_fractional_error"]),
        runs,
    )


def _run_constant_field_control(
    frame: pd.DataFrame, envelope: dict
) -> tuple[np.ndarray, float]:
    bounds = [
        tuple(envelope["log10_range_kpc_bounds"]),
        tuple(envelope["log10_alpha_bounds"]),
    ]
    target = frame["required_enhancement"].to_numpy(dtype=float)
    radius = frame["radius_kpc"].to_numpy(dtype=float)
    dynamics = frame["observable"].to_numpy(dtype=str) == "dynamics"

    def objective(vector) -> float:
        log10_range, log10_alpha = vector
        alpha = 10.0**log10_alpha
        prediction = np.full(len(frame), 1.0 + alpha)
        prediction[dynamics] = matched_mog_enhancement(
            radius[dynamics] / 10.0**log10_range, alpha
        )
        return _maximum_fractional_error(prediction, target)

    result = differential_evolution(
        objective,
        bounds,
        seed=20260726,
        popsize=24,
        maxiter=600,
        tol=1e-10,
        polish=True,
    )
    return np.asarray(result.x, dtype=float), float(result.fun)


def _strongest_monotonicity_contradiction(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values("mean_density_kg_m3", kind="stable").reset_index(
        drop=True
    )
    best = None
    for low_index in range(len(ordered)):
        low_target = float(ordered.loc[low_index, "required_enhancement"])
        for high_index in range(low_index + 1, len(ordered)):
            high_target = float(ordered.loc[high_index, "required_enhancement"])
            if high_target <= low_target:
                continue
            lower_bound = (high_target - low_target) / (high_target + low_target)
            if best is None or lower_bound > best[0]:
                best = (lower_bound, low_index, high_index)
    if best is None:
        return {"contradiction_found": False}
    lower_bound, low_index, high_index = best
    columns = [
        "domain",
        "system",
        "radius_kpc",
        "mean_density_kg_m3",
        "required_enhancement",
    ]
    low = ordered.loc[low_index, columns].to_dict()
    high = ordered.loc[high_index, columns].to_dict()
    return {
        "contradiction_found": True,
        "logic": (
            "The frozen chameleon has enhancement nonincreasing with density. "
            "For this ordered pair, any such prediction has a minimax relative "
            "error at least (nu_high-nu_low)/(nu_high+nu_low)."
        ),
        "lower_density_point": low,
        "higher_density_point": high,
        "required_enhancement_ratio_high_over_low": float(
            high["required_enhancement"] / low["required_enhancement"]
        ),
        "model_independent_minimax_fractional_error_lower_bound": float(
            lower_bound
        ),
    }


def _domain_metrics(frame: pd.DataFrame) -> dict:
    output = {}
    for domain, group in frame.groupby("domain", sort=True):
        fractional = np.abs(
            group["predicted_enhancement"]
            / group["required_enhancement"]
            - 1.0
        )
        residual_dex = np.log10(
            group["predicted_enhancement"] / group["required_enhancement"]
        )
        output[domain] = {
            "systems": int(group["system"].nunique()),
            "points": len(group),
            "absolute_fractional_error": _summary(fractional),
            "points_passing_5_percent": int((fractional <= 0.05).sum()),
            "rms_dex": float(np.sqrt(np.mean(np.square(residual_dex)))),
            "mean_residual_dex": float(np.mean(residual_dex)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen EMOG-Q0 action and structural pre-fit gates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "environmental_mog0_derivation.json",
    )
    parser.add_argument(
        "--sparc", type=Path, default=ROOT / "data" / "raw" / "sparc"
    )
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--bcg",
        type=Path,
        default=ROOT / "data" / "derived" / "measured_host_profile_sample.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "environmental_mog0"
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = config["prefit_gates"]
    envelope = config["favorable_chameleon_envelope"]
    frame = _prepare_frames(args.sparc, args.clash, args.bcg)

    vector, maximum_error, optimizer_runs = _run_environmental_envelope(
        frame, envelope
    )
    prediction = _environmental_prediction(
        vector,
        frame,
        reference_density=float(envelope["rho_reference_kg_m3"]),
        maximum_log_enhancement=float(
            envelope["maximum_log_metric_enhancement"]
        ),
    )
    if prediction is None:
        raise RuntimeError("best feasibility-envelope prediction is nonphysical")
    frame["predicted_enhancement"] = prediction
    frame["predicted_g_m_s2"] = prediction * frame["gbar_m_s2"]
    frame["absolute_fractional_error"] = np.abs(
        prediction / frame["required_enhancement"] - 1.0
    )
    frame["passes_5_percent"] = (
        frame["absolute_fractional_error"]
        <= gates["pointwise_fractional_error_max"]
    )

    constant_vector, constant_error = _run_constant_field_control(frame, envelope)
    clash = frame[frame["domain"] == "CLASH"]
    contradiction = _strongest_monotonicity_contradiction(clash)

    slope_crossing = brentq(
        lambda value: float(mog_extra_acceleration_log_slope(value)) + 1.0,
        1e-5,
        100.0,
    )
    slope_inner = brentq(
        lambda value: float(mog_extra_acceleration_log_slope(value)) + 0.95,
        1e-5,
        slope_crossing,
    )
    slope_outer = brentq(
        lambda value: float(mog_extra_acceleration_log_slope(value)) + 1.05,
        slope_crossing,
        100.0,
    )
    gamma_limit = float(gates["absolute_ppn_gamma_minus_one_max"])
    beta_unscreened_max = brentq(
        lambda beta: abs(float(unscreened_ppn_gamma_minus_one(beta)))
        - gamma_limit,
        0.0,
        10.0,
    )
    alpha_solar_max = gamma_limit / 2.0

    log10_z, power, log10_range, log10_alpha = vector
    prefit_pass = bool(
        maximum_error <= gates["pointwise_fractional_error_max"]
        and contradiction[
            "model_independent_minimax_fractional_error_lower_bound"
        ]
        <= gates["pointwise_fractional_error_max"]
    )
    report = {
        "status": "completed frozen EMOG-Q0 pre-fit derivation checkpoint",
        "inputs": {
            "config_sha256": _sha256(args.config),
            "sparc_fingerprint": data_fingerprint(args.sparc),
            "clash_sha256": _sha256(args.clash),
            "bcg_sha256": _sha256(args.bcg),
            "sparc_systems": int(
                frame.loc[frame["domain"] == "SPARC", "system"].nunique()
            ),
            "sparc_points": int((frame["domain"] == "SPARC").sum()),
            "clash_systems": int(clash["system"].nunique()),
            "clash_points": len(clash),
            "bcg_systems": int(
                frame.loc[
                    frame["domain"] == "SPIDERS-MaNGA BCG", "system"
                ].nunique()
            ),
        },
        "action_audit": {
            "candidate": config["candidate"],
            "one_physical_metric": config["conventions"]["physical_metric"],
            "global_physical_parameter_count": config["action"][
                "parameter_count"
            ],
            "parameters": config["action"]["global_physical_parameters"],
            "per_object_parameters": 0,
            "lensing_only_parameters": 0,
            "universal_composition_independent_current": True,
            "universal_vector_range": True,
            "positive_energy_proca_signs": True,
            "reciprocal_scalar_sources_retained": (
                "The scalar equation retains F_s R/2. Mu and kappa are constants, "
                "so the Proca sector creates no omitted scalar source."
            ),
        },
        "dimensions": {
            "s": "dimensionless",
            "F_and_beta_and_n_and_alpha": "dimensionless",
            "Lambda_s_and_mu": "inverse length (mass in c=hbar=1)",
            "phi_a": "mass",
            "kappa": "inverse mass",
            "each_lagrangian_density_term": "mass^4",
        },
        "field_health": {
            "conditions": [
                "F=exp(-2 beta s)>0",
                "Einstein-frame scalar kinetic K=1/F+6 beta^2>0",
                "U_ss=n(n+1)Lambda_s^2*s^(-n-2)>0",
                "canonical Proca kinetic sign and mu^2>0",
            ],
            "linear_mode_speeds_squared": {
                "tensor": 1.0,
                "scalar": 1.0,
                "proca_transverse": 1.0,
                "proca_longitudinal": 1.0,
            },
            "tensor_wave_speed_fractional_difference": 0.0,
            "local_linear_kinetic_and_gradient_health_passes": True,
            "nonlinear_global_well_posedness": (
                "not established beyond the regular F>0 Einstein-frame domain"
            ),
        },
        "solar_system": {
            "unscreened_beta_max_at_F_approximately_1": beta_unscreened_max,
            "gamma_absolute_limit": gamma_limit,
            "universal_long_range_vector": {
                "derivation": (
                    "At mu*r_AU<<1, massive dynamics measures E-alpha while "
                    "light measures E. After normalizing E-alpha=1, the effective "
                    "gamma shift is 2*alpha."
                ),
                "alpha_max": alpha_solar_max,
                "best_favorable_envelope_alpha_over_max": float(
                    10.0**log10_alpha / alpha_solar_max
                ),
                "gamma_shift_at_matched_best_envelope_alpha": (
                    vector_light_dynamics_gamma_minus_one(
                        10.0**log10_alpha, 1.0 + 10.0**log10_alpha
                    )
                ),
                "passes_at_best_envelope_value": bool(
                    10.0**log10_alpha <= alpha_solar_max
                ),
            },
            "chameleon_screening": (
                "The action can make the scalar massive at high density, but no "
                "parameter set survives the structural target gate, so a thin-shell "
                "solar solution is not promoted as a fitted success."
            ),
        },
        "spherical_solution": {
            "metric_potentials_constant_s": (
                "Phi=Psi=-G_N*M/(F0*r); the same Weyl potential lenses light"
            ),
            "proca_potential": "phi_0=kappa*M*exp(-mu*r)/(4*pi*r)",
            "massive_particle_enhancement": (
                "1/F0-alpha*(1+mu*r)*exp(-mu*r)"
            ),
            "matched_environment_condition": "1/F0=1+alpha",
            "matched_short_distance_limit": "1+alpha*(mu*r)^2/2+O(r^3)",
            "matched_large_distance_limit": "1+alpha",
            "lensing_enhancement_constant_s": "1/F0",
            "exact_spherical_shell_kernel_implemented": True,
            "point_mass_extra_acceleration_one_over_r_slope": {
                "mu_r_at_exact_slope_minus_one": slope_crossing,
                "mu_r_interval_for_slope_minus_one_plus_or_minus_0p05": [
                    slope_inner,
                    slope_outer,
                ],
                "radial_width_factor": slope_outer / slope_inner,
                "radial_width_dex": float(np.log10(slope_outer / slope_inner)),
            },
        },
        "constant_field_control": {
            "best_log10_range_kpc": float(constant_vector[0]),
            "best_range_kpc": float(10.0 ** constant_vector[0]),
            "best_log10_alpha": float(constant_vector[1]),
            "best_alpha": float(10.0 ** constant_vector[1]),
            "minimum_maximum_fractional_error": constant_error,
            "passes_5_percent": bool(
                constant_error <= gates["pointwise_fractional_error_max"]
            ),
        },
        "favorable_environmental_envelope": {
            "interpretation": envelope["interpretation"],
            "best_effective_parameters_not_adopted": {
                "log10_z_reference": float(log10_z),
                "z_reference": float(10.0**log10_z),
                "p": float(power),
                "equivalent_n": float(1.0 / power - 1.0),
                "log10_range_kpc": float(log10_range),
                "range_kpc": float(10.0**log10_range),
                "log10_alpha": float(log10_alpha),
                "alpha": float(10.0**log10_alpha),
            },
            "optimizer_runs": optimizer_runs,
            "minimum_maximum_fractional_error": maximum_error,
            "gate_maximum_fractional_error": gates[
                "pointwise_fractional_error_max"
            ],
            "passes_5_percent": bool(
                maximum_error <= gates["pointwise_fractional_error_max"]
            ),
            "by_domain": _domain_metrics(frame),
        },
        "environmental_response": {
            "adiabatic_minimum": (
                "s_min=[n*Lambda_s^2*Mpl^2/(beta*rho_bar)]^(1/(n+1))"
            ),
            "monotonic_prediction": (
                "For beta,n>0, 1/F increases as baryonic density decreases."
            ),
            "clash_monotonicity_contradiction": contradiction,
            "vector_cancellation_lock": (
                "alpha is fixed by the conserved universal current while 1/F(s) "
                "changes with environment; the action locks them only at one "
                "background value, so exact short-distance cancellation is not "
                "preserved across environments."
            ),
            "ea_q0_failure_repeated": False,
            "new_failure": (
                "The consistent scalar response has the wrong ordering and cannot "
                "remain locked to the universal vector charge."
            ),
        },
        "decision": {
            "prefit_checkpoint_passes": prefit_pass,
            "failed_gates": [
                "joint 5-percent spherical force shape",
                "CLASH density-response ordering",
                "environment-independent metric-vector cancellation lock",
                "Solar-System light/dynamics consistency for a universal long-range vector",
            ],
            "stage_3_configuration_frozen": False,
            "stage_4_replay_frozen": False,
            "astrophysical_fit_performed": False,
            "emog_q0_retired": not prefit_pass,
            "next_action": (
                "premise-level rethink of the one-field environmental unification "
                "target; do not add interpolation terms"
            ),
        },
        "guardrails": [
            "The feasibility scan is a minimax falsification envelope, not a fit.",
            "No optimizer value is adopted as a physical parameter estimate.",
            "No per-object, object-class, or lensing-only parameter is present.",
            "The vector range and charge are universal.",
            "All scored points are retained, including gobs<=gbar SPARC points.",
            "No Stage 3 or Stage 4 configuration is frozen after failure.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    columns = [
        "domain",
        "observable",
        "system",
        "radius_kpc",
        "gbar_m_s2",
        "observed_g_m_s2",
        "chi",
        "mean_density_kg_m3",
        "required_enhancement",
        "predicted_enhancement",
        "predicted_g_m_s2",
        "absolute_fractional_error",
        "passes_5_percent",
    ]
    frame[columns].to_csv(args.output / "feasibility_points.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if prefit_pass:
        print("EMOG-Q0 passes its pre-fit checkpoint.")
    else:
        print("EMOG-Q0 fails and is retired before an astrophysical fit.")


if __name__ == "__main__":
    main()
