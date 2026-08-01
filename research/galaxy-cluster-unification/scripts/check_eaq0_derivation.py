from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.constitutive import standard_mu_acceleration
from voidscreen.eaq import (
    aether_feedback_energy_scale,
    beta_from_gamma_bound,
    environment_acceleration_scale,
    exterior_feedback_ratio,
    high_field_mode_speeds_squared,
    maximum_eta_for_feedback_gate,
    point_source_minimum_range_over_radius,
    ppn_restricted_aether_coefficients,
    required_exponential_coupling,
    scalar_tensor_gamma_minus_one,
    standard_mu_from_y,
)
from voidscreen.unified import (
    A0_M_S2,
    KPC_M,
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _feedback_rows(
    frame: pd.DataFrame,
    *,
    domain: str,
    system_column: str,
    chi_column: str,
    eta: float,
    beta: float,
    range_over_radius: float,
    grid_points: int,
) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        radius_kpc = float(getattr(row, "radius_kpc"))
        gbar = float(getattr(row, "gbar_m_s2"))
        target_chi = float(getattr(row, chi_column))
        ratio = exterior_feedback_ratio(
            radius_m=radius_kpc * KPC_M,
            gbar_m_s2=gbar,
            target_chi=target_chi,
            eta_per_chi=eta,
            beta=beta,
            range_over_radius=range_over_radius,
            grid_points=grid_points,
        )
        records.append(
            {
                "domain": domain,
                "system": str(getattr(row, system_column)),
                "radius_kpc": radius_kpc,
                "gbar_m_s2": gbar,
                "target_chi": target_chi,
                "conservative_delta_q_over_target_chi": ratio,
                "passes_5_percent": ratio <= 0.05,
            }
        )
    return pd.DataFrame.from_records(records)


def _monotonicity_check(gbar_min: float, gbar_max: float, a_q_max: float) -> dict:
    gbar = np.geomspace(gbar_min, gbar_max, 160)
    a_q = np.geomspace(A0_M_S2, a_q_max, 160)
    values = aether_feedback_energy_scale(gbar[:, None], a_q[None, :])
    scale = max(float(np.max(values)), np.finfo(float).tiny)
    tolerance = 1e-11 * scale
    increases_with_gbar = bool(np.all(np.diff(values, axis=0) >= -tolerance))
    increases_with_a_q = bool(np.all(np.diff(values, axis=1) >= -tolerance))
    return {
        "gbar_range_m_s2": [float(gbar_min), float(gbar_max)],
        "a_q_range_m_s2": [A0_M_S2, float(a_q_max)],
        "increases_with_gbar": increases_with_gbar,
        "increases_with_a_q": increases_with_a_q,
        "supports_enclosed_mass_only_lower_bound": (
            increases_with_gbar and increases_with_a_q
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen EA-Q0 pre-fit checks.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "eaq0_derivation.json"
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
        "--output", type=Path, default=ROOT / "results" / "eaq0_derivation"
    )
    parser.add_argument("--grid-points", type=int, default=6000)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = config["prefit_gates"]
    frozen = config["fixed_constants"]
    beta = beta_from_gamma_bound(gates["gamma_absolute_max"])
    range_over_radius = point_source_minimum_range_over_radius(
        gates["q_fractional_error_max"]
    )
    eta = required_exponential_coupling(
        frozen["h7s_amplitude_not_widened"], frozen["h7s_transition_chi"]
    )
    if not np.isclose(beta, gates["beta_most_permissive_for_feedback"], rtol=1e-12):
        raise RuntimeError("the frozen beta value changed")
    if not np.isclose(range_over_radius, gates["point_source_minimum_L_over_r"]):
        raise RuntimeError("the frozen range ratio changed")
    if not np.isclose(eta, frozen["eta_per_chi"]):
        raise RuntimeError("the frozen environment coupling changed")

    sparc = load_sparc_acceleration_frame(args.sparc)
    clash = load_clash_acceleration_frame(args.clash)
    bcg = pd.read_csv(args.bcg)
    frames = [
        _feedback_rows(
            sparc,
            domain="SPARC",
            system_column="system",
            chi_column="chi",
            eta=eta,
            beta=beta,
            range_over_radius=range_over_radius,
            grid_points=args.grid_points,
        ),
        _feedback_rows(
            clash,
            domain="CLASH",
            system_column="system",
            chi_column="chi",
            eta=eta,
            beta=beta,
            range_over_radius=range_over_radius,
            grid_points=args.grid_points,
        ),
        _feedback_rows(
            bcg,
            domain="SPIDERS-MaNGA BCG",
            system_column="spiders_id",
            chi_column="total_profile_chi",
            eta=eta,
            beta=beta,
            range_over_radius=range_over_radius,
            grid_points=args.grid_points,
        ),
    ]
    feedback = pd.concat(frames, ignore_index=True)

    all_input_gbar = np.concatenate(
        [
            sparc["gbar_m_s2"].to_numpy(dtype=float),
            clash["gbar_m_s2"].to_numpy(dtype=float),
            bcg["gbar_m_s2"].to_numpy(dtype=float),
        ]
    )
    all_target_chi = np.concatenate(
        [
            sparc["chi"].to_numpy(dtype=float),
            clash["chi"].to_numpy(dtype=float),
            bcg["total_profile_chi"].to_numpy(dtype=float),
        ]
    )
    a_q_max = float(environment_acceleration_scale(np.max(all_target_chi), eta))
    monotonicity = _monotonicity_check(
        max(float(np.min(all_input_gbar)) * 1e-8, 1e-30),
        float(np.max(all_input_gbar)),
        a_q_max,
    )
    if not monotonicity["supports_enclosed_mass_only_lower_bound"]:
        raise RuntimeError("the conservative exterior-source ordering was not monotonic")

    by_domain = {}
    for domain, group in feedback.groupby("domain", sort=True):
        values = group["conservative_delta_q_over_target_chi"].to_numpy(dtype=float)
        by_domain[domain] = {
            **_summary(values),
            "passes_5_percent": int(group["passes_5_percent"].sum()),
            "pass_fraction": float(group["passes_5_percent"].mean()),
        }

    bcg_rows = [
        (row.radius_kpc * KPC_M, row.gbar_m_s2, row.total_profile_chi)
        for row in bcg.itertuples()
    ]
    eta_max = maximum_eta_for_feedback_gate(
        bcg_rows,
        beta=beta,
        range_over_radius=range_over_radius,
        maximum_fraction=gates["q_fractional_error_max"],
        grid_points=max(2000, args.grid_points // 2),
    )
    allowed_midpoint_response = float(np.exp(eta_max * frozen["h7s_transition_chi"]))
    bcg_feedback = feedback[feedback["domain"] == "SPIDERS-MaNGA BCG"]
    beta_required = (
        beta
        * bcg_feedback["conservative_delta_q_over_target_chi"].to_numpy(dtype=float)
        / gates["q_fractional_error_max"]
    )

    deep_ratio = gates["deep_gbar_over_aq_max"]
    deep_prediction = float(standard_mu_acceleration(A0_M_S2 * deep_ratio, A0_M_S2))
    deep_target = float(A0_M_S2 * np.sqrt(deep_ratio))
    deep_error = abs(deep_prediction / deep_target - 1.0)
    high_ratio = gates["high_gbar_over_aq_min"]
    high_prediction = float(standard_mu_acceleration(A0_M_S2 * high_ratio, A0_M_S2))
    high_correction = high_prediction / (A0_M_S2 * high_ratio) - 1.0
    y_grid = np.geomspace(1e-20, 1e20, 10000)
    mu_grid = standard_mu_from_y(y_grid)
    response_monotonic = bool(np.all(np.diff(mu_grid * np.sqrt(y_grid)) > 0.0))

    c14 = 1e-5
    c1 = 2e-5
    coefficients = ppn_restricted_aether_coefficients(c1, c14)
    speeds = high_field_mode_speeds_squared(c1, c14)
    high_field_health = {
        "representative_c1": c1,
        "representative_c14": c14,
        "coefficients": coefficients,
        "mode_speeds_squared": speeds,
        "kinetic_conditions_positive": bool(
            beta > 0.0
            and coefficients["c2"] > 0.0
            and c14 > 0.0
            and c1 >= c14
        ),
        "c13_is_zero": coefficients["c13"] == 0.0,
    }

    reciprocal_source_pass = bool(feedback["passes_5_percent"].all())
    prefit_pass = bool(
        config["global_physical_parameter_count"] <= 5
        and high_field_health["kinetic_conditions_positive"]
        and high_field_health["c13_is_zero"]
        and all(value > 0.0 for value in speeds.values())
        and deep_error <= gates["deep_relative_error_max"]
        and high_correction <= gates["high_fractional_correction_max"]
        and response_monotonic
        and reciprocal_source_pass
    )

    args.output.mkdir(parents=True, exist_ok=True)
    feedback.to_csv(args.output / "feedback_points.csv", index=False)
    report = {
        "status": "completed frozen EA-Q0 derivation checkpoint",
        "inputs": {
            "config_sha256": _sha256(args.config),
            "clash_sha256": _sha256(args.clash),
            "bcg_sha256": _sha256(args.bcg),
            "sparc_systems": int(sparc["system"].nunique()),
            "sparc_points": len(sparc),
            "clash_systems": int(clash["system"].nunique()),
            "clash_points": len(clash),
            "bcg_systems": len(bcg),
        },
        "action_audit": {
            "universal_matter_metric": config["physical_metric"],
            "global_physical_parameter_count": config[
                "global_physical_parameter_count"
            ],
            "per_object_parameters": 0,
            "lensing_only_parameters": 0,
            "unit_constraint_from_lambda": "u^a u_a=-1",
            "on_shell_conservation": (
                "verified by the diffeomorphism Noether identity after imposing the "
                "metric, Aether, Q, constraint, and matter equations"
            ),
            "reciprocal_q_source_retained": True,
        },
        "frozen_parameters_for_most_conservative_check": {
            "eta_per_chi": eta,
            "beta": beta,
            "L_Q_over_each_test_radius": range_over_radius,
            "required_aq_over_a0_at_chi_t": 10.0,
        },
        "high_field_health": high_field_health,
        "weak_field_limits": {
            "deep_relative_error": deep_error,
            "deep_relative_error_max": gates["deep_relative_error_max"],
            "high_fractional_correction": high_correction,
            "high_fractional_correction_max": gates[
                "high_fractional_correction_max"
            ],
            "mu_positive": bool(np.all(mu_grid > 0.0)),
            "d_mu_times_g_dg_positive": response_monotonic,
            "passes": bool(
                deep_error <= gates["deep_relative_error_max"]
                and high_correction <= gates["high_fractional_correction_max"]
                and response_monotonic
            ),
        },
        "conservative_source_ordering": monotonicity,
        "q_feedback": {
            "gate_absolute_fraction_max": gates["q_fractional_error_max"],
            "by_domain": by_domain,
            "all_points_pass": reciprocal_source_pass,
            "eta_max_for_all_bcg_to_pass": eta_max,
            "aq_over_a0_at_chi_t_for_eta_max": allowed_midpoint_response,
            "required_eta_over_allowed_eta": eta / eta_max,
            "beta_required_for_best_bcg": float(np.min(beta_required)),
            "beta_required_for_median_bcg": float(np.median(beta_required)),
            "beta_required_for_all_bcg": float(np.max(beta_required)),
            "gamma_shift_at_beta_required_for_all_bcg": scalar_tensor_gamma_minus_one(
                float(np.max(beta_required))
            ),
        },
        "decision": {
            "prefit_checkpoint_passes": prefit_pass,
            "failed_gate": None if prefit_pass else "reciprocal Q-source backreaction",
            "stage_3_configuration_frozen": False,
            "stage_4_replay_frozen": False,
            "local_eaq0_coupling_retired": not prefit_pass,
            "next_action": (
                "freeze EA-Q0 Stage 3 and Stage 4 replay"
                if prefit_pass
                else "derive the environmental MOG control without adding an interpolation term"
            ),
        },
        "guardrails": [
            "No new astrophysical fit was performed.",
            "The H7s amplitude was not widened.",
            "The Q equation retained the reciprocal source required by the action.",
            "Only already enclosed baryonic mass was used in the exterior feedback test.",
            "The shortest point-source range compatible with 5% Q accuracy was used.",
            "Beta was maximized subject to the frozen PPN gamma gate, minimizing feedback.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
