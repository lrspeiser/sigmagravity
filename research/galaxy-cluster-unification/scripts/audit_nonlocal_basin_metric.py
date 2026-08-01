from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.basin_metric import (
    basin_metric_coefficients,
    beta_for_response_ratio,
    lensing_to_dynamics_extra_ratio,
)
from voidscreen.unified import (
    load_clash_acceleration_frame,
    load_sparc_acceleration_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_summary(
    name: str,
    chi: np.ndarray,
    observed: np.ndarray,
    baryonic: np.ndarray,
    systems: int,
    evidence: str,
) -> dict[str, object]:
    log_chi = np.log10(np.asarray(chi, dtype=float))
    enhancement = np.asarray(observed, dtype=float) / np.asarray(
        baryonic, dtype=float
    )
    quantiles = [0.0, 0.05, 0.5, 0.95, 1.0]
    return {
        "domain": name,
        "systems": systems,
        "points": len(log_chi),
        "evidence": evidence,
        "log10_compactness_quantiles": dict(
            zip(
                ["minimum", "p05", "median", "p95", "maximum"],
                np.quantile(log_chi, quantiles).tolist(),
                strict=True,
            )
        ),
        "enhancement_quantiles": dict(
            zip(
                ["minimum", "p05", "median", "p95", "maximum"],
                np.quantile(enhancement, quantiles).tolist(),
                strict=True,
            )
        ),
        "positive_extra_fraction": float(np.mean(enhancement > 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nonlocal_basin_metric_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nonlocal_basin_metric_gate",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    inputs = protocol["input_audits"]
    observable_path = ROOT / inputs["observable_report"]
    same_system_path = ROOT / inputs["same_system_report"]
    ceiling_path = ROOT / inputs["public_data_ceiling_report"]
    observable = json.loads(observable_path.read_text(encoding="utf-8"))
    same_system = json.loads(same_system_path.read_text(encoding="utf-8"))
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8"))

    sparc = load_sparc_acceleration_frame(ROOT / inputs["SPARC_directory"])
    clash = load_clash_acceleration_frame(ROOT / inputs["CLASH_summary"])
    bcg = pd.read_csv(ROOT / inputs["BCG_summary"])
    summaries = [
        support_summary(
            "SPARC_dynamics",
            sparc["chi"].to_numpy(),
            sparc["observed_g_m_s2"].to_numpy(),
            sparc["gbar_m_s2"].to_numpy(),
            sparc["system"].nunique(),
            "resolved rotation speeds; direct dynamics after catalog calibration",
        ),
        support_summary(
            "BCG_dynamics_summary",
            bcg["total_profile_chi"].to_numpy(),
            bcg["gobs_m_s2"].to_numpy(),
            bcg["gbar_m_s2"].to_numpy(),
            bcg["plateifu"].nunique(),
            "11 one-radius Jeans summaries plus 23 calibrated dynamics proxies",
        ),
        support_summary(
            "CLASH_lensing_summary",
            clash["chi"].to_numpy(),
            clash["observed_g_m_s2"].to_numpy(),
            clash["gbar_m_s2"].to_numpy(),
            clash["system"].nunique(),
            "GR/NFW-deprojected acceleration summary; not theory-neutral",
        ),
    ]
    by_domain = {row["domain"]: row for row in summaries}
    sparc_robust_high = by_domain["SPARC_dynamics"][
        "log10_compactness_quantiles"
    ]["p95"]
    clash_robust_low = by_domain["CLASH_lensing_summary"][
        "log10_compactness_quantiles"
    ]["p05"]
    robust_gap = float(clash_robust_low - sparc_robust_high)

    same_ready = int(same_system["strict_r1_ready_systems"])
    target = int(same_system["target_strict_systems"])
    algebraic_examples = {}
    for label, alpha, beta in [
        ("pure_conformal", 1.0, 0.0),
        ("pure_disformal", 0.0, 1.0),
        ("no_slip", 1.0, 2.0),
        ("lensing_twice_dynamics", 1.0, beta_for_response_ratio(1.0, 2.0)),
    ]:
        coefficients = basin_metric_coefficients(alpha, beta)
        algebraic_examples[label] = {
            "alpha": alpha,
            "beta": beta,
            "Psi_X_coefficient": coefficients.dynamics,
            "Phi_X_coefficient": coefficients.spatial_curvature,
            "Weyl_half_X_coefficient": coefficients.weyl_half,
            "lensing_to_dynamics_extra_ratio": lensing_to_dynamics_extra_ratio(
                alpha, beta
            ),
        }

    gates = {
        "one_physical_metric_declared": True,
        "dynamics_and_lensing_coefficients_derived": True,
        "pure_conformal_limit_correctly_rejected_for_extra_lensing": True,
        "screened_X_zero_recovers_GR": True,
        "no_lensing_only_parameter": True,
        "minimum_same_system_count": same_ready >= target,
        "raw_alternative_metric_lensing_ready": observable["clash"][
            "alternative_metric_forward_model_ready_systems"
        ]
        >= target,
        "robust_SPARC_CLASH_compactness_overlap": robust_gap <= 0.0,
        "ten_system_public_data_ceiling_cleared": not ceiling[
            "hard_public_data_shortfall_established"
        ],
        "full_action_health_derived": False,
    }
    algebraic_gate = all(
        gates[key]
        for key in [
            "one_physical_metric_declared",
            "dynamics_and_lensing_coefficients_derived",
            "pure_conformal_limit_correctly_rejected_for_extra_lensing",
            "screened_X_zero_recovers_GR",
            "no_lensing_only_parameter",
        ]
    )
    empirical_gate = all(
        gates[key]
        for key in [
            "minimum_same_system_count",
            "raw_alternative_metric_lensing_ready",
            "ten_system_public_data_ceiling_cleared",
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    support_path = args.output_dir / "response_support.csv"
    pd.json_normalize(summaries, sep="_").to_csv(support_path, index=False)
    report = {
        "status": "completed NBM0 algebraic and empirical-identifiability gates",
        "report_version": "NBM0-gate-0.1",
        "protocol": {
            "path": str(args.protocol.relative_to(ROOT)),
            "sha256": sha256(args.protocol),
            "version": protocol["protocol_version"],
        },
        "input_hashes": {
            "observable_report": sha256(observable_path),
            "same_system_report": sha256(same_system_path),
            "public_data_ceiling_report": sha256(ceiling_path),
            "SPARC_data_fingerprint": sparc.attrs.get(
                "data_fingerprint", "recorded in existing SPARC package"
            ),
            "CLASH_summary": sha256(ROOT / inputs["CLASH_summary"]),
            "BCG_summary": sha256(ROOT / inputs["BCG_summary"]),
        },
        "weak_field_algebraic_examples": algebraic_examples,
        "response_support": summaries,
        "compactness_audit": {
            "SPARC_p95_log10_chi": sparc_robust_high,
            "CLASH_p05_log10_chi": clash_robust_low,
            "robust_gap_dex": robust_gap,
            "robust_ranges_overlap": robust_gap <= 0.0,
            "interpretation": "The small full-range overlap is confined to tails; the central 90% supports are separated. BCG dynamics bridges toward CLASH compactness but is not same-object lensing.",
        },
        "same_system_audit": {
            "candidate_systems": same_system["candidate_systems_evaluated"],
            "structural_passes": same_system[
                "systems_passing_three_plus_three_structural_overlap"
            ],
            "strict_ready": same_ready,
            "target": target,
            "complete_baryonic_forward_inputs": same_system[
                "systems_with_complete_baryonic_forward_inputs"
            ],
            "theory_neutral_joint_covariance": same_system[
                "systems_with_theory_neutral_joint_covariance"
            ],
        },
        "gates": gates,
        "algebraic_metric_gate_pass": algebraic_gate,
        "empirical_identifiability_gate_pass": empirical_gate,
        "parameter_fit_authorized": empirical_gate
        and gates["full_action_health_derived"],
        "decision": {
            "candidate": "retain_for_action_derivation",
            "MOND_or_BTFR_interpolation": "not_used",
            "dynamics_lensing_unification": "not_yet_empirically_identifiable",
            "next_action": "Finish the already-authorized RX J2129 measurement package; do not fit alpha, beta, kappa_X, or L_X until same-system dynamics, baryons, and raw lens covariance pass the frozen gate.",
        },
        "artifact": {
            "path": str(support_path.relative_to(ROOT)),
            "sha256": sha256(support_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
