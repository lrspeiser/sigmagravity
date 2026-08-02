#!/usr/bin/env python3
"""Run the single frozen P0705 squared-planarity endpoint screen."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p0704_spent_planarity_endpoint_joint_screen as endpoint_runner

OVERLAY = ROOT / "configs" / "p0705_spent_planarity_intensity_joint_screen.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_text(value: float | None) -> str:
    return "inf" if value is None else f"{value:.4g}"


def main() -> None:
    raw_planarity = endpoint_runner.baryonic_mass_planarity

    def intensity_planarity(density, spacing):
        geometry = raw_planarity(density, spacing)
        return dataclasses.replace(geometry, planarity=geometry.planarity**2)

    endpoint_runner.OVERLAY = OVERLAY
    endpoint_runner.baryonic_mass_planarity = intensity_planarity
    endpoint_runner.main()

    protocol = json.loads(OVERLAY.read_text(encoding="utf-8"))
    report_path = ROOT / protocol["outputs"]["directory"] / protocol["outputs"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report_version"] = "P0705-SPENT-PLANARITY-INTENSITY-JOINT-SCREEN-RESULTS-1.0.0"
    report["intensity_wrapper_source_sha256"] = sha256(Path(__file__).resolve())
    report["planarity_transform"] = "P_I=P_A^2; one frozen integer square; no exponent scan"
    report["candidate_advanced_to_sealed_outcomes"] = False
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    galaxy = report["spent_DDO154"]
    cluster = report["spent_RXJ2129"]
    fit = cluster["fit"]
    topology = cluster["topology"]
    failed = report["failed_gates"]
    summary = f"""# P0705 spent mass-planarity intensity joint screen

- Status: **{'PASS' if report['all_progression_gates_pass'] else 'FAIL'}**.
- DDO154 / RX J2129 planarity intensity: **{galaxy['field_audit']['mass_planarity']:.6g} / {cluster['field']['mass_planarity']:.6g}**.
- DDO154 RMSE / weighted RMSE: **{galaxy['candidate_score']['RMSE_km_s']:.4g} / {galaxy['candidate_score']['weighted_RMSE_km_s']:.4g} km/s**.
- DDO154 ordinary / weighted algebraic-MOND ratios: **{galaxy['comparisons']['candidate_RMSE_to_algebraic_MOND_ratio']:.4g} / {galaxy['comparisons']['candidate_weighted_RMSE_to_algebraic_MOND_ratio']:.4g}**.
- RX J2129 training / heldout roots: **{fit['training_roots_converged']}/15 / {fit['heldout_roots_converged']}/7**.
- RX J2129 training / heldout RMS / compact-halo ratio: **{metric_text(fit['training_RMS_arcsec'])} / {metric_text(fit['heldout_RMS_arcsec'])} arcsec / {metric_text(cluster['candidate_to_compact_halo_heldout_RMS_ratio'])}**.
- Missing / surplus / parity / critical families: **{topology['missing_multiplicity_families']} / {topology['potentially_observable_surplus_families']} / {topology['parity_diverse_families']} / {topology['critical_curve_present_families']}**.
- Near-bound nuisance parameters: **{fit['nuisance_parameters_near_bound']}**.
- Failed gates: **{', '.join(failed) if failed else 'none'}**.
- P0633/P0640 outcomes opened: **no**.
"""
    (report_path.parent / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
