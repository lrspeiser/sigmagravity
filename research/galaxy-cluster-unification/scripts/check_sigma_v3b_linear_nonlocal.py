from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_nonlocal_spectral import (
    entire_ir_transfer,
    entire_ir_transfer_derivative,
    entire_point_force_correction,
    entire_point_force_ratio,
    periodic_lensing_hessian,
    positive_spectral_transfer,
    positive_spectral_transfer_derivative,
    rational_far_enhancing_transfer,
    rational_point_force_ratio,
    rational_propagator_residues,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_row(frame: pd.DataFrame, model: str) -> pd.Series:
    rows = frame[(frame.cluster == "ALL") & (frame.model == model)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one ALL row for {model}, found {len(rows)}")
    return rows.iloc[0]


def make_manufactured_density(size: int, extent: float) -> tuple[np.ndarray, float]:
    coordinate = np.linspace(-extent / 2.0, extent / 2.0, size, endpoint=False)
    x, y = np.meshgrid(coordinate, coordinate)
    first = np.exp(-((x + 1.35) ** 2 + (y - 0.35) ** 2) / (2.0 * 0.24**2))
    second = 0.62 * np.exp(
        -((x - 1.05) ** 2 + (y + 0.55) ** 2) / (2.0 * 0.38**2)
    )
    bridge = 0.16 * np.exp(-((x - 0.05) ** 2 / 1.8**2 + (y + 0.05) ** 2 / 0.32**2))
    return first + second + bridge, extent / size


def shear_norm(field: dict[str, np.ndarray]) -> float:
    return float(
        np.sqrt(np.sum(np.square(field["shear_1"]) + np.square(field["shear_2"])))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v3B linear nonlocal spectral lane."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3b_linear_nonlocal_spectral_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3b_linear_nonlocal_spectral_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    diagnostic = config["spent_diagnostic_input"]
    structure_path = ROOT / diagnostic["structure_summary"]
    structure = pd.read_csv(structure_path)
    aqual = selected_row(structure, diagnostic["base_model"])
    newtonian = selected_row(structure, diagnostic["newtonian_model"])
    halo_kappa = float(aqual.median_halo_convergence)
    aqual_kappa = float(aqual.median_model_convergence)
    newtonian_kappa = float(newtonian.median_model_convergence)
    required_from_aqual = halo_kappa / aqual_kappa
    required_from_newtonian = halo_kappa / newtonian_kappa
    entire_log_boost = float(np.log(required_from_aqual))
    rational_amplitude = required_from_aqual - 1.0

    dimensionless_s = np.geomspace(1e-8, 1e8, 4000)
    positive_masses = np.geomspace(1e-3, 1e3, 31)
    positive_residues = np.exp(-np.square(np.log(positive_masses)) / 8.0)
    positive_transfer = positive_spectral_transfer(
        dimensionless_s, positive_masses, positive_residues
    )
    positive_derivative = positive_spectral_transfer_derivative(
        dimensionless_s, positive_masses, positive_residues
    )
    standard_spectral_monotone = bool(np.all(positive_derivative >= 0.0))
    standard_spectral_ir_not_enhanced = bool(positive_transfer[0] <= 1.0)

    rational_transfer = rational_far_enhancing_transfer(
        dimensionless_s, rational_amplitude
    )
    residues = rational_propagator_residues(rational_amplitude)
    rational_residues_nonnegative = bool(residues["massive_residue"] >= 0.0)

    entire_transfer = entire_ir_transfer(dimensionless_s, entire_log_boost)
    entire_derivative = entire_ir_transfer_derivative(
        dimensionless_s, entire_log_boost
    )
    entire_standard_spectral_monotonicity = bool(np.all(entire_derivative >= 0.0))
    entire_no_finite_zero_or_added_pole = True
    entire_luminal_massless_pole = True
    entire_causal_lorentzian_prescription_proved = False

    radius_over_length = np.geomspace(1e-12, 1e2, 2500)
    rational_force = rational_point_force_ratio(radius_over_length, rational_amplitude)
    entire_force = entire_point_force_ratio(radius_over_length, entire_log_boost)
    scales = config["illustrative_scales"]
    response_length_kpc = float(scales["response_length_kpc"])
    au_per_kpc = 206264806.24709636
    solar_x = float(scales["solar_radius_au"]) / (response_length_kpc * au_per_kpc)
    galaxy_x = float(scales["galaxy_radius_kpc"]) / response_length_kpc
    cluster_x = float(scales["cluster_radius_kpc"]) / response_length_kpc

    manufactured = config["manufactured_map"]
    density, pixel_scale = make_manufactured_density(
        int(manufactured["grid_size"]),
        float(manufactured["extent_in_response_lengths"]),
    )
    base_field = periodic_lensing_hessian(density, pixel_scale)
    filtered_field = periodic_lensing_hessian(
        density,
        pixel_scale,
        log_ir_boost=entire_log_boost,
        response_length=1.0,
    )
    shear_difference = {
        "shear_1": filtered_field["shear_1"] - base_field["shear_1"],
        "shear_2": filtered_field["shear_2"] - base_field["shear_2"],
    }
    relative_shear_change = shear_norm(shear_difference) / shear_norm(base_field)
    nonzero_shear_response = bool(
        relative_shear_change
        >= float(manufactured["minimum_relative_shear_l2_change"])
    )

    transfer_frame = pd.DataFrame(
        {
            "momentum_squared_times_length_squared": dimensionless_s,
            "positive_spectral_transfer_uv_normalized": positive_transfer,
            "positive_spectral_derivative": positive_derivative,
            "rational_far_enhancing_transfer": rational_transfer,
            "entire_no_pole_transfer": entire_transfer,
            "entire_transfer_derivative": entire_derivative,
        }
    )
    transfer_frame.to_csv(args.output / "transfer_curves.csv", index=False)
    force_frame = pd.DataFrame(
        {
            "radius_over_response_length": radius_over_length,
            "rational_point_force_ratio": rational_force,
            "entire_point_force_ratio": entire_force,
        }
    )
    force_frame.to_csv(args.output / "point_force_curves.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].semilogx(np.sqrt(dimensionless_s), positive_transfer, label="positive spectrum")
    axes[0, 0].semilogx(np.sqrt(dimensionless_s), rational_transfer, label="rational IR")
    axes[0, 0].semilogx(np.sqrt(dimensionless_s), entire_transfer, label="entire IR")
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 0].set(xlabel=r"$kL_\Sigma$", ylabel="static transfer", title="Momentum response")
    axes[0, 0].legend()
    axes[0, 1].semilogx(radius_over_length, rational_force, label="rational IR")
    axes[0, 1].semilogx(radius_over_length, entire_force, label="entire IR")
    axes[0, 1].axhline(required_from_aqual, color="black", linestyle="--", label="spent amplitude anchor")
    axes[0, 1].set(xlabel=r"$r/L_\Sigma$", ylabel=r"$g/g_{\rm local}$", title="Point-source force")
    axes[0, 1].legend()
    image0 = axes[1, 0].imshow(base_field["convergence"], origin="lower", cmap="coolwarm")
    axes[1, 0].set_title("Manufactured baryonic convergence")
    figure.colorbar(image0, ax=axes[1, 0], shrink=0.8)
    extra_shear = np.hypot(shear_difference["shear_1"], shear_difference["shear_2"])
    image1 = axes[1, 1].imshow(extra_shear, origin="lower", cmap="magma")
    axes[1, 1].set_title("Entire-filter added shear magnitude")
    figure.colorbar(image1, ax=axes[1, 1], shrink=0.8)
    figure.savefig(args.output / "linear_nonlocal_spectral_audit.png", dpi=180)
    plt.close(figure)

    gates = config["gates"]
    parameter_count_pass = bool(
        config["parameters"]["total_provisional_physical_parameter_count"]
        <= gates["maximum_total_universal_constants"]
    )
    positive_exchange_can_enhance_ir = not standard_spectral_ir_not_enhanced
    one_transfer_passes_all_health_gates = bool(
        positive_exchange_can_enhance_ir
        or (
            rational_residues_nonnegative
            and residues["massive_residue"] >= 0.0
        )
        or (
            entire_no_finite_zero_or_added_pole
            and entire_standard_spectral_monotonicity
            and entire_causal_lorentzian_prescription_proved
        )
    )
    advances = bool(
        parameter_count_pass
        and nonzero_shear_response
        and entire_luminal_massless_pole
        and one_transfer_passes_all_health_gates
    )

    report = {
        "status": "completed Sigma v3B linear nonlocal spectral audit",
        "model_id": config["model_id"],
        "input_hashes": {
            "config": sha256(args.config),
            "structure_summary": sha256(structure_path),
        },
        "spent_amplitude_anchors": {
            "median_newtonian_baryon_convergence": newtonian_kappa,
            "median_sigma_v1_AQUAL_convergence": aqual_kappa,
            "median_halo_convergence": halo_kappa,
            "required_low_k_boost_from_AQUAL": required_from_aqual,
            "required_low_k_boost_from_Newtonian": required_from_newtonian,
            "entire_log_boost_A_from_AQUAL": entire_log_boost,
            "rational_amplitude_from_AQUAL": rational_amplitude,
        },
        "standard_positive_spectral_exchange": {
            "formula": "D_E(s)=Z0/s+sum rho_i/(s+m_i^2), Z0>0, rho_i>=0",
            "uv_normalized_transfer_is_monotone_increasing": standard_spectral_monotone,
            "infrared_transfer_no_greater_than_UV": standard_spectral_ir_not_enhanced,
            "can_supply_required_IR_enhancement": positive_exchange_can_enhance_ir,
            "analytic_reason": "d[s D_E(s)]/ds=sum rho_i m_i^2/(s+m_i^2)^2 >= 0",
        },
        "rational_retarded_control": {
            "massless_residue": residues["massless_residue"],
            "massive_residue": residues["massive_residue"],
            "all_residues_nonnegative": rational_residues_nonnegative,
            "negative_residue_fraction_of_massless": abs(residues["massive_residue"])
            / residues["massless_residue"],
        },
        "entire_no_pole_escape": {
            "transfer": "T(s)=exp[A exp(-s L_sigma^2)]",
            "no_finite_zero_or_added_propagator_pole": entire_no_finite_zero_or_added_pole,
            "massless_tensor_pole_remains_luminal": entire_luminal_massless_pole,
            "standard_positive_spectral_monotonicity": entire_standard_spectral_monotonicity,
            "causal_Lorentzian_prescription_proved": entire_causal_lorentzian_prescription_proved,
            "claim_boundary": "failure of the standard spectral test is not a theorem against every generalized nonlocal spectral representation",
        },
        "illustrative_scale_response": {
            "response_length_kpc": response_length_kpc,
            "solar_radius_over_length": solar_x,
            "solar_entire_fractional_force_correction": float(
                entire_point_force_correction(solar_x, entire_log_boost)
            ),
            "galaxy_radius_over_length": galaxy_x,
            "galaxy_entire_force_ratio": float(
                entire_point_force_ratio(galaxy_x, entire_log_boost)
            ),
            "cluster_radius_over_length": cluster_x,
            "cluster_entire_force_ratio": float(
                entire_point_force_ratio(cluster_x, entire_log_boost)
            ),
            "asymptotic_force_ratio": float(np.exp(entire_log_boost)),
        },
        "manufactured_tidal_response": {
            "relative_shear_L2_change": relative_shear_change,
            "nonzero_baryon_registered_shear_response": nonzero_shear_response,
            "interpretation": "the common metric filter carries trace-free orientation; this is not an astronomical score",
        },
        "parameter_economy": {
            **config["parameters"],
            "passes": parameter_count_pass,
        },
        "gate_results": {
            "positive_spectral_linear_exchange_can_enhance_IR": positive_exchange_can_enhance_ir,
            "rational_IR_filter_has_no_negative_residue": rational_residues_nonnegative,
            "entire_filter_has_no_extra_finite_pole": entire_no_finite_zero_or_added_pole,
            "entire_filter_passes_standard_positive_spectral_test": entire_standard_spectral_monotonicity,
            "entire_filter_has_proved_causal_Lorentzian_prescription": entire_causal_lorentzian_prescription_proved,
            "luminal_massless_tensor_pole": entire_luminal_massless_pole,
            "nonzero_baryon_registered_shear_response": nonzero_shear_response,
            "parameter_count": parameter_count_pass,
        },
        "advances_to_frozen_linear_sigma_v3": advances,
        "decision": "retire ordinary positive-spectrum and rational linear IR enhancement; do not freeze the entire escape until a causal generalized-positive completion is proved",
        "next_mechanism": "derive a nonlinear retarded tidal interaction whose quadratic propagator remains the Sigma-v1/GR propagator, so cluster-scale activation does not require a negative linear spectral residue",
        "claim_boundary": config["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["gate_results"], indent=2, sort_keys=True))
    print(json.dumps(report["illustrative_scale_response"], indent=2, sort_keys=True))
    print(report["decision"])


if __name__ == "__main__":
    main()
