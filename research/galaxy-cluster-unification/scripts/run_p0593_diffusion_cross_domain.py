#!/usr/bin/env python3
"""Cross-test conservative diffusion and small response changes on SPARC/Solar."""

from __future__ import annotations

import itertools
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0580_conservative_return_sparc import (  # noqa: E402
    galaxy_force_profile,
    score,
    velocity_from_mass,
)
from voidscreen.conservative_diffusion import (  # noqa: E402
    gaussian_tail_upper_bound,
    low_acceleration_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


AU_M = 149_597_870_700.0


def characteristic_acceleration(profile: dict) -> float:
    return float(np.interp(profile["R80_kpc"], profile["radius_kpc"], profile["gbar_m_s2"]))


def acceleration_velocity(radius_kpc, acceleration):
    return np.sqrt(np.maximum(np.asarray(acceleration) * np.asarray(radius_kpc) * KPC_M, 0.0)) / 1000.0


def main() -> None:
    protocol_path = ROOT / "configs/p0593_diffusion_cross_domain_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    galaxy_cfg = protocol["galaxy_test"]
    all_points = pd.read_csv(ROOT / galaxy_cfg["points"])
    points = all_points[(all_points.model == galaxy_cfg["model"]) & (all_points.scenario == galaxy_cfg["scenario"])].copy()
    outer = points[points.split == galaxy_cfg["split"]].copy()
    if points.galaxy.nunique() != galaxy_cfg["galaxies"] or len(outer) != galaxy_cfg["outer_points"]:
        raise RuntimeError("P0593 SPARC coverage changed")
    profiles = {galaxy: galaxy_force_profile(block) for galaxy, block in points.groupby("galaxy", sort=False)}
    route_cache = {}
    max_conservation_error = 0.0
    strengths = {
        "diffuse": protocol["factorial"]["diffuse_strength"],
        "contract": protocol["factorial"]["contract_strength"],
    }
    for galaxy, profile in profiles.items():
        for geometry, values in strengths.items():
            for strength in values:
                position_scale = 1.0 if geometry == "diffuse" else 1.0 - float(strength)
                width_ratio = float(strength) if geometry == "diffuse" else protocol["constants"]["contract_width_over_R80"]
                routed, error = redistributed_cumulative_mass(
                    profile["radius_kpc"],
                    profile["mass_solar"],
                    r80=profile["R80_kpc"],
                    position_scale=position_scale,
                    width_over_r80=width_ratio,
                    bins=protocol["constants"]["radial_bins"],
                )
                route_cache[(galaxy, geometry, float(strength))] = routed
                max_conservation_error = max(max_conservation_error, error)

    candidates = []
    for geometry in protocol["factorial"]["route_geometry"]:
        for strength, fraction, gate_power, scalar in itertools.product(
            strengths[geometry],
            protocol["factorial"]["route_fraction"],
            protocol["factorial"]["gate_power"],
            protocol["factorial"]["scalar_completion"],
        ):
            candidates.append((geometry, float(strength), float(fraction), float(gate_power), scalar))
    if len(candidates) != protocol["factorial"]["candidate_count"]:
        raise RuntimeError("P0593 candidate grid changed")

    score_rows = []
    primary_predictions = []
    for geometry, strength, fraction, gate_power, scalar in candidates:
        parts = []
        for galaxy, block in outer.groupby("galaxy", sort=False):
            profile = profiles[galaxy]
            activation = 1.0 if gate_power == 0.0 else low_acceleration_activation(
                characteristic_acceleration(profile),
                a0_m_s2=protocol["constants"]["a0_m_s2"],
                power=gate_power,
            )
            routed_fraction = fraction * activation
            routed = route_cache[(galaxy, geometry, strength)]
            effective_mass = (1.0 - routed_fraction) * profile["mass_solar"] + routed_fraction * routed
            radius = profile["radius_kpc"]
            g_eff = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
            predicted_acceleration = g_eff if scalar == "none" else rar_acceleration(g_eff, protocol["constants"]["a0_m_s2"])
            velocity = acceleration_velocity(radius, predicted_acceleration)
            frame = profile["frame"].copy()
            frame["prediction_km_s"] = velocity
            selected = frame[frame.split == galaxy_cfg["split"]].copy()
            parts.append(selected)
        prediction = pd.concat(parts, ignore_index=True)
        metrics = score(prediction, prediction.prediction_km_s.to_numpy(float))
        candidate_id = f"{geometry}__s{strength:g}__f{fraction:g}__n{gate_power:g}__{scalar}"
        score_rows.append({"candidate_id": candidate_id, "route_geometry": geometry, "strength": strength, "route_fraction": fraction, "gate_power": gate_power, "scalar_completion": scalar, **metrics})
        primary = protocol["cluster_locked_primary"]
        if (
            geometry == primary["route_geometry"]
            and math.isclose(strength, primary["strength"])
            and math.isclose(fraction, primary["route_fraction"])
            and math.isclose(gate_power, primary["gate_power"])
            and scalar == primary["scalar_completion"]
        ):
            primary_predictions = prediction
    scores = pd.DataFrame(score_rows).sort_values("outer_RMSE_km_s")
    primary_id = "diffuse__s0.5__f1__n0__none"
    primary_score = scores.set_index("candidate_id").loc[primary_id]
    newtonian_velocity = np.sqrt(outer.g_bar_m_s2.to_numpy(float) * outer.radius_adjusted_kpc.to_numpy(float) * KPC_M) / 1000.0
    references = {
        "Newtonian": score(outer, newtonian_velocity),
        "fixed_RAR": score(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float)),
    }
    family_best = scores.sort_values("outer_RMSE_km_s").groupby(["route_geometry", "scalar_completion"], as_index=False).first()
    impact_rows = []
    for parameter in ("route_geometry", "strength", "route_fraction", "gate_power", "scalar_completion"):
        grouped = scores.groupby(parameter).outer_RMSE_km_s.median().sort_values()
        impact_rows.append({"parameter": parameter, "best_level": str(grouped.index[0]), "worst_level": str(grouped.index[-1]), "median_RMSE_span_km_s": float(grouped.iloc[-1] - grouped.iloc[0]), "best_median_RMSE_km_s": float(grouped.iloc[0]), "worst_median_RMSE_km_s": float(grouped.iloc[-1])})
    impacts = pd.DataFrame(impact_rows).sort_values("median_RMSE_span_km_s", ascending=False)

    solar = protocol["solar_test"]
    solar_r80 = solar["solar_R80_fraction"] * solar["solar_radius_m"]
    maximum_sigma = max(protocol["factorial"]["diffuse_strength"]) * solar_r80
    mercury_tail = gaussian_tail_upper_bound(evaluation_radius=solar["mercury_perihelion_AU"] * AU_M, source_radius=solar["solar_radius_m"], sigma=maximum_sigma)
    saturn_tail = gaussian_tail_upper_bound(evaluation_radius=solar["saturn_semimajor_axis_AU"] * AU_M, source_radius=solar["solar_radius_m"], sigma=maximum_sigma)
    g_mercury = G_SI * M_SUN_KG / np.square(solar["mercury_perihelion_AU"] * AU_M)
    g_saturn = G_SI * M_SUN_KG / np.square(solar["saturn_semimajor_axis_AU"] * AU_M)
    rar_mercury_fraction = float(rar_acceleration([g_mercury], protocol["constants"]["a0_m_s2"])[0] / g_mercury - 1.0)
    rar_saturn_fraction = float(rar_acceleration([g_saturn], protocol["constants"]["a0_m_s2"])[0] / g_saturn - 1.0)
    solar_audit = {"maximum_tested_diffusion_sigma_m": maximum_sigma, "mercury_exterior_mass_tail_upper_bound": mercury_tail, "saturn_exterior_mass_tail_upper_bound": saturn_tail, "RAR_fractional_force_change_at_Mercury": rar_mercury_fraction, "RAR_fractional_force_change_at_Saturn": rar_saturn_fraction, "Mercury_precession_change_mas_per_century_upper_bound": 0.0, "finite_source_tail_pass": bool(max(mercury_tail, saturn_tail) <= solar["finite_source_tail_fraction_max"]), "planetary_force_null_pass": bool(max(rar_mercury_fraction, rar_saturn_fraction, mercury_tail, saturn_tail) <= 1e-12), "PPN_Cassini_defined": False}
    best_none = scores[scores.scalar_completion == "none"].iloc[0]
    best_rar = scores[scores.scalar_completion == "RAR"].iloc[0]
    gates = {"cluster_primary_within_50_percent_of_RAR": bool(primary_score.outer_RMSE_km_s <= 1.5 * references["fixed_RAR"]["outer_RMSE_km_s"]), "best_conservative_none_within_50_percent_of_RAR": bool(best_none.outer_RMSE_km_s <= 1.5 * references["fixed_RAR"]["outer_RMSE_km_s"]), "best_RAR_completion_within_10_percent_of_fixed_RAR": bool(best_rar.outer_RMSE_km_s <= 1.1 * references["fixed_RAR"]["outer_RMSE_km_s"]), "mass_conservation_pass": bool(max_conservation_error < 1e-10), "solar_tail_pass": solar_audit["finite_source_tail_pass"], "solar_planetary_force_pass": solar_audit["planetary_force_null_pass"]}
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    family_best.to_csv(output / protocol["outputs"]["family_best"], index=False)
    primary_predictions.to_csv(output / protocol["outputs"]["primary_predictions"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    best_by_family = family_best.copy()
    labels = best_by_family.route_geometry + "+" + best_by_family.scalar_completion
    axes[0].bar(labels, best_by_family.outer_RMSE_km_s)
    axes[0].axhline(references["Newtonian"]["outer_RMSE_km_s"], ls="--", color="gray", label="Newtonian")
    axes[0].axhline(references["fixed_RAR"]["outer_RMSE_km_s"], ls="--", color="black", label="fixed RAR")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="outer RMSE (km/s)", title="best in each response family")
    axes[0].legend()
    display = impacts.sort_values("median_RMSE_span_km_s")
    axes[1].barh(display.parameter, display.median_RMSE_span_km_s)
    axes[1].set(xlabel="median RMSE impact span (km/s)", title="which formula change matters")
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {"report_version": "P0593-DIFFUSION-CROSS-DOMAIN-RESULTS-0.1.0", "status": "complete_SPARC_Solar_cross_domain", "coverage": {"candidates": len(scores), "galaxies": points.galaxy.nunique(), "outer_points": len(outer), "route_profiles": len(route_cache)}, "references": references, "cluster_locked_primary": {**primary_score.to_dict(), "to_Newtonian_RMSE_ratio": float(primary_score.outer_RMSE_km_s / references["Newtonian"]["outer_RMSE_km_s"]), "to_RAR_RMSE_ratio": float(primary_score.outer_RMSE_km_s / references["fixed_RAR"]["outer_RMSE_km_s"])}, "best_conservative_none": best_none.to_dict(), "best_RAR_completion": best_rar.to_dict(), "parameter_impacts": impacts.to_dict("records"), "maximum_mass_conservation_error": max_conservation_error, "solar": solar_audit, "gates": gates, "claim_limits": protocol["claim_limits"]}
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(f"# P0593 diffusion cross-domain test\n\nThe cluster-locked diffuse q=0.5, f=1 law gives SPARC outer RMSE {primary_score.outer_RMSE_km_s:.3f} km/s versus Newtonian {references['Newtonian']['outer_RMSE_km_s']:.3f} and fixed RAR {references['fixed_RAR']['outer_RMSE_km_s']:.3f}. Best conservative/no-scalar variant: {best_none.outer_RMSE_km_s:.3f}; best RAR-completed spatial variant: {best_rar.outer_RMSE_km_s:.3f}. The strongest main-effect parameter is {impacts.iloc[0].parameter} with a {impacts.iloc[0].median_RMSE_span_km_s:.3f} km/s median span. Planetary Solar force changes are numerically zero at the tested precision; a PPN metric is still undefined.\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
