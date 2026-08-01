#!/usr/bin/env python3
"""Audit acceleration screening inside a Solar proxy and at MACS J0416."""

from __future__ import annotations

import json
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

from run_p0580_conservative_return_sparc import galaxy_force_profile, score  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity, characteristic_acceleration  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    radial_shape_activation,
    redistributed_cumulative_mass,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


def activation(acceleration: float, a0: float, power: float) -> float:
    return 1.0 if float(power) == 0.0 else low_acceleration_activation(
        acceleration, a0_m_s2=a0, power=float(power)
    )


def weighted_radii(frame: pd.DataFrame) -> tuple[float, float]:
    weight = frame.mass_msun.to_numpy(dtype=float, copy=True)
    weight /= weight.sum()
    x = frame.x_arcsec.to_numpy(float)
    y = frame.y_arcsec.to_numpy(float)
    center_x = float(np.sum(weight * x))
    center_y = float(np.sum(weight * y))
    radius = np.hypot(x - center_x, y - center_y)
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order])
    values = []
    for fraction in (0.5, 0.8):
        index = min(np.searchsorted(cumulative, fraction), len(order) - 1)
        values.append(float(radius[order[index]]))
    return values[0], values[1]


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0598_stellar_interior_screen_protocol.json").read_text(encoding="utf-8")
    )
    formula = protocol["parent_formula"]
    galaxy_protocol = json.loads(
        (ROOT / protocol["data"]["galaxy_protocol"]).read_text(encoding="utf-8")
    )
    galaxy_cfg = galaxy_protocol["galaxy_test"]
    raw = pd.read_csv(ROOT / galaxy_cfg["points"])
    points = raw[(raw.model == galaxy_cfg["model"]) & (raw.scenario == galaxy_cfg["scenario"])].copy()
    points["source_point_index"] = points.index
    outer = points[points.split == galaxy_cfg["split"]].copy().reset_index(drop=True)
    profiles = {galaxy: galaxy_force_profile(block) for galaxy, block in points.groupby("galaxy", sort=False)}
    route_cache = {}
    shape_cache = {}
    for galaxy, profile in profiles.items():
        routed, _ = redistributed_cumulative_mass(
            profile["radius_kpc"],
            profile["mass_solar"],
            r80=profile["R80_kpc"],
            position_scale=1.0,
            width_over_r80=formula["q_R80"],
            bins=galaxy_protocol["constants"]["radial_bins"],
        )
        route_cache[galaxy] = routed
        shape_cache[galaxy] = radial_shape_activation(
            profile["concentration_R50_over_R80"],
            midpoint=formula["shape_midpoint"],
            width=formula["shape_width"],
        )

    solar_radius = galaxy_protocol["solar_test"]["solar_radius_m"]
    solar_r = np.linspace(solar_radius / protocol["stellar_proxy"]["radial_samples"], solar_radius, protocol["stellar_proxy"]["radial_samples"])
    solar_mass = np.power(solar_r / solar_radius, 3.0)
    solar_r80 = solar_radius * 0.8 ** (1.0 / 3.0)
    solar_r50 = solar_radius * 0.5 ** (1.0 / 3.0)
    solar_c = solar_r50 / solar_r80
    solar_shape = radial_shape_activation(
        solar_c, midpoint=formula["shape_midpoint"], width=formula["shape_width"]
    )
    solar_routed, _ = redistributed_cumulative_mass(
        solar_r,
        solar_mass,
        r80=solar_r80,
        position_scale=1.0,
        width_over_r80=formula["q_R80"],
        bins=galaxy_protocol["constants"]["radial_bins"],
    )
    solar_g_r80 = G_SI * M_SUN_KG * 0.8 / np.square(solar_r80)

    macs_protocol = json.loads(
        (ROOT / protocol["data"]["macs0416_protocol"]).read_text(encoding="utf-8")
    )
    macs_report = json.loads((ROOT / protocol["data"]["macs0416_report"]).read_text(encoding="utf-8"))
    macs_sources = pd.read_csv(ROOT / protocol["data"]["macs0416_sources"])
    macs_r50_arcsec, macs_r80_arcsec = weighted_radii(macs_sources)
    scale = macs_protocol["coordinate_system"]["scale_kpc_per_arcsec_planck18"]
    macs_r80_m = macs_r80_arcsec * scale * KPC_M
    macs_total_mass = float(macs_sources.mass_msun.sum())
    macs_g_r80 = G_SI * M_SUN_KG * 0.8 * macs_total_mass / np.square(macs_r80_m)
    macs_c = macs_r50_arcsec / macs_r80_arcsec
    macs_shape = radial_shape_activation(
        macs_c, midpoint=formula["shape_midpoint"], width=formula["shape_width"]
    )

    gate_rows = []
    solar_profile_rows = []
    predictions = {}
    a0 = galaxy_protocol["constants"]["a0_m_s2"]
    for power in protocol["gate_powers"]:
        prediction = np.empty(len(outer), dtype=float)
        for galaxy, indices in outer.groupby("galaxy", sort=False).indices.items():
            profile = profiles[galaxy]
            screened = activation(characteristic_acceleration(profile), a0, power)
            fraction = formula["route_fraction_max"] * shape_cache[galaxy] * screened
            effective_mass = (1.0 - fraction) * profile["mass_solar"] + fraction * route_cache[galaxy]
            radius = profile["radius_kpc"]
            g_eff = G_SI * M_SUN_KG * effective_mass / np.square(radius * KPC_M)
            full_velocity = acceleration_velocity(radius, rar_acceleration(g_eff, a0))
            frame = profile["frame"]
            mask = frame.split.to_numpy(str) == galaxy_cfg["split"]
            velocity_by_source = dict(zip(frame.loc[mask, "source_point_index"], full_velocity[mask]))
            target = np.asarray(indices, dtype=int)
            prediction[target] = [
                velocity_by_source[source]
                for source in outer.loc[target, "source_point_index"].to_numpy(int)
            ]
        predictions[float(power)] = prediction
        metrics = score(outer, prediction)
        rar_metrics = score(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float))
        galaxy_gain = 1.0 - metrics["outer_equal_galaxy_RMSE_km_s"] / rar_metrics["outer_equal_galaxy_RMSE_km_s"]
        solar_screen = activation(solar_g_r80, a0, power)
        solar_fraction = formula["route_fraction_max"] * solar_shape * solar_screen
        solar_effective = (1.0 - solar_fraction) * solar_mass + solar_fraction * solar_routed
        solar_force_change = solar_effective / solar_mass - 1.0
        macs_screen = activation(macs_g_r80, a0, power)
        gate_rows.append(
            {
                "gate_power": power,
                **metrics,
                "galaxy_equal_RMSE_improvement_fraction_vs_fixed_RAR": galaxy_gain,
                "solar_g_R80_m_s2": solar_g_r80,
                "solar_activation": solar_screen,
                "solar_effective_route_fraction": solar_fraction,
                "solar_maximum_absolute_interior_force_change": float(np.max(np.abs(solar_force_change))),
                "macs0416_g_R80_m_s2": macs_g_r80,
                "macs0416_activation": macs_screen,
                "macs0416_effective_route_fraction": formula["route_fraction_max"] * macs_shape * macs_screen,
            }
        )
        for radius, change in zip(solar_r, solar_force_change):
            solar_profile_rows.append(
                {"gate_power": power, "radius_over_solar_radius": radius / solar_radius, "fractional_force_change": change}
            )
    scores = pd.DataFrame(gate_rows)
    solar_profiles = pd.DataFrame(solar_profile_rows)
    safe = scores[
        scores.solar_maximum_absolute_interior_force_change
        <= protocol["stellar_proxy"]["maximum_interior_fractional_force_change"]
    ]
    selected = safe.sort_values(["outer_equal_galaxy_RMSE_km_s", "gate_power"]).iloc[0]
    cfg = protocol["interpretation_gates"]
    gates = {
        "galaxy_improvement_pass": bool(selected.galaxy_equal_RMSE_improvement_fraction_vs_fixed_RAR >= cfg["galaxy_equal_RMSE_improvement_fraction_min"]),
        "stellar_interior_screen_pass": bool(selected.solar_maximum_absolute_interior_force_change <= cfg["stellar_maximum_interior_force_change_max"]),
        "macs0416_activation_pass": bool(selected.macs0416_activation >= cfg["macs0416_activation_min"]),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["gate_scores"], index=False)
    solar_profiles.to_csv(output / protocol["outputs"]["solar_profiles"], index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(scores.gate_power, scores.outer_equal_galaxy_RMSE_km_s, marker="o")
    axes[0].axhline(rar_metrics["outer_equal_galaxy_RMSE_km_s"], color="black", ls="--", label="fixed RAR")
    axes[0].set(xlabel="acceleration gate power n", ylabel="galaxy equal RMSE (km/s)", title="galaxy response")
    axes[0].legend()
    for power, block in solar_profiles.groupby("gate_power"):
        axes[1].plot(block.radius_over_solar_radius, np.abs(block.fractional_force_change), label=f"n={power:g}")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="radius / Solar radius", ylabel="|fractional force change|", title="uniform-Sun interior proxy")
    axes[1].legend()
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)
    report = {
        "report_version": "P0598-STELLAR-INTERIOR-SCREEN-RESULTS-0.1.0",
        "status": "complete_stellar_interior_screen",
        "selected_safe_gate": selected.to_dict(),
        "fixed_RAR_reference": rar_metrics,
        "macs0416": {"source_mass_solar": macs_total_mass, "R50_arcsec": macs_r50_arcsec, "R80_arcsec": macs_r80_arcsec, "C_R50_over_R80": macs_c, "shape_activation": macs_shape, "g_R80_m_s2": macs_g_r80, "report_R80_arcsec_crosscheck": macs_report["nominal"]["r80_arcsec"]},
        "solar_proxy": {"C_R50_over_R80": solar_c, "shape_activation": solar_shape, "g_R80_m_s2": solar_g_r80},
        "gate_comparison": scores.to_dict("records"),
        "gates": gates,
        "all_interpretation_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0598 stellar-interior acceleration screen\n\n"
        f"The ungated n=0 spatial law changes the uniform-Sun interior force by up to "
        f"{scores.loc[scores.gate_power == 0, 'solar_maximum_absolute_interior_force_change'].iloc[0]:.2%}. "
        f"The best galaxy-scoring gate that passes the 1e-8 interior proxy is n={selected.gate_power:g}; "
        f"its galaxy equal-weighted RMSE is {selected.outer_equal_galaxy_RMSE_km_s:.3f} km/s, "
        f"its Solar proxy force change is {selected.solar_maximum_absolute_interior_force_change:.3e}, "
        f"and its physically normalized MACS J0416 activation is {selected.macs0416_activation:.6f}.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
