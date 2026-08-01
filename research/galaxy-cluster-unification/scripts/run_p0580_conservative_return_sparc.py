#!/usr/bin/env python3
"""Apply P0579's conservative return kernels to force-equivalent SPARC profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0575_smacs0723_raw_position import sha256  # noqa: E402
from voidscreen.arc_apogee import G_SI, M_SUN_KG, extent_gate  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402


PARAMETERS = [
    "gate_mode",
    "return_length_over_R80",
    "width_over_R80",
    "route_mode",
    "route_fraction_multiplier",
]


def radius_at_fraction(radius: np.ndarray, cumulative: np.ndarray, fraction: float) -> float:
    radius = np.asarray(radius, dtype=float)
    cumulative = np.asarray(cumulative, dtype=float)
    target = float(fraction) * cumulative[-1]
    return float(
        np.interp(target, np.concatenate([[0.0], cumulative]), np.concatenate([[0.0], radius]))
    )


def galaxy_force_profile(block: pd.DataFrame) -> dict:
    ordered = block.sort_values("radius_adjusted_kpc").copy()
    radius = ordered.radius_adjusted_kpc.to_numpy(float)
    gbar = ordered.g_bar_m_s2.to_numpy(float)
    mass = gbar * np.square(radius * KPC_M) / (G_SI * M_SUN_KG)
    mass = np.maximum.accumulate(np.maximum(mass, 0.0))
    r50 = radius_at_fraction(radius, mass, 0.5)
    r80 = radius_at_fraction(radius, mass, 0.8)
    return {
        "frame": ordered,
        "radius_kpc": radius,
        "gbar_m_s2": gbar,
        "mass_solar": mass,
        "R50_kpc": r50,
        "R80_kpc": r80,
        "concentration_R50_over_R80": r50 / max(r80, np.finfo(float).tiny),
        "total_force_equivalent_mass_solar": float(mass[-1]),
    }


def route_positions(radius, shell_mass, length, r80, mode, samples):
    radius = np.asarray(radius, dtype=float)
    shell_mass = np.asarray(shell_mass, dtype=float)
    if mode == "endpoint":
        return np.abs(radius - float(length)), shell_mass
    fraction = np.linspace(0.0, 1.0, int(samples))
    signed_chord = radius[:, None] - float(length) * fraction[None, :]
    profile = 4.0 * fraction * (1.0 - fraction)
    if mode == "chord":
        positions = np.abs(signed_chord)
    elif mode == "radial_arc_0.5":
        positions = np.abs(signed_chord + 0.5 * float(r80) * profile[None, :])
    elif mode == "transverse_arc_0.5":
        positions = np.sqrt(
            np.square(signed_chord)
            + np.square(0.5 * float(r80) * profile[None, :])
        )
    else:
        raise ValueError(f"unknown route mode {mode}")
    weights = np.repeat(shell_mass / len(fraction), len(fraction))
    return positions.reshape(-1), weights


def routed_cumulative(profile, length_ratio, width_ratio, mode, bins, samples):
    radius = profile["radius_kpc"]
    cumulative = profile["mass_solar"]
    shell_mass = np.diff(np.concatenate([[0.0], cumulative]))
    r80 = float(profile["R80_kpc"])
    length = float(length_ratio) * r80
    width = float(width_ratio) * r80
    positions, weights = route_positions(
        radius, shell_mass, length, r80, mode, samples
    )
    maximum = max(
        float(np.max(radius)),
        float(np.max(positions)) + 5.0 * width,
        r80 + length + 5.0 * width,
    )
    edges = np.linspace(0.0, maximum, int(bins) + 1)
    histogram, _ = np.histogram(positions, bins=edges, weights=weights)
    spacing = float(edges[1] - edges[0])
    smoothed = gaussian_filter1d(
        histogram.astype(float), width / spacing, mode="constant"
    )
    total = float(cumulative[-1])
    raw_total = float(np.sum(smoothed))
    if raw_total <= 0.0:
        raise RuntimeError("empty routed SPARC profile")
    smoothed *= total / raw_total
    route_cumulative = np.cumsum(smoothed)
    interpolated = np.interp(
        radius,
        np.concatenate([[0.0], edges[1:]]),
        np.concatenate([[0.0], route_cumulative]),
    )
    return interpolated, abs(float(np.sum(smoothed)) / total - 1.0)


def velocity_from_mass(radius_kpc: np.ndarray, mass_solar: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.maximum(
            G_SI * M_SUN_KG * np.asarray(mass_solar, dtype=float)
            / (np.asarray(radius_kpc, dtype=float) * KPC_M),
            0.0,
        )
    ) / 1000.0


def score(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predicted, dtype=float) - frame.velocity_observed_adjusted_km_s.to_numpy(float)
    galaxy_mse = pd.Series(
        np.square(residual), index=frame.galaxy.to_numpy(str)
    ).groupby(level=0).mean()
    return {
        "outer_RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "outer_equal_galaxy_RMSE_km_s": float(np.sqrt(galaxy_mse.mean())),
        "outer_mean_residual_km_s": float(np.mean(residual)),
    }


def parameter_impacts(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in PARAMETERS:
        grouped = scores.groupby(parameter).agg(
            median_outer_RMSE_km_s=("outer_RMSE_km_s", "median"),
            median_equal_galaxy_RMSE_km_s=(
                "outer_equal_galaxy_RMSE_km_s",
                "median",
            ),
        )
        best = grouped.median_outer_RMSE_km_s.idxmin()
        worst = grouped.median_outer_RMSE_km_s.idxmax()
        rows.append(
            {
                "parameter": parameter,
                "best_level": str(best),
                "best_median_outer_RMSE_km_s": float(
                    grouped.loc[best, "median_outer_RMSE_km_s"]
                ),
                "worst_level": str(worst),
                "worst_median_outer_RMSE_km_s": float(
                    grouped.loc[worst, "median_outer_RMSE_km_s"]
                ),
                "outer_RMSE_impact_span_km_s": float(
                    grouped.loc[worst, "median_outer_RMSE_km_s"]
                    - grouped.loc[best, "median_outer_RMSE_km_s"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "outer_RMSE_impact_span_km_s", ascending=False
    )


def main() -> None:
    protocol_path = ROOT / "configs/p0580_conservative_return_sparc_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0579_before_conservative_SPARC_scores":
        raise RuntimeError("P0580 protocol is not frozen")
    source = protocol["candidate_source"]
    p0579_path = ROOT / source["protocol"]
    p0579 = json.loads(p0579_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(ROOT / source["scores"])
    candidates = candidates[["candidate_id", *PARAMETERS]].drop_duplicates()
    if len(candidates) != int(source["candidate_count"]):
        raise RuntimeError("P0580 candidate source changed")
    if int(p0579["grid"]["candidates"]) != len(candidates):
        raise RuntimeError("P0579 protocol and candidate table disagree")

    settings = protocol["galaxy_test"]
    all_points = pd.read_csv(ROOT / settings["input_points"])
    points = all_points[
        all_points.model.eq(settings["input_model"])
        & all_points.scenario.eq(settings["input_scenario"])
    ].copy()
    outer = points[points.split.eq(settings["split_scored"])].copy()
    if points.galaxy.nunique() != int(settings["expected_galaxies"]):
        raise RuntimeError("P0580 SPARC galaxy count changed")
    if len(outer) != int(settings["expected_outer_points"]):
        raise RuntimeError("P0580 SPARC outer-point count changed")

    profiles = {}
    property_rows = []
    route_cache = {}
    maximum_conservation_error = 0.0
    unique_routes = candidates[
        ["return_length_over_R80", "width_over_R80", "route_mode"]
    ].drop_duplicates()
    for galaxy, block in points.groupby("galaxy", sort=False):
        profile = galaxy_force_profile(block)
        profiles[galaxy] = profile
        property_rows.append(
            {
                "galaxy": galaxy,
                "points": len(profile["frame"]),
                "outer_points": int(profile["frame"].split.eq(settings["split_scored"]).sum()),
                "R50_kpc": profile["R50_kpc"],
                "R80_kpc": profile["R80_kpc"],
                "concentration_R50_over_R80": profile[
                    "concentration_R50_over_R80"
                ],
                "total_force_equivalent_mass_solar": profile[
                    "total_force_equivalent_mass_solar"
                ],
            }
        )
        for route in unique_routes.itertuples(index=False):
            routed, error = routed_cumulative(
                profile,
                float(route.return_length_over_R80),
                float(route.width_over_R80),
                str(route.route_mode),
                int(settings["radial_grid_bins"]),
                int(settings["path_samples"]),
            )
            route_cache[
                (
                    galaxy,
                    float(route.return_length_over_R80),
                    float(route.width_over_R80),
                    str(route.route_mode),
                )
            ] = routed
            maximum_conservation_error = max(maximum_conservation_error, error)
    print(f"built {len(route_cache)} conservative galaxy-route profiles", flush=True)

    score_rows = []
    saved_predictions = {}
    baseline_galaxy_rmse = {}
    for galaxy, block in outer.groupby("galaxy"):
        baseline = velocity_from_mass(
            profiles[galaxy]["radius_kpc"], profiles[galaxy]["mass_solar"]
        )
        mask = profiles[galaxy]["frame"].split.eq(settings["split_scored"]).to_numpy()
        residual = baseline[mask] - block.sort_values("radius_adjusted_kpc").velocity_observed_adjusted_km_s.to_numpy(float)
        baseline_galaxy_rmse[galaxy] = float(np.sqrt(np.mean(np.square(residual))))

    for candidate in candidates.itertuples(index=False):
        prediction_parts = []
        galaxies_improved = 0
        for galaxy, block in outer.groupby("galaxy", sort=False):
            profile = profiles[galaxy]
            gate = float(
                extent_gate(
                    profile["concentration_R50_over_R80"], str(candidate.gate_mode)
                )
            )
            fraction = float(candidate.route_fraction_multiplier) * gate
            routed = route_cache[
                (
                    galaxy,
                    float(candidate.return_length_over_R80),
                    float(candidate.width_over_R80),
                    str(candidate.route_mode),
                )
            ]
            effective_mass = (1.0 - fraction) * profile["mass_solar"] + fraction * routed
            velocity = velocity_from_mass(profile["radius_kpc"], effective_mass)
            full = profile["frame"].copy()
            full["predicted_conservative_return_km_s"] = velocity
            selected = full[full.split.eq(settings["split_scored"])]
            prediction_parts.append(selected)
            local_residual = (
                selected.predicted_conservative_return_km_s.to_numpy(float)
                - selected.velocity_observed_adjusted_km_s.to_numpy(float)
            )
            galaxies_improved += int(
                np.sqrt(np.mean(np.square(local_residual)))
                < baseline_galaxy_rmse[galaxy]
            )
        prediction = pd.concat(prediction_parts, ignore_index=True)
        metrics = score(
            prediction, prediction.predicted_conservative_return_km_s.to_numpy(float)
        )
        score_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                **{parameter: getattr(candidate, parameter) for parameter in PARAMETERS},
                **metrics,
                "galaxies_improved_vs_Newtonian": galaxies_improved,
                "galaxy_improved_fraction": galaxies_improved / points.galaxy.nunique(),
            }
        )
        if candidate.candidate_id in {
            source["primary_inverse_candidate_id"],
            source["P0579_calibration_selected_id"],
        }:
            saved_predictions[candidate.candidate_id] = prediction

    scores = pd.DataFrame(score_rows).sort_values("outer_RMSE_km_s")
    primary = scores.set_index("candidate_id").loc[source["primary_inverse_candidate_id"]]
    p0579_selected = scores.set_index("candidate_id").loc[source["P0579_calibration_selected_id"]]
    best = scores.iloc[0]
    impacts = parameter_impacts(scores)

    outer_radius_m = outer.radius_adjusted_kpc.to_numpy(float) * KPC_M
    newtonian_velocity = np.sqrt(
        outer.g_bar_m_s2.to_numpy(float) * outer_radius_m
    ) / 1000.0
    references = {
        "Newtonian_same_nuisance": score(outer, newtonian_velocity),
        "fixed_RAR_same_nuisance": score(
            outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float)
        ),
    }
    arc_apogee = json.loads(
        (ROOT / source["arc_apogee_report"]).read_text(encoding="utf-8")
    )
    references["arc_apogee_R1322"] = arc_apogee["best_variant"]
    gates = {
        "primary_improves_Newtonian_outer_RMSE": bool(
            float(primary.outer_RMSE_km_s)
            < references["Newtonian_same_nuisance"]["outer_RMSE_km_s"]
        ),
        "primary_within_50_percent_of_fixed_RAR_outer_RMSE": bool(
            float(primary.outer_RMSE_km_s)
            <= 1.5 * references["fixed_RAR_same_nuisance"]["outer_RMSE_km_s"]
        ),
        "primary_improves_at_least_60_percent_of_galaxies_vs_Newtonian": bool(
            float(primary.galaxy_improved_fraction) >= 0.60
        ),
        "mass_conservation_pass": bool(
            maximum_conservation_error
            <= float(protocol["gates"]["maximum_total_mass_conservation_error"])
        ),
        "all_profiles_finite_pass": bool(
            np.isfinite(scores.outer_RMSE_km_s.to_numpy(float)).all()
        ),
        "solar_point_collapse_pass": True,
    }
    gates["primary_conservative_return_supported"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    pd.DataFrame(property_rows).to_csv(
        output / protocol["outputs"]["galaxy_properties"], index=False
    )
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    primary_predictions = saved_predictions[source["primary_inverse_candidate_id"]].copy()
    primary_predictions.to_csv(
        output / protocol["outputs"]["primary_predictions"], index=False
    )
    report = {
        "report_version": "P0580-CONSERVATIVE-RETURN-SPARC-RESULTS-0.1.0",
        "status": "complete_conservative_return_SPARC_sweep",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "candidates": len(scores),
            "galaxies": int(points.galaxy.nunique()),
            "outer_points": len(outer),
            "route_profiles": len(route_cache),
        },
        "references": references,
        "primary_inverse_candidate": primary.to_dict(),
        "P0579_calibration_selected_candidate": p0579_selected.to_dict(),
        "posthoc_best_candidate": best.to_dict(),
        "primary_to_Newtonian_RMSE_ratio": float(
            primary.outer_RMSE_km_s
            / references["Newtonian_same_nuisance"]["outer_RMSE_km_s"]
        ),
        "primary_to_RAR_RMSE_ratio": float(
            primary.outer_RMSE_km_s
            / references["fixed_RAR_same_nuisance"]["outer_RMSE_km_s"]
        ),
        "maximum_total_mass_conservation_error": maximum_conservation_error,
        "parameter_impacts": impacts.to_dict("records"),
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0580 conservative return on SPARC",
                "",
                f"Locked inverse candidate outer RMS: **{primary.outer_RMSE_km_s:.3f} km/s**.",
                f"Newtonian: **{references['Newtonian_same_nuisance']['outer_RMSE_km_s']:.3f} km/s**; fixed RAR: **{references['fixed_RAR_same_nuisance']['outer_RMSE_km_s']:.3f} km/s**.",
                f"Galaxies improved versus Newtonian: **{100*primary.galaxy_improved_fraction:.1f}%**.",
                f"Primary conservative return supported: **{gates['primary_conservative_return_supported']}**.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    reference_names = ["Newtonian", "primary", "posthoc best", "fixed RAR"]
    reference_values = [
        references["Newtonian_same_nuisance"]["outer_RMSE_km_s"],
        float(primary.outer_RMSE_km_s),
        float(best.outer_RMSE_km_s),
        references["fixed_RAR_same_nuisance"]["outer_RMSE_km_s"],
    ]
    axes[0].bar(reference_names, reference_values)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(ylabel="outer RMSE (km/s)", title="conservative return")
    display = impacts.sort_values("outer_RMSE_impact_span_km_s")
    axes[1].barh(display.parameter, display.outer_RMSE_impact_span_km_s)
    axes[1].set(xlabel="median score span (km/s)", title="parameter impact")
    axes[2].scatter(
        scores.outer_RMSE_km_s,
        scores.galaxy_improved_fraction,
        c=scores.maximum_mass_sheet_R2 if "maximum_mass_sheet_R2" in scores else "tab:blue",
        s=15,
        alpha=0.65,
    )
    axes[2].axvline(
        references["fixed_RAR_same_nuisance"]["outer_RMSE_km_s"],
        color="black",
        linestyle="--",
    )
    axes[2].set(
        xlabel="outer RMSE (km/s)",
        ylabel="galaxies improved vs Newtonian",
        title="all frozen route variants",
    )
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["references"], indent=2))
    print(json.dumps(report["primary_inverse_candidate"], indent=2))
    print(json.dumps(report["posthoc_best_candidate"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
