from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_tidal_misalignment import (
    bounded_misalignment_gradients,
    bounded_misalignment_potential,
    spectral_tidal_misalignment,
    tidal_commutator,
)
from voidscreen.sigma_triaxial_memory import (
    centered_axis,
    gaussian_mixture_density,
    high_acceleration_screen,
    integrated_response,
    symmetric_trace_free,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT.parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def random_rotation(generator: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def directional_error(
    local: np.ndarray,
    memory: np.ndarray,
    local_direction: np.ndarray,
    memory_direction: np.ndarray,
) -> float:
    gradient_local, gradient_memory = bounded_misalignment_gradients(local, memory)
    scale = max(1.0, np.linalg.norm(local), np.linalg.norm(memory))
    step = 1e-6 * scale
    finite = float(
        (
            bounded_misalignment_potential(
                local + step * local_direction,
                memory + step * memory_direction,
            )
            - bounded_misalignment_potential(
                local - step * local_direction,
                memory - step * memory_direction,
            )
        )
        / (2.0 * step)
    )
    analytic = float(
        np.sum(gradient_local * local_direction) + np.sum(gradient_memory * memory_direction)
    )
    return abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-8)


def algebra_audit() -> dict[str, float]:
    generator = np.random.default_rng(20260803)
    minimum_potential = math.inf
    maximum_potential = -math.inf
    maximum_rotation_error = 0.0
    maximum_gradient_error = 0.0
    maximum_trace_residual = 0.0
    for _ in range(256):
        local = symmetric_trace_free(generator.normal(size=(3, 3))) * float(
            10.0 ** generator.uniform(-1.0, 1.0)
        )
        memory = symmetric_trace_free(generator.normal(size=(3, 3))) * float(
            10.0 ** generator.uniform(-1.0, 1.0)
        )
        value = float(bounded_misalignment_potential(local, memory))
        minimum_potential = min(minimum_potential, value)
        maximum_potential = max(maximum_potential, value)
        rotation = random_rotation(generator)
        rotated = float(
            bounded_misalignment_potential(
                rotation @ local @ rotation.T,
                rotation @ memory @ rotation.T,
            )
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            abs(value - rotated) / max(abs(value), 1e-14),
        )
        local_direction = symmetric_trace_free(generator.normal(size=(3, 3)))
        memory_direction = symmetric_trace_free(generator.normal(size=(3, 3)))
        normalization = math.sqrt(float(np.sum(local_direction**2) + np.sum(memory_direction**2)))
        local_direction /= normalization
        memory_direction /= normalization
        maximum_gradient_error = max(
            maximum_gradient_error,
            directional_error(local, memory, local_direction, memory_direction),
        )
        gradient_local, gradient_memory = bounded_misalignment_gradients(local, memory)
        maximum_trace_residual = max(
            maximum_trace_residual,
            abs(float(np.trace(local))),
            abs(float(np.trace(memory))),
            abs(float(np.trace(gradient_local))),
            abs(float(np.trace(gradient_memory))),
        )

    rotation = random_rotation(generator)
    local_diagonal = np.diag([1.0, 0.0, -1.0])
    memory_diagonal = np.diag([0.2, 0.5, -0.7])
    commuting_local = rotation @ local_diagonal @ rotation.T
    commuting_memory = rotation @ memory_diagonal @ rotation.T
    commuting = abs(float(bounded_misalignment_potential(commuting_local, commuting_memory)))
    angle = 0.4
    tilted = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    noncommuting_memory = tilted @ memory_diagonal @ tilted.T
    noncommuting = float(bounded_misalignment_potential(local_diagonal, noncommuting_memory))
    weak_scale = 1e-3
    commutator = tidal_commutator(local_diagonal, noncommuting_memory)
    leading = weak_scale**4 * float(np.sum(commutator**2)) / 2.0
    exact = float(
        bounded_misalignment_potential(
            weak_scale * local_diagonal,
            weak_scale * noncommuting_memory,
        )
    )
    quartic_error = abs(exact / leading - 1.0)
    return {
        "minimum_random_potential": minimum_potential,
        "maximum_random_potential": maximum_potential,
        "maximum_rotation_invariance_relative_error": maximum_rotation_error,
        "maximum_directional_gradient_relative_error": maximum_gradient_error,
        "maximum_trace_residual": maximum_trace_residual,
        "commuting_absolute_potential": commuting,
        "fixed_noncommuting_potential": noncommuting,
        "quartic_onset_relative_error": quartic_error,
        "screen_at_g_over_a_sigma_1e5": float(high_acceleration_screen(1e5)),
    }


def run_system(
    source: dict,
    audit: dict,
    *,
    system: str,
    mass: float,
    points: int,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    fixture = source["dimensionless_fixture"]
    axis = centered_axis(points, float(fixture["box_half_width_L_sigma"]))
    density = gaussian_mixture_density(
        axis,
        fixture[f"{system}_components"],
        total_mass=mass,
    )
    spacing = float(axis[1] - axis[0])
    field = spectral_tidal_misalignment(
        density,
        spacing=spacing,
        gravitational_constant=float(fixture["G"]),
        a_sigma=float(fixture["a_sigma"]),
        memory_length=float(fixture["L_sigma"]),
    )
    response = integrated_response(
        field.bounded_potential,
        axis,
        analysis_half_width=float(audit["dimensionless_fixture"]["analysis_half_width_L_sigma"]),
    )
    local_trace = float(np.max(np.abs(np.trace(field.local_tide, axis1=-2, axis2=-1))))
    memory_trace = float(np.max(np.abs(np.trace(field.memory_tide, axis1=-2, axis2=-1))))
    center = points // 2
    return (
        {
            "grid_points": points,
            "system": system,
            "total_mass": mass,
            "spacing_L_sigma": spacing,
            "integrated_response": response,
            "maximum_potential": float(np.max(field.bounded_potential)),
            "mean_potential": float(np.mean(field.bounded_potential)),
            "mean_screen": float(np.mean(field.base.screen)),
            "minimum_screen": float(np.min(field.base.screen)),
            "maximum_g_over_a_sigma": float(
                np.max(np.linalg.norm(field.base.acceleration, axis=-1)) / float(fixture["a_sigma"])
            ),
            "maximum_trace_residual": max(local_trace, memory_trace),
        },
        {
            "axis": axis,
            "potential_slice": field.bounded_potential[:, :, center],
        },
    )


def plot_results(
    output: Path,
    ratios: pd.DataFrame,
    primary_slices: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    acceleration = np.logspace(-3, 7, 400)
    axes[0, 0].loglog(acceleration, high_acceleration_screen(acceleration), color="#2a6fbb")
    axes[0, 0].axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_xlabel(r"$g/a_\Sigma$")
    axes[0, 0].set_ylabel(r"$\mathcal{S}$")
    axes[0, 0].set_title("Local high-field screen")
    axes[0, 0].grid(alpha=0.25)

    local = np.diag([1.0, 0.0, -1.0])
    memory = np.diag([0.2, 0.5, -0.7])
    angles = np.linspace(0.0, np.pi / 2.0, 300)
    values = []
    for angle in angles:
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        values.append(bounded_misalignment_potential(local, rotation @ memory @ rotation.T))
    axes[0, 1].plot(np.degrees(angles), values, color="#8a3ffc")
    axes[0, 1].set_xlabel("tidal eigenframe rotation (degrees)")
    axes[0, 1].set_ylabel(r"$\mathcal{V}_{\rm mis}$")
    axes[0, 1].set_title("Commuting null and orientation activation")
    axes[0, 1].grid(alpha=0.25)

    for points, group in ratios.groupby("grid_points"):
        axes[1, 0].plot(
            group.total_mass,
            group.cluster_to_galaxy_response_ratio,
            marker="o",
            label=f"{points}$^3$ grid",
        )
    axes[1, 0].axhline(10.0, color="black", linewidth=0.8, linestyle="--", label="primary gate")
    axes[1, 0].axhline(2.0, color="gray", linewidth=0.8, linestyle=":", label="per-mass gate")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("equal total mass normalization")
    axes[1, 0].set_ylabel("distributed / compact response")
    axes[1, 0].set_title("Universal morphology separation")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    galaxy = primary_slices["galaxy"]
    cluster = primary_slices["cluster"]
    maximum = max(
        float(np.max(galaxy["potential_slice"])),
        float(np.max(cluster["potential_slice"])),
    )
    floor = max(np.finfo(float).tiny, 1e-8 * maximum)
    difference = np.log10(cluster["potential_slice"] + floor) - np.log10(
        galaxy["potential_slice"] + floor
    )
    limit = max(1.0, float(np.nanpercentile(np.abs(difference), 98)))
    extent = [
        float(cluster["axis"][0]),
        float(cluster["axis"][-1]),
        float(cluster["axis"][0]),
        float(cluster["axis"][-1]),
    ]
    image = axes[1, 1].imshow(
        difference.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[1, 1].set_xlabel(r"$x/L_\Sigma$")
    axes[1, 1].set_ylabel(r"$y/L_\Sigma$")
    axes[1, 1].set_title("log response: distributed minus compact")
    figure.colorbar(image, ax=axes[1, 1], label="dex")
    figure.suptitle("Sigma v3E preregistered tidal-misalignment audit", fontsize=14)
    figure.savefig(output / "tidal_misalignment_audit.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Sigma v3E structural audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3e_tidal_misalignment_action_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3e_tidal_misalignment_action_audit",
    )
    args = parser.parse_args()
    audit = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = ROOT / audit["dimensionless_fixture"]["source_config"]
    if sha256(source_path) != audit["dimensionless_fixture"]["source_config_sha256"]:
        raise RuntimeError("source fixture configuration does not match its frozen hash")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    algebra = algebra_audit()

    settings = audit["dimensionless_fixture"]
    primary_points = int(settings["primary_grid_points"])
    resolution_points = int(settings["resolution_grid_points"])
    rows = []
    primary_slices: dict[str, dict[str, np.ndarray]] = {}
    for points in (primary_points, resolution_points):
        for mass in map(float, settings["mass_normalizations"]):
            for system in ("galaxy", "cluster"):
                row, slices = run_system(
                    source,
                    audit,
                    system=system,
                    mass=mass,
                    points=points,
                )
                rows.append(row)
                if points == primary_points and mass == 1.0:
                    primary_slices[system] = slices
    responses = pd.DataFrame.from_records(rows)
    responses.to_csv(args.output / "fixture_responses.csv", index=False)
    ratio_rows = []
    for (points, mass), group in responses.groupby(["grid_points", "total_mass"]):
        values = group.set_index("system").integrated_response
        ratio_rows.append(
            {
                "grid_points": int(points),
                "total_mass": float(mass),
                "cluster_to_galaxy_response_ratio": float(values["cluster"] / values["galaxy"]),
            }
        )
    ratios = pd.DataFrame.from_records(ratio_rows).sort_values(["grid_points", "total_mass"])
    ratios.to_csv(args.output / "response_ratios.csv", index=False)
    primary = ratios.loc[ratios.grid_points == primary_points]
    resolution = ratios.loc[ratios.grid_points == resolution_points]
    primary_median = float(primary.cluster_to_galaxy_response_ratio.median())
    resolution_median = float(resolution.cluster_to_galaxy_response_ratio.median())
    resolution_change = abs(resolution_median / primary_median - 1.0)
    maximum_field_trace = float(responses.maximum_trace_residual.max())

    thresholds = audit["preregistered_gates"]
    gates = {
        "trace_free": bool(
            max(algebra["maximum_trace_residual"], maximum_field_trace)
            <= float(thresholds["maximum_trace_residual"])
        ),
        "rotation_invariance": bool(
            algebra["maximum_rotation_invariance_relative_error"]
            <= float(thresholds["maximum_rotation_invariance_relative_error"])
        ),
        "analytic_gradient": bool(
            algebra["maximum_directional_gradient_relative_error"]
            <= float(thresholds["maximum_directional_gradient_relative_error"])
        ),
        "bounded_random_potential": bool(
            algebra["minimum_random_potential"] >= float(thresholds["minimum_random_potential"])
            and algebra["maximum_random_potential"] <= float(thresholds["maximum_random_potential"])
        ),
        "commuting_null": bool(
            algebra["commuting_absolute_potential"]
            <= float(thresholds["maximum_commuting_absolute_potential"])
        ),
        "noncommuting_activation": bool(
            algebra["fixed_noncommuting_potential"]
            >= float(thresholds["minimum_noncommuting_potential"])
        ),
        "quartic_onset": bool(
            algebra["quartic_onset_relative_error"]
            <= float(thresholds["maximum_quartic_onset_relative_error"])
        ),
        "solar_screen": bool(
            algebra["screen_at_g_over_a_sigma_1e5"]
            <= float(thresholds["maximum_screen_at_g_over_a_sigma_1e5"])
        ),
        "primary_morphology_separation": bool(
            primary_median
            >= float(thresholds["minimum_primary_median_cluster_to_galaxy_response_ratio"])
        ),
        "each_mass_morphology_separation": bool(
            (
                primary.cluster_to_galaxy_response_ratio
                >= float(thresholds["minimum_each_mass_cluster_to_galaxy_response_ratio"])
            ).all()
        ),
        "resolution_stability": bool(
            resolution_change
            <= float(thresholds["maximum_resolution_fractional_change_in_primary_ratio"])
        ),
    }
    all_pass = bool(all(gates.values()))
    report = {
        "protocol_id": audit["protocol_id"],
        "config_sha256": sha256(args.config),
        "preregistered_git_revision": git_revision(),
        "source_fixture_sha256": audit["dimensionless_fixture"]["source_config_sha256"],
        "evidence_status": audit["evidence_status"],
        "algebra": algebra,
        "maximum_field_trace_residual": maximum_field_trace,
        "primary_grid_points": primary_points,
        "resolution_grid_points": resolution_points,
        "primary_median_cluster_to_galaxy_response_ratio": primary_median,
        "resolution_median_cluster_to_galaxy_response_ratio": resolution_median,
        "resolution_fractional_change_in_primary_ratio": resolution_change,
        "minimum_primary_mass_ratio": float(primary.cluster_to_galaxy_response_ratio.min()),
        "maximum_primary_mass_ratio": float(primary.cluster_to_galaxy_response_ratio.max()),
        "gates": gates,
        "all_preregistered_gates_pass": all_pass,
        "decision": (
            "advance_to_covariant_causal_action_completion_before_empirical_fit"
            if all_pass
            else "retire_v3e_misalignment_as_frozen_structural_mechanism"
        ),
        "raw_holdout_opened": False,
        "raw_holdout_failure_count": 0,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(args.output, ratios, primary_slices)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
