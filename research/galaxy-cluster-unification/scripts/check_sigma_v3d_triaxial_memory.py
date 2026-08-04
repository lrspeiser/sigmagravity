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

from voidscreen.sigma_triaxial_memory import (
    axisymmetric_tidal_tensor,
    bounded_triaxial_gradient,
    bounded_triaxial_potential,
    centered_axis,
    gaussian_mixture_density,
    high_acceleration_screen,
    integrated_response,
    spectral_tidal_memory,
    symmetric_trace_free,
    triaxial_invariants,
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
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def algebra_audit() -> dict[str, float]:
    generator = np.random.default_rng(20260803)
    minimum_discriminant = math.inf
    maximum_rotation_error = 0.0
    maximum_gradient_error = 0.0
    maximum_trace_residual = 0.0
    for _ in range(256):
        scale = float(10.0 ** generator.uniform(-1.0, 1.0))
        matrix = symmetric_trace_free(generator.normal(size=(3, 3))) * scale
        rotation = random_rotation(generator)
        rotated = rotation @ matrix @ rotation.T
        original_potential = float(bounded_triaxial_potential(matrix))
        rotated_potential = float(bounded_triaxial_potential(rotated))
        maximum_rotation_error = max(
            maximum_rotation_error,
            abs(rotated_potential - original_potential) / max(abs(original_potential), 1e-14),
        )
        direction = symmetric_trace_free(generator.normal(size=(3, 3)))
        direction /= np.linalg.norm(direction)
        step = 1e-6 * max(1.0, np.linalg.norm(matrix))
        finite = float(
            (
                bounded_triaxial_potential(matrix + step * direction)
                - bounded_triaxial_potential(matrix - step * direction)
            )
            / (2.0 * step)
        )
        analytic = float(np.sum(bounded_triaxial_gradient(matrix) * direction))
        maximum_gradient_error = max(
            maximum_gradient_error,
            abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-8),
        )
        _, _, discriminant = triaxial_invariants(matrix)
        minimum_discriminant = min(minimum_discriminant, float(discriminant))
        maximum_trace_residual = max(
            maximum_trace_residual,
            abs(float(np.trace(matrix))),
            abs(float(np.trace(bounded_triaxial_gradient(matrix)))),
        )

    maximum_axisymmetric = 0.0
    for amplitude in np.geomspace(1e-3, 1e3, 31):
        rotation = random_rotation(generator)
        matrix = rotation @ axisymmetric_tidal_tensor([1.0, 0.0, 0.0], amplitude) @ rotation.T
        maximum_axisymmetric = max(
            maximum_axisymmetric, abs(float(bounded_triaxial_potential(matrix)))
        )

    first = axisymmetric_tidal_tensor([1.0, 0.0, 0.0], 1.0)
    second = axisymmetric_tidal_tensor([1.0, 1.0, 0.3], 0.7)
    overlap = float(bounded_triaxial_potential(first + second))
    return {
        "minimum_random_discriminant": minimum_discriminant,
        "maximum_rotation_invariance_relative_error": maximum_rotation_error,
        "maximum_directional_gradient_relative_error": maximum_gradient_error,
        "maximum_trace_residual": maximum_trace_residual,
        "maximum_axisymmetric_absolute_potential": maximum_axisymmetric,
        "fixed_overlap_potential": overlap,
        "screen_at_g_over_a_sigma_1e5": float(high_acceleration_screen(1e5)),
    }


def run_system(
    config: dict,
    *,
    system: str,
    mass: float,
    points: int,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    fixture = config["dimensionless_fixture"]
    axis = centered_axis(points, float(fixture["box_half_width_L_sigma"]))
    density = gaussian_mixture_density(
        axis,
        fixture[f"{system}_components"],
        total_mass=mass,
    )
    spacing = float(axis[1] - axis[0])
    field = spectral_tidal_memory(
        density,
        spacing=spacing,
        gravitational_constant=float(fixture["G"]),
        a_sigma=float(fixture["a_sigma"]),
        memory_length=float(fixture["L_sigma"]),
    )
    response = integrated_response(
        field.bounded_potential,
        axis,
        analysis_half_width=float(fixture["analysis_half_width_L_sigma"]),
    )
    trace_residual = float(np.max(np.abs(np.trace(field.memory, axis1=-2, axis2=-1))))
    center = points // 2
    return (
        {
            "grid_points": points,
            "system": system,
            "total_mass": mass,
            "spacing_L_sigma": spacing,
            "integrated_response": response,
            "maximum_potential": float(np.max(field.bounded_potential)),
            "mean_screen": float(np.mean(field.screen)),
            "minimum_screen": float(np.min(field.screen)),
            "maximum_g_over_a_sigma": float(
                np.max(np.linalg.norm(field.acceleration, axis=-1)) / float(fixture["a_sigma"])
            ),
            "maximum_memory_trace_residual": trace_residual,
        },
        {
            "axis": axis,
            "potential_slice": field.bounded_potential[:, :, center],
            "screen_slice": field.screen[:, :, center],
        },
    )


def plot_results(
    output: Path,
    rows: pd.DataFrame,
    ratios: pd.DataFrame,
    primary_slices: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    acceleration = np.logspace(-3, 7, 400)
    axes[0, 0].loglog(acceleration, high_acceleration_screen(acceleration), color="#2a6fbb")
    axes[0, 0].axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_xlabel(r"$g/a_\Sigma$")
    axes[0, 0].set_ylabel(r"$\mathcal{S}$")
    axes[0, 0].set_title("Universal high-field screen")
    axes[0, 0].grid(alpha=0.25)

    asymmetry = np.linspace(0.0, 2.0, 300)
    tensors = np.array([np.diag([2.0, -1.0 + value, -1.0 - value]) for value in asymmetry])
    axes[0, 1].plot(asymmetry, bounded_triaxial_potential(tensors), color="#8a3ffc")
    axes[0, 1].set_xlabel("eigenvalue splitting")
    axes[0, 1].set_ylabel(r"$\mathcal{V}$")
    axes[0, 1].set_title("Axisymmetric null and triaxial activation")
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
    floor = max(
        np.finfo(float).tiny,
        1e-8
        * max(float(np.max(galaxy["potential_slice"])), float(np.max(cluster["potential_slice"]))),
    )
    difference = np.log10(cluster["potential_slice"] + floor) - np.log10(
        galaxy["potential_slice"] + floor
    )
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
        vmin=-max(1.0, float(np.nanpercentile(np.abs(difference), 98))),
        vmax=max(1.0, float(np.nanpercentile(np.abs(difference), 98))),
    )
    axes[1, 1].set_xlabel(r"$x/L_\Sigma$")
    axes[1, 1].set_ylabel(r"$y/L_\Sigma$")
    axes[1, 1].set_title("log response: distributed minus compact")
    figure.colorbar(image, ax=axes[1, 1], label="dex")
    figure.suptitle("Sigma v3D preregistered triaxial-memory audit", fontsize=14)
    figure.savefig(output / "triaxial_memory_audit.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Sigma v3D structural audit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v3d_triaxial_memory_action_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v3d_triaxial_memory_action_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    algebra = algebra_audit()

    fixture = config["dimensionless_fixture"]
    primary_points = int(fixture["primary_grid_points"])
    resolution_points = int(fixture["resolution_grid_points"])
    rows: list[dict[str, float | int | str]] = []
    primary_slices: dict[str, dict[str, np.ndarray]] = {}
    for points in (primary_points, resolution_points):
        for mass in map(float, fixture["mass_normalizations"]):
            for system in ("galaxy", "cluster"):
                row, slices = run_system(
                    config,
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
    primary_ratios = ratios.loc[ratios.grid_points == primary_points]
    resolution_ratios = ratios.loc[ratios.grid_points == resolution_points]
    primary_median_ratio = float(primary_ratios.cluster_to_galaxy_response_ratio.median())
    resolution_median_ratio = float(resolution_ratios.cluster_to_galaxy_response_ratio.median())
    resolution_fractional_change = abs(resolution_median_ratio / primary_median_ratio - 1.0)
    maximum_field_trace = float(responses.maximum_memory_trace_residual.max())

    thresholds = config["preregistered_gates"]
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
        "nonnegative_discriminant": bool(
            algebra["minimum_random_discriminant"]
            >= float(thresholds["minimum_random_discriminant"])
        ),
        "axisymmetric_null": bool(
            algebra["maximum_axisymmetric_absolute_potential"]
            <= float(thresholds["maximum_axisymmetric_absolute_potential"])
        ),
        "overlap_activation": bool(
            algebra["fixed_overlap_potential"] >= float(thresholds["minimum_overlap_potential"])
        ),
        "solar_screen": bool(
            algebra["screen_at_g_over_a_sigma_1e5"]
            <= float(thresholds["maximum_screen_at_g_over_a_sigma_1e5"])
        ),
        "primary_morphology_separation": bool(
            primary_median_ratio
            >= float(thresholds["minimum_primary_median_cluster_to_galaxy_response_ratio"])
        ),
        "each_mass_morphology_separation": bool(
            (
                primary_ratios.cluster_to_galaxy_response_ratio
                >= float(thresholds["minimum_each_mass_cluster_to_galaxy_response_ratio"])
            ).all()
        ),
        "resolution_stability": bool(
            resolution_fractional_change
            <= float(thresholds["maximum_resolution_fractional_change_in_primary_ratio"])
        ),
    }
    all_pass = bool(all(gates.values()))
    report = {
        "protocol_id": config["protocol_id"],
        "config_sha256": sha256(args.config),
        "preregistered_git_revision": git_revision(),
        "evidence_status": config["evidence_status"],
        "algebra": algebra,
        "maximum_memory_trace_residual": maximum_field_trace,
        "primary_grid_points": primary_points,
        "resolution_grid_points": resolution_points,
        "primary_median_cluster_to_galaxy_response_ratio": primary_median_ratio,
        "resolution_median_cluster_to_galaxy_response_ratio": resolution_median_ratio,
        "resolution_fractional_change_in_primary_ratio": resolution_fractional_change,
        "minimum_primary_mass_ratio": float(primary_ratios.cluster_to_galaxy_response_ratio.min()),
        "maximum_primary_mass_ratio": float(primary_ratios.cluster_to_galaxy_response_ratio.max()),
        "gates": gates,
        "all_preregistered_gates_pass": all_pass,
        "decision": (
            "advance_to_causal_action_completion_before_empirical_fit"
            if all_pass
            else "retire_v3d_discriminant_as_frozen_structural_mechanism"
        ),
        "raw_holdout_opened": False,
        "raw_holdout_failure_count": 0,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(args.output, responses, ratios, primary_slices)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
