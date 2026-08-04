"""Spherical local-state closure for first-gradient Sigma completions.

For a regular, shift-symmetric quasistatic action whose extra local density is
``F(Y,Z,U)``, spherical symmetry reduces every conserved field equation to an
algebraic constitutive flux evaluated at the local baryonic surface field
``G M_b(<r)/r^2``.  If the constitutive map is single-valued, two systems with
the same baryonic acceleration must have the same local field state and the
same force enhancement.  This module compares that prediction with already
spent SPARC and CLASH-development products; it does not open a new holdout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

Array = np.ndarray
KPC_METERS = 3.085677581491367e19


def _finite_vector(values: Array, *, name: str) -> Array:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _quantiles(values: Array) -> dict[str, float]:
    vector = _finite_vector(values, name="values")
    probabilities = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    labels = ("minimum", "p05", "p25", "median", "p75", "p95", "maximum")
    return {
        label: float(value)
        for label, value in zip(
            labels,
            np.quantile(vector, probabilities),
            strict=True,
        )
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of an existing file."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"missing input file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scale_equivalent_spherical_pair(
    *,
    first_mass: float = 1.0,
    first_radius: float = 1.0,
    mass_ratio: float = 100.0,
) -> dict[str, float]:
    """Construct two spheres with identical ``M/r^2`` but different scale.

    The second radius is ``sqrt(mass_ratio)`` times the first.  Therefore the
    local baryonic acceleration agrees exactly, while potential depth ``M/r``
    and tidal/density scale ``M/r^3`` differ.  Units cancel from every ratio.
    """

    mass = float(first_mass)
    radius = float(first_radius)
    ratio = float(mass_ratio)
    if not all(np.isfinite(value) and value > 0.0 for value in (mass, radius, ratio)):
        raise ValueError("masses, radii, and mass_ratio must be finite and positive")
    second_mass = mass * ratio
    second_radius = radius * np.sqrt(ratio)
    first_surface = mass / radius**2
    second_surface = second_mass / second_radius**2
    first_potential = mass / radius
    second_potential = second_mass / second_radius
    first_tidal = mass / radius**3
    second_tidal = second_mass / second_radius**3
    return {
        "first_mass": mass,
        "first_radius": radius,
        "second_mass": second_mass,
        "second_radius": second_radius,
        "surface_field_ratio_second_to_first": second_surface / first_surface,
        "potential_depth_ratio_second_to_first": second_potential / first_potential,
        "tidal_or_mean_density_ratio_second_to_first": second_tidal / first_tidal,
    }


def local_state_overlap_metrics(
    *,
    sparc_log_gbar: Array,
    sparc_log_enhancement: Array,
    cluster_log_gbar: Array,
    cluster_log_enhancement: Array,
    nearest_neighbors: int = 10,
) -> dict[str, object]:
    """Compare required enhancements at matched baryonic acceleration.

    Every cluster point is matched both to its nearest SPARC point and to the
    median of its ``nearest_neighbors`` SPARC points in ``log10(g_bar)``.  This
    is a mechanism diagnostic, not a likelihood: it ignores correlated errors,
    cluster deprojection covariance, and object-level resampling.
    """

    sparc_g = _finite_vector(sparc_log_gbar, name="sparc_log_gbar")
    sparc_e = _finite_vector(
        sparc_log_enhancement,
        name="sparc_log_enhancement",
    )
    cluster_g = _finite_vector(cluster_log_gbar, name="cluster_log_gbar")
    cluster_e = _finite_vector(
        cluster_log_enhancement,
        name="cluster_log_enhancement",
    )
    if sparc_g.shape != sparc_e.shape or cluster_g.shape != cluster_e.shape:
        raise ValueError("each acceleration vector must match its enhancement vector")
    neighbors = int(nearest_neighbors)
    if not 1 <= neighbors <= sparc_g.size:
        raise ValueError("nearest_neighbors must lie within the SPARC sample size")
    order = np.argsort(sparc_g)
    sorted_g = sparc_g[order]
    sorted_e = sparc_e[order]
    nearest_distance = np.empty(cluster_g.size)
    nearest_gap = np.empty(cluster_g.size)
    neighbor_mean_distance = np.empty(cluster_g.size)
    neighbor_median_gap = np.empty(cluster_g.size)
    for index, (gbar, enhancement) in enumerate(
        zip(cluster_g, cluster_e, strict=True)
    ):
        insertion = int(np.searchsorted(sorted_g, gbar))
        candidates = {
            max(0, min(sorted_g.size - 1, insertion - 1)),
            max(0, min(sorted_g.size - 1, insertion)),
        }
        nearest = min(candidates, key=lambda item: abs(sorted_g[item] - gbar))
        nearest_distance[index] = abs(sorted_g[nearest] - gbar)
        nearest_gap[index] = enhancement - sorted_e[nearest]
        neighbor_indices = np.argpartition(np.abs(sorted_g - gbar), neighbors - 1)[
            :neighbors
        ]
        neighbor_mean_distance[index] = float(
            np.mean(np.abs(sorted_g[neighbor_indices] - gbar))
        )
        neighbor_median_gap[index] = enhancement - float(
            np.median(sorted_e[neighbor_indices])
        )
    overlap_lower = max(float(np.min(sparc_g)), float(np.min(cluster_g)))
    overlap_upper = min(float(np.max(sparc_g)), float(np.max(cluster_g)))
    cluster_in_overlap = (cluster_g >= overlap_lower) & (cluster_g <= overlap_upper)
    return {
        "SPARC_points": int(sparc_g.size),
        "cluster_points": int(cluster_g.size),
        "SPARC_log_gbar": _quantiles(sparc_g),
        "cluster_log_gbar": _quantiles(cluster_g),
        "SPARC_log_enhancement": _quantiles(sparc_e),
        "cluster_log_enhancement": _quantiles(cluster_e),
        "overlap_interval_log_gbar": [overlap_lower, overlap_upper],
        "cluster_fraction_inside_SPARC_range": float(np.mean(cluster_in_overlap)),
        "nearest_log_gbar_distance_dex": _quantiles(nearest_distance),
        "nearest_required_enhancement_gap_dex": _quantiles(nearest_gap),
        "nearest_required_enhancement_gap_factor_at_median": float(
            10.0 ** np.median(nearest_gap)
        ),
        "fraction_cluster_gap_above_0p2_dex": float(np.mean(nearest_gap > 0.2)),
        "fraction_cluster_gap_positive": float(np.mean(nearest_gap > 0.0)),
        "neighbor_count": neighbors,
        "neighbor_mean_log_gbar_distance_dex": _quantiles(neighbor_mean_distance),
        "neighbor_median_required_enhancement_gap_dex": _quantiles(
            neighbor_median_gap
        ),
        "neighbor_median_gap_factor_at_median": float(
            10.0 ** np.median(neighbor_median_gap)
        ),
        "local_first_gradient_conflict_gate": bool(
            np.mean(cluster_in_overlap) >= 0.95
            and np.median(nearest_distance) <= 0.01
            and np.median(neighbor_median_gap) >= 0.3
            and np.mean(nearest_gap > 0.2) >= 0.9
        ),
    }


def load_existing_development_states(
    *,
    sparc_predictions_path: Path,
    cluster_sample_path: Path,
) -> dict[str, Array | int | str]:
    """Load the exact spent-development rows used by the closure audit."""

    sparc_path = Path(sparc_predictions_path)
    cluster_path = Path(cluster_sample_path)
    if not sparc_path.is_file() or not cluster_path.is_file():
        raise ValueError("both existing development CSV inputs must exist")
    sparc = pd.read_csv(
        sparc_path,
        usecols=[
            "model",
            "scenario",
            "split",
            "g_bar_m_s2",
            "velocity_observed_adjusted_km_s",
            "radius_adjusted_kpc",
        ],
    )
    selected_sparc = sparc.loc[
        (sparc["model"] == "fixed_RAR")
        & (sparc["scenario"] == "invariant")
        & (sparc["split"] == "outer_holdout")
    ].copy()
    gbar = selected_sparc["g_bar_m_s2"].to_numpy(dtype=float)
    velocity = selected_sparc["velocity_observed_adjusted_km_s"].to_numpy(
        dtype=float
    )
    radius = selected_sparc["radius_adjusted_kpc"].to_numpy(dtype=float)
    gobs = (velocity * 1000.0) ** 2 / (radius * KPC_METERS)
    valid_sparc = (
        np.isfinite(gbar)
        & np.isfinite(gobs)
        & (gbar > 0.0)
        & (gobs > 0.0)
    )
    cluster = pd.read_csv(
        cluster_path,
        usecols=["domain", "log_gbar", "log_gobs"],
    )
    selected_cluster = cluster.loc[cluster["domain"] == "cluster"].copy()
    cluster_log_gbar = selected_cluster["log_gbar"].to_numpy(dtype=float)
    cluster_log_gobs = selected_cluster["log_gobs"].to_numpy(dtype=float)
    valid_cluster = np.isfinite(cluster_log_gbar) & np.isfinite(cluster_log_gobs)
    return {
        "sparc_log_gbar": np.log10(gbar[valid_sparc]),
        "sparc_log_enhancement": np.log10(gobs[valid_sparc] / gbar[valid_sparc]),
        "cluster_log_gbar": cluster_log_gbar[valid_cluster],
        "cluster_log_enhancement": (
            cluster_log_gobs[valid_cluster] - cluster_log_gbar[valid_cluster]
        ),
        "sparc_rows_selected": int(np.sum(valid_sparc)),
        "cluster_rows_selected": int(np.sum(valid_cluster)),
        "sparc_sha256": sha256_file(sparc_path),
        "cluster_sha256": sha256_file(cluster_path),
    }


def audit_v9b_local_state_closure(
    *,
    sparc_predictions_path: Path,
    cluster_sample_path: Path,
    nearest_neighbors: int = 10,
) -> dict[str, object]:
    """Run the preregistered local first-gradient mechanism closure."""

    states = load_existing_development_states(
        sparc_predictions_path=sparc_predictions_path,
        cluster_sample_path=cluster_sample_path,
    )
    overlap = local_state_overlap_metrics(
        sparc_log_gbar=np.asarray(states["sparc_log_gbar"]),
        sparc_log_enhancement=np.asarray(states["sparc_log_enhancement"]),
        cluster_log_gbar=np.asarray(states["cluster_log_gbar"]),
        cluster_log_enhancement=np.asarray(states["cluster_log_enhancement"]),
        nearest_neighbors=nearest_neighbors,
    )
    theorem_conditions = {
        "static_spherical_symmetry": True,
        "local_shift_symmetric_first_gradient_density_F_of_Y_Z_U": True,
        "regular_single_valued_constitutive_inverse": True,
        "universal_boundary_condition_no_object_charge": True,
        "one_physical_metric": True,
    }
    return {
        "input_hashes": {
            "SPARC_point_predictions_sha256": states["sparc_sha256"],
            "cluster_sample_sha256": states["cluster_sha256"],
        },
        "theorem_conditions": theorem_conditions,
        "spherical_flux_identity": (
            "r^2 P_a(J,S)=c_a G M_b(<r), so P_a(J,S)=c_a g_bar"
        ),
        "theorem_conclusion": (
            "A regular unique constitutive inverse maps (J,S) and the physical "
            "force enhancement to a universal function of local g_bar alone."
        ),
        "scale_equivalent_pair": scale_equivalent_spherical_pair(),
        "development_overlap": overlap,
        "closure_gate_passed": bool(overlap["local_first_gradient_conflict_gate"]),
        "decision": "close_regular_local_first_gradient_F_Y_Z_U_lane",
        "existing_spent_observational_products_accessed": True,
        "new_observational_product_accessed": False,
        "new_holdout_opened": False,
    }
