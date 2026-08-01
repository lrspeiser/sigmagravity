"""Posterior halo-location targets for descriptive baryon-to-halo backtracking."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from voidscreen.gravity_arc_tomography import sinkhorn_transport


_OBJECT_HEADER = re.compile(r"^O(?P<object>\d+)\s*:\s*(?P<field>.+?)\s*$")


def bayes_headers(path: Path) -> list[str]:
    """Return Lenstool ``bayes.dat`` column labels in file order."""
    headers: list[str] = []
    with Path(path).open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("#"):
                headers.append(line[1:].strip())
            elif line.strip():
                break
    if not headers:
        raise ValueError(f"no bayes.dat headers in {path}")
    return headers


def cluster_halo_columns(headers: list[str]) -> dict[int, dict[str, int]]:
    """Find cluster-scale posterior objects with position, core, and strength.

    External shear objects have no x/y/core/sigma quartet and are excluded.
    Member ``potfile`` parameters are not ``O#`` objects and are also excluded.
    """
    objects: dict[int, dict[str, int]] = {}
    aliases = {
        "x (arcsec)": "x",
        "y (arcsec)": "y",
        "emass": "emass",
        "theta (deg)": "theta",
        "rc (arcsec)": "rc",
        "sigma (km/s)": "sigma",
    }
    for index, header in enumerate(headers):
        match = _OBJECT_HEADER.match(header)
        if match is None:
            continue
        field = aliases.get(match.group("field"))
        if field is not None:
            objects.setdefault(int(match.group("object")), {})[field] = index
    required = {"x", "y", "rc", "sigma"}
    return {key: value for key, value in objects.items() if required <= set(value)}


def thin_bayes_chain(path: Path, sample_count: int) -> tuple[list[str], np.ndarray, int]:
    """Read deterministic, evenly spaced posterior rows without loading the chain."""
    path = Path(path)
    headers = bayes_headers(path)
    with path.open("r", encoding="ascii") as handle:
        rows = sum(1 for line in handle if line.strip() and not line.startswith("#"))
    if rows < 1:
        raise ValueError(f"no posterior samples in {path}")
    count = min(int(sample_count), rows)
    selected = np.unique(np.linspace(0, rows - 1, count, dtype=int))
    selected_set = set(int(value) for value in selected)
    values = []
    data_index = -1
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            data_index += 1
            if data_index in selected_set:
                row = np.fromstring(line, sep=" ")
                if len(row) != len(headers):
                    raise ValueError(
                        f"posterior row has {len(row)} columns, expected {len(headers)}"
                    )
                values.append(row)
    result = np.asarray(values, dtype=float)
    if len(result) != len(selected):
        raise RuntimeError("posterior thinning lost selected rows")
    return headers, result, rows


def component_samples(
    headers: list[str], samples: np.ndarray, angular_scale_kpc_per_arcsec: float
) -> dict[int, dict[str, np.ndarray]]:
    """Extract cluster-halo posterior columns and convert geometry to kpc."""
    output: dict[int, dict[str, np.ndarray]] = {}
    scale = float(angular_scale_kpc_per_arcsec)
    for object_id, columns in cluster_halo_columns(headers).items():
        count = len(samples)
        output[object_id] = {
            # Lenstool reports x positive west; the project maps use east.
            "x_kpc": -samples[:, columns["x"]] * scale,
            "y_kpc": samples[:, columns["y"]] * scale,
            "core_kpc": np.maximum(samples[:, columns["rc"]] * scale, 0.0),
            "sigma_km_s": np.maximum(samples[:, columns["sigma"]], 0.0),
            "emass": (
                samples[:, columns["emass"]]
                if "emass" in columns
                else np.zeros(count, dtype=float)
            ),
            "theta_deg": (
                samples[:, columns["theta"]]
                if "theta" in columns
                else np.zeros(count, dtype=float)
            ),
        }
    if not output:
        raise ValueError("no cluster-scale halo posterior objects found")
    return output


def posterior_component_destinations(
    components: dict[int, dict[str, np.ndarray]],
    axis_kpc: np.ndarray,
    *,
    width_mode: str,
    width_kpc: float,
    weight_mode: str,
    maximum_radius_kpc: float,
    minimum_relative_density: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Make a posterior occupancy target and retain component identities.

    Each posterior center is convolved with a circular Gaussian.  This is a
    location proxy, not an exact dPIE convergence reconstruction.
    """
    axis = np.asarray(axis_kpc, dtype=float)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    aperture = np.hypot(grid_x, grid_y) <= float(maximum_radius_kpc)
    maps: dict[int, np.ndarray] = {}
    component_strength: dict[int, float] = {}
    for object_id, values in components.items():
        image = np.zeros_like(grid_x)
        sigma2 = np.square(values["sigma_km_s"])
        sample_strength = sigma2 if weight_mode == "sigma2" else np.ones_like(sigma2)
        for x, y, core, strength in zip(
            values["x_kpc"],
            values["y_kpc"],
            values["core_kpc"],
            sample_strength,
            strict=True,
        ):
            if width_mode == "fixed":
                width = float(width_kpc)
            elif width_mode == "core_plus_floor":
                width = float(np.hypot(float(width_kpc), core))
            else:
                raise ValueError(f"unknown width mode {width_mode}")
            profile = np.exp(
                -0.5 * (np.square(grid_x - x) + np.square(grid_y - y)) / width**2
            )
            profile[~aperture] = 0.0
            total = float(np.sum(profile))
            if total > 0.0:
                image += float(strength) * profile / total
        image /= len(values["x_kpc"])
        maps[object_id] = image
        component_strength[object_id] = float(np.sum(image))

    positions: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    for object_id, image in maps.items():
        threshold = float(minimum_relative_density) * float(np.max(image))
        use = aperture & (image >= threshold) & (image > 0.0)
        positions.append(np.column_stack([grid_x[use], grid_y[use]]))
        weights.append(image[use])
        identities.append(np.full(int(np.sum(use)), object_id, dtype=int))
    destination_positions = np.vstack(positions)
    destination_weights = np.concatenate(weights)
    destination_weights /= np.sum(destination_weights)
    component_ids = np.concatenate(identities)
    return destination_positions, destination_weights, component_ids, maps


def coarsen_source_map(
    image: np.ndarray,
    axis_arcsec: np.ndarray,
    angular_scale_kpc_per_arcsec: float,
    *,
    factor: int,
    maximum_radius_kpc: float,
    retained_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a registered morphology map to a sparse source distribution."""
    data = np.maximum(np.asarray(image, dtype=float), 0.0)
    axis = np.asarray(axis_arcsec, dtype=float)
    size = (len(axis) // int(factor)) * int(factor)
    block = data[:size, :size].reshape(
        size // factor, factor, size // factor, factor
    ).sum(axis=(1, 3))
    coarse_axis = axis[:size].reshape(size // factor, factor).mean(axis=1)
    x, y = np.meshgrid(coarse_axis, coarse_axis, indexing="xy")
    scale = float(angular_scale_kpc_per_arcsec)
    positions = np.column_stack([x.ravel() * scale, y.ravel() * scale])
    weights = block.ravel()
    use = (weights > 0.0) & (
        np.linalg.norm(positions, axis=1) <= float(maximum_radius_kpc)
    )
    positions = positions[use]
    weights = weights[use]
    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    count = int(np.searchsorted(cumulative, float(retained_weight), side="left") + 1)
    chosen = order[:count]
    positions = positions[chosen]
    weights = weights[chosen]
    weights /= np.sum(weights)
    return positions, weights


def halo_assignment(
    plan: np.ndarray, component_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional source-to-component shares and component marginals."""
    ids = np.unique(component_ids)
    flows = np.column_stack(
        [np.sum(plan[:, component_ids == object_id], axis=1) for object_id in ids]
    )
    source_mass = np.sum(plan, axis=1)
    conditional = flows / np.maximum(source_mass[:, None], np.finfo(float).tiny)
    return conditional, np.sum(flows, axis=0)


def solve_capacity_transport(
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    destination_positions: np.ndarray,
    destination_weights: np.ndarray,
    *,
    capacity_multiplier: float,
    entropy_length_kpc: float,
    iterations: int = 1000,
    tolerance: float = 1.0e-9,
) -> tuple[np.ndarray, dict[str, float]]:
    """Supply every arrival while allowing some baryonic capacity to stay local.

    ``capacity_multiplier=1`` is balanced transport.  For a multiplier q>1,
    a dummy local sink receives fraction (q-1)/q of the normalized plan.  The
    real target receives 1/q; multiplying that sub-plan by q restores unit
    target mass and gives each source a capacity q*b_i.  The dummy has equal
    zero cost from every source, so it only represents unused capacity.
    """
    multiplier = float(capacity_multiplier)
    if multiplier < 1.0:
        raise ValueError("capacity multiplier must be at least one")
    source = np.asarray(source_weights, dtype=float)
    source /= np.sum(source)
    target = np.asarray(destination_weights, dtype=float)
    target /= np.sum(target)
    displacement = (
        np.asarray(destination_positions, dtype=float)[None, :, :]
        - np.asarray(source_positions, dtype=float)[:, None, :]
    )
    real_cost = np.sum(np.square(displacement), axis=2)
    if multiplier == 1.0:
        augmented_cost = real_cost
        augmented_target = target
    else:
        augmented_cost = np.column_stack([real_cost, np.zeros(len(source))])
        augmented_target = np.r_[target / multiplier, (multiplier - 1.0) / multiplier]
    augmented = sinkhorn_transport(
        source,
        augmented_target,
        augmented_cost,
        entropy=float(entropy_length_kpc) ** 2,
        iterations=int(iterations),
        tolerance=float(tolerance),
    )
    real_plan = augmented[:, : len(target)] * multiplier
    source_capacity = multiplier * source
    source_outflow = np.sum(real_plan, axis=1)
    return real_plan, {
        "target_marginal_max_error": float(
            np.max(np.abs(np.sum(real_plan, axis=0) - target))
        ),
        "maximum_source_capacity_excess": float(
            np.max(source_outflow - source_capacity)
        ),
        "routed_source_capacity_fraction": float(
            np.sum(real_plan) / np.sum(source_capacity)
        ),
        "maximum_individual_source_capacity_used_fraction": float(
            np.max(source_outflow / np.maximum(source_capacity, np.finfo(float).tiny))
        ),
    }
