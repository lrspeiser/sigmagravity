from __future__ import annotations

from pathlib import Path

import numpy as np

from voidscreen.galaxy_replica import (
    generate_replica_particles,
    load_replica_seed,
    render_observed_replica,
    render_replica,
    score_replica,
    valid_rotation_mask,
)


ROOT = Path(__file__).resolve().parents[1]
SPARC = ROOT / "data" / "raw" / "sparc"
PHOTOMETRY = ROOT / "data" / "raw" / "sparc_replica" / "photometric_profiles"
DECOMPOSITIONS = ROOT / "data" / "raw" / "sparc_replica" / "bulge_disk_decompositions"


def seed(name: str = "DDO154"):
    return load_replica_seed(name, SPARC, PHOTOMETRY, DECOMPOSITIONS)


def test_official_photometry_and_decomposition_reconstruct_each_other():
    item = seed("NGC2841")
    rendered = render_observed_replica(item, pixels=257)
    scores = score_replica(item, rendered)
    assert scores["angular_photometry_knots"] >= 30
    assert scores["angular_photometry_rmse_dex"] < 1.0e-3
    assert scores["light_rmse_dex"] < 1.0e-12


def test_observation_replica_is_deterministic_and_conserves_light():
    item = seed()
    first = render_observed_replica(item, pixels=129)
    second = render_observed_replica(item, pixels=129)
    np.testing.assert_array_equal(first.total_lsun_pc2, second.total_lsun_pc2)
    np.testing.assert_array_equal(
        first.line_of_sight_velocity_km_s,
        second.line_of_sight_velocity_km_s,
    )
    scores = score_replica(item, first)
    assert abs(scores["total_light_fractional_error"]) < 0.03
    assert scores["rotation_rmse_km_s"] < 1.0e-12


def test_blind_renderer_requires_explicit_theory_velocities():
    item = seed()
    mask = valid_rotation_mask(item)
    radius = item.rotation.radius_kpc[mask]
    theory_velocity = np.full(len(radius), 12.5)
    rendered = render_replica(item, radius, theory_velocity, pixels=129)
    maximum_los = np.nanmax(np.abs(rendered.line_of_sight_velocity_km_s))
    expected = 12.5 * np.sin(np.radians(item.inclination_deg))
    assert np.isclose(maximum_los, expected, rtol=1.0e-3)
    assert not np.isclose(maximum_los, np.nanmax(item.rotation.velocity_observed_kms))


def test_particles_are_deterministic_and_luminosity_conserving():
    item = seed("NGC2841")
    first = generate_replica_particles(item, particle_count=4096)
    second = generate_replica_particles(item, particle_count=4096)
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.positions_kpc, second.positions_kpc)
    assert len(first.positions_kpc) == 4096
    assert np.isclose(first.luminosities_lsun.sum(), item.light.total_lsun)
    assert set(first.components) == {"disk", "bulge"}
