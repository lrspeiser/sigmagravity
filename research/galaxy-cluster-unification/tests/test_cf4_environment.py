import hashlib
import json
from pathlib import Path

import numpy as np

from voidscreen.data import pack_dataset
from voidscreen.environment import (
    CF4_H0_KMS_MPC,
    GRID_SPECS,
    build_cf4_environment_table,
    density_acceleration_and_tidal_fields,
    load_density_grid,
    validate_catalog_coordinates,
)

ROOT = Path(__file__).resolve().parents[1]
SPARC = ROOT / "data" / "raw" / "sparc"
CF4 = ROOT / "data" / "raw" / "cosmicflows4"


def test_cf4_provenance_hashes_match_downloaded_bytes() -> None:
    provenance = json.loads((CF4 / "provenance.json").read_text(encoding="utf-8-sig"))
    assert len(provenance["files"]) == 7
    for record in provenance["files"]:
        path = CF4 / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_cf4_density_grids_have_declared_shapes() -> None:
    for spec in GRID_SPECS:
        grid = load_density_grid(CF4, spec)
        assert grid.shape == spec.shape
        assert np.isfinite(grid).all()
        assert grid.std() > 0.0


def test_catalog_coordinate_columns_match_astropy_and_cartesian_axes() -> None:
    report = validate_catalog_coordinates(CF4 / "CF4_table4_groups.dat.gz")
    assert report["rows"] == 38053
    assert report["max_astropy_sgl_residual_deg"] < 0.002
    assert report["max_astropy_sgb_residual_deg"] < 0.001
    assert report["max_xyz_direction_residual_deg"] < 0.2


def test_environment_table_covers_every_sparc_galaxy_and_loads(tmp_path: Path) -> None:
    table = build_cf4_environment_table(SPARC, CF4)
    assert len(table) == 175
    assert table["galaxy"].nunique() == 175
    assert np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all()
    assert table["void_score"].std(ddof=0) > 0.0
    assert table["void_score"].equals(table["void_score_grouped_64"])
    assert (
        table["void_score_grouped_64"].corr(table["void_score_ungrouped_64"], method="spearman")
        > 0.7
    )

    environment_csv = tmp_path / "environment.csv"
    table.to_csv(environment_csv, index=False)
    packed = pack_dataset(SPARC, environment_csv=environment_csv)
    assert packed.n_galaxies == 131
    assert np.isfinite(packed.environment_standardized).all()
    assert np.isclose(packed.environment_standardized.mean(), 0.0, atol=1e-12)
    assert np.isclose(packed.environment_standardized.std(ddof=0), 1.0, atol=1e-12)

    alternate = pack_dataset(
        SPARC,
        environment_csv=environment_csv,
        environment_score_column="void_score_ungrouped_64",
    )
    assert alternate.environment_score_column == "void_score_ungrouped_64"
    assert alternate.environment_fingerprint == packed.environment_fingerprint
    assert not np.allclose(alternate.environment_raw, packed.environment_raw)


def test_fft_tidal_solver_recovers_single_plane_wave() -> None:
    side = 24
    coordinate = 2.0 * np.pi * np.arange(side) / side
    delta = np.cos(coordinate)[:, None, None] * np.ones((1, side, side))
    acceleration, tidal = density_acceleration_and_tidal_fields(
        delta,
        box_size_hmpc=240.0,
        h0_km_s_mpc=CF4_H0_KMS_MPC,
        omega_m=0.3,
        padding_factor=1,
    )
    h0_s = CF4_H0_KMS_MPC * 1000.0 / 3.085677581491367e22
    coefficient = 1.5 * 0.3 * h0_s**2
    assert acceleration.shape == (3, side, side, side)
    assert tidal.shape == (3, 3, side, side, side)
    assert np.allclose(tidal[0, 0], -coefficient * delta, rtol=1e-10, atol=1e-50)
    assert np.max(np.abs(tidal[1:, 1:])) < coefficient * 1e-10
    assert np.allclose(tidal, np.swapaxes(tidal, 0, 1), rtol=0.0, atol=0.0)
