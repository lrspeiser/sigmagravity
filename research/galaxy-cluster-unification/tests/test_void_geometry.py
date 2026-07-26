import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.void_geometry import (
    LOCAL_VOID_OVERLAP_THRESHOLD,
    build_local_void_wall_table,
    icrs_to_local_void_box_hmpc,
    sample_cloud_membership_and_wall_distance,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "raw" / "local_voids"


def test_local_void_provenance_hashes_match_downloaded_bytes() -> None:
    provenance = json.loads((CATALOG / "provenance.json").read_text(encoding="utf-8-sig"))
    assert provenance["commit"] == "bbbc34594d92eeef32897d67d291d54eb384be6e"
    assert len(provenance["files"]) == 104
    for record in provenance["files"]:
        path = CATALOG / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_catalog_sky_coordinates_reproduce_cartesian_centers() -> None:
    catalog = pd.read_csv(CATALOG / "voids_catalog.csv")
    calculated = icrs_to_local_void_box_hmpc(
        catalog["center RA [deg]"].to_numpy(),
        catalog["center Dec [deg]"].to_numpy(),
        catalog["center dist [Mpc/h]"].to_numpy() / 0.681,
    )
    published = catalog[["center x (Mpc/h)", "center y (Mpc/h)", "center z (Mpc/h)"]].to_numpy()
    assert np.allclose(calculated, published, rtol=0.0, atol=2e-12)


def test_void_center_is_inside_and_has_positive_wall_distance() -> None:
    catalog = pd.read_csv(CATALOG / "voids_catalog.csv")
    center = catalog.loc[0, ["center x (Mpc/h)", "center y (Mpc/h)", "center z (Mpc/h)"]].to_numpy(
        dtype=float
    )[None, :]
    overlap, wall = sample_cloud_membership_and_wall_distance(
        CATALOG / "VoronoiClouds" / "Voronoi_cloud_void_0_N32.npy", center
    )
    assert overlap[0] > LOCAL_VOID_OVERLAP_THRESHOLD
    assert wall[0] > 0.0


def test_frozen_wall_table_has_declared_coverage_and_bounds() -> None:
    table = build_local_void_wall_table(ROOT / "data" / "raw" / "sparc", CATALOG)
    assert len(table) == 175
    assert table["galaxy"].nunique() == 175
    assert int(table["inside_catalog_void"].sum()) == 72
    assert table.loc[table["inside_catalog_void"], "void_index"].nunique() == 12
    assert table["void_wall_score"].between(0.0, 1.0).all()
    assert table["void_score"].equals(table["void_wall_score"])
