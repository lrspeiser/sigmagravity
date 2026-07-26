import hashlib
import json
from pathlib import Path

import numpy as np

from voidscreen.data import load_curves, pack_dataset, parse_table1

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "sparc"


def test_complete_sparc_snapshot_parses() -> None:
    metadata = parse_table1(DATA / "table1.dat")
    curves = load_curves(DATA)
    assert len(metadata) == 175
    assert len(curves) == 175
    assert sum(curve.radius_kpc.size for curve in curves) == 3391


def test_default_radial_split_is_disjoint_and_ordered() -> None:
    data = pack_dataset(DATA)
    assert data.n_galaxies > 100
    assert data.n_points == data.n_train + data.n_holdout
    for galaxy_index in range(data.n_galaxies):
        selected = data.galaxy_index == galaxy_index
        train_radius = data.radius_kpc[selected & data.train_mask]
        holdout_radius = data.radius_kpc[selected & ~data.train_mask]
        assert train_radius.size >= 5
        assert holdout_radius.size >= 2
        assert np.max(train_radius) <= np.min(holdout_radius)


def test_data_fingerprint_is_stable_sha256() -> None:
    data = pack_dataset(DATA)
    assert len(data.data_fingerprint) == 64
    int(data.data_fingerprint, 16)


def test_data_fingerprint_ignores_import_timestamp(tmp_path: Path) -> None:
    from voidscreen.data import data_fingerprint

    manifest = tmp_path / "provenance.json"
    original = {
        "imported_utc": "2026-07-25T00:00:00Z",
        "files": [{"path": "table1.dat", "bytes": 10, "sha256": "a" * 64}],
    }
    manifest.write_text(json.dumps(original), encoding="utf-8")
    first = data_fingerprint(tmp_path)
    changed = dict(original)
    changed["imported_utc"] = "1900-01-01T00:00:00Z"
    manifest.write_text(json.dumps(changed, indent=2), encoding="utf-8")
    assert data_fingerprint(tmp_path) == first


def test_provenance_hashes_match_imported_bytes() -> None:
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8-sig"))
    assert provenance["rotmod_file_count"] == 175
    for record in provenance["files"]:
        path = DATA / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
