from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0630_synthetic_universe import (  # noqa: E402
    load_cluster_records,
    load_galaxy_records,
    load_protocol,
)


def protocol():
    path = ROOT / "configs/p0630_synthetic_universe_protocol.json"
    return load_protocol(path)


def test_frozen_whole_system_splits_are_disjoint_and_complete():
    settings = protocol()
    galaxies = load_galaxy_records(settings)
    clusters = load_cluster_records(settings)
    assert len(galaxies) == 131
    assert Counter(record.split for record in galaxies) == {
        "train": 81,
        "development": 27,
        "holdout": 23,
    }
    assert Counter(record.split for record in clusters) == {
        "train": 12,
        "development": 4,
        "holdout": 4,
    }
    for records in (galaxies, clusters):
        names = [record.name for record in records]
        assert len(names) == len(set(names))


def test_targets_are_not_stored_inside_baryonic_seed_objects():
    settings = protocol()
    records = [*load_galaxy_records(settings), *load_cluster_records(settings)]
    for record in records:
        seed_fields = set(record.seed.__dataclass_fields__)
        assert "target_g_m_s2" not in seed_fields
        assert "target_velocity_km_s" not in seed_fields
        assert len(record.target_g_m_s2) == len(record.profile.radius_kpc)


def test_full_result_covers_million_sweep_and_every_locked_holdout():
    report_path = ROOT / "results/p0630_synthetic_universe/report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split_counts"]["galaxy"]["holdout"] == 23
    assert report["split_counts"]["cluster"]["holdout"] == 4
    assert report["simulator"]["million_scale_sweep"]["synthetic_systems"] == 1_048_576
    assert report["simulator"]["injected_truth"][
        "injected_law_recovered_on_holdout"
    ]
    assert report["simulator"]["universal_fit"]["per_object_gravity_parameters"] == 0
    raw = {row["model"]: row for row in report["heldout"]["raw_cluster_images"]}
    assert raw["transport"]["clusters"] == 2
    assert raw["transport"]["heldout_images"] == 6
    assert raw["transport"]["all_roots_converged"]
    assert not raw["transport"]["all_training_roots_converged"]
    assert raw["GR_plus_cluster_halo"]["gravity_parameters_per_object"] == 2
