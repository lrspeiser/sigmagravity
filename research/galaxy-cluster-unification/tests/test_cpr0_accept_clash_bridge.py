from __future__ import annotations

import json
from pathlib import Path

from scripts.run_cpr0_accept_clash_bcg_stellar import (
    build_stellar_augmented_sample,
    load_clash_bcg_properties,
)
from scripts.run_cpr0_accept_clash_bridge import build_clash_sample

ROOT = Path(__file__).resolve().parents[1]
ACCEPT = ROOT / "data" / "raw" / "accept_cavagnolo2009" / "all_profiles.dat.txt"
CLASH = ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat"
TABLE1 = ROOT / "data" / "raw" / "clash_tian2020" / "table1.dat"
PROTOCOL = ROOT / "configs" / "cpr0_accept_clash_bridge_protocol.json"


def name_map() -> dict[str, str]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))["cluster_name_map"]


def test_primary_accept_clash_selection_is_frozen() -> None:
    sample, audit = build_clash_sample(
        ACCEPT, CLASH, name_map(), minimum_radius_kpc=100.0
    )
    assert int(audit["match"].sum()) == 18
    assert sample["system"].nunique() == 18
    assert len(sample) == 52
    assert sample["radius_kpc"].min() == 100.0
    assert sample["radius_kpc"].max() == 600.0


def test_stellar_augmentation_adds_all_twenty_central_points() -> None:
    properties = load_clash_bcg_properties(TABLE1)
    sample = build_stellar_augmented_sample(ACCEPT, CLASH, TABLE1, name_map())
    central = sample[sample["radius_kpc"] < 100.0]
    outer = sample[sample["radius_kpc"] >= 100.0]
    assert properties["cluster"].nunique() == 20
    assert sample["system"].nunique() == 20
    assert len(sample) == 72
    assert len(central) == 20
    assert len(outer) == 52
    assert (central["stellar_density_g_cm3"] > 0.0).all()
    assert (sample["local_density_g_cm3"] > 0.0).all()
