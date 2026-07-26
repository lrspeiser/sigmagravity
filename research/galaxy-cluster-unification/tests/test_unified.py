from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_manga_bcg_table import parse_table
from voidscreen.data import KPC_M
from voidscreen.unified import (
    A0_M_S2,
    C_M_S,
    assign_system_folds,
    baryonic_potential_profile,
    load_clash_acceleration_frame,
    predict_acceleration,
    rar_acceleration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_point_mass_potential_and_scale_length() -> None:
    radius_kpc = np.geomspace(0.5, 500.0, 20_000)
    amplitude = 2.0e31
    radius_m = radius_kpc * KPC_M
    gbar = amplitude / radius_m**2
    potential = baryonic_potential_profile(radius_kpc, gbar)
    expected = amplitude / radius_m
    np.testing.assert_allclose(potential, expected, rtol=2e-7)
    np.testing.assert_allclose(potential / gbar / KPC_M, radius_kpc, rtol=2e-7)


def test_rar_has_newtonian_and_deep_acceleration_limits() -> None:
    high = rar_acceleration(np.asarray([1e-7]))[0]
    low_gbar = 1e-14
    low = rar_acceleration(np.asarray([low_gbar]))[0]
    assert np.isclose(high, 1e-7, rtol=1e-4)
    assert np.isclose(low, np.sqrt(A0_M_S2 * low_gbar), rtol=0.005)


def test_unified_models_use_baryonic_geometry() -> None:
    gbar = np.full(2, 1e-11)
    ell = np.asarray([1.0, 500.0])
    chi = np.asarray([1e-9, 1e-4])
    u1 = predict_acceleration(
        "U1_coherence_length", gbar, chi, ell, [1.0, 1.0], domain="galaxy"
    )
    u0 = predict_acceleration(
        "U0_emond_like", gbar, chi, ell, [1.0, -6.0, 0.1], domain="cluster"
    )
    assert u1[1] > u1[0]
    assert u0[1] > u0[0]
    assert np.all(chi * C_M_S**2 > 0.0)


def test_clash_loader_and_fold_assignment() -> None:
    frame = load_clash_acceleration_frame(
        ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat"
    )
    assert len(frame) == 84
    assert frame["system"].nunique() == 20
    assert np.isfinite(frame[["chi", "ell_bar_kpc"]].to_numpy()).all()
    assert (frame["ell_bar_kpc"] > 0.0).all()

    assigned = assign_system_folds(frame, folds=5, seed=20260726)
    per_system = assigned.groupby("system")["fold"].nunique()
    assert (per_system == 1).all()
    counts = assigned[["system", "fold"]].drop_duplicates()["fold"].value_counts()
    assert sorted(counts.tolist()) == [4, 4, 4, 4, 4]


def test_domain_oracle_changes_only_cluster_scale() -> None:
    frame = pd.DataFrame(
        {"gbar": [1e-11], "chi": [1e-6], "ell": [100.0]}
    )
    common = (frame["gbar"], frame["chi"], frame["ell"], [-9.0])
    galaxy = predict_acceleration("domain_oracle", *common, domain="galaxy")
    cluster = predict_acceleration("domain_oracle", *common, domain="cluster")
    fixed = predict_acceleration("fixed_rar", *common[:3], [], domain="galaxy")
    np.testing.assert_allclose(galaxy, fixed)
    assert cluster[0] > galaxy[0]


def test_manga_bcg_table_extraction() -> None:
    frame = parse_table(
        ROOT / "data" / "raw" / "manga_bcg_tian2024" / "RAR_BCG.tex"
    )
    assert len(frame) == 50
    assert frame["plateifu"].nunique() == 50
    assert (frame["radius_kpc"] > 0.0).all()
    assert frame["log_gbar"].between(-10.5, -8.5).all()
