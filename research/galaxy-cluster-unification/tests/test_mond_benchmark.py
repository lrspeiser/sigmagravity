from __future__ import annotations

from pathlib import Path

import numpy as np

from voidscreen.data import load_curves
from voidscreen.mond_benchmark import (
    catalog_curve,
    li2018_rar_mond_acceleration,
    parse_li2018_table,
    precision_mask,
    published_fit_curve,
    reduced_chi_square,
    simple_mond_acceleration,
    standard_mond_acceleration,
)


ROOT = Path(__file__).resolve().parents[1]
SPARC = ROOT / "data" / "raw" / "sparc"
TABLE = ROOT / "data" / "raw" / "li2018_rar" / "source" / "Table.tex"


def test_source_table_parses_all_published_fits():
    fits = parse_li2018_table(TABLE)
    assert len(fits) == 175
    ngc2841 = fits["NGC2841"]
    assert np.isclose(ngc2841.disk_mass_to_light, 0.81)
    assert np.isclose(ngc2841.bulge_mass_to_light, 0.93)
    assert np.isclose(ngc2841.distance_mpc, 15.5)
    assert np.isclose(ngc2841.inclination_deg, 81.9)
    assert np.isclose(ngc2841.reduced_chi_square, 1.515)


def test_mond_laws_have_newtonian_and_deep_mond_limits():
    gbar = np.array([1.0e-14, 1.0e-10, 1.0e-6])
    for law in (
        li2018_rar_mond_acceleration,
        simple_mond_acceleration,
        standard_mond_acceleration,
    ):
        predicted = law(gbar)
        assert np.all(predicted >= gbar)
        assert np.isclose(predicted[-1], gbar[-1], rtol=2.0e-4)
        assert np.isclose(predicted[0], np.sqrt(1.2e-10 * gbar[0]), rtol=0.01)


def test_one_published_galaxy_replays_reduced_chi_square():
    fits = parse_li2018_table(TABLE)
    curve = next(item for item in load_curves(SPARC) if item.metadata.name == "NGC2841")
    evaluation = published_fit_curve(curve, fits["NGC2841"])
    calculated = reduced_chi_square(evaluation, fitted_parameters=4)
    assert abs(calculated - fits["NGC2841"].reduced_chi_square) < 0.02


def test_precision_selection_is_strict_and_reproducible():
    curves = load_curves(SPARC)
    points = 0
    galaxies = 0
    for curve in curves:
        if curve.metadata.quality <= 2 and curve.metadata.inclination_deg >= 30.0:
            evaluation = catalog_curve(curve, "li2018_rar_mond")
            points += int(precision_mask(evaluation).sum())
            galaxies += 1
    assert galaxies == 153
    assert points == 2694
