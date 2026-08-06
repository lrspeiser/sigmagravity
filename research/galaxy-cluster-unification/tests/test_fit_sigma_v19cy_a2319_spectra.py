import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fit_sigma_v19cy_a2319_spectra as fitter


def future_config() -> dict:
    config = json.loads(fitter.CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["fit_protocol"]["nxb_constraint_band_keV"] = [1.0, 17.0]
    return config


def test_public_nxb_model_is_parsed_without_changing_its_56_parameters():
    config = future_config()
    path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    expression, specs = fitter.parse_nxb_model(path.read_text(encoding="utf-8"))
    assert expression.startswith("constant*powerlaw")
    assert len(specs) == 56
    assert specs[7] == "= nxb1:p5/5.8980*5.8876"
    assert specs[10] == "= nxb1:p1"


def test_second_branch_nxb_internal_links_shift_and_background_copies_tie():
    config = future_config()
    path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    _, specs = fitter.parse_nxb_model(path.read_text(encoding="utf-8"))
    lines = fitter.nxb_model_lines(specs, 2)
    assert len(lines) == 224
    assert lines[56 + 7] == "= nxb1:p61/5.8980*5.8876"
    assert lines[112] == "= nxb1:p1"
    assert lines[168] == "= nxb1:p57"


def test_primary_source_linkage_has_independent_branch_normalizations_and_zero_nxb_source():
    lines = fitter.primary_source_model_lines(2)
    assert len(lines) == 24
    assert lines[0].startswith("0.112 -1")
    assert lines[6:11] == ["= p1", "= p2", "= p3", "= p4", "= p5"]
    assert not lines[11].startswith("=") and " -1 " not in lines[11]
    assert lines[12:17] == ["= p1", "= p2", "= p3", "= p4", "= p5"]
    assert lines[17].startswith("0 -1")
    assert lines[23].startswith("0 -1")


def test_two_temperature_linkage_shares_velocity_but_not_two_branch_normalizations():
    lines = fitter.two_temperature_source_model_lines(2)
    assert len(lines) == 44
    assert lines[7:10] == ["= p3", "= p4", "= p5"]
    assert lines[11 + 3] == "= p4"
    assert not lines[11 + 5].startswith("=")
    assert not lines[11 + 10].startswith("=")
    assert lines[22 + 5].startswith("0 -1")
    assert lines[22 + 10].startswith("0 -1")


def test_xspec_deck_uses_mixed_statistics_and_separate_responses(tmp_path: Path):
    config = future_config()
    nxb_path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    expression, specs = fitter.parse_nxb_model(nxb_path.read_text(encoding="utf-8"))
    bundle = []
    for branch in ("one", "two"):
        bundle.append(
            {
                "source_pha": tmp_path / branch / "source.pha",
                "nxb_pha": tmp_path / branch / "nxb.pha",
                "rmf": tmp_path / branch / "source.rmf",
                "arf": tmp_path / branch / "source.arf",
            }
        )
    deck, metadata = fitter.build_xspec_deck(
        config,
        bundle,
        variant={"name": "primary", "band_keV": [3.0, 9.5]},
        nxb_expression=expression,
        nxb_specs=specs,
        log_path=tmp_path / "xspec.log",
        session_path=tmp_path / "best.xcm",
    )
    assert "statistic cstat 1-2" in deck
    assert "statistic chi standard 3-4" in deck
    assert "response 1:1" in deck and "source.rmf" in deck
    assert "arf 1:1" in deck and "source.arf" in deck
    assert deck.count("newdiag60000.rmf") == 6
    assert "ignore 1:**\n" in deck and "ignore 2:**\n" in deck
    assert "ignore 3:**-1.0 17.0-**" in deck
    assert "ignore 1:**-3.0 9.5-**" in deck
    assert metadata["source_group_count"] == 2
    assert metadata["source_statistic"] == "cstat"
    assert metadata["nxb_statistic"] == "chi standard"


def test_markers_and_velocity_conversion_are_machine_readable():
    parsed = fitter.parse_markers(
        "noise\n"
        f"{fitter.MARKER} statistic 123.5\n"
        f"{fitter.MARKER} redshift_error 0.053 0.055 FFFFFFFFF\n"
    )
    assert parsed["statistic"] == "123.5"
    assert fitter.parse_error(parsed["redshift_error"]) == (0.053, 0.055, "FFFFFFFFF")
    config = future_config()
    assert abs(fitter.velocity_km_s(config, config["fit_protocol"]["bcg_redshift"]) + 12.42) < 1e-12


def test_interval_overlap_uses_combined_two_sigma_support():
    assert fitter.intervals_overlap([-100.0, 100.0], [250.0, 350.0])
    assert not fitter.intervals_overlap([-100.0, 100.0], [500.0, 600.0])


def test_published_comparison_is_diagnostic_and_uses_directional_error():
    config = future_config()
    primary = {
        region: {"velocity_km_s": row["velocity_km_s"] + 10.0}
        for region, row in config["published_no_ssm_benchmark"]["regions"].items()
    }
    comparison = fitter.published_comparison(config, primary)
    assert set(comparison) == set(primary)
    assert all(row["diagnostic_only"] for row in comparison.values())
    for region, row in comparison.items():
        expected = 10.0 / config["published_no_ssm_benchmark"]["regions"][region]["plus_1sigma"]
        assert abs(row["difference_over_published_directional_1sigma"] - expected) < 1e-12
