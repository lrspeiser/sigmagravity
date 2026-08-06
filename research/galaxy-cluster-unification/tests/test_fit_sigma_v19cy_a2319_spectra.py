import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_frozen_component_and_arf_hash_chain_is_accepted_before_source_fit():
    config, component_report, arf_report = fitter.validate_inputs()
    assert config["protocol_version"] == fitter.EXPECTED_PROTOCOL
    assert component_report["component_gate_passed"] is True
    assert arf_report["arf_gate_passed"] is True
    assert arf_report["config_sha256"] == config["pre_fit_grouping_amendment"][
        "arf_generation_config_sha256"
    ]


def test_json_checkpoint_writer_replaces_atomically(tmp_path: Path):
    output = tmp_path / "checkpoint.json"
    fitter.write_json_atomic(output, {"step": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"step": 1}
    fitter.write_json_atomic(output, {"step": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"step": 2}
    assert not (tmp_path / "checkpoint.json.writing").exists()


def test_public_nxb_model_is_parsed_without_changing_its_56_parameters():
    config = future_config()
    path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    expression, specs = fitter.parse_nxb_model(path.read_text(encoding="utf-8"))
    assert expression.startswith("constant*powerlaw")
    assert len(specs) == 56
    assert specs[7] == "= nxb1:p5/5.8980*5.8876"
    assert specs[10] == "= nxb1:p1"


def test_nxb_grouping_command_is_optsnmin3_and_restores_derived_class(tmp_path: Path):
    config = future_config()
    command = fitter.nxb_grouping_command(
        config,
        tmp_path / "original.pha",
        tmp_path / "scratch.pha",
        tmp_path / "grouped.pha",
        tmp_path,
    )
    assert "grouptype=optsnmin groupscale=3.0" in command
    assert "value=BKG" in command
    assert "value=DERIVED" in command
    assert command.index("value=BKG") < command.index("ftgrouppha")
    assert command.index("ftgrouppha") < command.index("value=DERIVED")
    assert "keyword=RESPFILE operation=add value=NONE" in command
    assert command.count("&& punlearn") == 4
    assert fitter.xspec_path(config, tmp_path / "original.pha") in command


def test_grouped_nxb_summary_has_no_zero_variance_and_reports_boundary_group():
    rate = np.asarray([3.0, 3.0, 6.0, 6.0])
    error = np.asarray([1.0, 1.0, 2.0, 2.0])
    grouping = np.asarray([1, -1, 1, -1])
    energy_min = np.asarray([1.0, 1.5, 2.0, 2.5])
    energy_max = energy_min + 0.5
    report = fitter.summarize_nxb_groups(
        rate, error, grouping, energy_min, energy_max, [1.0, 3.0]
    )
    assert report["groups_in_band"] == 2
    assert report["zero_variance_groups_in_band"] == 0
    assert report["maximum_channels_per_group_in_band"] == 2
    assert report["minimum_effective_counts_in_band"] == pytest.approx(18.0)
    assert report["minimum_signal_to_noise_in_band"] == pytest.approx(18.0**0.5)


def test_grouped_nxb_summary_fails_closed_on_zero_variance_group():
    with pytest.raises(RuntimeError, match="zero-variance"):
        fitter.summarize_nxb_groups(
            np.asarray([1.0, 1.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([1, -1]),
            np.asarray([1.0, 1.5]),
            np.asarray([1.5, 2.0]),
            [1.0, 2.0],
        )


def test_grouped_nxb_summary_fails_closed_below_frozen_signal_to_noise():
    with pytest.raises(RuntimeError, match="minimum signal-to-noise"):
        fitter.summarize_nxb_groups(
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, 1.0]),
            np.asarray([1, -1]),
            np.asarray([1.0, 1.5]),
            np.asarray([1.5, 2.0]),
            [1.0, 2.0],
        )


def test_nxb_grouping_gate_requires_all_ten_complete_contracts():
    row = {
        "rate_and_stat_err_preserved_exactly": True,
        "zero_variance_groups_in_band": 0,
        "minimum_signal_to_noise_in_band": 3.01,
        "grouped": {
            "hduclas2": "DERIVED",
            "poiserr": False,
            "respfile": "NONE",
            "grouping_type": "optsnmin",
            "grouping_scale": 3.0,
        },
    }
    assert fitter.nxb_grouping_gate_passed([copy.deepcopy(row) for _ in range(10)])
    assert not fitter.nxb_grouping_gate_passed([copy.deepcopy(row) for _ in range(9)])
    failed = [copy.deepcopy(row) for _ in range(10)]
    failed[4]["minimum_signal_to_noise_in_band"] = 2.99
    assert not fitter.nxb_grouping_gate_passed(failed)


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
    assert all(f"tclout stat {index}" in deck for index in range(1, 5))
    assert metadata["source_group_count"] == 2
    assert metadata["source_statistic"] == "cstat"
    assert metadata["nxb_statistic"] == "chi standard"
    numeric = fitter.nxb_numeric_parameter_indices(specs, 2)
    thawed = fitter.nxb_free_parameter_indices(specs, 2)
    assert numeric
    assert thawed == [
        *(fitter.NXB_THAWED_NORMALIZATIONS),
        *(56 + index for index in fitter.NXB_THAWED_NORMALIZATIONS),
    ]
    assert all(f"freeze nxb1:{index}" in deck for index in numeric)
    assert all(f"thaw nxb1:{index}" in deck for index in thawed)
    assert "thaw nxb1:2\n" not in deck


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


def test_fit_audit_separates_source_and_nxb_statistic_contributions():
    config = future_config()
    nxb_path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    _, nxb_specs = fitter.parse_nxb_model(nxb_path.read_text(encoding="utf-8"))
    source_specs = fitter.primary_source_model_lines(1)
    nxb_free = fitter.nxb_free_parameter_indices(nxb_specs, 1)
    metadata = {
        "source_group_count": 1,
        "source_parameter_specs": source_specs,
        "source_free_parameter_indices": fitter.source_free_parameter_indices(1, False),
        "nxb_free_parameter_indices": nxb_free,
    }
    markers = {
        "statistic": "30.0",
        "statistic_spectrum_1": "10.0",
        "statistic_spectrum_2": "20.0",
        "dof": "50",
        "covariance": "1 0 1",
        "variable_parameters": "17",
        "redshift": "0.05458",
        "redshift_error": "0.054 0.055 FFFFFFFFF",
        "redshift_sigma": "0.0001",
    }
    for index, spec in enumerate(source_specs, start=1):
        markers[f"source_parameter_{index}"] = (
            "1.0" if spec.startswith("=") else spec.split()[0]
        )
    for index in nxb_free:
        markers[f"nxb_parameter_{index}"] = nxb_specs[index - 1].split()[0]
    result = fitter.inspect_fit(config, markers, metadata, nxb_specs)
    assert result["source_cstat_contribution"] == 10.0
    assert result["nxb_chi_square_contribution"] == 20.0
    assert result["statistic_by_spectrum"] == {"1": 10.0, "2": 20.0}
    markers["statistic"] = "31.0"
    with pytest.raises(RuntimeError, match="do not sum"):
        fitter.inspect_fit(config, markers, metadata, nxb_specs)


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
