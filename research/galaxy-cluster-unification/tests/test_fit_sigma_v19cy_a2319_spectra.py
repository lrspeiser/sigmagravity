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


def test_xspec_deck_writer_is_lf_only_even_on_windows(tmp_path: Path):
    path = tmp_path / "fit.xcm"
    fitter.write_xspec_deck(path, "model tbabs*bapec\n1\n")
    assert path.read_bytes() == b"model tbabs*bapec\n1\n"
    with pytest.raises(RuntimeError, match="carriage return"):
        fitter.write_xspec_deck(path, "model tbabs*bapec\r\n")


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


def test_grouping_resume_requires_every_exact_checkpoint_hash(tmp_path: Path):
    reports = []
    bundles = {}
    for index in range(10):
        region = f"r{index}"
        branch = f"b{index}"
        original = tmp_path / "raw" / branch / region / "nxb.pha"
        original.parent.mkdir(parents=True)
        original.write_bytes(b"original")
        grouped = (
            tmp_path / "staging" / "grouped_nxb" / branch / region / "nxb_optsnmin3.pha"
        )
        grouped.parent.mkdir(parents=True)
        grouped.write_bytes(f"grouped-{index}".encode())
        bundles[region] = [
            {
                "source_pha": original,
                "nxb_pha": original,
                "rmf": original,
                "arf": original,
            }
        ]
        reports.append(
            {
                "branch": branch,
                "region": region,
                "rate_and_stat_err_preserved_exactly": True,
                "zero_variance_groups_in_band": 0,
                "minimum_signal_to_noise_in_band": 3.01,
                "grouped": {
                    "sha256": fitter.preparation.sha256(grouped),
                    "hduclas2": "DERIVED",
                    "poiserr": False,
                    "respfile": "NONE",
                    "grouping_type": "optsnmin",
                    "grouping_scale": 3.0,
                },
            }
        )
    resumed = fitter.resume_grouped_nxb_bundles(
        bundles, tmp_path / "staging", reports
    )
    assert len(resumed) == 10
    assert all(row[0]["nxb_pha"].name == "nxb_optsnmin3.pha" for row in resumed.values())
    next(iter((tmp_path / "staging").rglob("nxb_optsnmin3.pha"))).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        fitter.resume_grouped_nxb_bundles(bundles, tmp_path / "staging", reports)


def test_second_branch_nxb_internal_links_shift_and_background_copies_tie():
    config = future_config()
    path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    _, specs = fitter.parse_nxb_model(path.read_text(encoding="utf-8"))
    lines = fitter.nxb_model_lines(specs, 2)
    assert len(lines) == 224
    assert lines[56 + 7] == "= nxb1:p61/5.8980*5.8876"
    assert lines[112] == "= nxb1:p1"
    assert lines[168] == "= nxb1:p57"


def initial_nxb_numeric_values(specs: list[str], source_group_count: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for source_index in range(source_group_count):
        offset = source_index * fitter.NXB_PARAMETER_COUNT
        for local_index, spec in enumerate(specs, start=1):
            if not spec.startswith("="):
                values[str(offset + local_index)] = float(spec.split()[0])
    return values


def test_nxb_prefit_transfer_freezes_shape_and_preserves_independent_branches():
    config = future_config()
    path = fitter.ROOT / config["nxb_protocol"]["empirical_model_path"]
    _, specs = fitter.parse_nxb_model(path.read_text(encoding="utf-8"))
    values = initial_nxb_numeric_values(specs, 2)
    values["2"] = 0.161
    values["58"] = 0.171
    transferred = fitter.nxb_specs_after_prefit(specs, 2, values)
    assert len(transferred) == 112
    assert transferred[1].split()[:2] == ["0.161", "-1"]
    assert float(transferred[57].split()[0]) == pytest.approx(0.171)
    assert transferred[57].split()[1] == "-1"
    assert transferred[7] == "= nxb1:p5/5.8980*5.8876"
    assert transferred[56 + 7] == "= nxb1:p61/5.8980*5.8876"
    joint = fitter.nxb_joint_model_lines(transferred, 2)
    assert len(joint) == 224
    assert joint[112] == "= nxb1:p1"
    assert joint[168] == "= nxb1:p57"


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
        nxb_prefit_values=initial_nxb_numeric_values(specs, 2),
        log_path=tmp_path / "xspec.log",
        session_path=tmp_path / "best.xcm",
    )
    assert "statistic cstat 1-2" in deck
    assert "statistic chi standard 3-4" in deck
    assert "response 1:1" in deck and "source.rmf" in deck
    assert "arf 1:1" in deck and "source.arf" in deck
    assert deck.count("newdiag60000.rmf") == 6
    assert "ignore 1:**\n" not in deck and "ignore 2:**\n" not in deck
    assert "ignore 3:**-1.0 17.0-**" in deck
    assert "ignore 1:**-3.0 9.5-**" in deck
    assert deck.splitlines().count("fit") == 1
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


def test_nxb_only_prefit_deck_cannot_read_source_spectra(tmp_path: Path):
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
    deck, metadata = fitter.build_nxb_prefit_deck(
        config,
        bundle,
        nxb_expression=expression,
        nxb_specs=specs,
        log_path=tmp_path / "nxb_prefit.log",
        session_path=tmp_path / "nxb_prefit.xcm",
    )
    assert "source.pha" not in deck
    assert "source.rmf" not in deck
    assert "source.arf" not in deck
    assert deck.count("nxb.pha") == 2
    assert deck.count("newdiag60000.rmf") == 2
    assert "statistic chi standard 1-2" in deck
    assert "model 1:nxb1 constant*powerlaw" in deck
    assert "ignore 1:**-1.0 17.0-**" in deck
    assert "ignore 1:**\n" not in deck
    assert deck.splitlines().count("fit") == 2
    assert metadata["source_spectra_loaded"] is False


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


def test_published_comparison_reports_all_frozen_aggregate_diagnostics():
    config = future_config()
    primary = {
        region: {
            "velocity_km_s": benchmark["velocity_km_s"],
            "velocity_interval_halfwidth_km_s": 25.0,
        }
        for region, benchmark in config["published_no_ssm_benchmark"]["regions"].items()
    }
    comparison = fitter.published_comparison(config, primary)
    assert set(comparison["per_region"]) == set(primary)
    aggregate = comparison["aggregate"]
    assert aggregate["unweighted_rms_difference_km_s"] == pytest.approx(0.0)
    assert aggregate[
        "inverse_combined_variance_weighted_rms_difference_km_s"
    ] == pytest.approx(0.0)
    assert aggregate["pearson_velocity_correlation"] == pytest.approx(1.0)
    assert aggregate["spearman_velocity_rank_correlation"] == pytest.approx(1.0)
    assert aggregate["pairwise_velocity_rank_agreement_fraction"] == pytest.approx(1.0)
    assert aggregate["sign_agreement_fraction"] == pytest.approx(1.0)


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
        region: {
            "velocity_km_s": row["velocity_km_s"] + 10.0,
            "velocity_interval_halfwidth_km_s": 25.0,
        }
        for region, row in config["published_no_ssm_benchmark"]["regions"].items()
    }
    comparison = fitter.published_comparison(config, primary)
    per_region = comparison["per_region"]
    assert set(per_region) == set(primary)
    assert all(row["diagnostic_only"] for row in per_region.values())
    for region, row in per_region.items():
        expected = 10.0 / config["published_no_ssm_benchmark"]["regions"][region]["plus_1sigma"]
        assert abs(row["difference_over_published_directional_1sigma"] - expected) < 1e-12
