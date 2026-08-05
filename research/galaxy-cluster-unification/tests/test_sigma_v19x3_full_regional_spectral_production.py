from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x3_full_regional_spectral_production as runner


def fixture_config() -> dict:
    return {
        "registered_workload": {
            "clusters": {
                "BULLET": {"total_regions": 2, "total_cells": 2},
                "ABELL2146": {"total_regions": 1, "total_cells": 1},
            }
        },
        "runtime_authorization": {"required_unified_cells": 3},
        "regional_gates": {
            "expected_total_regions": 3,
            "minimum_quality_passes_per_cluster": 1,
        },
    }


def fixture_manifest() -> list[dict[str, str]]:
    return [
        {
            "production_index": "2",
            "cluster": "BULLET",
            "bin_id": "1",
            "obsid": "11",
            "ccd_id": "0",
        },
        {
            "production_index": "1",
            "cluster": "BULLET",
            "bin_id": "0",
            "obsid": "10",
            "ccd_id": "0",
        },
        {
            "production_index": "3",
            "cluster": "ABELL2146",
            "bin_id": "8",
            "obsid": "20",
            "ccd_id": "1",
        },
    ]


def fixture_validated(tmp_path: Path) -> dict:
    output = {}
    for index, row in enumerate(fixture_manifest()):
        key = runner.adapter.task_key(row)
        output[key] = {
            "cluster": key[0],
            "bin_id": key[1],
            "obsid": key[2],
            "ccd_id": key[3],
            "cell_name": runner.adapter.cell_name(row),
            "source_pha_sha256": str(index) * 64,
            "source_pha_total_counts": 20 + index,
            "source_band_events": 10 + index,
            "background_band_events": 2,
            "source_pha": tmp_path / f"source{index}.pi",
        }
    return output


def passing_combination(label: str, cells: list[dict], *_args) -> dict:
    return {
        "label": label,
        "cells": len(cells),
        "full_pha_count_conservation_exact": True,
        "grouped_pha_links": {"BACKFILE": "bkg"},
        "expected_grouped_pha_links": {"BACKFILE": "bkg"},
        "frozen_snapshot": {"files": 4, "products": []},
    }


def passing_fit(_config, cluster: str, combination: dict, abundance: float) -> dict:
    return {
        "cluster": cluster,
        "label": combination["label"],
        "fit_completed": True,
        "source_spectrum_sha256": "a" * 64,
        "parameters": {
            "temperature_keV": 8.0,
            "abundance_solar": abundance,
            "normalization": 0.001,
        },
        "gates": {"all_passed": True},
    }


def test_region_plan_uses_every_manifest_cell_once_and_preserves_order() -> None:
    plan = runner.build_full_region_plan(fixture_config(), fixture_manifest())
    assert list(plan) == ["ABELL2146", "BULLET"]
    assert list(plan["BULLET"]) == [0, 1]
    assert list(plan["ABELL2146"]) == [8]
    assert sum(len(rows) for regions in plan.values() for rows in regions.values()) == 3


def test_full_production_requires_finite_fits_but_not_every_quality_subgate(
    tmp_path: Path, monkeypatch
) -> None:
    config = fixture_config()
    plan = runner.build_full_region_plan(config, fixture_manifest())
    validated = fixture_validated(tmp_path)
    monkeypatch.setattr(runner.inherited_v19x, "combine_aperture", passing_combination)

    def one_quality_failure(config_arg, cluster, combination, abundance):
        fit = passing_fit(config_arg, cluster, combination, abundance)
        if combination["label"] == "BULLET_bin1":
            fit["gates"]["all_passed"] = False
        return fit

    monkeypatch.setattr(runner, "fit_regional_gas_spectrum", one_quality_failure)
    result = runner.run_full_regional_production(
        config,
        tmp_path / "output",
        tmp_path / "scratch",
        plan,
        validated,
        {"BULLET": 0.3, "ABELL2146": 0.2},
    )
    assert result["source_map_construction_authorized"]
    assert result["gates"]["every_region_has_finite_best_fit"]
    assert result["cluster_summaries"]["BULLET"]["individual_quality_pass_regions"] == 1


def test_nonfinite_or_failed_best_fit_blocks_source_map(tmp_path: Path, monkeypatch) -> None:
    config = fixture_config()
    plan = runner.build_full_region_plan(config, fixture_manifest())
    validated = fixture_validated(tmp_path)
    monkeypatch.setattr(runner.inherited_v19x, "combine_aperture", passing_combination)

    def failed_best_fit(config_arg, cluster, combination, abundance):
        fit = passing_fit(config_arg, cluster, combination, abundance)
        if combination["label"] == "BULLET_bin1":
            fit["fit_completed"] = False
            fit["gates"]["all_passed"] = False
        return fit

    monkeypatch.setattr(runner, "fit_regional_gas_spectrum", failed_best_fit)
    result = runner.run_full_regional_production(
        config,
        tmp_path / "output",
        tmp_path / "scratch",
        plan,
        validated,
        {"BULLET": 0.3, "ABELL2146": 0.2},
    )
    assert not result["source_map_construction_authorized"]
    assert not result["gates"]["every_region_has_finite_best_fit"]


def test_region_checkpoints_prevent_recombination_and_refitting(
    tmp_path: Path, monkeypatch
) -> None:
    config = fixture_config()
    cell = next(iter(fixture_validated(tmp_path).values()))
    calls = {"combine": 0, "fit": 0}

    def combine(*args):
        calls["combine"] += 1
        return passing_combination(args[0], args[1])

    def fit(*args):
        calls["fit"] += 1
        return passing_fit(*args)

    monkeypatch.setattr(runner.inherited_v19x, "combine_aperture", combine)
    monkeypatch.setattr(runner, "fit_regional_gas_spectrum", fit)
    first = runner.process_region(
        config, tmp_path / "output", tmp_path / "scratch", "BULLET", 0, [cell], 0.3
    )
    monkeypatch.setattr(
        runner,
        "validate_combination_checkpoint",
        lambda record, *_args: record["combination"],
    )
    monkeypatch.setattr(
        runner,
        "validate_fit_checkpoint",
        lambda record, *_args: record["fit"],
    )
    second = runner.process_region(
        config, tmp_path / "output", tmp_path / "scratch", "BULLET", 0, [cell], 0.3
    )
    assert calls == {"combine": 1, "fit": 1}
    assert not first["combination_reused"] and not first["fit_reused"]
    assert second["combination_reused"] and second["fit_reused"]


def test_x2_authorization_extracts_only_passing_integrated_abundances(
    tmp_path: Path,
) -> None:
    config = fixture_config()
    config["runtime_authorization"].update(
        {
            "required_v19x2_config_sha256": "c" * 64,
            "required_v19x2_runner_sha256": "d" * 64,
        }
    )
    report = {
        "status": runner.AUTHORIZED_X2_STATUS,
        "config_sha256": "c" * 64,
        "runner_sha256": "d" * 64,
        "gates": {"all": True},
        "full_494_region_combination_and_fit_authorized": True,
        "replacement_cluster_lensing_target_opened": False,
        "integrated_fits": [
            {
                "cluster": cluster,
                "fit_completed": True,
                "gates": {"all_passed": True},
                "parameters": {"abundance_solar": abundance},
            }
            for cluster, abundance in (("BULLET", 0.3), ("ABELL2146", 0.2))
        ],
    }
    path = tmp_path / "x2.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    _, abundances = runner.validate_x2_authorization(config, path)
    assert abundances == {"BULLET": 0.3, "ABELL2146": 0.2}

    report["integrated_fits"][0]["gates"]["all_passed"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrated fit did not pass"):
        runner.validate_x2_authorization(config, path)


def test_profile_interval_converts_sherpa_deltas_and_rejects_bad_order() -> None:
    result = SimpleNamespace(
        parnames=["xsapec.component.norm"], parmins=[-0.25], parmaxes=[0.5]
    )
    lower, upper, width = runner.confidence_interval_from_profile(
        1.0, "xsapec.component.norm", result
    )
    assert (lower, upper, width) == pytest.approx((0.75, 1.5, 0.375))

    result.parmins = [0.0]
    _, _, width = runner.confidence_interval_from_profile(
        1.0, "xsapec.component.norm", result
    )
    assert math.isnan(width)


def test_regional_fit_profiles_apec_normalization(monkeypatch) -> None:
    combination = {"label": "BULLET_bin7"}
    monkeypatch.setattr(runner.inherited_v19x, "fit_spectrum", passing_fit)

    class FakeSherpaError(Exception):
        pass

    norm = SimpleNamespace(fullname="xsapec.apec_bullet_bin7.norm")
    thermal = SimpleNamespace(norm=norm)
    profile = SimpleNamespace(
        parnames=[norm.fullname],
        parmins=[-0.0002],
        parmaxes=[0.0004],
    )
    ui = SimpleNamespace(
        get_model_component=lambda name: thermal,
        set_conf_opt=lambda *_args: None,
        conf=lambda *_args: None,
        get_conf_results=lambda: profile,
    )
    sherpa = ModuleType("sherpa")
    astro = ModuleType("sherpa.astro")
    astro.ui = ui
    utils = ModuleType("sherpa.utils")
    errors = ModuleType("sherpa.utils.err")
    errors.SherpaErr = FakeSherpaError
    sherpa.astro = astro
    sherpa.utils = utils
    utils.err = errors
    for name, module in (
        ("sherpa", sherpa),
        ("sherpa.astro", astro),
        ("sherpa.utils", utils),
        ("sherpa.utils.err", errors),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    fit = runner.fit_regional_gas_spectrum({}, "BULLET", combination, 0.3)
    interval = fit["normalization_confidence_68_percent"]
    assert interval["lower"] == pytest.approx(0.0008)
    assert interval["upper"] == pytest.approx(0.0014)
    assert fit["gates"]["finite_and_ordered_normalization_interval"]
    assert fit["gates"]["all_passed"]
