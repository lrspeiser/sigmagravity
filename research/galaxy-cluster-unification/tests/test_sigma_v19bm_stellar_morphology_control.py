from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bm_stellar_morphology_control.json"
REPORT = ROOT / "results" / "sigma_v19bm_stellar_morphology_control" / "preflight_report.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19bm_stellar_morphology_control.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19bm", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_passes_without_observed_gas_or_target_payload() -> None:
    report = load_runner().build_preflight_report(CONFIG)
    assert all(report["gates"].values())
    assert not report["observed_v19x4_gas_posterior_opened"]
    assert not report["stellar_control_computed"]
    assert not report["cross_filter_luminosity_amplitudes_compared"]
    assert not report["stellar_mass_inferred"]
    assert not report["lensing_halo_action_or_gravity_payload_opened"]


def test_frozen_preflight_report_matches_current_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == "passed_stellar_control_preflight_awaiting_terminal_v19x4"
    assert all(report["gates"].values())


def test_member_batches_preserve_sample_order_inventory_and_unit_light(tmp_path: Path) -> None:
    runner = load_runner()
    path = tmp_path / "members.csv.gz"
    fields = ["sample_id", "member_id", "ra_deg", "dec_deg", "relative_light"]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in range(3):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "member_id": "A",
                    "ra_deg": 5.0 + 0.1 * sample_id,
                    "dec_deg": 5.0,
                    "relative_light": 1.0,
                }
            )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "member_id": "B",
                    "ra_deg": 6.0,
                    "dec_deg": 5.0 + 0.1 * sample_id,
                    "relative_light": 3.0,
                }
            )

    class IdentityWCS:
        @staticmethod
        def world_to_pixel_values(ra, dec):
            return np.asarray(ra), np.asarray(dec)

    batches = list(
        runner.member_map_batches(
            path,
            cluster="SYNTHETIC",
            spec={
                "expected_members_per_draw": 2,
                "luminosity_field": "relative_light",
                "kpc_per_arcsec": 1.0,
            },
            wcs=IdentityWCS(),
            center={"logicalx": 6.0, "logicaly": 6.0},
            common_axis=np.arange(-10.0, 11.0),
            draws=3,
            batch_size=2,
            output_pixel_arcsec=1.0,
        )
    )
    assert [ids.tolist() for ids, _ in batches] == [[0, 1], [2]]
    maps = np.concatenate([values for _, values in batches])
    np.testing.assert_allclose(np.sum(maps, axis=(-2, -1)), 1.0, atol=1.0e-14)
