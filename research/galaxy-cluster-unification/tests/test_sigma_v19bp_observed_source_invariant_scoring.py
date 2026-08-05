from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
CONFIG = ROOT / "configs" / "sigma_v19bp_observed_source_invariant_scoring.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19bp_observed_source_invariant_scoring.py"
PREFLIGHT = (
    ROOT
    / "results"
    / "sigma_v19bp_observed_source_invariant_scoring"
    / "preflight_report.json"
)
SPEC = importlib.util.spec_from_file_location("sigma_v19bp", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def manufactured_features(draws: int = 64, regions: int = 48):
    rng = np.random.default_rng(1902)
    token = MODULE.variant_token(50.0, 350.0)
    region_phase = np.linspace(0.0, 2.0 * np.pi, regions, endpoint=False)[None, :]
    draw_phase = np.linspace(0.0, 2.0 * np.pi, draws, endpoint=False)[:, None]

    def field(name: str, values: np.ndarray) -> tuple[str, np.ndarray]:
        return f"{name}_{token}", np.asarray(values, dtype=float)

    noise = lambda scale: rng.normal(0.0, scale, size=(draws, regions))
    features = dict(
        [
            field(
                "electron_density_gradient_east_kpc_inv",
                0.045 + 0.004 * np.sin(region_phase) + noise(0.001),
            ),
            field(
                "electron_density_gradient_north_kpc_inv",
                0.025 + 0.003 * np.cos(region_phase) + noise(0.001),
            ),
            field(
                "entropy_gradient_east_kpc_inv",
                0.032 + 0.003 * np.cos(region_phase) + noise(0.001),
            ),
            field(
                "entropy_gradient_north_kpc_inv",
                0.014 + 0.002 * np.sin(region_phase) + noise(0.001),
            ),
            field(
                "pressure_gradient_east_kpc_inv",
                0.018 + 0.004 * np.sin(region_phase) + noise(0.001),
            ),
            field(
                "pressure_gradient_north_kpc_inv",
                0.037 + 0.003 * np.cos(region_phase) + noise(0.001),
            ),
            field(
                "i4_q_plus",
                0.60
                + 0.035 * np.sin(region_phase)
                + 0.006 * np.sin(draw_phase)
                + noise(0.002),
            ),
            field(
                "i4_q_cross",
                0.24
                + 0.025 * np.cos(region_phase)
                + 0.004 * np.cos(draw_phase)
                + noise(0.002),
            ),
            field(
                "i5_baroclinicity",
                np.clip(
                    0.43
                    + 0.05 * np.sin(2.0 * region_phase)
                    + 0.008 * np.cos(draw_phase)
                    + noise(0.003),
                    0.0,
                    1.0,
                ),
            ),
            field("control_log_gas_surface_density", noise(1.0)),
            field("control_log_surface_gradient", noise(1.0)),
            field("control_surface_hessian_trace", noise(1.0)),
            field("control_surface_hessian_anisotropy", noise(1.0)),
        ]
    )
    ranks = np.tile((np.arange(regions) + 0.5) / regions, (draws, 1))
    for draw in range(draws):
        ranks[draw] = ranks[draw, rng.permutation(regions)]
    return features, ranks


def test_manufactured_i4_and_i5_pass_the_terminal_variant_algebra() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    features, ranks = manufactured_features()
    i4, i4_arrays = MODULE.score_variant(
        features,
        ranks,
        fwhm_kpc=50.0,
        radius_kpc=350.0,
        candidate="I4",
        thresholds=config["thresholds"],
    )
    i5, i5_arrays = MODULE.score_variant(
        features,
        ranks,
        fwhm_kpc=50.0,
        radius_kpc=350.0,
        candidate="I5",
        thresholds=config["thresholds"],
    )
    assert i4["supported_regions"] == 48
    assert all(i4["direction_gates"].values())
    assert all(i4["amplitude_or_scalar_gates"].values())
    assert all(i5["amplitude_or_scalar_gates"].values())
    assert i4_arrays["response"].shape == (64, 48, 2)
    assert i5_arrays["response"].shape == (64, 48)


def test_variant_stability_separates_amplitude_and_direction() -> None:
    primary = {
        "activation": np.ones(32),
        "axis_deg": np.full(32, 20.0),
    }
    stable = MODULE.variant_stability(
        primary,
        [{"activation": np.full(32, 1.02), "axis_deg": np.full(32, 23.0)}],
        {
            "maximum_activation_change_fraction": 0.1,
            "maximum_axis_change_deg": 10.0,
        },
    )
    assert stable == {
        "activation_draw_pass_fraction": 1.0,
        "axis_draw_pass_fraction": 1.0,
        "joint_draw_pass_fraction": 1.0,
    }
    amplitude_failure = MODULE.variant_stability(
        primary,
        [{"activation": np.full(32, 1.3), "axis_deg": np.full(32, 23.0)}],
        {
            "maximum_activation_change_fraction": 0.1,
            "maximum_axis_change_deg": 10.0,
        },
    )
    assert amplitude_failure["activation_draw_pass_fraction"] == 0.0
    assert amplitude_failure["axis_draw_pass_fraction"] == 1.0


def test_i5_can_rescue_amplitude_but_never_direction() -> None:
    def branch(direction: bool, amplitude: bool, scalar: bool):
        return {
            "candidates": {
                "I4": {
                    "direction_pass": direction,
                    "amplitude_or_scalar_pass": amplitude,
                },
                "I5": {
                    "direction_pass": False,
                    "amplitude_or_scalar_pass": scalar,
                },
            }
        }

    rescued = MODULE.aggregate_source_decision(
        [branch(True, False, True) for _ in range(6)], 6
    )
    assert rescued["action_derivation_authorized"]
    no_direction = MODULE.aggregate_source_decision(
        [branch(False, True, True) for _ in range(6)], 6
    )
    assert not no_direction["action_derivation_authorized"]


def test_preflight_passes_with_terminal_and_target_payloads_sealed() -> None:
    report = MODULE.build_preflight_report(CONFIG)
    assert report["decision"].startswith("passed_observed_source_executor_preflight")
    assert all(report["gates"].values())
    assert not report["terminal_v19x4_or_v19bm_opened"]
    assert not report["observed_source_score_computed"]
    assert not report["lensing_halo_action_or_gravity_payload_opened"]


def test_frozen_preflight_report_matches_the_current_executor() -> None:
    frozen = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    rebuilt = MODULE.build_preflight_report(CONFIG)

    assert frozen == rebuilt
    assert frozen["runner_sha256"] == MODULE.sha256(RUNNER)
    assert frozen["config_sha256"] == MODULE.sha256(CONFIG)
    assert all(frozen["gates"].values())
