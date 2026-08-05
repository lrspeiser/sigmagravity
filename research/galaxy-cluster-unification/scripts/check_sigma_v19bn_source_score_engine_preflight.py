#!/usr/bin/env python3
"""Commission the frozen V19BN posterior source-score decision engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voidscreen.sigma_source_score_engine import (
    i4_draw_summary,
    joint_variant_draw_pass_fraction,
    leave_one_region_out_stability,
    posterior_novelty_scores,
)

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bn_source_score_engine_preflight.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19BN parent changed: {name}")
        hashes[name] = actual
    implementation = config["implementation"]
    checker = ROOT / implementation["checker"]
    engine = ROOT / implementation["engine_module"]
    if checker.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BN configuration names another checker")
    if sha256(checker) != implementation["checker_sha256"]:
        raise RuntimeError("V19BN checker changed after freeze")
    if sha256(engine) != implementation["engine_module_sha256"]:
        raise RuntimeError("V19BN engine changed after freeze")

    rng = np.random.default_rng(1914)
    draws, regions = 16, 80
    controls = rng.normal(size=(draws, regions, 5))
    controlled = 3.0 * controls[..., 0] - 2.0 * controls[..., 1]
    independent = rng.normal(size=(draws, regions))
    support = np.ones(regions, dtype=bool)
    controlled_score = posterior_novelty_scores(
        controls, controlled, support, minimum_unexplained_fraction=0.2
    )
    independent_score = posterior_novelty_scores(
        controls, independent, support, minimum_unexplained_fraction=0.2
    )
    plus = np.ones((draws, regions))
    cross = np.zeros_like(plus)
    primary = i4_draw_summary(plus, cross, support)
    close = i4_draw_summary(1.02 * plus, cross, support)
    rotated = i4_draw_summary(np.zeros_like(plus), plus, support)
    loo = leave_one_region_out_stability(
        np.stack([plus, cross], axis=-1),
        support,
        candidate="I4",
        maximum_activation_change_fraction=0.1,
        maximum_axis_change_deg=10.0,
    )
    evidence = {
        "controlled_novelty_pass_fraction": controlled_score["pass_fraction"],
        "independent_novelty_pass_fraction": independent_score["pass_fraction"],
        "close_variant_pass_fraction": joint_variant_draw_pass_fraction(
            primary,
            [close],
            maximum_activation_change_fraction=0.1,
            maximum_axis_change_deg=10.0,
        ),
        "rotated_variant_pass_fraction": joint_variant_draw_pass_fraction(
            primary,
            [rotated],
            maximum_activation_change_fraction=0.1,
            maximum_axis_change_deg=10.0,
        ),
        "uniform_tensor_leave_one_out_pass_fraction": loo["pass_fraction"],
    }
    bl = load_json(ROOT / config["parents"]["v19bl_config"]["path"])
    threshold = config["inherited_thresholds"]
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hashes),
        "thresholds_match_v19bl": (
            threshold["minimum_gradient_detection_sigma"]
            == bl["gradient_support"]["minimum_gradient_detection_sigma"]
            and threshold["minimum_supported_regions"]
            == bl["gradient_support"]["minimum_supported_regions"]
            and threshold["minimum_unexplained_variance_fraction"]
            == bl["density_novelty_control"]["minimum_unexplained_variance_fraction"]
        ),
        "controlled_response_rejected": evidence["controlled_novelty_pass_fraction"] == 0.0,
        "independent_response_admitted": evidence["independent_novelty_pass_fraction"] >= 0.9,
        "small_variant_admitted_large_rotation_rejected": (
            evidence["close_variant_pass_fraction"] == 1.0
            and evidence["rotated_variant_pass_fraction"] == 0.0
        ),
        "uniform_tensor_leave_one_out_stable": evidence[
            "uniform_tensor_leave_one_out_pass_fraction"
        ]
        == 1.0,
        "observed_scores_and_targets_sealed": (
            not config["authorization"]["compute_observed_score_now"]
            and not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["select_action_or_gravity_parameter"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "decision": "passed_source_score_engine_preflight" if all(gates.values()) else "failed_closed",
        "config_sha256": sha256(config_path),
        "checker_sha256": sha256(Path(__file__).resolve()),
        "engine_module_sha256": sha256(engine),
        "input_hashes": hashes,
        "manufactured_evidence": evidence,
        "gates": gates,
        "observed_source_score_computed": False,
        "lensing_halo_action_or_gravity_payload_opened": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    report = build_report(config_path)
    config = load_json(config_path)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, indent=2))
    if report["decision"] == "failed_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
