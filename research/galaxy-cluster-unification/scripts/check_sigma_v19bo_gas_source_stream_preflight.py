#!/usr/bin/env python3
"""Commission the V19BO gas posterior-to-region streaming layer."""

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

from voidscreen.sigma_gas_source_stream import gas_feature_batch, smooth_masked_draws

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bo_gas_source_stream_preflight.json"


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
            raise RuntimeError(f"V19BO parent changed: {name}")
        hashes[name] = actual
    implementation = config["implementation"]
    checker = ROOT / implementation["checker"]
    module = ROOT / implementation["module"]
    if checker.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19BO configuration names another checker")
    if sha256(checker) != implementation["checker_sha256"]:
        raise RuntimeError("V19BO checker changed after freeze")
    if sha256(module) != implementation["module_sha256"]:
        raise RuntimeError("V19BO module changed after freeze")

    axis = np.arange(-50.0, 51.0, 10.0)
    labels = np.arange(axis.size**2).reshape(axis.size, axis.size)
    ids = labels.ravel()
    east, north = np.meshgrid(axis, axis)
    draws = 4
    fields = {
        "electron_density_cm3": np.stack([np.exp(0.005 * east) * (1 + 0.01 * d) for d in range(draws)]).reshape(draws, -1),
        "entropy_proxy_keV_cm2": np.stack([np.exp(0.004 * north) * (1 + 0.01 * d) for d in range(draws)]).reshape(draws, -1),
        "thermal_pressure_erg_cm3": np.stack([np.exp(0.003 * (east + north)) * (1 + 0.01 * d) for d in range(draws)]).reshape(draws, -1),
        "gas_surface_density_msun_kpc2": np.stack([np.exp(0.002 * east) * (1 + 0.01 * d) for d in range(draws)]).reshape(draws, -1),
    }
    features = gas_feature_batch(
        fields,
        region_ids=ids,
        label_grid=labels,
        east_axis_kpc=axis,
        north_axis_kpc=axis,
        spacing_kpc=10.0,
        smoothing_fwhm_kpc=[20.0, 40.0],
        radii_kpc=[30.0, 50.0],
    )
    masked = np.full((2, 15, 15), np.nan)
    masked[0, 2:13, 2:13] = 1.0
    masked[1, 2:13, 2:13] = 3.0
    smoothed = smooth_masked_draws(masked, sigma_pixels=1.5, conserve_integral=True)
    mass_error = float(
        np.max(
            np.abs(
                np.nansum(smoothed, axis=(-2, -1))
                / np.nansum(masked, axis=(-2, -1))
                - 1.0
            )
        )
    )
    i5 = np.concatenate(
        [value[np.isfinite(value)] for name, value in features.items() if name.startswith("i5_baroclinicity")]
    )
    gates = {
        "all_parent_and_implementation_hashes_exact": bool(hashes),
        "exact_14_features_four_variants": len(features) == 14 * 2 * 2,
        "every_feature_has_draw_region_shape": all(value.shape == (draws, ids.size) for value in features.values()),
        "surface_density_conserved_to_1e_12": mass_error <= 1.0e-12,
        "I5_finite_values_bounded_zero_one": bool(np.all((i5 >= 0.0) & (i5 <= 1.0))),
        "observed_and_target_payloads_sealed": (
            not config["authorization"]["read_terminal_v19x4_now"]
            and not config["authorization"]["read_lensing_or_halo_payload"]
            and not config["authorization"]["select_action_or_gravity_parameter"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "decision": "passed_gas_source_stream_preflight" if all(gates.values()) else "failed_closed",
        "config_sha256": sha256(config_path),
        "checker_sha256": sha256(Path(__file__).resolve()),
        "module_sha256": sha256(module),
        "input_hashes": hashes,
        "manufactured_feature_arrays": len(features),
        "manufactured_surface_density_relative_error": mass_error,
        "gates": gates,
        "terminal_v19x4_opened": False,
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
