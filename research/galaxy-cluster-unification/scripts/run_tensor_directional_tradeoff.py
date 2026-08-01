#!/usr/bin/env python3
"""Exact post-failure refits for moderate tensor directionality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cpr0_accept_clash_bridge import (  # noqa: E402
    assign_group_folds,
    domain_metrics,
    equal_group_domain_mse,
)
from run_sigma_field_exploration import run_diagnostic_lensing  # noqa: E402
from run_sparc_independent_nuisance_refit import build_frame  # noqa: E402
from run_vector_completion_full_test import (  # noqa: E402
    at_boundary,
    completion_summary,
    json_safe,
    raw_lensing_profile,
    run_sparc_transfer,
    solar_diagnostics,
)
from voidscreen.tensor_completion import (  # noqa: E402
    predict_tensor_acceleration,
    spherical_tidal_eigenvalues,
)


def predict(frame: pd.DataFrame, base_model: str, fixed_q: float, free) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(float))
    eigenvalues = spherical_tidal_eigenvalues(
        gbar,
        frame["radius_kpc"].to_numpy(float),
        frame["local_density_g_cm3"].to_numpy(float),
    )
    parameters = np.asarray([*np.asarray(free, dtype=float), fixed_q])
    result = predict_tensor_acceleration(gbar, eigenvalues, base_model, parameters)
    return np.log10(result["predicted_acceleration_m_s2"])


def fit(frame: pd.DataFrame, option: dict, protocol: dict, seed: int) -> np.ndarray:
    bounds = list(map(tuple, option["bounds"]))
    objective = lambda free: equal_group_domain_mse(
        frame,
        predict(frame, option["base_model"], float(option["fixed_q"]), free),
    )
    settings = protocol["optimization"]
    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(settings["differential_evolution_maxiter"]),
        popsize=int(settings["differential_evolution_popsize"]),
        tol=1.0e-10,
        polish=False,
        workers=1,
    )
    local = minimize(
        objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 8000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local.x if local.success else global_fit.x, dtype=float)


def cross_validate(frame: pd.DataFrame, option: dict, protocol: dict):
    predictions = np.full(len(frame), np.nan)
    fits = []
    for fold in range(int(protocol["bridge_folds"])):
        training = frame[frame["fold"] != fold]
        heldout = frame[frame["fold"] == fold]
        free = fit(training, option, protocol, int(protocol["bridge_seed"]) + fold)
        predictions[heldout.index] = predict(
            heldout, option["base_model"], float(option["fixed_q"]), free
        )
        fits.append(
            {
                "fold": fold,
                "free_parameters": dict(zip(option["parameters"], map(float, free), strict=True)),
            }
        )
    full = fit(frame, option, protocol, int(protocol["bridge_seed"]) + 100)
    return predictions, fits, full


def main() -> None:
    config_path = ROOT / "configs/tensor_directional_tradeoff_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "postfailure_frozen_before_exact_refits":
        raise RuntimeError("post-failure options were not frozen before exact refits")
    primary_protocol = json.loads(
        (ROOT / protocol["inputs"]["primary_protocol"]).read_text(encoding="utf-8")
    )
    primary_report = json.loads(
        (ROOT / protocol["inputs"]["primary_report"]).read_text(encoding="utf-8")
    )
    sample = pd.read_csv(ROOT / primary_protocol["inputs"]["bridge_sample"])
    sample = assign_group_folds(
        sample.drop(columns=["fold"], errors="ignore"),
        int(protocol["bridge_folds"]),
        int(protocol["bridge_seed"]),
    )
    sparc_protocol = json.loads(
        (ROOT / primary_protocol["inputs"]["SPARC_protocol"]).read_text(encoding="utf-8")
    )
    sparc_frame = build_frame(
        sparc_protocol,
        ROOT / primary_protocol["inputs"]["SPARC_raw"],
        ROOT / primary_protocol["inputs"]["SPARC_morphology"],
    )

    results = {}
    bridge_tables = []
    sparc_tables = []
    raw_tables = []
    for name, option in protocol["options"].items():
        base_model = option["base_model"]
        fixed_q = float(option["fixed_q"])
        print(f"exact bridge refit option={name}", flush=True)
        heldout, fold_fits, free = cross_validate(sample, option, protocol)
        parameters = np.asarray([*free, fixed_q])
        bridge_metrics = domain_metrics(sample, heldout)
        bridge_table = sample.copy()
        bridge_table.insert(0, "option", name)
        bridge_table["predicted_log_gobs"] = heldout
        bridge_table["residual_dex"] = heldout - bridge_table["log_gobs"]
        bridge_tables.append(bridge_table)

        print(f"fresh SPARC nuisance fits option={name}", flush=True)
        sparc_points, sparc_metrics = run_sparc_transfer(
            base_model, parameters, sparc_protocol, sparc_frame
        )
        sparc_points.insert(0, "tradeoff_option", name)
        sparc_tables.append(sparc_points)

        print(f"raw lensing option={name}", flush=True)
        profile = raw_lensing_profile(base_model, parameters, primary_protocol)
        raw_predictions, raw_summary = run_diagnostic_lensing(
            pd.Series(dict(zip([*option["parameters"], "q"], parameters, strict=True))),
            primary_protocol,
            profile,
        )
        raw_predictions["tradeoff_option"] = name
        raw_tables.append(raw_predictions)

        results[name] = {
            "base_model": base_model,
            "fixed_q": fixed_q,
            "selection_disclosure": protocol["disclosure"],
            "fold_fits": fold_fits,
            "full_parameters": dict(
                zip([*option["parameters"], "q"], map(float, parameters), strict=True)
            ),
            "full_fit_at_boundary": dict(
                zip(
                    option["parameters"],
                    at_boundary(
                        free,
                        option["bounds"],
                        float(protocol["optimization"]["boundary_fraction_tolerance"]),
                    ),
                    strict=True,
                )
            ),
            "bridge_metrics": bridge_metrics,
            "completion": completion_summary(base_model, parameters, sample),
            "solar": solar_diagnostics(base_model, parameters, primary_protocol),
            "SPARC_metrics": sparc_metrics,
            "raw_lensing": raw_summary,
        }

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed exploratory post-failure directional refits",
        "disclosure": protocol["disclosure"],
        "coverage": primary_report["coverage"],
        "results": results,
        "references": primary_report["references"],
        "claim": "tradeoff diagnostic only; not independent model selection",
    }
    output = ROOT / Path(protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(bridge_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["bridge_predictions"], index=False
    )
    pd.concat(sparc_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["SPARC_predictions"], index=False
    )
    pd.concat(raw_tables, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["raw_lensing_predictions"], index=False
    )
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Post-failure tensor directionality tradeoff",
        "",
        protocol["disclosure"],
        "",
        "| option | bridge (dex) | cluster (dex) | SPARC outer (km/s) | raw lensing (arcsec) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['bridge_metrics']['equal_domain_RMSE_dex']:.4f} | "
            f"{result['bridge_metrics']['cluster']['equal_system_RMSE_dex']:.4f} | "
            f"{result['SPARC_metrics']['outer_holdout']['RMSE_km_s']:.3f} | "
            f"{result['raw_lensing']['heldout']['exact_radial_RMS_arcsec']:.3f} |"
        )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
