#!/usr/bin/env python3
"""Replay galaxy-selected spatial diffusion on ten cluster maps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, lens_source_map  # noqa: E402
from run_p0568_baryon_only_tensor_forward import build_contexts  # noqa: E402
from run_p0592_diffusive_propagator_transfer import baryon_r80, normalize_inside  # noqa: E402


def gain_fraction(candidate: float, local: float) -> float:
    return 1.0 - float(candidate) / float(local)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0594_galaxy_locked_cluster_replay_protocol.json").read_text(encoding="utf-8")
    )
    p0592 = json.loads((ROOT / protocol["source_protocols"][1]).read_text(encoding="utf-8"))
    p0568 = json.loads((ROOT / p0592["data"]["p0568_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / p0592["data"]["p0567_protocol"]).read_text(encoding="utf-8"))
    contexts = build_contexts(p0568, p0567)
    development = set(p0592["data"]["development_systems"])
    holdouts = set(p0592["data"]["holdout_systems"])
    spacing = float(p0592["preprocessing"]["grid_spacing_kpc"])
    candidates = protocol["candidates"]
    cache: dict[tuple[str, str], np.ndarray] = {}
    system_rows: list[dict] = []
    uncertainty_rows: list[dict] = []
    glafic_rows: list[dict] = []

    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        r80 = baryon_r80(context.data)
        local = normalize_inside(
            deposit_baryons(context.data, p0592["preprocessing"]["base_baryon_smoothing_kpc"]),
            context.aperture,
        )
        for name, spec in candidates.items():
            if name == "local":
                prediction = local
            else:
                endpoint = normalize_inside(
                    gaussian_filter(local, float(spec["q_R80"]) * r80 / spacing, mode="constant"),
                    context.aperture,
                )
                fraction = float(spec["routed_fraction"])
                prediction = normalize_inside(
                    (1.0 - fraction) * local + fraction * endpoint, context.aperture
                )
            cache[(label, name)] = prediction
            system_rows.append(
                {
                    "system": label,
                    "cohort": cohort,
                    "candidate": name,
                    "R80_kpc": r80,
                    "q_R80": float(spec["q_R80"]),
                    "routed_fraction": float(spec["routed_fraction"]),
                    **shape_metrics(prediction, context.target, context.aperture),
                }
            )
            glafic_rows.append(
                {
                    "system": label,
                    "cohort": cohort,
                    "candidate": name,
                    **shape_metrics(prediction, context.glafic_target, context.aperture),
                }
            )
        for realization, raw in enumerate(context.data.range_maps):
            target = lens_source_map(raw, context.data.radius, spacing, 20.0, (250.0, 300.0))
            metrics = {
                name: shape_metrics(cache[(label, name)], target, context.aperture)["jensen_shannon"]
                for name in candidates
            }
            uncertainty_rows.append(
                {
                    "system": label,
                    "cohort": cohort,
                    "realization": realization,
                    **{f"{name}_jsd": value for name, value in metrics.items()},
                    "galaxy_locked_improves_local": metrics["galaxy_locked"] < metrics["local"],
                    "galaxy_locked_beats_cluster_locked": metrics["galaxy_locked"] < metrics["cluster_locked"],
                }
            )

    system_scores = pd.DataFrame(system_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    glafic = pd.DataFrame(glafic_rows)
    holdout_system = system_scores[system_scores.cohort == "holdout"].pivot(
        index="system", columns="candidate", values="jensen_shannon"
    )
    means = holdout_system.mean()
    holdout_gain = gain_fraction(means.galaxy_locked, means.local)
    cluster_gain = gain_fraction(means.cluster_locked, means.local)
    retained = holdout_gain / cluster_gain
    systems_improved = int(np.sum(holdout_system.galaxy_locked < holdout_system.local))
    holdout_uncertainty = uncertainty[uncertainty.cohort == "holdout"]
    realization_improved_fraction = float(holdout_uncertainty.galaxy_locked_improves_local.mean())
    realization_beats_cluster_fraction = float(holdout_uncertainty.galaxy_locked_beats_cluster_locked.mean())
    glafic_holdout = glafic[glafic.cohort == "holdout"].groupby("candidate").jensen_shannon.mean()
    glafic_gain = gain_fraction(glafic_holdout.galaxy_locked, glafic_holdout.local)

    gate_cfg = protocol["interpretation_gates"]
    gates = {
        "holdout_mean_improvement_vs_local_pass": bool(
            holdout_gain >= gate_cfg["holdout_mean_jsd_improvement_fraction_vs_local_min"]
        ),
        "holdout_system_count_pass": bool(
            systems_improved >= gate_cfg["holdout_systems_improved_vs_local_min"]
        ),
        "holdout_realization_fraction_pass": bool(
            realization_improved_fraction
            >= gate_cfg["holdout_realizations_improved_vs_local_fraction_min"]
        ),
        "holdout_glafic_improvement_pass": bool(
            glafic_gain >= gate_cfg["holdout_glafic_improvement_fraction_vs_local_min"]
        ),
        "cluster_locked_gain_retained_pass": bool(
            retained >= gate_cfg["holdout_cluster_locked_gain_retained_fraction_min"]
        ),
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    system_scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    glafic.to_csv(output / protocol["outputs"]["glafic_scores"], index=False)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    holdout_system.plot(kind="bar", ax=axes[0])
    axes[0].set(ylabel="Jensen-Shannon distance", title="three cluster holdouts")
    axes[0].tick_params(axis="x", rotation=20)
    all_means = system_scores.groupby(["cohort", "candidate"]).jensen_shannon.mean().unstack()
    all_means.plot(kind="bar", ax=axes[1])
    axes[1].set(ylabel="mean Jensen-Shannon distance", title="cohort transfer")
    axes[1].tick_params(axis="x", rotation=0)
    figure.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0594-GALAXY-LOCKED-CLUSTER-REPLAY-RESULTS-0.1.0",
        "status": "complete_cluster_replay",
        "coverage": {
            "clusters": len(contexts),
            "development": len(development),
            "holdout": len(holdouts),
            "candidates": len(candidates),
            "lenstool_realizations": len(uncertainty),
        },
        "galaxy_locked_candidate": candidates["galaxy_locked"],
        "cluster_gate_treatment": protocol["cluster_gate_treatment"],
        "holdout_mean_jsd": {key: float(value) for key, value in means.items()},
        "holdout_improvement_fraction_vs_local": holdout_gain,
        "cluster_locked_improvement_fraction_vs_local": cluster_gain,
        "cluster_locked_gain_retained_fraction": retained,
        "holdout_systems_improved_vs_local": systems_improved,
        "holdout_realizations_improved_vs_local_fraction": realization_improved_fraction,
        "holdout_realizations_beating_cluster_locked_fraction": realization_beats_cluster_fraction,
        "glafic_holdout_mean_jsd": {key: float(value) for key, value in glafic_holdout.items()},
        "glafic_holdout_improvement_fraction_vs_local": glafic_gain,
        "gates": gates,
        "all_interpretation_gates_pass": bool(all(gates.values())),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0594 galaxy-locked cluster replay\n\n"
        f"The galaxy-selected q=0.75, f=0.25 spatial law improves the three cluster holdouts by "
        f"{100.0 * holdout_gain:.2f}% versus local baryon light and improves {systems_improved}/3 systems. "
        f"It retains {100.0 * retained:.1f}% of the morphology gain achieved by the cluster-selected "
        f"q=0.5, f=1 law. It improves {100.0 * realization_improved_fraction:.1f}% of the 300 holdout "
        f"Lenstool realizations and changes the independent GLAFIC score by {100.0 * glafic_gain:+.2f}%. "
        "The cluster acceleration gate was unavailable and was set to its favorable S=1 asymptote.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
