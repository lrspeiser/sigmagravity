#!/usr/bin/env python3
"""Map the exact two-sided coupling response of the P0559 physical tensor."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import build_contexts, fit_context  # noqa: E402
from run_p0557_baryon_proxy_tidal import (  # noqa: E402
    build_candidate_context,
    json_safe,
    sha256,
)
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)


def equal_system_rms(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def coupling_id(value):
    return f"t_{float(value):+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def main(config_path=None):
    config_path = Path(config_path) if config_path else ROOT / "configs/p0560_accept_tensor_coupling_response_protocol.json"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    protocol = json.loads(config_path.read_text())
    if not protocol["status"].startswith("frozen_before_any_"):
        raise RuntimeError("P0560 protocol is not frozen")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text())
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    member = json.loads((ROOT / protocol["inputs"]["member_tidal_protocol"]).read_text())
    member["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    locked = protocol["locked_map"]
    contexts, _, input_hashes = build_contexts(
        member, softening_kpc=float(locked["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    catalogs, physical_audits = physical_catalogs(p0559, contexts, registered)
    catalog_key = (
        locked["gas_normalization"],
        float(locked["gas_power"]),
        bool(locked["include_stars"]),
    )
    operator = {"operator_id": "contrast", "subtract_circular_mean": True}
    tensor_audits = []
    tensor_contexts = {}
    for base in contexts:
        label = base.system["label"]
        tensor_contexts[label] = build_candidate_context(
            base,
            catalogs[label][catalog_key],
            p0557,
            "accept_absolute_sqrt",
            operator,
            pixels_per_axis=int(locked["pixels_per_axis"]),
            softening_kpc=float(locked["softening_kpc"]),
            audit_rows=tensor_audits,
            stage=f"{protocol['protocol_version'].split('-')[0].lower()}_exact_response",
        )

    maximum_q = {
        row["system_label"]: float(row["maximum_Q_eigenvalue"])
        for row in tensor_audits
    }
    minimum_margin = float(protocol.get("ellipticity", {}).get("minimum_eigenvalue", 0.0))
    for coupling in map(float, protocol["coupling_grid"]):
        for label, value in maximum_q.items():
            minimum_eigenvalue = 1.0 - abs(coupling) * value
            if minimum_eigenvalue <= minimum_margin:
                raise ValueError(
                    f"coupling {coupling:g} violates ellipticity margin for "
                    f"{label}: {minimum_eigenvalue:g} <= {minimum_margin:g}"
                )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []
    grid = [float(value) for value in protocol["coupling_grid"]]
    starts = int(protocol["optimization"]["starts_per_exact_fit"])
    seed = int(protocol["optimization"]["random_seed"])
    for coupling_index, coupling in enumerate(grid):
        for system_index, base in enumerate(contexts):
            label = base.system["label"]
            context = base if coupling == 0.0 else tensor_contexts[label]
            print(f"P0560 exact response: {label} t={coupling:+.2f}", flush=True)
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=seed + coupling_index * 100 + system_index,
            )
            score = fitted["heldout"]
            rows.append(
                {
                    "coupling": coupling,
                    "coupling_id": coupling_id(coupling),
                    "system_label": label,
                    "heldout_exact_RMS_arcsec": score["exact_radial_RMS_arcsec"],
                    "all_heldout_roots": score["all_roots_converged"],
                    "fit_success": bool(fitted["fit"]["result"].success),
                    "fit_cost": float(fitted["fit"]["result"].cost),
                    "minimum_permittivity_eigenvalue": (
                        1.0 - abs(coupling) * maximum_q[label]
                    ),
                }
            )
            for frame in (fitted["training_predictions"], fitted["heldout_predictions"]):
                frame = frame.copy()
                frame["coupling"] = coupling
                predictions.append(frame)

    scores = pd.DataFrame(rows)
    zero = scores[scores.coupling.eq(0.0)].set_index("system_label")
    scores["improvement_fraction_vs_zero"] = [
        1.0
        - float(row.heldout_exact_RMS_arcsec)
        / float(zero.loc[row.system_label, "heldout_exact_RMS_arcsec"])
        for row in scores.itertuples(index=False)
    ]
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(tensor_audits).to_csv(
        output / protocol["outputs"]["tensor_audits"], index=False
    )

    aggregate_rows = []
    for coupling, group in scores.groupby("coupling", sort=True):
        aggregate_rows.append(
            {
                "coupling": float(coupling),
                "equal_system_exact_RMS_arcsec": equal_system_rms(
                    group.heldout_exact_RMS_arcsec
                ),
                "all_roots": bool(group.all_heldout_roots.all()),
                "all_systems_improve": bool(
                    (group.improvement_fraction_vs_zero > 0.0).all()
                ) if coupling != 0.0 else False,
                "mean_improvement_fraction": float(
                    group.improvement_fraction_vs_zero.mean()
                ),
            }
        )
    aggregates = pd.DataFrame(aggregate_rows).sort_values("coupling")
    aggregate_zero = float(
        aggregates.loc[
            aggregates.coupling.eq(0.0), "equal_system_exact_RMS_arcsec"
        ].iloc[0]
    )
    aggregates["aggregate_improvement_fraction_vs_zero"] = (
        1.0 - aggregates.equal_system_exact_RMS_arcsec / aggregate_zero
    )

    summary_rows = []
    near_negative, near_positive = map(
        float, protocol["response_checks"]["near_zero_pair"]
    )
    for label, group in scores.groupby("system_label", sort=True):
        complete = group[group.all_heldout_roots].sort_values(
            ["heldout_exact_RMS_arcsec", "coupling"]
        )
        best = complete.iloc[0]
        negative = group[group.coupling.eq(near_negative)].iloc[0]
        positive = group[group.coupling.eq(near_positive)].iloc[0]
        local_slope = (
            float(positive.heldout_exact_RMS_arcsec)
            - float(negative.heldout_exact_RMS_arcsec)
        ) / (near_positive - near_negative)
        summary_rows.append(
            {
                "system_label": label,
                "zero_RMS_arcsec": float(zero.loc[label, "heldout_exact_RMS_arcsec"]),
                "best_grid_coupling": float(best.coupling),
                "best_grid_RMS_arcsec": float(best.heldout_exact_RMS_arcsec),
                "best_grid_improvement_fraction": float(
                    best.improvement_fraction_vs_zero
                ),
                "near_zero_dRMS_dt_arcsec": local_slope,
                "near_zero_preferred_sign": (
                    "positive" if local_slope < 0.0 else "negative" if local_slope > 0.0 else "flat"
                ),
            }
        )
    response = pd.DataFrame(summary_rows)

    loo_rows = []
    labels = sorted(scores.system_label.unique())
    for heldout_label in labels:
        discovery = scores[scores.system_label.ne(heldout_label)]
        choices = []
        for coupling, group in discovery.groupby("coupling"):
            if bool(group.all_heldout_roots.all()):
                choices.append(
                    (equal_system_rms(group.heldout_exact_RMS_arcsec), float(coupling))
                )
        discovery_rms, chosen = min(choices)
        held = scores[
            scores.system_label.eq(heldout_label) & scores.coupling.eq(chosen)
        ].iloc[0]
        loo_rows.append(
            {
                "heldout_system": heldout_label,
                "chosen_coupling_from_other_three": chosen,
                "other_three_RMS_arcsec": discovery_rms,
                "heldout_RMS_arcsec": float(held.heldout_exact_RMS_arcsec),
                "heldout_all_roots": bool(held.all_heldout_roots),
                "heldout_improvement_fraction_vs_zero": float(
                    held.improvement_fraction_vs_zero
                ),
            }
        )
    leave_one_out = pd.DataFrame(loo_rows)
    response.to_csv(output / protocol["outputs"]["response_summary"], index=False)
    leave_one_out.to_csv(output / protocol["outputs"]["leave_one_out"], index=False)

    complete_common = aggregates[aggregates.all_roots].sort_values(
        ["equal_system_exact_RMS_arcsec", "coupling"]
    )
    common_best = complete_common.iloc[0]
    nonzero_common = complete_common[~complete_common.coupling.eq(0.0)].iloc[0]
    sign_counts = response.near_zero_preferred_sign.value_counts().to_dict()
    report = {
        "report_version": protocol["protocol_version"].replace(
            "-0.1.0", "-RESULTS-0.1.0"
        ),
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "input_hashes": input_hashes,
        "aggregate_response": aggregates.to_dict("records"),
        "per_system_response": response.to_dict("records"),
        "leave_one_out": leave_one_out.to_dict("records"),
        "primary": {
            "zero_RMS_arcsec": aggregate_zero,
            "best_common_coupling": float(common_best.coupling),
            "best_common_RMS_arcsec": float(common_best.equal_system_exact_RMS_arcsec),
            "best_common_improvement_fraction": float(
                common_best.aggregate_improvement_fraction_vs_zero
            ),
            "best_nonzero_common_coupling": float(nonzero_common.coupling),
            "best_nonzero_common_RMS_arcsec": float(
                nonzero_common.equal_system_exact_RMS_arcsec
            ),
            "best_nonzero_common_improvement_fraction": float(
                nonzero_common.aggregate_improvement_fraction_vs_zero
            ),
            "best_nonzero_all_systems_improve": bool(
                nonzero_common.all_systems_improve
            ),
            "near_zero_preferred_sign_counts": sign_counts,
            "leave_one_out_all_improve": bool(
                (leave_one_out.heldout_improvement_fraction_vs_zero > 0.0).all()
            ),
        },
        "physical_map_audits": physical_audits.to_dict("records"),
        "verdict": {
            "common_sign_supported": len(sign_counts) == 1,
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for label, group in scores.groupby("system_label"):
        group = group.sort_values("coupling")
        axes[0].plot(
            group.coupling,
            100.0 * group.improvement_fraction_vs_zero,
            marker="o",
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0].set(
        xlabel="universal tensor coupling t",
        ylabel="exact held-out improvement vs t=0 (%)",
        title="Cluster-specific response",
    )
    axes[0].legend()
    axes[1].plot(
        aggregates.coupling,
        100.0 * aggregates.aggregate_improvement_fraction_vs_zero,
        marker="o",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axvline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[1].set(
        xlabel="universal tensor coupling t",
        ylabel="four-cluster exact improvement vs t=0 (%)",
        title="Common-coupling response",
    )
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    (output / protocol["outputs"]["summary"]).write_text(
        f"""# {protocol['protocol_version'].split('-')[0]} physical tensor coupling response

Best common grid coupling: {float(common_best.coupling):+.2f}; exact RMS
{float(common_best.equal_system_exact_RMS_arcsec):.3f} arcsec versus
{aggregate_zero:.3f} at zero. Near-zero preferred signs: {sign_counts}.
Leave-one-system-out transfers all improve: {bool((leave_one_out.heldout_improvement_fraction_vs_zero > 0.0).all())}.
This is a spent-sample diagnostic; no formula is promoted.
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["primary"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/p0560_accept_tensor_coupling_response_protocol.json",
    )
    args = parser.parse_args()
    main(args.config)
