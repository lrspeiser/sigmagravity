#!/usr/bin/env python3
"""Direct observed-image response of the physical ACCEPT tensor."""

from __future__ import annotations

import json
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

from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    build_contexts,
    fixed_source_local_rms,
    model_name,
)
from run_p0557_baryon_proxy_tidal import (  # noqa: E402
    build_candidate_context,
    json_safe,
    sha256,
)
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)


GEOMETRY_LABELS = ["axis_ratio", "phi_radian", "center_x", "center_y", "gamma1", "gamma2"]


def rms(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def grid_from(protocol):
    rule = protocol["response_grid"]
    lo, hi, step = map(
        float,
        [rule["minimum_coupling"], rule["maximum_coupling"], rule["step"]],
    )
    count = int(round((hi - lo) / step)) + 1
    grid = lo + step * np.arange(count)
    if not np.isclose(grid[-1], hi):
        raise ValueError("response grid does not land on its maximum")
    return grid


def main():
    config_path = ROOT / "configs/p0562_accept_tensor_direct_response_protocol.json"
    protocol = json.loads(config_path.read_text())
    if not protocol["status"].startswith("frozen_before_any_"):
        raise RuntimeError("P0562 protocol is not frozen")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text())
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    member = json.loads((ROOT / protocol["inputs"]["member_tidal_protocol"]).read_text())
    member["optimization"]["maximum_function_evaluations"] = int(
        protocol["baseline_fit"]["maximum_function_evaluations"]
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
            stage="p0562_direct_response",
        )
    maximum_q = {
        row["system_label"]: float(row["maximum_Q_eigenvalue"])
        for row in tensor_audits
    }
    coupling_grid = grid_from(protocol)
    margin = float(protocol["ellipticity"]["minimum_eigenvalue"])
    if any(
        1.0 - abs(t) * maximum_q[label] <= margin
        for t in coupling_grid
        for label in maximum_q
    ):
        raise ValueError("response grid violates the ellipticity margin")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    geometry_rows = []
    response_rows = []
    seeds = list(map(int, protocol["baseline_fit"]["seed_ensembles"]))
    starts = int(protocol["baseline_fit"]["starts_per_seed_ensemble"])
    for ensemble_index, seed in enumerate(seeds):
        ensemble = f"seed_{ensemble_index + 1}"
        for system_index, context in enumerate(contexts):
            label = context.system["label"]
            tensor_context = tensor_contexts[label]
            zero_lens = MemberTidalLens(
                tensor_context.local_protocol,
                tensor_context.fields,
                tensor_context.correction,
                0.0,
            )
            print(f"P0562 zero geometry: {ensemble} {label}", flush=True)
            fitted = zero_lens.fit(
                model_name(0.0),
                tensor_context.training,
                starts=starts,
                seed=seed + system_index,
                initial_override=tensor_context.initial_geometry,
            )
            parameters = fitted["result"].x
            geometry_rows.append(
                {
                    "ensemble": ensemble,
                    "seed": seed + system_index,
                    "system_label": label,
                    "fit_cost": float(fitted["result"].cost),
                    "fit_success": bool(fitted["result"].success),
                    **dict(zip(GEOMETRY_LABELS, parameters, strict=True)),
                }
            )
            for coupling in coupling_grid:
                lens = MemberTidalLens(
                    tensor_context.local_protocol,
                    tensor_context.fields,
                    tensor_context.correction,
                    float(coupling),
                )
                name = model_name(float(coupling))
                residual, sources = lens.profiled_residuals(
                    name, parameters, tensor_context.training
                )
                training = residual.reshape(-1, 2) * lens.sigma
                training_rms = float(
                    np.sqrt(np.mean(np.sum(training * training, axis=1)))
                )
                heldout_rms = fixed_source_local_rms(
                    lens,
                    name,
                    parameters,
                    sources,
                    tensor_context.heldout,
                )
                response_rows.append(
                    {
                        "ensemble": ensemble,
                        "system_label": label,
                        "coupling": float(coupling),
                        "training_profiled_local_RMS_arcsec": training_rms,
                        "heldout_fixed_source_local_RMS_arcsec": heldout_rms,
                        "minimum_permittivity_eigenvalue": (
                            1.0 - abs(float(coupling)) * maximum_q[label]
                        ),
                    }
                )

    geometry = pd.DataFrame(geometry_rows)
    response = pd.DataFrame(response_rows)
    geometry.to_csv(output / protocol["outputs"]["geometry_ensembles"], index=False)
    response.to_csv(output / protocol["outputs"]["response_scores"], index=False)
    pd.DataFrame(tensor_audits).to_csv(
        output / protocol["outputs"]["tensor_audits"], index=False
    )

    aggregate_rows = []
    for (ensemble, coupling), group in response.groupby(["ensemble", "coupling"]):
        aggregate_rows.append(
            {
                "ensemble": ensemble,
                "coupling": float(coupling),
                "equal_system_training_RMS_arcsec": rms(
                    group.training_profiled_local_RMS_arcsec
                ),
                "equal_system_heldout_RMS_arcsec": rms(
                    group.heldout_fixed_source_local_RMS_arcsec
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / protocol["outputs"]["aggregate_response"], index=False)

    near_negative, near_positive = map(
        float, protocol["response_grid"]["near_zero_pair"]
    )
    system_rows = []
    for (ensemble, label), group in response.groupby(["ensemble", "system_label"]):
        group = group.sort_values("coupling")
        best = group.loc[group.heldout_fixed_source_local_RMS_arcsec.idxmin()]
        zero = group[group.coupling.eq(0.0)].iloc[0]
        negative = group[group.coupling.eq(near_negative)].iloc[0]
        positive = group[group.coupling.eq(near_positive)].iloc[0]
        slope = (
            float(positive.heldout_fixed_source_local_RMS_arcsec)
            - float(negative.heldout_fixed_source_local_RMS_arcsec)
        ) / (near_positive - near_negative)
        system_rows.append(
            {
                "ensemble": ensemble,
                "system_label": label,
                "zero_heldout_RMS_arcsec": float(
                    zero.heldout_fixed_source_local_RMS_arcsec
                ),
                "best_grid_coupling": float(best.coupling),
                "best_grid_heldout_RMS_arcsec": float(
                    best.heldout_fixed_source_local_RMS_arcsec
                ),
                "best_grid_improvement_fraction": 1.0
                - float(best.heldout_fixed_source_local_RMS_arcsec)
                / float(zero.heldout_fixed_source_local_RMS_arcsec),
                "near_zero_dRMS_dt_arcsec": slope,
                "near_zero_preferred_sign": (
                    "positive" if slope < 0.0 else "negative" if slope > 0.0 else "flat"
                ),
            }
        )
    per_system = pd.DataFrame(system_rows)
    per_system.to_csv(output / protocol["outputs"]["per_system_summary"], index=False)

    ensemble_summary = []
    for ensemble, group in aggregate.groupby("ensemble"):
        zero = group[group.coupling.eq(0.0)].iloc[0]
        best = group.loc[group.equal_system_heldout_RMS_arcsec.idxmin()]
        ensemble_summary.append(
            {
                "ensemble": ensemble,
                "zero_heldout_RMS_arcsec": float(zero.equal_system_heldout_RMS_arcsec),
                "best_common_coupling": float(best.coupling),
                "best_common_heldout_RMS_arcsec": float(
                    best.equal_system_heldout_RMS_arcsec
                ),
                "best_common_improvement_fraction": 1.0
                - float(best.equal_system_heldout_RMS_arcsec)
                / float(zero.equal_system_heldout_RMS_arcsec),
            }
        )
    ensemble_frame = pd.DataFrame(ensemble_summary)
    sign_table = per_system.pivot(
        index="system_label", columns="ensemble", values="near_zero_preferred_sign"
    )
    optimum_table = per_system.pivot(
        index="system_label", columns="ensemble", values="best_grid_coupling"
    )
    report = {
        "report_version": "P0562-ACCEPT-TENSOR-DIRECT-RESPONSE-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "input_hashes": input_hashes,
        "ensemble_summary": ensemble_frame.to_dict("records"),
        "per_system_summary": per_system.to_dict("records"),
        "primary": {
            "near_zero_sign_agreement_between_seed_ensembles": bool(
                (sign_table.iloc[:, 0] == sign_table.iloc[:, 1]).all()
            ),
            "grid_optimum_agreement_between_seed_ensembles": bool(
                np.allclose(optimum_table.iloc[:, 0], optimum_table.iloc[:, 1])
            ),
            "common_optimum_agreement_between_seed_ensembles": bool(
                ensemble_frame.best_common_coupling.nunique() == 1
            ),
            "all_systems_share_one_near_zero_sign": bool(
                per_system.near_zero_preferred_sign.nunique() == 1
            ),
        },
        "physical_map_audits": physical_audits.to_dict("records"),
        "verdict": {"formula_promoted": False},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for (ensemble, label), group in response.groupby(["ensemble", "system_label"]):
        group = group.sort_values("coupling")
        zero = float(
            group.loc[
                group.coupling.eq(0.0), "heldout_fixed_source_local_RMS_arcsec"
            ].iloc[0]
        )
        axes[0].plot(
            group.coupling,
            100.0 * (1.0 - group.heldout_fixed_source_local_RMS_arcsec / zero),
            label=f"{label} {ensemble}",
            alpha=0.8,
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0].set(
        xlabel="tensor coupling t",
        ylabel="held-out local improvement vs t=0 (%)",
        title="Fixed-geometry direct response",
    )
    axes[0].legend(fontsize=7, ncol=2)
    for ensemble, group in aggregate.groupby("ensemble"):
        group = group.sort_values("coupling")
        zero = float(
            group.loc[group.coupling.eq(0.0), "equal_system_heldout_RMS_arcsec"].iloc[0]
        )
        axes[1].plot(
            group.coupling,
            100.0 * (1.0 - group.equal_system_heldout_RMS_arcsec / zero),
            label=ensemble,
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axvline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[1].set(
        xlabel="tensor coupling t",
        ylabel="four-cluster held-out local improvement (%)",
        title="Common response by geometry basin",
    )
    axes[1].legend()
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    summary_lines = [
        "# P0562 direct physical-tensor response",
        "",
        "| Ensemble | Zero RMS | Best common t | Best RMS | Improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ensemble_frame.itertuples(index=False):
        summary_lines.append(
            f"| {row.ensemble} | {row.zero_heldout_RMS_arcsec:.4f} | "
            f"{row.best_common_coupling:+.2f} | "
            f"{row.best_common_heldout_RMS_arcsec:.4f} | "
            f"{100.0 * row.best_common_improvement_fraction:+.3f}% |"
        )
    summary_lines.extend(
        [
            "",
            f"Near-zero signs agree across seed ensembles: "
            f"{report['primary']['near_zero_sign_agreement_between_seed_ensembles']}.",
            f"All clusters share one sign: "
            f"{report['primary']['all_systems_share_one_near_zero_sign']}.",
            "No formula is promoted.",
        ]
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report["ensemble_summary"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["primary"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
