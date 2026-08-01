#!/usr/bin/env python3
"""Conditioning-robust source-plane response for the physical ACCEPT tensor."""

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

from run_member_tidal_metric import MemberTidalLens, build_contexts, model_name  # noqa: E402
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


def coupling_grid(protocol):
    rule = protocol["response_grid"]
    lo, hi, step = map(
        float,
        [rule["minimum_coupling"], rule["maximum_coupling"], rule["step"]],
    )
    return lo + step * np.arange(int(round((hi - lo) / step)) + 1)


def unweighted_source_closure(lens, name, parameters, training, heldout):
    sources = {}
    training_residuals = []
    for family, group in training.groupby("source_family", sort=True):
        x = group.x_arcsec.to_numpy(float)
        y = group.y_arcsec.to_numpy(float)
        redshift = float(group.source_redshift.median())
        beta_x, beta_y = lens.ray_shooting(name, parameters, x, y, redshift)
        beta = np.column_stack([beta_x, beta_y])
        source = beta.mean(axis=0)
        sources[int(family)] = source
        training_residuals.append(beta - source)
    heldout_residuals = []
    singular_values = []
    for family, group in heldout.groupby("source_family", sort=True):
        x = group.x_arcsec.to_numpy(float)
        y = group.y_arcsec.to_numpy(float)
        redshift = float(group.source_redshift.median())
        beta_x, beta_y = lens.ray_shooting(name, parameters, x, y, redshift)
        heldout_residuals.append(
            np.column_stack([beta_x, beta_y]) - sources[int(family)]
        )
        jacobian = lens.jacobian(name, parameters, x, y, redshift)
        singular_values.extend(np.linalg.svd(jacobian, compute_uv=False).ravel())
    training_delta = np.vstack(training_residuals)
    heldout_delta = np.vstack(heldout_residuals)
    singular_values = np.asarray(singular_values, dtype=float)
    return {
        "training_unweighted_source_plane_RMS_arcsec": float(
            np.sqrt(np.mean(np.sum(training_delta**2, axis=1)))
        ),
        "heldout_unweighted_source_plane_RMS_arcsec": float(
            np.sqrt(np.mean(np.sum(heldout_delta**2, axis=1)))
        ),
        "heldout_min_jacobian_singular_value": float(singular_values.min()),
        "heldout_max_inverse_jacobian_gain": float(1.0 / singular_values.min()),
        "heldout_jacobian_condition_proxy": float(
            singular_values.max() / singular_values.min()
        ),
    }


def main():
    config_path = ROOT / "configs/p0563_accept_tensor_source_plane_response_protocol.json"
    protocol = json.loads(config_path.read_text())
    if not protocol["status"].startswith("frozen_after_p0562_"):
        raise RuntimeError("P0563 protocol is not frozen with the conditioning disclosure")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text())
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    member = json.loads((ROOT / protocol["inputs"]["member_tidal_protocol"]).read_text())
    contexts, _, input_hashes = build_contexts(
        member, softening_kpc=float(protocol["locked_map"]["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    catalogs, physical_audits = physical_catalogs(p0559, contexts, registered)
    locked = protocol["locked_map"]
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
            stage="p0563_source_plane_response",
        )
    maximum_q = {
        row["system_label"]: float(row["maximum_Q_eigenvalue"])
        for row in tensor_audits
    }
    grid = coupling_grid(protocol)
    margin = float(protocol["ellipticity"]["minimum_eigenvalue"])
    if any(
        1.0 - abs(t) * maximum_q[label] <= margin
        for t in grid
        for label in maximum_q
    ):
        raise ValueError("source-plane grid violates ellipticity margin")

    geometry_path = ROOT / protocol["inputs"]["geometry_ensembles"]
    local_path = ROOT / protocol["inputs"]["p0562_local_response"]
    geometry = pd.read_csv(geometry_path)
    local = pd.read_csv(local_path)
    input_hashes["geometry_ensembles"] = sha256(geometry_path)
    input_hashes["p0562_local_response"] = sha256(local_path)
    rows = []
    for item in geometry.itertuples(index=False):
        context = tensor_contexts[item.system_label]
        parameters = np.asarray([getattr(item, key) for key in GEOMETRY_LABELS])
        for coupling in grid:
            lens = MemberTidalLens(
                context.local_protocol,
                context.fields,
                context.correction,
                float(coupling),
            )
            values = unweighted_source_closure(
                lens,
                model_name(float(coupling)),
                parameters,
                context.training,
                context.heldout,
            )
            rows.append(
                {
                    "ensemble": item.ensemble,
                    "system_label": item.system_label,
                    "coupling": float(coupling),
                    **values,
                    "minimum_permittivity_eigenvalue": (
                        1.0 - abs(float(coupling)) * maximum_q[item.system_label]
                    ),
                }
            )
    response = pd.DataFrame(rows)
    response = response.merge(
        local[
            [
                "ensemble",
                "system_label",
                "coupling",
                "heldout_fixed_source_local_RMS_arcsec",
            ]
        ],
        on=["ensemble", "system_label", "coupling"],
        how="left",
        validate="one_to_one",
    )
    response["local_to_source_plane_RMS_ratio"] = (
        response.heldout_fixed_source_local_RMS_arcsec
        / response.heldout_unweighted_source_plane_RMS_arcsec
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
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
                "equal_system_training_source_plane_RMS_arcsec": rms(
                    group.training_unweighted_source_plane_RMS_arcsec
                ),
                "equal_system_heldout_source_plane_RMS_arcsec": rms(
                    group.heldout_unweighted_source_plane_RMS_arcsec
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
        best = group.loc[group.heldout_unweighted_source_plane_RMS_arcsec.idxmin()]
        zero = group[group.coupling.eq(0.0)].iloc[0]
        negative = group[group.coupling.eq(near_negative)].iloc[0]
        positive = group[group.coupling.eq(near_positive)].iloc[0]
        slope = (
            float(positive.heldout_unweighted_source_plane_RMS_arcsec)
            - float(negative.heldout_unweighted_source_plane_RMS_arcsec)
        ) / (near_positive - near_negative)
        system_rows.append(
            {
                "ensemble": ensemble,
                "system_label": label,
                "zero_source_plane_RMS_arcsec": float(
                    zero.heldout_unweighted_source_plane_RMS_arcsec
                ),
                "best_grid_coupling": float(best.coupling),
                "best_grid_source_plane_RMS_arcsec": float(
                    best.heldout_unweighted_source_plane_RMS_arcsec
                ),
                "best_grid_improvement_fraction": 1.0
                - float(best.heldout_unweighted_source_plane_RMS_arcsec)
                / float(zero.heldout_unweighted_source_plane_RMS_arcsec),
                "near_zero_dRMS_dt_arcsec": slope,
                "near_zero_preferred_sign": (
                    "positive" if slope < 0.0 else "negative" if slope > 0.0 else "flat"
                ),
            }
        )
    per_system = pd.DataFrame(system_rows)
    per_system.to_csv(output / protocol["outputs"]["per_system_summary"], index=False)

    conditioning = response[
        [
            "ensemble",
            "system_label",
            "coupling",
            "heldout_min_jacobian_singular_value",
            "heldout_max_inverse_jacobian_gain",
            "heldout_jacobian_condition_proxy",
            "heldout_fixed_source_local_RMS_arcsec",
            "heldout_unweighted_source_plane_RMS_arcsec",
            "local_to_source_plane_RMS_ratio",
        ]
    ].copy()
    conditioning.to_csv(output / protocol["outputs"]["conditioning_audit"], index=False)
    finite = conditioning.replace([np.inf, -np.inf], np.nan).dropna()
    log_gain = np.log10(finite.heldout_max_inverse_jacobian_gain.to_numpy(float))
    log_ratio = np.log10(finite.local_to_source_plane_RMS_ratio.to_numpy(float))
    conditioning_correlation = float(np.corrcoef(log_gain, log_ratio)[0, 1])

    ensemble_rows = []
    for ensemble, group in aggregate.groupby("ensemble"):
        zero = group[group.coupling.eq(0.0)].iloc[0]
        best = group.loc[
            group.equal_system_heldout_source_plane_RMS_arcsec.idxmin()
        ]
        ensemble_rows.append(
            {
                "ensemble": ensemble,
                "zero_source_plane_RMS_arcsec": float(
                    zero.equal_system_heldout_source_plane_RMS_arcsec
                ),
                "best_common_coupling": float(best.coupling),
                "best_common_source_plane_RMS_arcsec": float(
                    best.equal_system_heldout_source_plane_RMS_arcsec
                ),
                "best_common_improvement_fraction": 1.0
                - float(best.equal_system_heldout_source_plane_RMS_arcsec)
                / float(zero.equal_system_heldout_source_plane_RMS_arcsec),
            }
        )
    ensemble_summary = pd.DataFrame(ensemble_rows)
    sign_table = per_system.pivot(
        index="system_label", columns="ensemble", values="near_zero_preferred_sign"
    )
    report = {
        "report_version": "P0563-ACCEPT-TENSOR-SOURCE-PLANE-RESPONSE-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "input_hashes": input_hashes,
        "ensemble_summary": ensemble_summary.to_dict("records"),
        "per_system_summary": per_system.to_dict("records"),
        "conditioning": {
            "log_inverse_gain_vs_log_local_to_source_ratio_correlation": conditioning_correlation,
            "maximum_inverse_jacobian_gain": float(
                conditioning.heldout_max_inverse_jacobian_gain.max()
            ),
            "maximum_local_to_source_plane_RMS_ratio": float(
                conditioning.local_to_source_plane_RMS_ratio.max()
            ),
        },
        "primary": {
            "near_zero_sign_agreement_between_geometry_ensembles": bool(
                (sign_table.iloc[:, 0] == sign_table.iloc[:, 1]).all()
            ),
            "all_systems_share_one_near_zero_sign": bool(
                per_system.near_zero_preferred_sign.nunique() == 1
            ),
            "common_optimum_agreement_between_geometry_ensembles": bool(
                ensemble_summary.best_common_coupling.nunique() == 1
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
                group.coupling.eq(0.0),
                "heldout_unweighted_source_plane_RMS_arcsec",
            ].iloc[0]
        )
        axes[0].plot(
            group.coupling,
            100.0 * (1.0 - group.heldout_unweighted_source_plane_RMS_arcsec / zero),
            label=f"{label} {ensemble}",
            alpha=0.8,
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[0].set(
        xlabel="tensor coupling t",
        ylabel="held-out source-plane improvement vs t=0 (%)",
        title="Conditioning-robust directional response",
    )
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].scatter(
        finite.heldout_max_inverse_jacobian_gain,
        finite.local_to_source_plane_RMS_ratio,
        alpha=0.45,
        s=14,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="maximum inverse-Jacobian gain",
        ylabel="local-image RMS / source-plane RMS",
        title="Why the P0562 local metric spikes",
    )
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    lines = [
        "# P0563 conditioning-robust tensor response",
        "",
        "| Ensemble | Zero source RMS | Best common t | Best source RMS | Improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ensemble_summary.itertuples(index=False):
        lines.append(
            f"| {row.ensemble} | {row.zero_source_plane_RMS_arcsec:.4f} | "
            f"{row.best_common_coupling:+.2f} | "
            f"{row.best_common_source_plane_RMS_arcsec:.4f} | "
            f"{100.0 * row.best_common_improvement_fraction:+.3f}% |"
        )
    lines.extend(
        [
            "",
            f"Inverse-gain correlation: {conditioning_correlation:.4f}.",
            f"All systems share one near-zero sign: "
            f"{report['primary']['all_systems_share_one_near_zero_sign']}.",
            "No formula is promoted.",
        ]
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(report["ensemble_summary"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["conditioning"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["primary"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
