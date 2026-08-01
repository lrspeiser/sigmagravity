#!/usr/bin/env python3
"""Map the signed four-cluster response of the continuous baryonic metric."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import MemberTidalLens, build_contexts, fit_context  # noqa: E402
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)
from run_p0570_physical_baryon_residual_lensing import source_plane_rms  # noqa: E402
from run_p0586_continuous_baryonic_metric import (  # noqa: E402
    candidate_id,
    json_safe,
    sha256,
)
from voidscreen.baryonic_metric import (  # noqa: E402
    build_baryonic_metric_correction_field,
    prepare_baryonic_metric_state,
    prepare_baryonic_metric_workspace,
)


def main():
    protocol_path = ROOT / "configs/p0586c_signed_metric_response_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_signed_metric_response_score":
        raise RuntimeError("P0586C protocol is not frozen")
    p0586 = json.loads(
        (ROOT / protocol["inputs"]["p0586_protocol"]).read_text(encoding="utf-8")
    )
    p0559 = json.loads(
        (ROOT / protocol["inputs"]["p0559_protocol"]).read_text(encoding="utf-8")
    )
    member = json.loads(
        (ROOT / p0559["inputs"]["member_tidal_protocol"]).read_text(encoding="utf-8")
    )
    member["optimization"]["maximum_function_evaluations"] = int(
        p0559["optimization"]["maximum_function_evaluations"]
    )
    contexts, _, _ = build_contexts(
        member, softening_kpc=float(p0559["locked_field"]["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    physical, physical_audits = physical_catalogs(p0559, contexts, registered)
    physical_audits = physical_audits.set_index("system_label")
    locked = protocol["locked"]
    factorial = protocol["factorial"]
    numerical = p0586["numerics"]

    catalogs = {}
    masses = {}
    workspaces = {}
    states = {}
    for context in contexts:
        label = context.system["label"]
        catalog = physical[label][("accept_absolute", 0.5, True)]
        total_mass = float(
            physical_audits.loc[label, "stellar_mass_assigned_to_map_msun"]
            + physical_audits.loc[label, "projected_ACCEPT_gas_mass_on_map_msun"]
        )
        scale = float(
            context.local_protocol["cosmology_and_coordinates"][
                "angular_scale_kpc_per_arcsec"
            ]
        )
        catalogs[label] = catalog
        masses[label] = total_mass
        print(f"P0586C workspace {label}", flush=True)
        workspaces[label] = prepare_baryonic_metric_workspace(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=total_mass,
            scale_kpc_per_arcsec=scale,
            half_width_arcsec=float(locked["field_half_width_arcsec"]),
            pixels_per_axis=int(locked["field_pixels_per_axis"]),
            point_softening_arcsec=float(locked["point_softening_arcsec"]),
        )
        for width in map(float, factorial["smoothing_r80_fraction"]):
            states[(label, width)] = prepare_baryonic_metric_state(
                workspaces[label], width
            )

    baseline_fits = {}
    zero_scores = {}
    for index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"P0586C zero fit {label}", flush=True)
        fitted = fit_context(
            context,
            0.0,
            starts=8,
            seed=20261400 + index,
        )
        baseline_fits[label] = fitted
        zero_lens = MemberTidalLens(
            context.local_protocol, context.fields, context.correction, 0.0
        )
        zero_scores[label] = source_plane_rms(
            zero_lens,
            0.0,
            fitted["fit"]["result"].x,
            fitted["fit"]["sources"],
            context.heldout,
        )

    grid = list(
        itertools.product(
            map(float, factorial["minimum_permittivity"]),
            map(float, factorial["anisotropy_tau"]),
            map(float, factorial["smoothing_r80_fraction"]),
        )
    )
    if len(grid) != int(factorial["candidate_count"]):
        raise RuntimeError("P0586C candidate count differs from the protocol")
    rows = []
    for index, (epsilon, tau, width) in enumerate(grid):
        cid = candidate_id(
            epsilon,
            float(locked["a0_m_s2"]),
            float(locked["gate_power"]),
            tau,
            width,
        )
        for context in contexts:
            label = context.system["label"]
            catalog = catalogs[label]
            field = build_baryonic_metric_correction_field(
                catalog.x_arcsec.to_numpy(float),
                catalog.y_arcsec.to_numpy(float),
                catalog.normalized_light_weight.to_numpy(float),
                total_mass_msun=masses[label],
                scale_kpc_per_arcsec=workspaces[label].scale_kpc_per_arcsec,
                minimum_permittivity=epsilon,
                a0_m_s2=float(locked["a0_m_s2"]),
                gate_power=float(locked["gate_power"]),
                anisotropy=tau,
                smoothing_r80_fraction=width,
                asymmetry_threshold=float(numerical["asymmetry_threshold"]),
                asymmetry_power=float(numerical["asymmetry_power"]),
                workspace=workspaces[label],
                state=states[(label, width)],
            )
            fitted = baseline_fits[label]
            lens = MemberTidalLens(context.local_protocol, context.fields, field, 1.0)
            score = source_plane_rms(
                lens,
                1.0,
                fitted["fit"]["result"].x,
                fitted["fit"]["sources"],
                context.heldout,
            )
            rows.append(
                {
                    "candidate_id": cid,
                    "system_label": label,
                    "minimum_permittivity": epsilon,
                    "anisotropy_tau": tau,
                    "smoothing_r80_fraction": width,
                    "source_plane_RMS_arcsec": score,
                    "zero_source_plane_RMS_arcsec": zero_scores[label],
                    "improvement_fraction": 1.0 - score / zero_scores[label],
                }
            )
        if (index + 1) % 18 == 0:
            print(f"P0586C screen {index + 1}/{len(grid)}", flush=True)
    screen = pd.DataFrame(rows)
    candidate_rows = []
    for cid, block in screen.groupby("candidate_id", sort=False):
        first = block.iloc[0]
        candidate_rows.append(
            {
                "candidate_id": cid,
                "minimum_permittivity": float(first.minimum_permittivity),
                "anisotropy_tau": float(first.anisotropy_tau),
                "smoothing_r80_fraction": float(first.smoothing_r80_fraction),
                "all_four_RMS_arcsec": float(
                    np.sqrt(np.mean(np.square(block.source_plane_RMS_arcsec)))
                ),
                "systems_improved": int((block.improvement_fraction > 0.0).sum()),
                "all_four_improve": bool((block.improvement_fraction > 0.0).all()),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    optima = (
        screen.sort_values("source_plane_RMS_arcsec")
        .groupby("system_label", as_index=False)
        .first()
    )

    impact_rows = []
    for system, block in screen.groupby("system_label"):
        for coordinate in (
            "minimum_permittivity",
            "anisotropy_tau",
            "smoothing_r80_fraction",
        ):
            means = block.groupby(coordinate).source_plane_RMS_arcsec.mean()
            impact_rows.append(
                {
                    "system_label": system,
                    "coordinate": coordinate,
                    "best_main_effect_level": float(means.idxmin()),
                    "main_effect_span_arcsec": float(means.max() - means.min()),
                }
            )
    impacts = pd.DataFrame(impact_rows)

    wide = screen.pivot(
        index="candidate_id", columns="system_label", values="improvement_fraction"
    )
    correlation_rows = []
    for left, right in itertools.combinations(wide.columns, 2):
        rho, pvalue = spearmanr(wide[left], wide[right])
        correlation_rows.append(
            {
                "left_system": left,
                "right_system": right,
                "Spearman_response_correlation": float(rho),
                "two_sided_p": float(pvalue),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    all_four = candidates[candidates.all_four_improve].sort_values(
        "all_four_RMS_arcsec"
    )
    best_common = candidates.sort_values("all_four_RMS_arcsec").iloc[0]
    optimum_signs = np.sign(optima.anisotropy_tau.to_numpy(float))
    nonzero_signs = set(optimum_signs[optimum_signs != 0.0].tolist())
    same_sign = len(nonzero_signs) <= 1

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    optima.to_csv(output / protocol["outputs"]["per_system_optima"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    correlations.to_csv(
        output / protocol["outputs"]["response_correlations"], index=False
    )
    report = {
        "report_version": "P0586C-SIGNED-METRIC-RESPONSE-RESULTS-0.1.0",
        "status": "complete_signed_metric_response",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "clusters": len(contexts),
            "candidates": len(candidates),
            "system_scores": len(screen),
        },
        "zero_source_plane_RMS_arcsec": zero_scores,
        "best_common_candidate": json_safe(best_common.to_dict()),
        "candidates_improving_all_four": int(len(all_four)),
        "best_all_four_if_any": json_safe(all_four.head(1).to_dict(orient="records")),
        "per_system_optima": json_safe(optima.to_dict(orient="records")),
        "response_correlations": json_safe(correlations.to_dict(orient="records")),
        "parameter_impacts": json_safe(impacts.to_dict(orient="records")),
        "gates": {
            "at_least_one_candidate_improves_all_four": bool(len(all_four) > 0),
            "common_optimum_same_tau_sign": bool(same_sign),
            "per_cluster_gravity_parameters": 0,
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0586C signed continuous-metric response",
        "",
        f"Candidates improving all four clusters: **{len(all_four)}/{len(candidates)}**.",
        f"Best common candidate: `{best_common.candidate_id}`, improving **{int(best_common.systems_improved)}/4** systems.",
        f"Independent optimum tau values: **{', '.join(f'{row.system_label}={row.anisotropy_tau:+g}' for row in optima.itertuples(index=False))}**.",
        f"All nonzero optimum signs agree: **{same_sign}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for label, block in screen[
        screen.minimum_permittivity.eq(1.0)
        & screen.smoothing_r80_fraction.eq(0.5)
    ].groupby("system_label"):
        block = block.sort_values("anisotropy_tau")
        axes[0].plot(
            block.anisotropy_tau,
            100.0 * block.improvement_fraction,
            marker="o",
            label=label,
        )
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set_xlabel("signed anisotropy tau")
    axes[0].set_ylabel("fixed-geometry gain (%)")
    axes[0].legend(fontsize=7)
    axes[1].barh(
        [f"{row.left_system}\n{row.right_system}" for row in correlations.itertuples(index=False)],
        correlations.Spearman_response_correlation,
    )
    axes[1].axvline(0.0, color="black", lw=1)
    axes[1].set_xlabel("Spearman response correlation")
    matrix = optima.pivot_table(
        index="system_label",
        values=["minimum_permittivity", "anisotropy_tau", "smoothing_r80_fraction"],
    )
    image = axes[2].imshow(matrix, aspect="auto", cmap="coolwarm")
    axes[2].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    axes[2].set_yticks(range(len(matrix.index)), matrix.index)
    axes[2].set_title("independent diagnostic optima")
    fig.colorbar(image, ax=axes[2])
    fig.suptitle("P0586C signed continuous-metric response")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["best_common_candidate"], indent=2), flush=True)
    print(json.dumps(report["per_system_optima"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
