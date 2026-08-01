#!/usr/bin/env python3
"""Transfer the frozen P0564 morphology-sign rule to RX J2129."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    SystemContext,
    fit_context,
    model_name,
)
from run_metric_slip_raw_lensing import build_fields, model_name as slip_model_name  # noqa: E402
from run_p0554_local_cross_domain_sensitivity import raw_contexts  # noqa: E402
from run_p0557_baryon_proxy_tidal import (  # noqa: E402
    build_candidate_context,
    json_safe,
    sha256,
)
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)
from run_p0563_accept_tensor_source_plane_response import (  # noqa: E402
    unweighted_source_closure,
)
from run_p0564_baryon_morphology_sign_audit import (  # noqa: E402
    acute_quadrupole_misalignment,
    component_descriptors,
)


def build_rx_context(protocol):
    interaction = json.loads(
        (ROOT / protocol["inputs"]["p0554_interaction_protocol"]).read_text()
    )
    raw = next(context for context in raw_contexts(interaction) if context.label == "RXJ2129")
    metric = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_protocol"]).read_text()
    )
    slip = 5.0
    raw_fields, _ = build_fields(
        raw.anchors,
        raw.local,
        [-2.0, slip],
        cutoff_kpc=float(metric["field_closure"]["primary_maximum_radius_kpc"]),
        a_dagger=float(metric["matter_law"]["a_dagger_m_s2"]),
    )
    fields = {
        "baryon": raw_fields[slip_model_name(0)],
        "scalar_slip": raw_fields[slip_model_name(1)],
    }

    def extra_alpha(radius):
        return fields["scalar_slip"].reduced_alpha_arcsec(
            radius, 1.0
        ) - fields["baryon"].reduced_alpha_arcsec(radius, 1.0)

    system = {"label": "RXJ2129", "system": raw.system}
    return SystemContext(
        system=system,
        local_protocol=raw.local,
        training=raw.training,
        heldout=raw.heldout,
        members=pd.DataFrame(),
        fields=fields,
        correction=None,
        initial_geometry=raw.geometry,
        extra_alpha=extra_alpha,
    )


def rx_morphology(protocol, p0559, registered):
    axis = registered["axis"]
    spacing = float(axis[1] - axis[0])
    star = np.maximum(registered["star"], 0.0)
    gas = np.sqrt(
        np.maximum(
            gaussian_filter(
                np.maximum(registered["gas"], 0.0),
                sigma=float(p0559["gas_map"]["smoothing_sigma_arcsec"]) / spacing,
                mode="nearest",
            ),
            0.0,
        )
    )
    gate = protocol["sign_gate"]
    inner_aperture = float(gate["inner_correlation_aperture_arcsec"])
    outer_aperture = float(gate["outer_alignment_aperture_arcsec"])
    inner_star = component_descriptors(axis, star, inner_aperture)
    inner_gas = component_descriptors(axis, gas, inner_aperture)
    mask = inner_star["_mask"] & inner_gas["_mask"]
    inner_correlation = float(
        np.corrcoef(
            inner_star["_normalized_image"][mask],
            inner_gas["_normalized_image"][mask],
        )[0, 1]
    )
    outer_star = component_descriptors(axis, star, outer_aperture)
    outer_gas = component_descriptors(axis, gas, outer_aperture)
    misalignment = acute_quadrupole_misalignment(
        outer_star["quadrupole_angle_deg"], outer_gas["quadrupole_angle_deg"]
    )
    outer_cos2 = float(np.cos(np.radians(2.0 * misalignment)))
    inner_trigger = inner_correlation > float(
        gate["inner_star_gas_correlation_threshold"]
    )
    outer_trigger = outer_cos2 < float(gate["outer_quadrupole_cos2_threshold"])
    predicted_sign = "negative" if inner_trigger and outer_trigger else "positive"
    coupling = -float(gate["universal_magnitude"]) if predicted_sign == "negative" else float(gate["universal_magnitude"])
    return {
        "system_label": "RXJ2129",
        "inner_star_gas_correlation": inner_correlation,
        "inner_threshold": float(gate["inner_star_gas_correlation_threshold"]),
        "inner_negative_trigger": inner_trigger,
        "outer_quadrupole_misalignment_deg": misalignment,
        "outer_quadrupole_cos2_alignment": outer_cos2,
        "outer_threshold": float(gate["outer_quadrupole_cos2_threshold"]),
        "outer_negative_trigger": outer_trigger,
        "predicted_sign": predicted_sign,
        "predicted_coupling": coupling,
    }


def main():
    config_path = ROOT / "configs/p0565_rxj2129_morphology_gate_transfer_protocol.json"
    protocol = json.loads(config_path.read_text())
    if not protocol["status"].startswith("frozen_before_rxj2129_"):
        raise RuntimeError("P0565 protocol is not frozen before RXJ2129 scores")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text())
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    base = build_rx_context(protocol)
    registered = prepare_registered_maps(p0559, [base])
    morphology = rx_morphology(protocol, p0559, registered["RXJ2129"])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([morphology]).to_csv(
        output / protocol["outputs"]["morphology"], index=False
    )
    print("P0565 frozen morphology prediction:", morphology, flush=True)

    catalogs, physical_audits = physical_catalogs(
        p0559, [base], registered
    )
    key = (
        protocol["map"]["gas_normalization"],
        float(protocol["map"]["gas_power"]),
        bool(protocol["map"]["include_stars"]),
    )
    tensor_audits = []
    tensor = build_candidate_context(
        base,
        catalogs["RXJ2129"][key],
        p0557,
        "accept_absolute_sqrt",
        {"operator_id": "contrast", "subtract_circular_mean": True},
        pixels_per_axis=int(protocol["map"]["pixels_per_axis"]),
        softening_kpc=float(protocol["map"]["softening_kpc"]),
        audit_rows=tensor_audits,
        stage="p0565_rxj2129_transfer",
    )
    pd.DataFrame(tensor_audits).to_csv(
        output / protocol["outputs"]["tensor_audit"], index=False
    )
    physical_audits.to_csv(
        output / protocol["outputs"]["physical_map_audit"], index=False
    )
    max_q = float(tensor_audits[0]["maximum_Q_eigenvalue"])
    response_rule = protocol["response"]
    lo, hi, step = map(
        float,
        [
            response_rule["coupling_grid_min"],
            response_rule["coupling_grid_max"],
            response_rule["coupling_grid_step"],
        ],
    )
    grid = lo + step * np.arange(int(round((hi - lo) / step)) + 1)
    if np.min(1.0 - np.abs(grid) * max_q) <= 0.05:
        raise ValueError("RXJ2129 response grid violates ellipticity margin")

    exact_rows = []
    response_rows = []
    predictions = []
    zero_fits = {}
    candidate_coupling = float(morphology["predicted_coupling"])
    seeds = list(map(int, response_rule["optimizer_seed_ensembles"]))
    starts = int(response_rule["starts_per_exact_fit"])
    base.local_protocol["optimization"]["maximum_function_evaluations"] = int(
        response_rule["maximum_function_evaluations"]
    )
    tensor.local_protocol["optimization"]["maximum_function_evaluations"] = int(
        response_rule["maximum_function_evaluations"]
    )
    for ensemble_index, seed in enumerate(seeds):
        ensemble = f"seed_{ensemble_index + 1}"
        for model_id, coupling, context in [
            ("zero", 0.0, tensor),
            ("morphology_gated_t", candidate_coupling, tensor),
        ]:
            print(f"P0565 exact {ensemble} {model_id} t={coupling:+.2f}", flush=True)
            fitted = fit_context(context, coupling, starts=starts, seed=seed)
            if model_id == "zero":
                zero_fits[ensemble] = fitted
            exact_rows.append(
                {
                    "ensemble": ensemble,
                    "model_id": model_id,
                    "coupling": coupling,
                    "fit_cost": float(fitted["fit"]["result"].cost),
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
            for frame in [fitted["training_predictions"], fitted["heldout_predictions"]]:
                frame = frame.copy()
                frame["ensemble"] = ensemble
                frame["model_id"] = model_id
                predictions.append(frame)
        parameters = zero_fits[ensemble]["fit"]["result"].x
        for coupling in grid:
            lens = MemberTidalLens(
                tensor.local_protocol,
                tensor.fields,
                tensor.correction,
                float(coupling),
            )
            values = unweighted_source_closure(
                lens,
                model_name(float(coupling)),
                parameters,
                tensor.training,
                tensor.heldout,
            )
            response_rows.append(
                {
                    "ensemble": ensemble,
                    "coupling": float(coupling),
                    **values,
                    "minimum_permittivity_eigenvalue": 1.0 - abs(float(coupling)) * max_q,
                }
            )
    exact = pd.DataFrame(exact_rows)
    zero_score = exact[exact.model_id.eq("zero")].set_index("ensemble")
    candidate_score = exact[exact.model_id.eq("morphology_gated_t")].set_index("ensemble")
    exact["improvement_fraction_vs_ensemble_zero"] = [
        0.0
        if row.model_id == "zero"
        else 1.0
        - float(row.heldout_exact_RMS_arcsec)
        / float(zero_score.loc[row.ensemble, "heldout_exact_RMS_arcsec"])
        if np.isfinite(float(row.heldout_exact_RMS_arcsec))
        else -np.inf
        for row in exact.itertuples(index=False)
    ]
    response = pd.DataFrame(response_rows)
    near_negative, near_positive = map(float, response_rule["near_zero_pair"])
    response_signs = []
    for ensemble, group in response.groupby("ensemble"):
        negative = group[group.coupling.eq(near_negative)].iloc[0]
        positive = group[group.coupling.eq(near_positive)].iloc[0]
        slope = (
            float(positive.heldout_unweighted_source_plane_RMS_arcsec)
            - float(negative.heldout_unweighted_source_plane_RMS_arcsec)
        ) / (near_positive - near_negative)
        response_signs.append(
            {
                "ensemble": ensemble,
                "near_zero_dRMS_dt_arcsec": slope,
                "near_zero_preferred_sign": "positive" if slope < 0 else "negative" if slope > 0 else "flat",
            }
        )
    sign_frame = pd.DataFrame(response_signs)
    response = response.merge(sign_frame, on="ensemble", how="left")
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    response.to_csv(output / protocol["outputs"]["source_plane_response"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )

    sign_pass = bool(
        (sign_frame.near_zero_preferred_sign == morphology["predicted_sign"]).all()
    )
    roots_pass = bool(candidate_score.all_heldout_roots.all())
    improve_pass = bool(
        (
            candidate_score.heldout_exact_RMS_arcsec
            < zero_score.heldout_exact_RMS_arcsec
        ).all()
    )
    report = {
        "report_version": "P0565-RXJ2129-MORPHOLOGY-GATE-TRANSFER-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "morphology": morphology,
        "response_signs": sign_frame.to_dict("records"),
        "exact_scores": exact.to_dict("records"),
        "gate_audit": {
            "morphology_sign_matches_near_zero_sign_in_both_ensembles": sign_pass,
            "exact_candidate_all_roots_in_both_ensembles": roots_pass,
            "exact_candidate_improves_zero_in_both_ensembles": improve_pass,
        },
        "primary": {
            "candidate_gate_validated": bool(sign_pass and roots_pass and improve_pass),
            "formula_promoted": False,
        },
        "physical_map_audit": physical_audits.to_dict("records"),
        "tensor_audit": tensor_audits,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ensemble, group in response.groupby("ensemble"):
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
            label=ensemble,
        )
    axes[0].axvline(candidate_coupling, color="black", linestyle="--", label="morphology gate")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set(
        xlabel="tensor coupling t",
        ylabel="RXJ2129 held-out source-plane improvement (%)",
        title="Prospective sign response",
    )
    axes[0].legend()
    pivot = exact.pivot(index="ensemble", columns="model_id", values="heldout_exact_RMS_arcsec")
    pivot.plot.bar(ax=axes[1])
    axes[1].set(ylabel="held-out exact RMS (arcsec)", title="Exact morphology-gate check")
    axes[1].tick_params(axis="x", rotation=0)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    (output / protocol["outputs"]["summary"]).write_text(
        f"""# P0565 RX J2129 morphology-gate transfer

The frozen morphology rule predicted `{morphology['predicted_sign']}` with
`t={candidate_coupling:+.1f}`. It matched both near-zero source-plane signs:
{sign_pass}. Exact roots passed in both ensembles: {roots_pass}. Exact RMS
improved in both ensembles: {improve_pass}. Candidate gate validated:
{bool(sign_pass and roots_pass and improve_pass)}. No formula is promoted.
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["morphology"]), indent=2), flush=True)
    print(sign_frame.to_string(index=False), flush=True)
    print(exact.to_string(index=False), flush=True)
    print(json.dumps(json_safe(report["gate_audit"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
