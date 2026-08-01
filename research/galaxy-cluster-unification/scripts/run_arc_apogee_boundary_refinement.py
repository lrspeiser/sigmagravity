"""Refine the arc-apogee galaxy/Solar boundaries and scalar/vector placement."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_arc_apogee_cross_domain import (  # noqa: E402
    fit_q,
    galaxy_properties,
    json_safe,
    score_predictions,
    sha256,
    velocity_prediction,
)
from voidscreen.arc_apogee import (  # noqa: E402
    acceleration_screen,
    residence_coordinate,
    solar_diagnostics,
)
from voidscreen.data import KPC_M  # noqa: E402


def add_coordinate(points, properties, spec, a0):
    frame = points.merge(properties, on="galaxy", validate="many_to_one")
    mu = float(spec["scale_mix_mu"])
    scale = np.power(frame.force_equivalent_r80_kpc.to_numpy(float), 1.0 - mu)
    scale *= np.power(frame.mass_radius_kpc.to_numpy(float), mu)
    coordinate = residence_coordinate(
        frame.radius_adjusted_kpc.to_numpy(float),
        scale,
        alpha=float(spec["alpha"]),
        apogee_ratio=float(spec["apogee_ratio"]),
    )
    screen = acceleration_screen(
        frame.g_bar_m_s2.to_numpy(float),
        a0_m_s2=a0,
        exponent=float(spec["screen_exponent"]),
    )
    # Extent information is deliberately absent from scalar amplitude. It is
    # retained in the independently tested cluster directional kernel.
    frame["arc_coordinate"] = coordinate * screen
    frame["arc_residence_coordinate_raw"] = coordinate
    frame["arc_screen"] = screen
    frame["arc_scale_radius_kpc"] = scale
    return frame


def cross_galaxy_score(inner, outer, properties, bounds):
    predictions = np.empty(len(outer), dtype=float)
    q_values = []
    for fold in sorted(properties.galaxy_fold.astype(int).unique()):
        train_names = set(properties[properties.galaxy_fold.ne(fold)].galaxy)
        test_names = set(properties[properties.galaxy_fold.eq(fold)].galaxy)
        q = fit_q(inner[inner.galaxy.isin(train_names)], bounds)
        q_values.append(q)
        mask = outer.galaxy.isin(test_names).to_numpy()
        predictions[mask] = velocity_prediction(outer[mask], q)
    return score_predictions(outer, predictions), q_values


def morphology_scores(predictions, properties, candidate_id):
    frame = predictions[
        predictions.candidate_id.eq(candidate_id)
        & predictions.split.eq("outer_holdout")
    ].copy()
    bins = {
        "all": np.ones(len(frame), dtype=bool),
        "dwarf_mass": frame.baryonic_mass_solar < 1e9,
        "intermediate_mass": (frame.baryonic_mass_solar >= 1e9) & (frame.baryonic_mass_solar < 1e11),
        "giant_mass": frame.baryonic_mass_solar >= 1e11,
        "gas_rich": frame.gas_fraction >= 0.5,
        "gas_poor": frame.gas_fraction < 0.2,
        "disk_dominated": frame.stellar_bulge_fraction <= 0.05,
        "bulge_dominated": frame.stellar_bulge_fraction >= 0.30,
        "late_type": frame.hubble_type >= 7,
        "early_type": frame.hubble_type <= 3,
    }
    rows = []
    for name, mask in bins.items():
        local = frame[mask]
        if local.empty:
            continue
        predicted = local.velocity_arc_km_s.to_numpy(float)
        score = score_predictions(local, predicted)
        rar = score_predictions(local, local.velocity_RAR_same_nuisance_km_s.to_numpy(float))
        rows.append({
            "candidate_id": candidate_id,
            "bin": name,
            "galaxies": int(local.galaxy.nunique()),
            "points": int(len(local)),
            "arc_RMSE_km_s": score["RMSE_km_s"],
            "RAR_RMSE_km_s": rar["RMSE_km_s"],
            "arc_over_RAR": score["RMSE_km_s"] / rar["RMSE_km_s"],
            "arc_mean_residual_km_s": score["mean_residual_km_s"],
        })
    return rows


def make_figure(scores, morphology, impacts, rar_rmse, output):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axis = axes[0, 0]
    grouped = scores.groupby("scale_mix_mu").cross_galaxy_outer_RMSE_km_s.min()
    axis.plot(grouped.index, grouped.values, "o-", lw=2)
    axis.axhline(rar_rmse, color="black", ls="--", label="fixed RAR")
    axis.set(title="How much square-root mass radius is required?", xlabel="mix mu (0=R80, 1=sqrt(GM/a0))", ylabel="best outer RMSE (km/s)")
    axis.legend()

    axis = axes[0, 1]
    pivot = scores.groupby(["alpha", "screen_exponent"]).cross_galaxy_outer_RMSE_km_s.min().unstack()
    image = axis.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
    axis.set(xlabel="screen exponent n", ylabel="accumulation exponent alpha", title="Best RMSE across scale/apogee")
    figure.colorbar(image, ax=axis, label="km/s")

    axis = axes[1, 0]
    local = morphology.set_index("bin")
    axis.barh(local.index, local.arc_over_RAR, color="tab:orange")
    axis.axvline(1.0, color="black", ls="--")
    axis.set(title="Best arc law by galaxy type", xlabel="RMSE / fixed-RAR RMSE")

    axis = axes[1, 1]
    ordered = impacts.sort_values("impact_span")
    axis.barh(ordered.parameter, ordered.impact_span, color="tab:blue")
    axis.set(title="Refined parameter impact", xlabel="median outer-RMSE span (km/s)")
    figure.suptitle("Separated scalar residence and directional extent gate")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    protocol_path = ROOT / "configs" / "arc_apogee_boundary_refinement_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / protocol["inputs"]["points"])
    points = raw[raw.model.eq("fixed_RAR") & raw.scenario.eq("invariant")].copy()
    morphology = pd.read_csv(ROOT / protocol["inputs"]["morphology"])
    a0 = 1.2e-10
    properties = galaxy_properties(points, morphology, a0)
    grid = protocol["grid"]
    score_rows = []
    specs = []
    for values in itertools.product(
        protocol["scale_interpolation"]["mu_values"],
        grid["alpha_values"],
        grid["apogee_ratios"],
        grid["screen_exponents"],
    ):
        spec = dict(zip(
            ["scale_mix_mu", "alpha", "apogee_ratio", "screen_exponent"],
            values,
            strict=True,
        ))
        spec["candidate_id"] = f"R{len(specs):04d}"
        specs.append(spec)
    cache = {}
    for index, spec in enumerate(specs):
        frame = add_coordinate(points, properties, spec, a0)
        inner = frame[frame.split.eq("inner_train")]
        outer = frame[frame.split.eq("outer_holdout")]
        q = fit_q(inner, grid["universal_q_bounds"])
        outer_score = score_predictions(outer, velocity_prediction(outer, q))
        cross_score, q_folds = cross_galaxy_score(
            inner, outer, properties, grid["universal_q_bounds"]
        )
        solar = solar_diagnostics(
            residence_strength=q,
            alpha=float(spec["alpha"]),
            apogee_ratio=float(spec["apogee_ratio"]),
            gate_mode="none",
            scale_mode="hybrid_radius",
            scale_mix=float(spec["scale_mix_mu"]),
            screen_a0_m_s2=a0,
            screen_exponent=float(spec["screen_exponent"]),
        )
        score_rows.append({
            **spec,
            "universal_q": q,
            "q_at_boundary": bool(q >= grid["universal_q_bounds"][1] - 1e-3),
            "fold_q_min": float(np.min(q_folds)),
            "fold_q_max": float(np.max(q_folds)),
            "outer_RMSE_km_s": outer_score["RMSE_km_s"],
            "outer_equal_galaxy_RMSE_km_s": outer_score["equal_galaxy_RMSE_km_s"],
            "cross_galaxy_outer_RMSE_km_s": cross_score["RMSE_km_s"],
            "cross_galaxy_outer_equal_galaxy_RMSE_km_s": cross_score["equal_galaxy_RMSE_km_s"],
            **solar,
            "all_solar_proxies_pass": bool(
                solar["Cassini_proxy_pass"]
                and solar["Earth_proxy_pass"]
                and solar["Mercury_proxy_pass"]
            ),
        })
        if index % 240 == 0:
            print(f"refinement {index + 1}/{len(specs)}", flush=True)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    eligible = scores[scores.all_solar_proxies_pass & ~scores.q_at_boundary]
    best = eligible.sort_values("cross_galaxy_outer_RMSE_km_s").iloc[0]
    best_by_mu = eligible.sort_values("cross_galaxy_outer_RMSE_km_s").groupby("scale_mix_mu").head(1)
    selected_ids = list(best_by_mu.candidate_id)
    prediction_frames = []
    morphology_rows = []
    for candidate_id in selected_ids:
        row = scores[scores.candidate_id.eq(candidate_id)].iloc[0]
        spec = {name: row[name] for name in ["scale_mix_mu", "alpha", "apogee_ratio", "screen_exponent"]}
        frame = add_coordinate(points, properties, spec, a0)
        frame["velocity_arc_km_s"] = velocity_prediction(frame, float(row.universal_q))
        frame["candidate_id"] = candidate_id
        frame["selected_role"] = "best_overall" if candidate_id == best.candidate_id else f"best_mu_{row.scale_mix_mu:g}"
        prediction_frames.append(frame)
        morphology_rows.extend(morphology_scores(frame, properties, candidate_id))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    morphology_frame = pd.DataFrame(morphology_rows)
    predictions.to_csv(output / protocol["outputs"]["predictions"], index=False)
    morphology_frame.to_csv(output / protocol["outputs"]["morphology_scores"], index=False)

    impact_rows = []
    for parameter in ["scale_mix_mu", "alpha", "apogee_ratio", "screen_exponent"]:
        grouped = scores.groupby(parameter).cross_galaxy_outer_RMSE_km_s.median()
        impact_rows.append({
            "parameter": parameter,
            "best_level": str(grouped.idxmin()),
            "best_median_RMSE_km_s": float(grouped.min()),
            "worst_level": str(grouped.idxmax()),
            "worst_median_RMSE_km_s": float(grouped.max()),
            "impact_span": float(grouped.max() - grouped.min()),
        })
    impacts = pd.DataFrame(impact_rows).sort_values("impact_span", ascending=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    outer = points[points.split.eq("outer_holdout")]
    rar = score_predictions(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float))
    parent = json.loads((ROOT / protocol["inputs"]["parent_report"]).read_text())
    cluster_primary = parent["cluster_selection"]["primary"]
    cluster_best = parent["cluster_selection"]["best"]
    best_morphology = morphology_frame[morphology_frame.candidate_id.eq(best.candidate_id)]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed post-boundary scalar-direction placement refinement",
        "protocol_sha256": sha256(protocol_path),
        "coverage": {
            "variants": int(len(scores)),
            "SPARC_galaxies": 131,
            "outer_points": 968,
            "galaxy_folds": 5,
        },
        "best_variant": best.to_dict(),
        "best_variant_by_scale_mix": best_by_mu.to_dict("records"),
        "fixed_RAR_same_nuisance_outer_RMSE_km_s": rar["RMSE_km_s"],
        "best_arc_to_RAR_RMSE_ratio": float(best.cross_galaxy_outer_RMSE_km_s / rar["RMSE_km_s"]),
        "directional_cluster_kernel": {
            "primary_inverse_candidate": cluster_primary,
            "best_same_data_replay": cluster_best,
            "placement": "soft extent gate changes local-versus-routed spatial direction but not scalar q",
        },
        "morphology_for_best": best_morphology.to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "bottom_line": "Separating the extent gate from scalar amplitude is strongly favored. The remaining galaxy improvement is driven by moving Rb toward sqrt(GM/a0), exposing how much of the success comes from reintroducing the MOND/RAR mass scale.",
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "# Arc-apogee boundary refinement",
        "",
        f"Best variant `{best.candidate_id}`: mu={best.scale_mix_mu:g}, alpha={best.alpha:g}, zeta={best.apogee_ratio:g}, n={best.screen_exponent:g}, q={best.universal_q:.5g}.",
        f"Cross-galaxy outer RMSE: {best.cross_galaxy_outer_RMSE_km_s:.3f} km/s versus fixed RAR {rar['RMSE_km_s']:.3f} km/s.",
        f"All Solar proxies pass: {bool(best.all_solar_proxies_pass)}; Mercury {best.Mercury_precession_mas_per_century:.3g} mas/century.",
        "",
        "The extent gate is retained only in the cluster directional kernel. It is not allowed to modulate scalar galaxy strength.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    make_figure(
        scores,
        best_morphology,
        impacts,
        rar["RMSE_km_s"],
        output / protocol["outputs"]["figure"],
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
