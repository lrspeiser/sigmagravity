"""Explore the inverse-derived gravity-arc apogee law across three domains."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import (  # noqa: E402
    build_source_context,
    target_from_path,
)
from run_gravity_arc_tomography import (  # noqa: E402
    deposit_points,
    normalized_in_aperture,
    shape_metrics,
)
from voidscreen.arc_apogee import (  # noqa: E402
    G_SI,
    M_SUN_KG,
    acceleration_screen,
    extent_gate,
    mass_radius_kpc,
    residence_coordinate,
    solar_diagnostics,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.phenomenology import simple_mond_enhancement  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def radius_at_mass_fraction(radius: np.ndarray, mass: np.ndarray, fraction: float) -> float:
    order = np.argsort(radius)
    r = np.asarray(radius, dtype=float)[order]
    m = np.maximum.accumulate(np.asarray(mass, dtype=float)[order])
    target = float(fraction) * float(m[-1])
    index = int(np.searchsorted(m, target, side="left"))
    if index == 0:
        return float(r[0] * target / max(m[0], np.finfo(float).tiny))
    return float(np.interp(target, m[index - 1 : index + 1], r[index - 1 : index + 1]))


def galaxy_properties(points: pd.DataFrame, morphology: pd.DataFrame, a0: float) -> pd.DataFrame:
    rows = []
    morphology = morphology.set_index("galaxy")
    for galaxy, block in points.groupby("galaxy", sort=False):
        radius = block.radius_adjusted_kpc.to_numpy(float)
        gbar = block.g_bar_m_s2.to_numpy(float)
        mass = gbar * np.square(radius * KPC_M) / (G_SI * M_SUN_KG)
        order = np.argsort(radius)
        monotonic_mass = np.maximum.accumulate(mass[order])
        total = float(monotonic_mass[-1])
        r50 = radius_at_mass_fraction(radius, mass, 0.5)
        r80 = radius_at_mass_fraction(radius, mass, 0.8)
        row = {
            "galaxy": galaxy,
            "galaxy_fold": int(block.galaxy_index.iloc[0]) % 5,
            "force_equivalent_mass_solar": total,
            "force_equivalent_r50_kpc": r50,
            "force_equivalent_r80_kpc": r80,
            "force_equivalent_concentration_r50_over_r80": r50 / r80,
            "mass_radius_kpc": float(mass_radius_kpc(total, a0_m_s2=a0)),
        }
        if galaxy in morphology.index:
            for name in [
                "fold",
                "hubble_type",
                "baryonic_mass_solar",
                "gas_fraction",
                "stellar_bulge_fraction",
                "effective_radius_kpc",
                "disk_scale_kpc",
            ]:
                row[name] = morphology.loc[galaxy, name]
        rows.append(row)
    return pd.DataFrame(rows)


def add_arc_coordinates(points: pd.DataFrame, properties: pd.DataFrame, spec: dict, a0: float):
    frame = points.merge(properties, on="galaxy", validate="many_to_one")
    if spec["scale_mode"] == "baryon_r80":
        scale = frame.force_equivalent_r80_kpc.to_numpy(float)
    elif spec["scale_mode"] == "mass_radius":
        scale = frame.mass_radius_kpc.to_numpy(float)
    elif spec["scale_mode"] == "fixed_200kpc":
        scale = np.full(len(frame), 200.0)
    else:
        raise ValueError(spec["scale_mode"])
    coordinate = residence_coordinate(
        frame.radius_adjusted_kpc.to_numpy(float),
        scale,
        alpha=spec["alpha"],
        apogee_ratio=spec["apogee_ratio"],
    )
    gate = extent_gate(
        frame.force_equivalent_concentration_r50_over_r80.to_numpy(float),
        spec["gate_mode"],
    )
    screen = acceleration_screen(
        frame.g_bar_m_s2.to_numpy(float),
        a0_m_s2=a0,
        exponent=spec["screen_exponent"],
    )
    frame["arc_coordinate"] = coordinate * gate * screen
    frame["arc_residence_coordinate_raw"] = coordinate
    frame["arc_extent_gate"] = gate
    frame["arc_screen"] = screen
    frame["arc_scale_radius_kpc"] = scale
    return frame


def equal_galaxy_mse(frame: pd.DataFrame, q: float) -> float:
    radius_m = frame.radius_adjusted_kpc.to_numpy(float) * KPC_M
    gbar = frame.g_bar_m_s2.to_numpy(float)
    predicted = np.sqrt(
        np.maximum(gbar * (1.0 + float(q) * frame.arc_coordinate.to_numpy(float)) * radius_m / 1e6, 0.0)
    )
    residual2 = np.square(predicted - frame.velocity_observed_adjusted_km_s.to_numpy(float))
    return float(pd.Series(residual2, index=frame.galaxy).groupby(level=0).mean().mean())


def fit_q(frame: pd.DataFrame, bounds: list[float]) -> float:
    result = minimize_scalar(
        lambda value: equal_galaxy_mse(frame, value),
        bounds=(float(bounds[0]), float(bounds[1])),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 300},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("universal q fit failed")
    return float(result.x)


def velocity_prediction(frame: pd.DataFrame, q: float) -> np.ndarray:
    return np.sqrt(
        np.maximum(
            frame.g_bar_m_s2.to_numpy(float)
            * (1.0 + float(q) * frame.arc_coordinate.to_numpy(float))
            * frame.radius_adjusted_kpc.to_numpy(float)
            * KPC_M
            / 1e6,
            0.0,
        )
    )


def score_predictions(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - frame.velocity_observed_adjusted_km_s.to_numpy(float)
    per_galaxy = pd.Series(np.square(residual), index=frame.galaxy).groupby(level=0).mean()
    return {
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_galaxy_RMSE_km_s": float(np.sqrt(per_galaxy.mean())),
        "mean_residual_km_s": float(np.mean(residual)),
    }


def galaxy_grid(protocol: dict):
    settings = protocol["galaxy_test"]
    raw = pd.read_csv(ROOT / settings["input_points"])
    points = raw[
        raw.model.eq(settings["input_model"])
        & raw.scenario.eq(settings["input_scenario"])
    ].copy()
    morphology = pd.read_csv(ROOT / settings["morphology"])
    a0 = float(protocol["law"]["screen_a0_m_s2"])
    properties = galaxy_properties(points, morphology, a0)
    specs = []
    for values in itertools.product(
        settings["scale_modes"],
        settings["gate_modes"],
        settings["alpha_values"],
        settings["apogee_ratios"],
        settings["screen_exponents"],
    ):
        spec = dict(zip(
            ["scale_mode", "gate_mode", "alpha", "apogee_ratio", "screen_exponent"],
            values,
            strict=True,
        ))
        spec["candidate_id"] = f"A{len(specs):04d}"
        specs.append(spec)
    score_rows = []
    cached = {}
    for index, spec in enumerate(specs):
        frame = add_arc_coordinates(points, properties, spec, a0)
        inner = frame[frame.split.eq("inner_train")]
        outer = frame[frame.split.eq("outer_holdout")]
        q = fit_q(inner, settings["q_bounds"])
        inner_score = score_predictions(inner, velocity_prediction(inner, q))
        outer_score = score_predictions(outer, velocity_prediction(outer, q))
        fold_q = []
        cross_predictions = np.empty(len(outer), dtype=float)
        for fold in sorted(properties.galaxy_fold.astype(int).unique()):
            train_galaxies = set(properties[properties.galaxy_fold.ne(fold)].galaxy)
            test_galaxies = set(properties[properties.galaxy_fold.eq(fold)].galaxy)
            q_fold = fit_q(inner[inner.galaxy.isin(train_galaxies)], settings["q_bounds"])
            fold_q.append(q_fold)
            mask = outer.galaxy.isin(test_galaxies).to_numpy()
            cross_predictions[mask] = velocity_prediction(outer[mask], q_fold)
        cross_score = score_predictions(outer, cross_predictions)
        solar = solar_diagnostics(
            residence_strength=q,
            alpha=float(spec["alpha"]),
            apogee_ratio=float(spec["apogee_ratio"]),
            gate_mode=spec["gate_mode"],
            scale_mode=spec["scale_mode"],
            screen_a0_m_s2=a0,
            screen_exponent=float(spec["screen_exponent"]),
        )
        score_rows.append(
            {
                **spec,
                "universal_q": q,
                "q_at_boundary": bool(
                    q <= settings["q_bounds"][0] + 1e-4
                    or q >= settings["q_bounds"][1] - 1e-3
                ),
                "fold_q_min": float(np.min(fold_q)),
                "fold_q_max": float(np.max(fold_q)),
                "inner_equal_galaxy_RMSE_km_s": inner_score["equal_galaxy_RMSE_km_s"],
                "outer_RMSE_km_s": outer_score["RMSE_km_s"],
                "outer_equal_galaxy_RMSE_km_s": outer_score["equal_galaxy_RMSE_km_s"],
                "outer_mean_residual_km_s": outer_score["mean_residual_km_s"],
                "cross_galaxy_outer_RMSE_km_s": cross_score["RMSE_km_s"],
                "cross_galaxy_outer_equal_galaxy_RMSE_km_s": cross_score["equal_galaxy_RMSE_km_s"],
                **solar,
                "all_solar_proxies_pass": bool(
                    solar["Cassini_proxy_pass"]
                    and solar["Earth_proxy_pass"]
                    and solar["Mercury_proxy_pass"]
                ),
            }
        )
        if index % 90 == 0:
            print(f"galaxy grid {index + 1}/{len(specs)}", flush=True)
    scores = pd.DataFrame(score_rows)
    primary_values = settings["primary_candidate"]
    primary_mask = np.ones(len(scores), dtype=bool)
    for name, value in primary_values.items():
        primary_mask &= scores[name].eq(value).to_numpy()
    primary = scores[primary_mask].iloc[0]
    eligible = scores[scores.all_solar_proxies_pass & ~scores.q_at_boundary]
    best = eligible.sort_values("cross_galaxy_outer_RMSE_km_s").iloc[0]
    best_baryon = eligible[eligible.scale_mode.eq("baryon_r80")].sort_values(
        "cross_galaxy_outer_RMSE_km_s"
    ).iloc[0]
    best_mass = eligible[eligible.scale_mode.eq("mass_radius")].sort_values(
        "cross_galaxy_outer_RMSE_km_s"
    ).iloc[0]
    selected = pd.DataFrame([primary, best, best_baryon, best_mass]).drop_duplicates(
        "candidate_id"
    )
    prediction_rows = []
    for _, row in selected.iterrows():
        spec = {name: row[name] for name in [
            "scale_mode", "gate_mode", "alpha", "apogee_ratio", "screen_exponent"
        ]}
        frame = add_arc_coordinates(points, properties, spec, a0)
        frame["velocity_arc_km_s"] = velocity_prediction(frame, float(row.universal_q))
        frame["selected_role"] = (
            "primary" if row.candidate_id == primary.candidate_id else
            "best_overall" if row.candidate_id == best.candidate_id else
            "best_baryon_extent" if row.candidate_id == best_baryon.candidate_id else
            "best_mass_radius"
        )
        frame["candidate_id"] = row.candidate_id
        prediction_rows.append(frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    outer = points[points.split.eq("outer_holdout")]
    radius_m = outer.radius_adjusted_kpc.to_numpy(float) * KPC_M
    v_newton = np.sqrt(outer.g_bar_m_s2.to_numpy(float) * radius_m / 1e6)
    v_mond = np.sqrt(
        outer.g_bar_m_s2.to_numpy(float)
        * simple_mond_enhancement(outer.g_bar_m_s2.to_numpy(float), a0)
        * radius_m
        / 1e6
    )
    references = {
        "Newtonian_same_nuisance": score_predictions(outer, v_newton),
        "RAR_same_nuisance": score_predictions(
            outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float)
        ),
        "simple_MOND_same_nuisance": score_predictions(outer, v_mond),
    }
    return scores, predictions, properties, references, {
        "primary": primary.to_dict(),
        "best_overall": best.to_dict(),
        "best_baryon_extent": best_baryon.to_dict(),
        "best_mass_radius": best_mass.to_dict(),
    }


def cluster_prediction(context, morphology_row, spec):
    r80 = float(morphology_row.r80_kpc)
    concentration = float(morphology_row.radial_concentration_r50_over_r80)
    gate = float(extent_gate(concentration, spec["gate_mode"]))
    width = float(spec["width_over_R80"]) * r80
    length = float(spec["return_length_over_R80"]) * r80
    center = np.sum(context.positions * context.soft_weights[:, None], axis=0)
    inward = center[None, :] - context.positions
    radius = np.linalg.norm(inward, axis=1)
    inward /= np.maximum(radius[:, None], np.finfo(float).tiny)
    endpoint = context.positions + length * inward
    local = deposit_points(context, context.positions, context.soft_weights, width)
    mode = spec["deposition_mode"]
    if mode == "endpoint":
        routed = deposit_points(context, endpoint, context.soft_weights, width)
    else:
        samples = 17
        fraction = np.linspace(0.0, 1.0, samples)
        path = context.positions[:, None, :] + fraction[None, :, None] * (
            endpoint[:, None, :] - context.positions[:, None, :]
        )
        if mode.startswith("outward_arc_"):
            height = float(mode.removeprefix("outward_arc_")) * r80
            outward = -inward
            path += (
                4.0
                * height
                * fraction[None, :, None]
                * (1.0 - fraction[None, :, None])
                * outward[:, None, :]
            )
        routed = deposit_points(
            context,
            path.reshape(-1, 2),
            np.repeat(context.soft_weights / samples, samples),
            width,
        )
    return normalized_in_aperture((1.0 - gate) * local + gate * routed, context.aperture)


def cluster_grid(protocol: dict):
    settings = protocol["cluster_shape_test"]
    acquisition = json.loads((ROOT / settings["acquisition_protocol"]).read_text())
    audit = json.loads((ROOT / settings["input_audit"]).read_text())
    if not audit["coverage_gate_passed"]:
        raise RuntimeError("cluster input audit failed")
    sources = pd.read_csv(ROOT / settings["sources"])
    systems = pd.read_csv(ROOT / settings["systems"]).set_index("system")
    morphology = pd.read_csv(ROOT / settings["morphology"]).set_index("system")
    raw = ROOT / acquisition["acquisition"]["output_directory"]
    specs = []
    for values in itertools.product(
        settings["gate_modes"],
        settings["return_length_over_R80"],
        settings["width_over_R80"],
        settings["deposition_modes"],
    ):
        spec = dict(zip(
            ["gate_mode", "return_length_over_R80", "width_over_R80", "deposition_mode"],
            values,
            strict=True,
        ))
        spec["candidate_id"] = f"K{len(specs):04d}"
        specs.append(spec)
    rows = []
    for system in acquisition["systems"]:
        label = system["label"]
        context, world = build_source_context(
            system, systems.loc[label], sources, acquisition["spatial_preprocessing"]
        )
        predictions = {spec["candidate_id"]: cluster_prediction(
            context, morphology.loc[label], spec
        ) for spec in specs}
        models = {model["method"]: model for model in system["models"]}
        len_dir = raw / "models" / system["slug"] / "lenstool"
        target_sum = np.zeros_like(context.x_grid)
        paths = sorted((len_dir / "range").glob("*_kappa.fits"))
        for path in paths:
            target_sum += target_from_path(path, world, context, acquisition["spatial_preprocessing"])
        targets = {
            "lenstool_ensemble_mean": normalized_in_aperture(
                target_sum / len(paths), context.aperture
            ),
            "glafic_best": target_from_path(
                raw / "models" / system["slug"] / "glafic" / models["glafic"]["best_filename"],
                world,
                context,
                acquisition["spatial_preprocessing"],
            ),
        }
        for target_kind, target in targets.items():
            for spec in specs:
                rows.append({
                    "system": label,
                    "target_kind": target_kind,
                    **spec,
                    **shape_metrics(predictions[spec["candidate_id"]], target, context.aperture),
                })
        print(f"cluster grid completed {label}", flush=True)
    scores = pd.DataFrame(rows)
    aggregate = scores.groupby(
        ["candidate_id", "gate_mode", "return_length_over_R80", "width_over_R80", "deposition_mode"],
        as_index=False,
    ).agg(
        median_JS=("jensen_shannon", "median"),
        mean_JS=("jensen_shannon", "mean"),
        median_Pearson=("pearson", "median"),
    )
    best = aggregate.sort_values(["median_JS", "mean_JS"]).iloc[0].to_dict()
    primary = aggregate[
        aggregate.gate_mode.eq("cluster_logistic")
        & aggregate.return_length_over_R80.eq(0.36)
        & aggregate.width_over_R80.eq(0.23)
        & aggregate.deposition_mode.eq("endpoint")
    ].iloc[0].to_dict()
    impact_rows = []
    parameters = ["gate_mode", "return_length_over_R80", "width_over_R80", "deposition_mode"]
    for parameter in parameters:
        grouped = aggregate.groupby(parameter).median_JS.median()
        impact_rows.append({
            "domain": "cluster_shape",
            "parameter": parameter,
            "best_level": str(grouped.idxmin()),
            "best_median_JS": float(grouped.min()),
            "worst_level": str(grouped.idxmax()),
            "worst_median_JS": float(grouped.max()),
            "impact_span": float(grouped.max() - grouped.min()),
        })
    return scores, pd.DataFrame(impact_rows), {"primary": primary, "best": best}


def galaxy_impacts(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in ["scale_mode", "gate_mode", "alpha", "apogee_ratio", "screen_exponent"]:
        grouped = scores.groupby(parameter).cross_galaxy_outer_RMSE_km_s.median()
        rows.append({
            "domain": "SPARC",
            "parameter": parameter,
            "best_level": str(grouped.idxmin()),
            "best_score": float(grouped.min()),
            "worst_level": str(grouped.idxmax()),
            "worst_score": float(grouped.max()),
            "impact_span": float(grouped.max() - grouped.min()),
            "score_unit": "km/s outer RMSE",
        })
    for parameter in ["scale_mode", "gate_mode", "alpha", "apogee_ratio", "screen_exponent"]:
        grouped = scores.groupby(parameter).maximum_fractional_change_limb_to_Saturn.median()
        rows.append({
            "domain": "Solar_proxy",
            "parameter": parameter,
            "best_level": str(grouped.idxmin()),
            "best_score": float(grouped.min()),
            "worst_level": str(grouped.idxmax()),
            "worst_score": float(grouped.max()),
            "impact_span": float(grouped.max() - grouped.min()),
            "score_unit": "fractional force",
        })
    return pd.DataFrame(rows)


def make_figure(galaxy_scores, cluster_scores, impacts, references, selections, output):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axis = axes[0, 0]
    for scale, block in galaxy_scores.groupby("scale_mode"):
        values = np.sort(block.cross_galaxy_outer_RMSE_km_s.to_numpy(float))
        axis.plot(np.arange(1, len(values) + 1), values, label=scale)
    axis.axhline(references["RAR_same_nuisance"]["RMSE_km_s"], color="black", ls="--", label="fixed RAR")
    axis.set(title="SPARC variation distribution", xlabel="variant rank within scale", ylabel="outer RMSE (km/s)")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    cluster_aggregate = cluster_scores.groupby("deposition_mode").jensen_shannon.median().sort_values()
    axis.barh(cluster_aggregate.index, cluster_aggregate.values, color="tab:purple")
    axis.set(title="Cluster path-deposition impact", xlabel="median JS (lower is better)")

    axis = axes[1, 0]
    for exponent, block in galaxy_scores.groupby("screen_exponent"):
        axis.scatter(
            block.cross_galaxy_outer_RMSE_km_s,
            np.maximum(block.Earth_orbit_fractional_change, 1e-40),
            s=10,
            alpha=0.45,
            label=f"n={exponent:g}",
        )
    axis.axhline(1e-10, color="black", ls="--")
    axis.set(yscale="log", title="Galaxy accuracy versus Solar suppression", xlabel="SPARC outer RMSE (km/s)", ylabel="Earth fractional extra force")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    local = impacts.sort_values("impact_span", ascending=True)
    axis.barh(local.domain + ": " + local.parameter, local.impact_span, color="tab:green")
    axis.set(title="Parameter impact spans (domain-specific units)", xlabel="best-to-worst median score span")
    figure.suptitle("Gravity-arc apogee: tiny variations across galaxies, clusters, and Solar proxies")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    protocol_path = ROOT / "configs" / "arc_apogee_cross_domain_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    galaxy_scores, predictions, properties, references, galaxy_selection = galaxy_grid(protocol)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    predictions.to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    properties.to_csv(output / protocol["outputs"]["galaxy_properties"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["solar_scores"], index=False)
    cluster_scores, cluster_impacts, cluster_selection = cluster_grid(protocol)
    cluster_scores.to_csv(output / protocol["outputs"]["cluster_scores"], index=False)
    cluster_impacts.to_csv(output / protocol["outputs"]["cluster_impacts"], index=False)
    impacts = pd.concat([galaxy_impacts(galaxy_scores), cluster_impacts.rename(columns={
        "best_median_JS": "best_score", "worst_median_JS": "worst_score"
    }).assign(score_unit="JS divergence")], ignore_index=True, sort=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    best = galaxy_selection["best_overall"]
    best_baryon = galaxy_selection["best_baryon_extent"]
    mass = galaxy_selection["best_mass_radius"]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed exploratory arc-apogee cross-domain parameter-impact sweep",
        "protocol_sha256": sha256(protocol_path),
        "coverage": {
            "galaxy_variants": int(len(galaxy_scores)),
            "SPARC_galaxies": int(properties.galaxy.nunique()),
            "SPARC_outer_points": 968,
            "cluster_variants": int(cluster_scores.candidate_id.nunique()),
            "cluster_systems": int(cluster_scores.system.nunique()),
            "cluster_method_scores": int(len(cluster_scores)),
        },
        "references_same_RAR_nuisances": references,
        "galaxy_selection": galaxy_selection,
        "cluster_selection": cluster_selection,
        "most_impactful_parameters": impacts.sort_values("impact_span", ascending=False).to_dict("records"),
        "bottom_line": (
            "The sweep determines whether measured baryonic size or square-root mass radius, arc apogee, extent gating, and Solar screening carry the cross-domain leverage. All winners remain exploratory because the formula family followed inspection of the cluster inverse maps."
        ),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "# Arc-apogee cross-domain sweep",
        "",
        f"Best universal galaxy variant: `{best['candidate_id']}` with {best['cross_galaxy_outer_RMSE_km_s']:.3f} km/s cross-galaxy outer RMSE and q={best['universal_q']:.4g}.",
        f"Best measured-baryon-size variant: `{best_baryon['candidate_id']}` with {best_baryon['cross_galaxy_outer_RMSE_km_s']:.3f} km/s.",
        f"Best square-root-mass-radius control: `{mass['candidate_id']}` with {mass['cross_galaxy_outer_RMSE_km_s']:.3f} km/s.",
        f"Fixed-RAR same-nuisance outer RMSE: {references['RAR_same_nuisance']['RMSE_km_s']:.3f} km/s.",
        f"Primary inverse-derived cluster candidate median JS: {cluster_selection['primary']['median_JS']:.4f}; post-hoc best {cluster_selection['best']['median_JS']:.4f}.",
        "",
        "See report.json and the CSV tables for parameter impacts and Solar proxy results.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    make_figure(
        galaxy_scores,
        cluster_scores,
        impacts,
        references,
        galaxy_selection,
        output / protocol["outputs"]["figure"],
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
