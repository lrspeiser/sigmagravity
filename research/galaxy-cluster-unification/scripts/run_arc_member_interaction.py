#!/usr/bin/env python3
"""Test radial-budget-preserving member routing on new potential/path parents."""

from __future__ import annotations

import hashlib
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

from run_arc_invariant_absolute_lensing import raw_field  # noqa: E402
from run_arc_invariant_pareto_refinement import build_specs  # noqa: E402
from run_rxj2129_member_geometry import (  # noqa: E402
    load_members,
    randomized_layout,
    split_images,
)
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_LABELS,
    RawLens,
    load_baryonic_anchors,
    load_images,
    near_bound,
    score,
)
from voidscreen.member_lensing import (  # noqa: E402
    member_geometry_delta_deflection,
    point_mass_einstein_radius_squared_arcsec2,
)
from voidscreen.member_routing import normalized_member_weights  # noqa: E402


MODEL = "locked_universal_candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class ArcMemberLens(RawLens):
    """A scalar parent plus a clumpy-minus-circularized member field."""

    def __init__(
        self,
        protocol: dict,
        field,
        members: pd.DataFrame,
        routed_mass: np.ndarray,
        *,
        routing_fraction: float,
        softening_scale: float,
        layout_x: np.ndarray | None = None,
        layout_y: np.ndarray | None = None,
    ):
        super().__init__(protocol, {MODEL: field})
        self.member_x = (
            members.x_arcsec.to_numpy(float)
            if layout_x is None
            else np.asarray(layout_x, dtype=float)
        )
        self.member_y = (
            members.y_arcsec.to_numpy(float)
            if layout_y is None
            else np.asarray(layout_y, dtype=float)
        )
        self.routed_mass = np.asarray(routed_mass, dtype=float)
        self.softening = (
            members.base_softening_arcsec.to_numpy(float) * float(softening_scale)
        )
        self.routing_fraction = float(routing_fraction)
        self.lens_distance_m = float(
            self.cosmo.angular_diameter_distance(self.z_lens).to_value("m")
        )

    def alpha(self, model, parameters, x_arcsec, y_arcsec, source_redshift):
        base_x, base_y = super().alpha(
            model, parameters, x_arcsec, y_arcsec, source_redshift
        )
        if self.routing_fraction == 0.0:
            return base_x, base_y
        strength = point_mass_einstein_radius_squared_arcsec2(
            self.routed_mass,
            lens_angular_distance_m=self.lens_distance_m,
            distance_ratio=self.distance_ratio(source_redshift),
        )
        delta_x, delta_y = member_geometry_delta_deflection(
            x_arcsec,
            y_arcsec,
            self.member_x,
            self.member_y,
            strength,
            self.softening,
        )
        return (
            base_x + self.routing_fraction * delta_x,
            base_y + self.routing_fraction * delta_y,
        )


def candidate_id(parent: str, fraction: float, power: float, softening: float, dressing: str) -> str:
    return f"{parent}|f={fraction:g}|eta={power:g}|s={softening:g}|d={dressing}"


def parent_initial(parameters: pd.DataFrame, parent: str) -> np.ndarray:
    block = parameters[parameters.candidate_id.eq(parent)].set_index("parameter")
    return np.asarray([float(block.loc[label, "value"]) for label in FIXED_LABELS])


def dressing_vector(
    profile: pd.DataFrame,
    members: pd.DataFrame,
    raw_protocol: dict,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return np.ones(len(members))
    numerator = {
        "dynamic_enhancement": "dynamic_acceleration_m_s2",
        "lensing_enhancement": "lensing_acceleration_m_s2",
    }[mode]
    ratio = profile[numerator].to_numpy(float) / profile.gbar_m_s2.to_numpy(float)
    radius_kpc = (
        members.radius_arcsec_recomputed.to_numpy(float)
        * float(raw_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    )
    return np.exp(
        np.interp(
            np.log(radius_kpc),
            np.log(profile.radius_kpc.to_numpy(float)),
            np.log(ratio),
        )
    )


def routed_weights(
    members: pd.DataFrame,
    profile: pd.DataFrame,
    raw_protocol: dict,
    spec: dict,
) -> np.ndarray:
    return normalized_member_weights(
        members.expected_stellar_mass_msun.to_numpy(float),
        mass_power=float(spec["member_mass_power"]),
        radial_dressing=dressing_vector(
            profile, members, raw_protocol, str(spec["radial_dressing"])
        ),
    )


def build_lens(
    raw_protocol,
    field,
    profile,
    members,
    spec,
    *,
    layout_x=None,
    layout_y=None,
):
    weights = routed_weights(members, profile, raw_protocol, spec)
    return ArcMemberLens(
        raw_protocol,
        field,
        members,
        weights,
        routing_fraction=float(spec["routing_fraction"]),
        softening_scale=float(spec["softening_scale"]),
        layout_x=layout_x,
        layout_y=layout_y,
    )


def profiled_rms(lens: RawLens, parameters: np.ndarray, training: pd.DataFrame) -> float:
    residual, _ = lens.profiled_residuals(MODEL, parameters, training)
    xy = residual.reshape(-1, 2) * lens.sigma
    return float(np.sqrt(np.mean(np.sum(xy**2, axis=1))))


def exact_fit(
    lens: RawLens,
    training: pd.DataFrame,
    heldout: pd.DataFrame | None,
    *,
    initial: np.ndarray,
    starts: int,
    seed: int,
):
    fit = lens.fit(MODEL, training, starts=starts, seed=seed, initial_override=initial)
    train_prediction = lens.exact_predictions(
        MODEL, fit["result"].x, fit["sources"], training, stage="training"
    )
    heldout_prediction = None
    if heldout is not None:
        heldout_prediction = lens.exact_predictions(
            MODEL, fit["result"].x, fit["sources"], heldout, stage="heldout"
        )
    return {
        "parameters": fit["result"].x,
        "optimizer_cost": float(fit["result"].cost),
        "optimizer_success": bool(fit["result"].success),
        "optimization_RMS_arcsec": fit["optimization_radial_RMS_arcsec"],
        "training_predictions": train_prediction,
        "training_score": score(train_prediction, lens.sigma, free_parameters=20),
        "heldout_predictions": heldout_prediction,
        "heldout_score": (
            score(heldout_prediction, lens.sigma) if heldout_prediction is not None else None
        ),
    }


def grid_specs(protocol: dict, parent: str) -> list[dict]:
    grid = protocol["grid"]
    specs = [
        {
            "parent": parent,
            "routing_fraction": 0.0,
            "member_mass_power": 1.0,
            "softening_scale": 1.0,
            "radial_dressing": "none",
        }
    ]
    for fraction, power, softening, dressing in itertools.product(
        grid["routing_fraction"],
        grid["member_mass_power"],
        grid["softening_scale"],
        grid["radial_dressing"],
    ):
        if float(fraction) == 0.0:
            continue
        specs.append(
            {
                "parent": parent,
                "routing_fraction": float(fraction),
                "member_mass_power": float(power),
                "softening_scale": float(softening),
                "radial_dressing": str(dressing),
            }
        )
    for spec in specs:
        spec["candidate_id"] = candidate_id(
            parent,
            spec["routing_fraction"],
            spec["member_mass_power"],
            spec["softening_scale"],
            spec["radial_dressing"],
        )
    return specs


def factor_impacts(screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in (
        "parent",
        "routing_fraction",
        "member_mass_power",
        "softening_scale",
        "radial_dressing",
    ):
        medians = screen.groupby(factor).screen_training_RMS_arcsec.median()
        rows.append(
            {
                "factor": factor,
                "marginal_median_RMS_span_arcsec": float(medians.max() - medians.min()),
                "best_marginal_level": str(medians.idxmin()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "marginal_median_RMS_span_arcsec", ascending=False
    )


def make_figure(screen, final, randomizations, members, selected_weights, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    minima = screen.groupby(["parent", "routing_fraction"], as_index=False).screen_training_RMS_arcsec.min()
    for parent, block in minima.groupby("parent"):
        ax.plot(block.routing_fraction, block.screen_training_RMS_arcsec, marker="o", label=parent)
    ax.set(xlabel="routing fraction f", ylabel="best training screen RMS (arcsec)", title="Directional interaction screen")
    ax.legend()

    ax = axes[0, 1]
    positions = np.arange(len(final))
    ax.bar(positions - 0.18, final.training_RMS_arcsec, width=0.36, label="training")
    ax.bar(positions + 0.18, final.heldout_RMS_arcsec.fillna(0), width=0.36, label="heldout")
    for position, row in enumerate(final.itertuples(index=False)):
        if not np.isfinite(row.heldout_RMS_arcsec):
            ax.text(position + 0.18, 0.05, f"{row.heldout_converged_roots}/7 roots", rotation=90, ha="center", color="crimson")
    ax.set_xticks(positions, [f"{p}\n{v}" for p, v in zip(final.parent, final.variant)], rotation=15)
    ax.set(ylabel="exact RMS (arcsec)", title="Final matched-effort fits")
    ax.legend()

    ax = axes[1, 0]
    finite = randomizations[np.isfinite(randomizations.heldout_RMS_arcsec)]
    for parent, block in finite.groupby("parent"):
        ax.hist(block.heldout_RMS_arcsec, bins=14, alpha=0.45, label=parent)
    for row in final[final.variant.eq("selected")].itertuples(index=False):
        if np.isfinite(row.heldout_RMS_arcsec):
            ax.axvline(row.heldout_RMS_arcsec, ls="--", label=f"{row.parent} actual")
    ax.set(xlabel="random-angle heldout RMS (arcsec)", title="Is the measured layout special?")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    size = 15 + 90 * selected_weights / np.max(selected_weights)
    color_weight = np.log10(np.maximum(selected_weights, np.max(selected_weights) * 1.0e-6))
    scatter = ax.scatter(members.x_arcsec, members.y_arcsec, s=size, c=color_weight, cmap="viridis", alpha=0.8)
    ax.set(aspect="equal", xlabel="x (arcsec)", ylabel="y (arcsec)", title="Observed members; size = selected route weight")
    fig.colorbar(scatter, ax=ax, label="log10 routed mass weight")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    config_path = ROOT / "configs/arc_member_interaction_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    parent_protocol_path = ROOT / protocol["inputs"]["parent_protocol"]
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    raw_protocol_path = ROOT / protocol["inputs"]["raw_lensing_protocol"]
    raw_protocol = json.loads(raw_protocol_path.read_text(encoding="utf-8"))
    images = load_images(raw_protocol)
    training, heldout = split_images(images, raw_protocol)
    anchors = load_baryonic_anchors(raw_protocol)
    member_settings = {
        "path": protocol["inputs"]["member_catalog"],
        **protocol["member_catalog"],
    }
    members = load_members(member_settings, raw_protocol)

    score_path = ROOT / protocol["inputs"]["parent_scores"]
    parameter_path = ROOT / protocol["inputs"]["parent_parameters"]
    parent_scores = pd.read_csv(score_path)
    parent_parameters = pd.read_csv(parameter_path)
    parent_spec_map = {item["candidate_id"]: item for item in build_specs(parent_protocol)}
    fields, profiles, initials, scalar_rows = {}, {}, {}, {}
    for parent_item in protocol["parents"]:
        parent = parent_item["candidate_id"]
        scalar = parent_scores[parent_scores.candidate_id.eq(parent)].iloc[0]
        field, profile = raw_field(
            parent_spec_map[parent], float(scalar.universal_q), anchors, raw_protocol, 1.2e-10
        )
        fields[parent] = field
        profiles[parent] = profile
        initials[parent] = parent_initial(parent_parameters, parent)
        scalar_rows[parent] = scalar

    screen_records, spec_map = [], {}
    for parent_item in protocol["parents"]:
        parent = parent_item["candidate_id"]
        specs = grid_specs(protocol, parent)
        for index, spec in enumerate(specs):
            spec_map[spec["candidate_id"]] = spec
            lens = build_lens(raw_protocol, fields[parent], profiles[parent], members, spec)
            screen_records.append(
                {**spec, "screen_training_RMS_arcsec": profiled_rms(lens, initials[parent], training)}
            )
            if (index + 1) % 90 == 0:
                print(f"{parent} screen {index + 1}/{len(specs)}", flush=True)
    screen = pd.DataFrame(screen_records)
    screen.to_csv(output / protocol["outputs"]["screen"], index=False)

    shortlist_ids = []
    for parent, block in screen.groupby("parent", sort=False):
        for _, fraction_block in block.groupby("routing_fraction", sort=True):
            shortlist_ids.append(
                str(fraction_block.sort_values("screen_training_RMS_arcsec").iloc[0].candidate_id)
            )
    shortlist_ids = list(dict.fromkeys(shortlist_ids))
    shortlist_records, shortlist_fits = [], {}
    starts = int(protocol["selection"]["shortlist_fit_starts"])
    for index, key in enumerate(shortlist_ids):
        spec = spec_map[key]
        parent = spec["parent"]
        lens = build_lens(raw_protocol, fields[parent], profiles[parent], members, spec)
        fitted = exact_fit(
            lens,
            training,
            None,
            initial=initials[parent],
            starts=starts,
            seed=31000 + index,
        )
        shortlist_fits[key] = fitted
        metrics = fitted["training_score"]
        shortlist_records.append(
            {
                **spec,
                "training_RMS_arcsec": metrics["exact_radial_RMS_arcsec"],
                "training_converged_roots": metrics["converged_roots"],
                "training_all_roots_converged": metrics["all_roots_converged"],
                "optimizer_cost": fitted["optimizer_cost"],
            }
        )
    shortlist = pd.DataFrame(shortlist_records)
    shortlist.to_csv(output / protocol["outputs"]["shortlist"], index=False)

    chosen = {}
    for parent, block in shortlist.groupby("parent", sort=False):
        eligible = block[block.training_all_roots_converged.astype(bool)].copy()
        if eligible.empty:
            eligible = block.copy()
        eligible["selection_RMS"] = pd.to_numeric(eligible.training_RMS_arcsec, errors="coerce").fillna(np.inf)
        eligible["complexity"] = eligible.routing_fraction.abs()
        chosen[parent] = str(eligible.sort_values(["selection_RMS", "complexity"]).iloc[0].candidate_id)

    final_records, predictions, parameter_records, final_fits = [], [], [], {}
    final_starts = int(protocol["selection"]["final_fit_starts"])
    for parent_index, parent_item in enumerate(protocol["parents"]):
        parent = parent_item["candidate_id"]
        baseline_key = candidate_id(parent, 0.0, 1.0, 1.0, "none")
        for variant, key in (("baseline", baseline_key), ("selected", chosen[parent])):
            if variant == "selected" and key == baseline_key:
                fitted = final_fits[(parent, "baseline")]
            else:
                spec = spec_map[key]
                lens = build_lens(raw_protocol, fields[parent], profiles[parent], members, spec)
                seed_fit = shortlist_fits.get(key)
                start = seed_fit["parameters"] if seed_fit is not None else initials[parent]
                fitted = exact_fit(
                    lens,
                    training,
                    heldout,
                    initial=start,
                    starts=final_starts,
                    seed=41000 + 100 * parent_index + (variant == "selected"),
                )
            final_fits[(parent, variant)] = fitted
            spec = spec_map[key]
            train_score, hold_score = fitted["training_score"], fitted["heldout_score"]
            scalar = scalar_rows[parent]
            final_records.append(
                {
                    "parent": parent,
                    "variant": variant,
                    "candidate_id": key,
                    **{name: spec[name] for name in ("routing_fraction", "member_mass_power", "softening_scale", "radial_dressing")},
                    "training_RMS_arcsec": train_score["exact_radial_RMS_arcsec"],
                    "training_converged_roots": train_score["converged_roots"],
                    "heldout_RMS_arcsec": hold_score["exact_radial_RMS_arcsec"],
                    "heldout_converged_roots": hold_score["converged_roots"],
                    "optimizer_cost": fitted["optimizer_cost"],
                    "galaxy_outer_RMSE_km_s": float(scalar.cross_galaxy_outer_RMSE_km_s),
                    "CLASH_absolute_RMSE_dex": float(scalar.cluster_RMSE_dex),
                    "Solar_all_proxies_pass": bool(scalar.all_solar_proxies_pass),
                    "galaxy_and_Solar_change_by_construction": 0.0,
                }
            )
            joined = pd.concat(
                [fitted["training_predictions"], fitted["heldout_predictions"]], ignore_index=True
            )
            joined["parent"] = parent
            joined["variant"] = variant
            joined["candidate_id"] = key
            predictions.append(joined)
            bounds = near_bound(MODEL, fitted["parameters"])
            for label, value in zip(FIXED_LABELS, fitted["parameters"]):
                parameter_records.append(
                    {"parent": parent, "variant": variant, "parameter": label, "value": value, "near_bound": bounds[label]}
                )
    final = pd.DataFrame(final_records)
    final["fractional_heldout_improvement_vs_parent_baseline"] = 0.0
    for parent, block in final.groupby("parent"):
        base = float(block[block.variant.eq("baseline")].heldout_RMS_arcsec.iloc[0])
        mask = final.parent.eq(parent)
        final.loc[mask, "fractional_heldout_improvement_vs_parent_baseline"] = (
            base - final.loc[mask, "heldout_RMS_arcsec"]
        ) / base
    final.to_csv(output / protocol["outputs"]["final"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / protocol["outputs"]["predictions"], index=False)
    pd.DataFrame(parameter_records).to_csv(output / protocol["outputs"]["parameters"], index=False)

    rng = np.random.default_rng(int(protocol["randomization"]["seed"]))
    random_records = []
    for parent_item in protocol["parents"]:
        parent = parent_item["candidate_id"]
        key = chosen[parent]
        spec = spec_map[key]
        actual_parameters = final_fits[(parent, "selected")]["parameters"]
        for trial in range(int(protocol["randomization"]["fixed_geometry_trials"])):
            layout_x, layout_y = randomized_layout(members, rng)
            lens = build_lens(raw_protocol, fields[parent], profiles[parent], members, spec, layout_x=layout_x, layout_y=layout_y)
            _, sources = lens.profiled_residuals(MODEL, actual_parameters, training)
            prediction = lens.exact_predictions(MODEL, actual_parameters, sources, heldout, stage="heldout")
            metrics = score(prediction, lens.sigma)
            random_records.append(
                {"parent": parent, "mode": "fixed_geometry", "trial": trial, "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"], "heldout_converged_roots": metrics["converged_roots"]}
            )
        for trial in range(int(protocol["randomization"]["one_start_refit_trials"])):
            layout_x, layout_y = randomized_layout(members, rng)
            lens = build_lens(raw_protocol, fields[parent], profiles[parent], members, spec, layout_x=layout_x, layout_y=layout_y)
            fitted = exact_fit(lens, training, heldout, initial=actual_parameters, starts=1, seed=51000 + trial)
            metrics = fitted["heldout_score"]
            random_records.append(
                {"parent": parent, "mode": "one_start_refit", "trial": trial, "heldout_RMS_arcsec": metrics["exact_radial_RMS_arcsec"], "heldout_converged_roots": metrics["converged_roots"]}
            )
    randomizations = pd.DataFrame(random_records)
    randomizations.to_csv(output / protocol["outputs"]["randomizations"], index=False)

    impacts = factor_impacts(screen)
    impacts.to_csv(output / protocol["outputs"]["impacts"], index=False)
    weight_tables = []
    for parent, key in chosen.items():
        spec = spec_map[key]
        weights = routed_weights(members, profiles[parent], raw_protocol, spec)
        table = members[["clash_id", "x_arcsec", "y_arcsec", "radius_arcsec_recomputed", "expected_stellar_mass_msun"]].copy()
        table["parent"] = parent
        table["selected_route_weight_msun"] = weights
        table["selected_weight_share"] = weights / weights.sum()
        weight_tables.append(table)
    weights_table = pd.concat(weight_tables, ignore_index=True)
    weights_table.to_csv(output / protocol["outputs"]["member_weights"], index=False)

    random_summary = {}
    for parent in chosen:
        actual = float(final[(final.parent.eq(parent)) & final.variant.eq("selected")].heldout_RMS_arcsec.iloc[0])
        random_summary[parent] = {}
        for mode, block in randomizations[randomizations.parent.eq(parent)].groupby("mode"):
            values = pd.to_numeric(block.heldout_RMS_arcsec, errors="coerce").fillna(np.inf).to_numpy(float)
            random_summary[parent][mode] = {
                "trials": int(len(values)),
                "finite_trials": int(np.isfinite(values).sum()),
                "median_RMS_arcsec": float(np.median(values)),
                "empirical_p_random_as_good_or_better": float((1 + np.sum(values <= actual)) / (1 + len(values))),
            }

    best_parent = str(final[final.variant.eq("selected")].sort_values("heldout_RMS_arcsec").iloc[0].parent)
    best_row = final[(final.parent.eq(best_parent)) & final.variant.eq("selected")].iloc[0]
    gate = protocol["interpretation_gates"]
    report = {
        "protocol_version": protocol["protocol_version"],
        "equation": protocol["equation"],
        "inputs": {
            "members": int(len(members)),
            "training_images": int(len(training)),
            "heldout_images": int(len(heldout)),
            "screen_variants": int(len(screen)),
            "total_effective_member_stellar_mass_msun": float(members.expected_stellar_mass_msun.sum()),
        },
        "preservation_checks": {
            "net_added_radial_member_mass_msun": 0.0,
            "galaxy_prediction_change": 0.0,
            "Solar_System_prediction_change": 0.0,
            "reason": "The contrast term is exactly subtracted from its azimuthal average and is absent for the isolated axisymmetric controls.",
        },
        "selected_training_only": chosen,
        "final_scores": final.to_dict(orient="records"),
        "randomization": random_summary,
        "factor_impacts": impacts.to_dict(orient="records"),
        "verdict": {
            "best_parent": best_parent,
            "best_selected_heldout_RMS_arcsec": float(best_row.heldout_RMS_arcsec),
            "best_fractional_improvement": float(best_row.fractional_heldout_improvement_vs_parent_baseline),
            "meaningful_improvement_gate_pass": bool(best_row.fractional_heldout_improvement_vs_parent_baseline >= float(gate["meaningful_fractional_heldout_RMS_improvement"])),
            "strong_absolute_gate_pass": bool(best_row.heldout_RMS_arcsec <= float(gate["strong_absolute_heldout_RMS_arcsec"])),
            "measured_layout_randomization_gate_pass": bool(random_summary[best_parent]["fixed_geometry"]["empirical_p_random_as_good_or_better"] <= float(gate["randomization_empirical_p_max"])),
        },
        "claim_limits": protocol["claim_limits"],
        "hashes": {
            "protocol": sha256(config_path),
            "parent_protocol": sha256(parent_protocol_path),
            "raw_protocol": sha256(raw_protocol_path),
            "parent_scores": sha256(score_path),
            "parent_parameters": sha256(parameter_path),
            "member_catalog": sha256(ROOT / protocol["inputs"]["member_catalog"]),
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    selected_weight_for_plot = weights_table[weights_table.parent.eq(best_parent)].selected_route_weight_msun.to_numpy(float)
    make_figure(screen, final, randomizations, members, selected_weight_for_plot, output / protocol["outputs"]["figure"])
    summary = f"""# Arc-member interaction result

The training-only screen evaluated **{len(screen)}** radial-budget-preserving directional variants on two fixed scalar parents.  The best final parent was **{best_parent}** with held-out RX J2129 RMS **{float(best_row.heldout_RMS_arcsec):.4f} arcsec**, a **{100*float(best_row.fractional_heldout_improvement_vs_parent_baseline):.2f}%** change from the same parent's scalar baseline.

The operator changes neither the scalar galaxy predictions nor the Solar-System predictions.  It therefore isolates whether observed member directions contain useful field-routing information.  See `report.json` for the gates and random-angle controls.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report["verdict"]), indent=2))


if __name__ == "__main__":
    main()
