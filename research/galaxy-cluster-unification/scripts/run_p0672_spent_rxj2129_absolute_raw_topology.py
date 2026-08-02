#!/usr/bin/env python3
"""Fit and globally audit frozen absolute scalar/tensor RX J2129 lenses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_adaptive_route_raw_rxj2129 import json_safe
from run_p0554_multifamily_multiplicity import classify_family
from run_p0660_exact_tensor_activation_audit import sha256
from run_rxj2129_member_geometry import split_images
from run_rxj2129_raw_theory_lensing import RawLens, load_images, score

from voidscreen.raw_lensing import shear_deflection

DEFAULT_CONFIG = ROOT / "configs" / "p0672_spent_rxj2129_absolute_raw_topology.json"
MODELS = ("scalar_absolute_AQUAL", "tensor_absolute_P0669")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class PhysicalDeflectionGrid:
    def __init__(self, axis_arcsec, alpha_x_arcsec, alpha_y_arcsec):
        self.axis = np.asarray(axis_arcsec, dtype=float)
        self.minimum = float(self.axis[0])
        self.maximum = float(self.axis[-1])
        self.x_spline = RectBivariateSpline(
            self.axis,
            self.axis,
            np.asarray(alpha_x_arcsec, dtype=float),
            kx=3,
            ky=3,
            s=0.0,
        )
        self.y_spline = RectBivariateSpline(
            self.axis,
            self.axis,
            np.asarray(alpha_y_arcsec, dtype=float),
            kx=3,
            ky=3,
            s=0.0,
        )

    def sample(self, x_arcsec, y_arcsec):
        x, y = np.broadcast_arrays(
            np.asarray(x_arcsec, dtype=float),
            np.asarray(y_arcsec, dtype=float),
        )
        inside = (
            (x >= self.minimum)
            & (x <= self.maximum)
            & (y >= self.minimum)
            & (y <= self.maximum)
        )
        result_x = np.zeros_like(x)
        result_y = np.zeros_like(y)
        if np.any(inside):
            result_x[inside] = self.x_spline.ev(x[inside], y[inside])
            result_y[inside] = self.y_spline.ev(x[inside], y[inside])
        return result_x, result_y


class AbsoluteGridLens(RawLens):
    """A physical 2D deflection map with center and shear nuisances only."""

    lower = np.asarray([-3.0, -3.0, -0.25, -0.25])
    upper = np.asarray([3.0, 3.0, 0.25, 0.25])
    initial = np.zeros(4)
    labels = (
        "map_center_x_arcsec",
        "map_center_y_arcsec",
        "external_shear_gamma1_at_reference_source",
        "external_shear_gamma2_at_reference_source",
    )

    def __init__(self, protocol: dict, grids: dict[str, PhysicalDeflectionGrid]):
        super().__init__(protocol, {})
        self.grids = grids

    def alpha(self, model, parameters, x_arcsec, y_arcsec, source_redshift):
        cx, cy, gamma1, gamma2 = np.asarray(parameters, dtype=float)
        x = np.asarray(x_arcsec, dtype=float)
        y = np.asarray(y_arcsec, dtype=float)
        ratio = self.distance_ratio(float(source_redshift))
        base_x, base_y = self.grids[str(model)].sample(x - cx, y - cy)
        scale = ratio / self.distance_ratio_ref
        shear_x, shear_y = shear_deflection(x, y, gamma1 * scale, gamma2 * scale)
        return ratio * base_x + shear_x, ratio * base_y + shear_y

    def prior_residuals(self, model, parameters):
        del model
        cx, cy, gamma1, gamma2 = np.asarray(parameters, dtype=float)
        return np.asarray([cx / 1.5, cy / 1.5, gamma1 / 0.1, gamma2 / 0.1])

    def fit(self, model, rows, *, starts, seed, initial_override=None):
        rng = np.random.default_rng(seed)
        span = self.upper - self.lower
        initial = self.initial if initial_override is None else np.asarray(initial_override)
        candidates = [np.clip(initial, self.lower + 1e-6, self.upper - 1e-6)]
        for _ in range(int(starts) - 1):
            candidates.append(
                np.clip(
                    self.initial + rng.normal(0.0, 0.2, 4) * span,
                    self.lower + 1e-6,
                    self.upper - 1e-6,
                )
            )
        best = None
        for index, start in enumerate(candidates, start=1):
            result = least_squares(
                lambda values: self.objective(model, values, rows),
                start,
                bounds=(self.lower, self.upper),
                jac="2-point",
                diff_step=2e-3,
                x_scale=span,
                max_nfev=int(self.protocol["optimization"]["maximum_function_evaluations"]),
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if best is None or result.cost < best.cost:
                best = result
            print(
                f"{model} start {index:02d}/{starts}: cost={result.cost:.6g}; "
                f"best={best.cost:.6g}",
                flush=True,
            )
        data, sources = self.profiled_residuals(model, best.x, rows)
        return {
            "result": best,
            "sources": sources,
            "optimization_radial_RMS_arcsec": float(
                np.sqrt(np.mean(np.sum((data.reshape(-1, 2) * self.sigma) ** 2, axis=1)))
            ),
        }


def exact_fit(lens, model, training, heldout, protocol, seed_offset):
    nuisance = protocol["nuisance_fit"]
    fit = lens.fit(
        model,
        training,
        starts=int(nuisance["multi_starts"]),
        seed=int(nuisance["random_seed"]) + int(seed_offset),
    )
    training_prediction = lens.exact_predictions(
        model,
        fit["result"].x,
        fit["sources"],
        training,
        stage="training",
    )
    heldout_prediction = lens.exact_predictions(
        model,
        fit["result"].x,
        fit["sources"],
        heldout,
        stage="heldout",
    )
    return {
        "parameters": fit["result"].x,
        "sources": fit["sources"],
        "optimizer_cost": float(fit["result"].cost),
        "optimizer_success": bool(fit["result"].success),
        "optimization_RMS_arcsec": fit["optimization_radial_RMS_arcsec"],
        "training_prediction": training_prediction,
        "heldout_prediction": heldout_prediction,
        "training_score": score(training_prediction, lens.sigma, free_parameters=18),
        "heldout_score": score(heldout_prediction, lens.sigma),
    }


def near_bound_count(parameters: np.ndarray) -> int:
    values = np.asarray(parameters, dtype=float)
    span = AbsoluteGridLens.upper - AbsoluteGridLens.lower
    fraction = np.minimum(
        (values - AbsoluteGridLens.lower) / span,
        (AbsoluteGridLens.upper - values) / span,
    )
    return int(np.sum(fraction <= 0.01))


def critical_curve_cells(lens, model, parameters, redshift, settings):
    half = float(settings["critical_grid_half_width_arcsec"])
    spacing = float(settings["critical_grid_spacing_arcsec"])
    axis = np.arange(-half, half + 0.5 * spacing, spacing)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    jacobians = lens.jacobian(
        model,
        parameters,
        x.ravel(),
        y.ravel(),
        float(redshift),
        step=float(settings["jacobian_step_arcsec"]),
    )
    determinant = np.linalg.det(jacobians).reshape(x.shape)
    horizontal = determinant[:-1, :] * determinant[1:, :] <= 0.0
    vertical = determinant[:, :-1] * determinant[:, 1:] <= 0.0
    return int(np.sum(horizontal) + np.sum(vertical)), determinant


def global_topology(lens, model, fit, images, settings):
    root_rows = []
    assignment_rows = []
    family_rows = []
    critical_maps = {}
    variant = SimpleNamespace(variant_id=model)
    for family, group in images.groupby("source_family", sort=True):
        source = fit["sources"][int(family)]
        roots, assignments, summary = classify_family(
            lens,
            variant,
            fit["parameters"],
            source,
            group,
            settings,
            "RXJ2129",
        )
        critical_cells, determinant = critical_curve_cells(
            lens,
            model,
            fit["parameters"],
            float(group.source_redshift.median()),
            settings,
        )
        parities = {row["parity"] for row in roots}
        summary["parity_diverse"] = {"positive", "negative"}.issubset(parities)
        summary["critical_sign_change_cells"] = critical_cells
        summary["critical_curve_present"] = critical_cells > 0
        root_rows.extend(roots)
        assignment_rows.extend(assignments)
        family_rows.append(summary)
        critical_maps[int(family)] = determinant
        print(
            f"{model} family {family}: roots={summary['global_roots']} "
            f"observed={summary['observed_images']} class={summary['multiplicity_classification']} "
            f"parity={summary['parity_diverse']} critical={critical_cells}",
            flush=True,
        )
    return (
        pd.DataFrame(root_rows),
        pd.DataFrame(assignment_rows),
        pd.DataFrame(family_rows),
        critical_maps,
    )


def topology_summary(families: pd.DataFrame) -> dict:
    classes = families.multiplicity_classification
    return {
        "families": len(families),
        "missing_multiplicity_families": int(classes.eq("missing_multiplicity").sum()),
        "exact_multiplicity_families": int(classes.eq("exact_multiplicity").sum()),
        "demagnified_only_surplus_families": int(
            classes.eq("demagnified_only_surplus").sum()
        ),
        "potentially_observable_surplus_families": int(
            classes.eq("potentially_observable_surplus").sum()
        ),
        "exact_or_demagnified_only_families": int(
            classes.isin(["exact_multiplicity", "demagnified_only_surplus"]).sum()
        ),
        "parity_diverse_families": int(families.parity_diverse.astype(bool).sum()),
        "critical_curve_present_families": int(
            families.critical_curve_present.astype(bool).sum()
        ),
        "total_global_roots": int(families.global_roots.sum()),
        "total_observed_images": int(families.observed_images.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0672_raw_lens_or_topology_score":
        raise RuntimeError("P0672 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0671 parent no longer passes")
    raw = read_json(ROOT / protocol["raw_protocol"])
    images = load_images(raw)
    training, heldout = split_images(images, raw)
    scale = float(raw["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    field_path = ROOT / protocol["field_input"]
    with np.load(field_path) as data:
        axis_arcsec = data["axis_kpc"].astype(float) / scale
        grids = {
            "scalar_absolute_AQUAL": PhysicalDeflectionGrid(
                axis_arcsec,
                data["scalar_alpha_x_physical_arcsec"].astype(float),
                data["scalar_alpha_y_physical_arcsec"].astype(float),
            ),
            "tensor_absolute_P0669": PhysicalDeflectionGrid(
                axis_arcsec,
                data["tensor_alpha_x_physical_arcsec"].astype(float),
                data["tensor_alpha_y_physical_arcsec"].astype(float),
            ),
        }
    lens = AbsoluteGridLens(raw, grids)
    fitted = {}
    predictions = []
    fit_rows = []
    parameter_rows = []
    for index, model in enumerate(MODELS):
        result = exact_fit(lens, model, training, heldout, protocol, index)
        fitted[model] = result
        joined = pd.concat(
            [result["training_prediction"], result["heldout_prediction"]],
            ignore_index=True,
        )
        predictions.append(joined)
        fit_rows.append(
            {
                "model": model,
                "training_RMS_arcsec": result["training_score"]["exact_radial_RMS_arcsec"],
                "training_roots_converged": result["training_score"]["converged_roots"],
                "heldout_RMS_arcsec": result["heldout_score"]["exact_radial_RMS_arcsec"],
                "heldout_roots_converged": result["heldout_score"]["converged_roots"],
                "optimizer_cost": result["optimizer_cost"],
                "nuisance_parameters_near_bound": near_bound_count(result["parameters"]),
            }
        )
        parameter_rows.extend(
            {
                "model": model,
                "parameter": label,
                "value": float(value),
                "lower": float(lower),
                "upper": float(upper),
            }
            for label, value, lower, upper in zip(
                AbsoluteGridLens.labels,
                result["parameters"],
                AbsoluteGridLens.lower,
                AbsoluteGridLens.upper,
                strict=True,
            )
        )
    fit_scores = pd.DataFrame(fit_rows)
    topology_results = {}
    root_frames = []
    assignment_frames = []
    family_frames = []
    critical_maps = {}
    settings = protocol["global_topology"]
    for model in MODELS:
        roots, assignments, families, model_critical = global_topology(
            lens,
            model,
            fitted[model],
            images,
            settings,
        )
        root_frames.append(roots)
        assignment_frames.append(assignments)
        family_frames.append(families)
        critical_maps[model] = model_critical
        topology_results[model] = topology_summary(families)
    scalar_fit = fit_scores.set_index("model").loc["scalar_absolute_AQUAL"]
    tensor_fit = fit_scores.set_index("model").loc["tensor_absolute_P0669"]
    scalar_training = float(scalar_fit.training_RMS_arcsec)
    tensor_training = float(tensor_fit.training_RMS_arcsec)
    scalar_heldout = float(scalar_fit.heldout_RMS_arcsec)
    tensor_heldout = float(tensor_fit.heldout_RMS_arcsec)
    training_improvement = (
        1.0 - tensor_training / scalar_training
        if np.isfinite(scalar_training) and np.isfinite(tensor_training)
        else float("-inf")
    )
    heldout_worsening = (
        tensor_heldout / scalar_heldout - 1.0
        if np.isfinite(scalar_heldout) and np.isfinite(tensor_heldout)
        else float("inf")
    )
    comparator = read_json(ROOT / protocol["comparator_report"])
    compact_halo = float(
        comparator["model_scores"]["GR_plus_cluster_halo"]["heldout"][
            "exact_radial_RMS_arcsec"
        ]
    )
    tensor_halo_ratio = tensor_heldout / compact_halo if np.isfinite(tensor_heldout) else float("inf")
    tensor_topology = topology_results["tensor_absolute_P0669"]
    gates = protocol["predeclared_progression_gates"]
    accounting = protocol["models"]
    gate_results = {
        "P0671_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0671_all_progression_gates_pass"]),
        "training_roots": int(tensor_fit.training_roots_converged)
        == int(gates["tensor_training_roots_converged"]),
        "heldout_roots": int(tensor_fit.heldout_roots_converged)
        == int(gates["tensor_heldout_roots_converged"]),
        "training_improvement": training_improvement
        >= gates["tensor_training_RMS_improvement_fraction_vs_scalar_min"],
        "heldout_stability": heldout_worsening
        <= gates["tensor_heldout_RMS_worsening_fraction_vs_scalar_max"],
        "heldout_absolute_RMS": tensor_heldout <= gates["tensor_heldout_RMS_arcsec_max"],
        "compact_halo_comparison": tensor_halo_ratio
        <= gates["tensor_to_compact_halo_heldout_RMS_ratio_max"],
        "no_missing_multiplicity": tensor_topology["missing_multiplicity_families"]
        <= int(gates["tensor_missing_multiplicity_families_max"]),
        "observable_surplus": tensor_topology["potentially_observable_surplus_families"]
        <= int(gates["tensor_potentially_observable_surplus_families_max"]),
        "acceptable_multiplicity": tensor_topology["exact_or_demagnified_only_families"]
        >= int(gates["tensor_exact_or_demagnified_only_families_min"]),
        "parity_diversity": tensor_topology["parity_diverse_families"]
        >= int(gates["tensor_parity_diverse_families_min"]),
        "critical_curves": tensor_topology["critical_curve_present_families"]
        >= int(gates["tensor_critical_curve_present_families_min"]),
        "nuisance_bounds": int(tensor_fit.nuisance_parameters_near_bound)
        <= int(gates["tensor_nuisance_parameters_near_bound_max"]),
        "no_fitted_gravity": int(accounting["gravity_parameters_fit_to_RXJ2129"])
        == int(gates["gravity_parameters_fit_to_RXJ2129"]),
        "no_fitted_photon_amplitude": int(
            accounting["photon_amplitudes_fit_to_RXJ2129"]
        )
        == int(gates["photon_amplitudes_fit_to_RXJ2129"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    fit_scores.to_csv(output / "fit_scores.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "nuisance_parameters.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "exact_predictions.csv", index=False)
    roots = pd.concat(root_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    families = pd.concat(family_frames, ignore_index=True)
    roots.to_csv(output / "global_roots.csv", index=False)
    assignments.to_csv(output / "global_assignments.csv", index=False)
    families.to_csv(output / "family_topology.csv", index=False)
    report = {
        "report_version": "P0672-SPENT-RXJ2129-ABSOLUTE-RAW-TOPOLOGY-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_robustness": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "field_sha256": sha256(field_path),
        "coverage": {
            "training_images": len(training),
            "spent_heldout_images": len(heldout),
            "source_families": int(images.source_family.nunique()),
            "ordinary_nuisance_parameters": 4,
            "profiled_source_coordinates": 14,
            "gravity_parameters": int(accounting["gravity_parameters_fit_to_RXJ2129"]),
            "photon_amplitudes": int(accounting["photon_amplitudes_fit_to_RXJ2129"]),
        },
        "fit_scores": fit_scores.to_dict(orient="records"),
        "comparisons": {
            "tensor_training_improvement_fraction_vs_scalar": training_improvement,
            "tensor_heldout_worsening_fraction_vs_scalar": heldout_worsening,
            "compact_halo_heldout_RMS_arcsec": compact_halo,
            "tensor_to_compact_halo_heldout_RMS_ratio": tensor_halo_ratio,
            "published_multi_halo_reference_RMS_arcsec": 0.29,
        },
        "topology": topology_results,
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(fit_scores.model, fit_scores.training_RMS_arcsec, label="training")
    axes[0].scatter(
        fit_scores.model,
        fit_scores.heldout_RMS_arcsec,
        color="black",
        label="spent heldout",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(ylabel="exact root RMS (arcsec)", title="Absolute raw-lens score")
    axes[0].legend()
    tensor_families = families[families.variant_id.eq("tensor_absolute_P0669")]
    class_counts = tensor_families.multiplicity_classification.value_counts()
    axes[1].bar(class_counts.index, class_counts.values)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="families", title="Tensor multiplicity")
    determinant = critical_maps["tensor_absolute_P0669"][1]
    half = float(settings["critical_grid_half_width_arcsec"])
    image = axes[2].imshow(
        np.sign(determinant).T,
        origin="lower",
        extent=[-half, half, -half, half],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[2].set(title="Family 1 Jacobian sign", xlabel="x (arcsec)", ylabel="y (arcsec)")
    figure.colorbar(image, ax=axes[2], shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / "p0672_absolute_raw_topology.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0672 spent RX J2129 absolute raw topology

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Scalar/tensor training RMS: **{scalar_training:.4g} / {tensor_training:.4g} arcsec**.
- Scalar/tensor spent-heldout RMS: **{scalar_heldout:.4g} / {tensor_heldout:.4g} arcsec**.
- Tensor training improvement / heldout worsening: **{100*training_improvement:+.3g}% / {100*heldout_worsening:+.3g}%**.
- Tensor missing / exact-or-demagnified / observable-surplus families: **{tensor_topology['missing_multiplicity_families']} / {tensor_topology['exact_or_demagnified_only_families']} / {tensor_topology['potentially_observable_surplus_families']}**.
- Tensor parity-diverse / critical-curve families: **{tensor_topology['parity_diverse_families']} / {tensor_topology['critical_curve_present_families']}** of 7.
- Tensor/compact-halo heldout RMS ratio: **{tensor_halo_ratio:.4g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633/P0640 targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
