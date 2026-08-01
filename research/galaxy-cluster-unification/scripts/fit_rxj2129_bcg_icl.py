"""Fit frozen PSF-convolved one- and two-component RX J2129 light models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import fftconvolve
from scipy.special import gamma, gammainc, gammaincinv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_bcg_icl_protocol.json"


def _resolve(path: str) -> Path:
    return ROOT / path


def _regularized_whitener(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(np.diag(covariance))
    regularized = 0.95 * covariance + 0.05 * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    positive_diagonal = np.diag(regularized)[np.diag(regularized) > 0]
    floor = 1e-6 * float(np.median(positive_diagonal))
    eigenvalues = np.maximum(eigenvalues, floor)
    return np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def _sersic_flux_within(radius: float, re_arcsec: float, n: float, amplitude: float) -> float:
    bn = gammaincinv(2.0 * n, 0.5)
    total = (
        2.0
        * np.pi
        * amplitude
        * re_arcsec**2
        * n
        * np.exp(bn)
        * bn ** (-2.0 * n)
        * gamma(2.0 * n)
    )
    return float(total * gammainc(2.0 * n, bn * (radius / re_arcsec) ** (1.0 / n)))


@dataclass
class FitResult:
    model: str
    structural: np.ndarray
    coefficients: np.ndarray
    chi2: float
    residual: np.ndarray
    prediction: np.ndarray
    success: bool
    evaluations: int


class ProfileModeler:
    def __init__(
        self,
        profile: pd.DataFrame,
        psf_f125w: np.ndarray,
        psf_f814w: np.ndarray,
        pixel_scale: float,
        fit_radius: float,
    ) -> None:
        self.profile = profile
        self.usable = profile["profile_gate_usable"].astype(bool).to_numpy()
        self.usable_indices = np.flatnonzero(self.usable)
        self.edges = np.concatenate(
            [
                profile["radius_min_arcsec"].to_numpy()[:1],
                profile["radius_max_arcsec"].to_numpy(),
            ]
        )
        self.pixel_scale = pixel_scale
        self.fit_radius = fit_radius
        maximum_psf_half = max(psf_f125w.shape[0], psf_f814w.shape[0]) // 2
        grid_half = int(np.ceil(fit_radius / pixel_scale)) + maximum_psf_half + 2
        yy, xx = np.indices((2 * grid_half + 1, 2 * grid_half + 1), dtype=float)
        self.radius = np.hypot(xx - grid_half, yy - grid_half) * pixel_scale
        self.bin_index = np.searchsorted(self.edges, self.radius, side="right") - 1
        self.radial_valid = (
            (self.bin_index >= 0)
            & (self.bin_index < len(self.edges) - 1)
            & (self.radius <= fit_radius)
        )
        self.bin_counts = np.bincount(
            self.bin_index[self.radial_valid], minlength=len(self.edges) - 1
        ).astype(float)
        self.psfs = {
            "F125W": np.clip(np.asarray(psf_f125w, dtype=float), 0.0, None),
            "F814W": np.clip(np.asarray(psf_f814w, dtype=float), 0.0, None),
        }
        for key in self.psfs:
            self.psfs[key] /= self.psfs[key].sum()
        self._cache: dict[tuple[float, float, str], np.ndarray] = {}

    def template(self, re_arcsec: float, n: float, filter_name: str) -> np.ndarray:
        key = (float(re_arcsec), float(n), filter_name)
        if key in self._cache:
            return self._cache[key]
        bn = gammaincinv(2.0 * n, 0.5)
        image = np.exp(-bn * ((self.radius / re_arcsec) ** (1.0 / n) - 1.0))
        convolved = fftconvolve(image, self.psfs[filter_name], mode="same")
        sums = np.bincount(
            self.bin_index[self.radial_valid],
            weights=convolved[self.radial_valid],
            minlength=len(self.edges) - 1,
        )
        radial = np.divide(
            sums,
            self.bin_counts,
            out=np.full_like(sums, np.nan),
            where=self.bin_counts > 0,
        )[self.usable]
        self._cache[key] = radial
        return radial

    @staticmethod
    def decode(model: str, parameters: np.ndarray) -> list[tuple[float, float]]:
        if model == "one_component":
            return [(float(np.exp(parameters[0])), float(parameters[1]))]
        re_inner = float(np.exp(parameters[0]))
        re_outer = float(
            np.exp(np.log(2.0 * re_inner) + parameters[2] * (np.log(80.0) - np.log(2.0 * re_inner)))
        )
        return [(re_inner, float(parameters[1])), (re_outer, float(parameters[3]))]

    def design(self, model: str, parameters: np.ndarray) -> np.ndarray:
        components = self.decode(model, parameters)
        count = len(self.usable_indices)
        columns = 2 * len(components) + 2
        design = np.zeros((2 * count, columns), dtype=float)
        for index, (re_arcsec, n) in enumerate(components):
            design[:count, index] = self.template(re_arcsec, n, "F125W")
            design[count:, len(components) + 1 + index] = self.template(
                re_arcsec, n, "F814W"
            )
        design[:count, len(components)] = 1.0
        design[count:, -1] = 1.0
        return design


def _coefficient_bounds(components: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate([np.zeros(components), [-np.inf], np.zeros(components), [-np.inf]])
    upper = np.full(2 * components + 2, np.inf)
    return lower, upper


def _fit_model(
    modeler: ProfileModeler,
    model: str,
    y: np.ndarray,
    covariance: np.ndarray,
    selected: np.ndarray,
    seed: int,
    multistarts: int = 20,
) -> FitResult:
    selected = np.asarray(selected, dtype=int)
    whitener = _regularized_whitener(covariance[np.ix_(selected, selected)])
    y_selected = y[selected]
    components = 1 if model == "one_component" else 2
    coefficient_bounds = _coefficient_bounds(components)
    if model == "one_component":
        lower = np.asarray([np.log(0.10), 0.3])
        upper = np.asarray([np.log(80.0), 8.0])
        fixed_starts = [np.asarray([np.log(11.06), 2.70])]
    else:
        lower = np.asarray([np.log(0.10), 0.3, 0.0, 0.3])
        upper = np.asarray([np.log(40.0), 8.0, 1.0, 8.0])
        fixed_starts = [np.asarray([np.log(3.0), 3.0, 0.35, 1.5])]
    rng = np.random.default_rng(seed)
    starts = fixed_starts + [rng.uniform(lower, upper) for _ in range(multistarts - 1)]
    best: FitResult | None = None

    def solve_coefficients(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = modeler.design(model, parameters)
        design_selected = design[selected]
        solution = lsq_linear(
            whitener @ design_selected,
            whitener @ y_selected,
            bounds=coefficient_bounds,
            method="trf",
            lsmr_tol="auto",
        )
        prediction = design @ solution.x
        residual_selected = whitener @ (y_selected - prediction[selected])
        return solution.x, prediction, residual_selected

    for start in starts:
        evaluation = least_squares(
            lambda parameters: solve_coefficients(parameters)[2],
            start,
            bounds=(lower, upper),
            max_nfev=100,
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
        )
        coefficients, prediction, residual = solve_coefficients(evaluation.x)
        candidate = FitResult(
            model=model,
            structural=evaluation.x,
            coefficients=coefficients,
            chi2=float(residual @ residual),
            residual=residual,
            prediction=prediction,
            success=bool(evaluation.success),
            evaluations=int(evaluation.nfev),
        )
        if best is None or candidate.chi2 < best.chi2:
            best = candidate
    assert best is not None
    return best


def _chi2(
    residual: np.ndarray, covariance: np.ndarray, selected: np.ndarray
) -> float:
    selected = np.asarray(selected, dtype=int)
    whitener = _regularized_whitener(covariance[np.ix_(selected, selected)])
    transformed = whitener @ residual[selected]
    return float(transformed @ transformed)


def _result_row(
    label: str,
    fit: FitResult,
    modeler: ProfileModeler,
    selected_count: int,
    heldout_f125_chi2: float | None = None,
    heldout_f814_chi2: float | None = None,
) -> dict[str, Any]:
    components = modeler.decode(fit.model, fit.structural)
    count = len(components)
    row: dict[str, Any] = {
        "variant": label,
        "model": fit.model,
        "fit_points": selected_count,
        "chi2": fit.chi2,
        "success": fit.success,
        "evaluations": fit.evaluations,
        "re_inner_arcsec": components[0][0],
        "n_inner": components[0][1],
        "f125w_inner_amplitude": fit.coefficients[0],
        "f125w_sky": fit.coefficients[count],
        "f814w_inner_amplitude": fit.coefficients[count + 1],
        "f814w_sky": fit.coefficients[-1],
        "heldout_f125w_chi2": heldout_f125_chi2,
        "heldout_f814w_chi2": heldout_f814_chi2,
    }
    if count == 2:
        row.update(
            {
                "re_outer_arcsec": components[1][0],
                "n_outer": components[1][1],
                "f125w_outer_amplitude": fit.coefficients[1],
                "f814w_outer_amplitude": fit.coefficients[count + 2],
                "outer_to_inner_re_ratio": components[1][0] / components[0][0],
            }
        )
        inner_flux = _sersic_flux_within(
            30.0, components[0][0], components[0][1], fit.coefficients[0]
        )
        outer_flux = _sersic_flux_within(
            30.0, components[1][0], components[1][1], fit.coefficients[1]
        )
        row["f125w_outer_light_fraction_within_30arcsec"] = outer_flux / (
            inner_flux + outer_flux
        )
    return row


def _plot(
    path: Path,
    profile: pd.DataFrame,
    one: FitResult,
    two: FitResult,
) -> None:
    usable = profile["profile_gate_usable"].astype(bool).to_numpy()
    radius = profile.loc[usable, "radius_mid_arcsec"].to_numpy()
    count = len(radius)
    data = [
        profile.loc[usable, "f125w_surface_brightness"].to_numpy(),
        profile.loc[usable, "f814w_surface_brightness"].to_numpy(),
    ]
    errors = [
        profile.loc[usable, "f125w_surface_brightness_error"].to_numpy(),
        profile.loc[usable, "f814w_surface_brightness_error"].to_numpy(),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    for index, label in enumerate(("F125W", "F814W")):
        slc = slice(index * count, (index + 1) * count)
        one_sky = one.coefficients[1 if index == 0 else -1]
        two_sky = two.coefficients[2 if index == 0 else -1]
        one_signal = one.prediction[slc] - one_sky
        two_signal = two.prediction[slc] - two_sky
        data_signal = data[index] - two_sky
        positive = (data_signal > 0) & (one_signal > 0) & (two_signal > 0)
        axes[index, 0].errorbar(
            radius[positive],
            data_signal[positive],
            yerr=errors[index][positive],
            fmt=".",
            ms=3,
            color="black",
            label="profile",
        )
        axes[index, 0].plot(radius[positive], one_signal[positive], label="one Sersic")
        axes[index, 0].plot(radius[positive], two_signal[positive], label="two Sersic")
        axes[index, 0].set(xscale="log", yscale="log", ylabel=f"{label} sky-subtracted")
        axes[index, 0].grid(alpha=0.25)
        axes[index, 0].legend(fontsize=8)
        axes[index, 1].plot(
            radius,
            (data[index] - one.prediction[slc]) / errors[index],
            label="one Sersic",
        )
        axes[index, 1].plot(
            radius,
            (data[index] - two.prediction[slc]) / errors[index],
            label="two Sersic",
        )
        axes[index, 1].axhline(0, color="black", linewidth=0.8)
        axes[index, 1].set(xscale="log", ylabel="diagonal residual / sigma")
        axes[index, 1].grid(alpha=0.25)
        axes[index, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel("radius (arcsec)")
    axes[1, 1].set_xlabel("radius (arcsec)")
    fig.suptitle("RX J2129 frozen PSF-convolved baseline and radial CV candidates")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["authorization"]["gravity_response_fit"]:
        raise ValueError("BCG/ICL protocol cannot authorize gravity fitting")
    profile_report = json.loads(
        _resolve(config["outputs"]["profile_extraction_report"]).read_text(encoding="utf-8")
    )
    if not profile_report["profile_extraction_gate_pass"]:
        raise ValueError("Nonparametric HST profile gate failed")
    profile = pd.read_csv(_resolve(config["outputs"]["nonparametric_profile"]))
    usable = profile["profile_gate_usable"].astype(bool).to_numpy()
    count = int(usable.sum())
    y = np.concatenate(
        [
            profile.loc[usable, "f125w_surface_brightness"].to_numpy(),
            profile.loc[usable, "f814w_surface_brightness"].to_numpy(),
        ]
    )
    covariance = pd.read_csv(
        _resolve(config["outputs"]["profile_covariance"]), index_col="row"
    ).to_numpy()
    psfs = np.load(_resolve(config["inputs"]["empirical_psf"]))
    modeler = ProfileModeler(
        profile,
        psfs["f125w"],
        psfs["f814w"],
        config["geometry"]["pixel_scale_arcsec"],
        config["geometry"]["fit_radius_arcsec"],
    )
    full = np.arange(2 * count)
    one = _fit_model(modeler, "one_component", y, covariance, full, seed=2129)
    two = _fit_model(modeler, "two_component", y, covariance, full, seed=2130)
    rows = [
        _result_row("baseline_full", one, modeler, len(full)),
        _result_row("baseline_full", two, modeler, len(full)),
    ]

    usable_bin_numbers = profile.loc[usable, "radial_bin"].to_numpy(dtype=int)
    heldout_totals = {
        "one_component": {"F125W": 0.0, "F814W": 0.0},
        "two_component": {"F125W": 0.0, "F814W": 0.0},
    }
    for fold in (0, 1):
        hold_radial = np.flatnonzero(usable_bin_numbers % 2 == fold)
        train_radial = np.flatnonzero(usable_bin_numbers % 2 != fold)
        train = np.concatenate([train_radial, count + train_radial])
        hold_f125 = hold_radial
        hold_f814 = count + hold_radial
        for model_index, model in enumerate(("one_component", "two_component")):
            result = _fit_model(
                modeler,
                model,
                y,
                covariance,
                train,
                seed=2200 + 10 * fold + model_index,
            )
            residual = y - result.prediction
            chi_f125 = _chi2(residual, covariance, hold_f125)
            chi_f814 = _chi2(residual, covariance, hold_f814)
            heldout_totals[model]["F125W"] += chi_f125
            heldout_totals[model]["F814W"] += chi_f814
            rows.append(
                _result_row(
                    f"radial_cv_fold_{fold}",
                    result,
                    modeler,
                    len(train),
                    chi_f125,
                    chi_f814,
                )
            )

    variants = pd.DataFrame(rows)
    output_path = _resolve(config["outputs"]["model_variants"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants.to_csv(output_path, index=False)
    _plot(_resolve(config["outputs"]["diagnostic"]), profile, one, two)

    improvements = {
        band: 1.0
        - heldout_totals["two_component"][band] / heldout_totals["one_component"][band]
        for band in ("F125W", "F814W")
    }
    baseline_two = rows[1]
    gates = config["component_identifiability_gate"]
    cross_validation_pass = all(
        value >= gates["minimum_two_component_heldout_chi2_improvement_fraction_both_bands"]
        for value in improvements.values()
    )
    radius_ratio_pass = bool(
        baseline_two["outer_to_inner_re_ratio"]
        >= gates["outer_to_inner_effective_radius_minimum"]
    )
    fraction_range = gates["icl_f125w_light_fraction_within_30arcsec_range"]
    fraction_pass = bool(
        fraction_range[0]
        <= baseline_two["f125w_outer_light_fraction_within_30arcsec"]
        <= fraction_range[1]
    )
    decisive_baseline_failure = not (
        cross_validation_pass and radius_ratio_pass and fraction_pass
    )
    if decisive_baseline_failure:
        status = "bcg_icl_nonidentifiable_predeclared_baseline_gate_failed"
        sensitivity_status = (
            "not_run_because_the_conjunctive_baseline_gate_already_failed"
        )
        next_action = (
            "Retain the frozen nonparametric total-light profile and record BCG/ICL "
            "non-identifiability. Do not map the failed component split to stellar "
            "mass; continue the independent gas, satellite, lens-likelihood, and "
            "30-host inventory workstreams without reading a gravity residual."
        )
    else:
        status = "baseline_and_radial_cross_validation_pass_sensitivity_grid_pending"
        sensitivity_status = "pending"
        next_action = (
            "Execute the frozen mask, background-annulus, and three leave-one-star-out "
            "PSF variants. Accept the two-component split only if its light fraction "
            "and cumulative profile remain within the predeclared ranges."
        )
    report = {
        "protocol_version": config["protocol_version"],
        "status": status,
        "gravity_or_lens_residual_read": False,
        "data_points_joint": int(2 * count),
        "baseline": {
            "one_component_chi2": one.chi2,
            "two_component_chi2": two.chi2,
            "two_component": baseline_two,
        },
        "radial_cross_validation": {
            "heldout_chi2": heldout_totals,
            "two_component_improvement_fraction": improvements,
            "minimum_improvement_fraction_each_band": gates[
                "minimum_two_component_heldout_chi2_improvement_fraction_both_bands"
            ],
            "gate_pass": cross_validation_pass,
        },
        "structural_gate": {
            "outer_to_inner_radius_ratio_pass": radius_ratio_pass,
            "f125w_outer_light_fraction_pass": fraction_pass,
            "allowed_f125w_outer_light_fraction_range": fraction_range,
        },
        "total_light_shape_result": (
            "two_sersic_terms_are_predictively_required_but_not_identifiable_as_bcg_and_icl"
            if cross_validation_pass
            else "two_sersic_terms_not_predictively_required"
        ),
        "bcg_icl_nonidentifiability_explicit": decisive_baseline_failure,
        "sensitivity_grid_status": sensitivity_status,
        "sensitivity_grid_complete": False,
        "component_identifiability_gate_pass": False,
        "stellar_mass_mapping_authorized": False,
        "strict_r1_ready": False,
        "outputs": {
            "model_variants": config["outputs"]["model_variants"],
            "diagnostic": config["outputs"]["diagnostic"],
        },
        "next_action": next_action,
    }
    report_path = _resolve(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    print(json.dumps(fit(arguments.config), indent=2))


if __name__ == "__main__":
    main()
