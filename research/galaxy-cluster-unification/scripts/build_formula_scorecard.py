"""Build one auditable scorecard for every scientifically distinct tested law.

The percentages are descriptive normalizations, not probabilities:

* velocity proximity = 100 * max(0, 1 - RMSE / RMS(observed velocity));
* log-acceleration proximity = 100 * 10**(-RMSE_dex);
* raw-lens proximity = 100 * max(0, 1 - image_RMS / RMS(observed radius)).

Raw image positions and GR/NFW-derived acceleration products are deliberately
kept in different columns.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "formula_scorecard"

SPARC_OUTER_OBS_RMS = 166.513237512293
SPARC_ALL_OBS_RMS = 173.48993550840146
RAW_DENOMINATORS = {
    "RXJ2129: 7 held-out images": 13.515543366344449,
    "MACS1115+1931: 6 held-out images": 27.177197079824552,
    "MACS0329+0429+1115+1931: 11 held-out images": math.sqrt(
        sum(x * x for x in (30.9227, 11.5525, 30.5024, 23.3838)) / 4.0
    ),
}


def load(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def velocity_proximity(rmse: float | None, observed_rms: float) -> float | None:
    if rmse is None or not math.isfinite(rmse):
        return None
    return 100.0 * max(0.0, 1.0 - rmse / observed_rms)


def dex_proximity(rmse: float | None) -> float | None:
    if rmse is None or not math.isfinite(rmse):
        return None
    return 100.0 * 10.0 ** (-rmse)


def raw_proximity(rmse: float | None, test: str) -> float | None:
    if rmse is None or not math.isfinite(rmse):
        return None
    return 100.0 * max(0.0, 1.0 - rmse / RAW_DENOMINATORS[test])


rows: list[dict] = []


def add(
    family: str,
    name: str,
    formula: str,
    *,
    galaxy_error: float | None = None,
    galaxy_kind: str | None = None,
    galaxy_test: str = "",
    derived_error: float | None = None,
    derived_test: str = "",
    raw_error: float | None = None,
    raw_test: str = "",
    raw_complete: bool | None = None,
    verdict: str,
    evidence: str,
) -> None:
    if galaxy_kind == "SPARC outer velocity":
        galaxy_pct = velocity_proximity(galaxy_error, SPARC_OUTER_OBS_RMS)
        galaxy_unit = "km/s"
    elif galaxy_kind == "SPARC all-point velocity":
        galaxy_pct = velocity_proximity(galaxy_error, SPARC_ALL_OBS_RMS)
        galaxy_unit = "km/s"
    elif galaxy_kind == "observed log acceleration":
        galaxy_pct = dex_proximity(galaxy_error)
        galaxy_unit = "dex"
    else:
        galaxy_pct = None
        galaxy_unit = ""

    if raw_complete is False:
        raw_pct = None
    elif raw_error is not None and raw_test:
        raw_pct = raw_proximity(raw_error, raw_test)
    else:
        raw_pct = None

    rows.append(
        {
            "family": family,
            "formula": name,
            "schematic_equation": formula,
            "galaxy_proximity_percent": galaxy_pct,
            "galaxy_error": galaxy_error,
            "galaxy_error_unit": galaxy_unit,
            "galaxy_test": galaxy_test,
            "derived_lensing_proximity_percent": dex_proximity(derived_error),
            "derived_lensing_error_dex": derived_error,
            "derived_lensing_test": derived_test,
            "raw_lensing_proximity_percent": raw_pct,
            "raw_lensing_error_arcsec": raw_error,
            "raw_lensing_test": raw_test,
            "all_raw_roots_complete": raw_complete,
            "verdict": verdict,
            "evidence": evidence,
        }
    )


# Established controls and the bridge formulas.
sparc = load("results/sparc_independent_nuisance_refit/report.json")
phen = load("results/phenomenology_formula_sweep/report.json")
raw_rxj = load("results/rxj2129_raw_theory_lensing/report.json")
spherical_raw = load("results/spherical_spacetime_cavity/raw_lensing_report.json")
one_parameter_raw = load("results/one_parameter_multicluster_lens/report.json")
solar_screened_raw = load("results/solar_screened_isothermal/report.json")
solar_screened_galaxies = load(
    "results/solar_screened_galaxy_morphology/report.json"
)
profile_diffusion = load(
    "results/reopened_hybrid_profile_diffusion_analysis/report.json"
)

outer = {
    key: value["outer_holdout"]["RMSE_km_s"] for key, value in sparc["scores"].items()
}
phen_equations = {
    "Newtonian": "g = g_bar",
    "fixed_galaxy_RAR": "g/g_bar = [1-exp(-sqrt(g_bar/a0))]^-1",
    "simple_MOND": "g/g_bar = (1+sqrt(1+4 a0/g_bar))/2",
    "cluster_scale_RAR_diagnostic": "RAR law with a0 retuned to cluster data",
    "RG": "g = g_bar / epsilon(rho_b)",
    "RG_acceleration_threshold": "g = g_bar / epsilon[rho_b; rho_c(g_bar)]",
    "RG_potential_threshold": "g = g_bar / epsilon[rho_b; rho_c(Phi_bar)]",
    "RG_acceleration_floor": "g = g_bar / epsilon[rho_b; epsilon0(g_bar)]",
    "RG_Sigma_additive": "E = E_RG + B E_Sigma",
    "RG_Sigma_quadrature": "E = sqrt(E_RG^2 + (B E_Sigma)^2)",
    "RG_Sigma_product": "E = E_RG (1 + B E_Sigma)",
    "RG_density_gated_Sigma": "E = E_RG + B S(rho_b) E_Sigma",
    "linear_g_rho": "log g = b0 + b1 log g_bar + b2 log rho_b",
    "quadratic_g_rho": "log g = quadratic(log g_bar, log rho_b)",
    "quadratic_g_rho_potential": "log g = quadratic(log g_bar, log rho_b, Phi_bar/c^2)",
}
pretty = {
    "fixed_galaxy_RAR": "Fixed RAR",
    "simple_MOND": "Simple MOND",
    "cluster_scale_RAR_diagnostic": "Cluster-retuned RAR",
    "RG": "Density-only refracted gravity",
    "RG_acceleration_threshold": "RG with acceleration-moving density threshold",
    "RG_potential_threshold": "RG with potential-moving density threshold",
    "RG_acceleration_floor": "RG with acceleration-dependent floor",
    "RG_Sigma_additive": "RG + Sigma additive",
    "RG_Sigma_quadrature": "RG + Sigma quadrature",
    "RG_Sigma_product": "RG x Sigma product",
    "RG_density_gated_Sigma": "Density-gated Sigma/RG",
    "linear_g_rho": "Linear g-rho surface",
    "quadratic_g_rho": "Quadratic g-rho surface",
    "quadratic_g_rho_potential": "Quadratic g-rho-potential surface",
}

raw_two = spherical_raw["comparators"]
for key, metric in phen["metrics"].items():
    galaxy_error = metric["BCG"]["equal_system_RMSE_dex"]
    galaxy_kind = "observed log acceleration"
    galaxy_test = "44 observed BCG dynamical-acceleration points"
    raw_error = None
    raw_test = ""
    raw_complete = None
    verdict = "diagnostic only; no independent SPARC transfer"

    if key == "Newtonian":
        galaxy_error = 60.721
        galaxy_kind = "SPARC all-point velocity"
        galaxy_test = "131 SPARC galaxies, 3,034 held-out whole-galaxy CV points"
        raw_error = raw_two["baryons_GR"]["equal_system_radial_RMS_arcsec"]
        raw_test = "MACS1115+1931: 6 held-out images"
        raw_complete = True
        verdict = "baseline fails galaxies and clusters"
    elif key == "fixed_galaxy_RAR":
        galaxy_error = outer["fixed_RAR:invariant"]
        galaxy_kind = "SPARC outer velocity"
        galaxy_test = "131 SPARC galaxies, 968 untouched outer points"
        raw_error = 25.673186
        raw_test = "MACS1115+1931: 6 held-out images"
        raw_complete = True
        verdict = "excellent galaxy control; fails raw cluster lensing"
    elif key == "simple_MOND":
        galaxy_error = outer["simple_MOND:invariant"]
        galaxy_kind = "SPARC outer velocity"
        galaxy_test = "131 SPARC galaxies, 968 untouched outer points"
        raw_error = raw_two["fixed_simple_MOND"]["equal_system_radial_RMS_arcsec"]
        raw_test = "MACS1115+1931: 6 held-out images"
        raw_complete = True
        verdict = "excellent galaxies; fails cluster lensing"
    elif key == "cluster_scale_RAR_diagnostic":
        rxj = raw_rxj["model_scores"]["cluster_retuned_RAR_diagnostic"]["heldout"]
        raw_error = rxj["exact_radial_RMS_arcsec"]
        raw_test = "RXJ2129: 7 held-out images"
        raw_complete = rxj["all_roots_converged"]
        verdict = "cluster-only retuning; not universal"

    add(
        "Controls and bridge",
        pretty.get(key, key.replace("_", " ")),
        phen_equations[key],
        galaxy_error=galaxy_error,
        galaxy_kind=galaxy_kind,
        galaxy_test=galaxy_test,
        derived_error=metric["cluster"]["equal_system_RMSE_dex"],
        derived_test="20 CLASH clusters, 72 GR/NFW-derived acceleration points",
        raw_error=raw_error,
        raw_test=raw_test,
        raw_complete=raw_complete,
        verdict=verdict,
        evidence="phenomenology_formula_sweep; SPARC/raw overrides where available",
    )

# The present RAR + coherence-gated RG/Sigma candidate.
candidate_raw = raw_rxj["model_scores"]["locked_universal_candidate"]["heldout"]
add(
    "Controls and bridge",
    "RAR + squared coherence-gated RG (current empirical bridge)",
    "E = E_RAR + [1-(3C^2-2C^3)]^2 (epsilon(rho_b)^-1 - 1)",
    galaxy_error=outer["RAR_sharp_coherence_gated_RG:primary"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 untouched outer points",
    derived_error=0.13870141328241087,
    derived_test="20 CLASH clusters, 72 GR/NFW-derived acceleration points",
    raw_error=candidate_raw["exact_radial_RMS_arcsec"],
    raw_test="RXJ2129: 7 held-out images",
    raw_complete=candidate_raw["all_roots_converged"],
    verdict="best empirical bridge; one-cluster raw success, not yet multi-cluster validated",
    evidence="sparc_independent_nuisance_refit; clash_lensing_universal_comparison; rxj2129_raw_theory_lensing",
)

# Dark-matter controls need separate rows because the lens datasets and halo
# flexibility differ.
add(
    "Dark-matter controls",
    "Per-galaxy NFW halo",
    "rho(r)=rho_s/[x(1+x)^2], x=r/r_s",
    galaxy_error=outer["NFW:invariant"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 outer points; two halo parameters fit per galaxy",
    verdict="dark-matter galaxy control; flexible per galaxy and weak outer extrapolation here",
    evidence="sparc_independent_nuisance_refit",
)
add(
    "Dark-matter controls",
    "Compact cluster halo (RXJ2129)",
    "alpha = alpha_baryons + alpha_compact_halo",
    raw_error=raw_rxj["model_scores"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"],
    raw_test="RXJ2129: 7 held-out images",
    raw_complete=True,
    verdict="limited one-halo dark-matter comparator",
    evidence="rxj2129_raw_theory_lensing",
)
add(
    "Dark-matter controls",
    "Compact cluster halo (four-cluster transfer)",
    "alpha = alpha_baryons + alpha_compact_halo",
    raw_error=9.048410306058654,
    raw_test="MACS0329+0429+1115+1931: 11 held-out images",
    raw_complete=True,
    verdict="best four-cluster raw comparator, though still inadequate in two systems",
    evidence="unbounded_running_multicluster_raw",
)
add(
    "Dark-matter controls",
    "CLASH NFW construction",
    "g_obs is deprojected from the same fitted NFW lensing profile",
    derived_error=0.0,
    derived_test="20 CLASH clusters; construction target, not independent validation",
    verdict="100% by construction; not an independent prediction",
    evidence="clash_lensing_universal_comparison",
)

# The frozen one-universal-parameter multi-cluster raw-lens search. This is kept
# separate from the galaxy bridge because no galaxy or Solar-System completion
# has yet been demonstrated.
add(
    "One-parameter cluster laws",
    "Baryon-normalized isothermal tail (lambda=9)",
    "g=g_bar+9 g_bar(200 kpc)(200 kpc/r)",
    raw_error=one_parameter_raw["validation"]["selected_law"][
        "equal_system_radial_RMS_arcsec"
    ],
    raw_test="MACS1115+1931: 6 held-out images",
    raw_complete=one_parameter_raw["validation"]["selected_law"][
        "all_roots_converged"
    ],
    verdict=(
        "one shared parameter improves both replay-holdout clusters and narrowly "
        "beats the compact-halo equal-cluster aggregate, but fails the 2-arcsec "
        "gate and the RXJ2129 stress comparison"
    ),
    evidence="one_parameter_multicluster_lens",
)
add(
    "One-parameter cluster laws",
    "Solar-screened baryon-normalized isothermal tail",
    "g=g_bar+lambda g_bar(200 kpc)(200 kpc/r) a0/(a0+g_bar); lambda=10.5",
    galaxy_error=solar_screened_galaxies["overall_outer_scores"][
        "solar_screened_isothermal"
    ]["RMSE_km_s"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 untouched outer points; morphology-stratified",
    raw_error=solar_screened_raw["validation"]["aggregate"][
        "equal_system_radial_RMS_arcsec"
    ],
    raw_test="MACS1115+1931: 6 held-out images",
    raw_complete=solar_screened_raw["validation"]["aggregate"][
        "all_roots_converged"
    ],
    verdict=(
        "one shared parameter passes the published Mercury-margin diagnostic and "
        "beats the limited compact-halo replay aggregate, but fails the frozen galaxy "
        "transfer (18.60 versus 10.35 km/s for RAR), the earlier 2-arcsec target, "
        "and the RXJ2129 halo comparison"
    ),
    evidence="solar_screened_isothermal; solar_screened_galaxy_morphology",
)

# Unified low-acceleration laws.
unified = load("results/unified_cv/report.json")
unified_eq = {
    "joint_a0": "RAR with one a0 fit jointly to galaxies and clusters",
    "U0_emond_like": "a_eff=a0 exp[ln(F) S(Phi_bar/c^2)]; insert a_eff in RAR",
    "U1_coherence_length": "U0 multiplied by a coherence/length gate",
    "domain_oracle": "fixed galaxy RAR for galaxies; cluster RAR for clusters",
}
for key in ("joint_a0", "U0_emond_like", "U1_coherence_length", "domain_oracle"):
    metric = unified["heldout_metrics"][key]
    add(
        "Unified acceleration laws",
        key.replace("_", " "),
        unified_eq[key],
        galaxy_error=metric["galaxy"]["rmse"],
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        derived_error=metric["cluster"]["rmse"],
        derived_test="20 CLASH clusters, 84 GR/NFW-derived acceleration points",
        verdict="not universal" if key == "domain_oracle" else "failed joint universal gates",
        evidence="unified_cv",
    )

# Original CF4 low-acceleration/void laws (map-resolution changes are reported
# in the test label rather than treated as new force equations).
for name, formula, rmse, verdict in (
    ("Free-p low-acceleration law", "Delta g/g_bar = A S(g_bar) (g_bar/a_t)^(-p)", 25.6222, "beats Newtonian but loses to RAR"),
    ("Fixed p=1/2 low-acceleration law", "Delta g/g_bar = A S(g_bar) sqrt(a_t/g_bar)", 23.085, "radial holdout strong; strict galaxy CV no better than RAR"),
    ("CF4 grouped-64 environment law", "Delta g -> Delta g (E_CF4/E0)^beta", 25.8224, "environment worsens held-out prediction"),
    ("CF4 ungrouped-64 environment law", "Delta g -> Delta g (E_CF4/E0)^beta", 26.4373, "environment worsens held-out prediction"),
    ("CF4 ungrouped-128 environment law", "Delta g -> Delta g (E_CF4/E0)^beta", 25.2835, "sign is unstable; no robust detection"),
):
    add(
        "Void and environment",
        name,
        formula,
        galaxy_error=rmse,
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        verdict=verdict,
        evidence="cf4_theory_test",
    )

# Literal and screened void-cage formulas. Reconstruction-only copies are
# retained because the user explicitly asked for every tested variation.
void_report = load("results/void_cage_test/report.json")
void_formula = {
    "direct_harmonic_blind": "Delta v^2 = kappa r^2 (no measured environment)",
    "direct_harmonic_primary": "Delta v^2 = kappa(E_CF4) r^2",
    "screened_radial_blind": "Delta v^2 = V0^2 r^2/[r^2+(c_R R_d)^2]",
    "screened_primary": "Delta v^2 = V0^2 E_CF4^m r^2/[r^2+(c_R R_d)^2]",
    "screened_primary_shuffled": "same screened law with shuffled E_CF4 control",
    "screened_grouped_64_power_p3": "screened law; E from inverse-cube exterior force",
    "screened_grouped_64_yukawa_l31p25": "screened law; E from Yukawa lambda=31.25 Mpc/h",
    "screened_grouped_64_yukawa_l62p5": "screened law; E from Yukawa lambda=62.5 Mpc/h",
    "screened_grouped_64_yukawa_l7p8125": "screened law; E from Yukawa lambda=7.8125 Mpc/h",
    "screened_ungrouped_128_yukawa_l15p625": "screened law; E from ungrouped-128 Yukawa map",
    "screened_ungrouped_64_yukawa_l15p625": "screened law; E from ungrouped-64 Yukawa map",
}
for key, formula in void_formula.items():
    metric = void_report["variant_metrics"][key]
    add(
        "Void and environment",
        key.replace("_", " "),
        formula,
        galaxy_error=metric["rmse_kms"],
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        verdict="failed frozen void-cage gates",
        evidence="void_cage_test",
    )

scaling = load("results/void_cage_galaxy_scaling_test/report.json")
scaling_formula = {
    "catalog_mass_concentration_internal": "Delta v^2=V0^2 M^eta r^2/[r^2+(c_R R_d C^gamma)^2]",
    "catalog_mass_concentration_void": "mass/concentration law x E_CF4^m",
    "catalog_mass_surface_internal": "Delta v^2=V0^2 M^eta r^2/[r^2+(c_R R_d Sigma^beta)^2]",
    "catalog_mass_surface_void": "mass/surface law x E_CF4^m",
    "catalog_mass_surface_void_shuffled": "mass/surface law with shuffled E_CF4",
    "legacy_size_only": "Delta v^2=V0^2 r^2/[r^2+(c_R R_d)^2]",
    "local_acceleration_internal": "screened law with S(g_bar)=[1+(g_bar/g*)^n]^-1",
    "local_acceleration_void": "local-acceleration law x E_CF4^m",
    "local_acceleration_void_shuffled": "local-acceleration law with shuffled E_CF4",
    "local_acceleration_void_ungrouped_128": "local-acceleration law x ungrouped-128 E_CF4^m",
    "local_acceleration_void_ungrouped_64": "local-acceleration law x ungrouped-64 E_CF4^m",
}
for key, formula in scaling_formula.items():
    metric = scaling["variant_metrics"][key]
    add(
        "Galaxy-scaled void laws",
        key.replace("_", " "),
        formula,
        galaxy_error=metric["rmse_kms"],
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        verdict="mass scaling is competitive but void exponent is zero" if "catalog_mass" in key else "failed",
        evidence="void_cage_galaxy_scaling_test",
    )

transition = load("results/void_cage_transition_isolation/report.json")
transition_formula = {
    "mass_amplitude_only": "Delta v^2=V0^2 M^eta r^2/[r^2+(c_R R_d)^2]",
    "mass_transition": "same, with r_t=c_R R_d M^alpha",
    "surface_transition": "same, with r_t=c_R R_d Sigma^beta",
    "concentration_transition": "same, with r_t=c_R R_d C^gamma",
}
for key, formula in transition_formula.items():
    metric = transition["variant_metrics"][key]
    add(
        "Galaxy-scaled void laws",
        key.replace("_", " "),
        formula,
        galaxy_error=metric["rmse_kms"],
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        verdict="retained empirical mass-amplitude control" if key == "mass_amplitude_only" else "transition dependence not supported",
        evidence="void_cage_transition_isolation",
    )

# Unbounded scalar, path, tensor, vector and variable-exponent completions.
completion_specs = (
    ("results/unbounded_running_full_test/report.json", "Unbounded running", {
        "curvature_log": "E=1+alpha[ln(1+T*/T)]^p",
        "curvature_loglog": "E=1+alpha{ln[1+ln(1+T*/T)]}^p",
        "curvature_rootlog": "E=1+alpha sqrt[ln(1+T*/T)]^p",
        "curvature_power": "E=[1+(T*/T)^p]^epsilon",
        "path_log_running": "E=1+alpha ln[1+(ell/L)^p]",
        "path_power_running": "E=[1+(ell/L)^p]^epsilon",
        "tensor_alignment_log": "log running x tidal-eigenvector alignment",
        "tensor_dominance_log": "log running x tidal-eigenvalue dominance",
        "tensor_alignment_power": "power running x tidal-eigenvector alignment",
        "tensor_dominance_power": "power running x tidal-eigenvalue dominance",
    }),
    ("results/unbounded_running_variable_exponent/report.json", "Variable exponent", {
        "curvature_variable_mass_power": "E=[1+(T*/T)^p(M_eq)]^epsilon",
        "curvature_variable_density_power": "E=[1+(T*/T)^p(rho_b)]^epsilon",
        "curvature_variable_shape_power": "E=[1+(T*/T)^p(rho_b/rho_mean)]^epsilon",
    }),
    ("results/tensor_completion_full_test/report.json", "Bounded tensor completion", {
        "tensor_isotropic": "C_ij=C_solar delta_ij+(1-C_solar) A_T delta_ij",
        "tensor_alignment": "C_ij=C_solar delta_ij+(1-C_solar) A_T P_align,ij",
        "tensor_competition": "C_ij=C_solar delta_ij+(1-C_solar) A_T P_compete,ij",
        "tensor_dominance": "C_ij=C_solar delta_ij+(1-C_solar) A_T P_dom,ij",
    }),
    ("results/vector_completion_full_test/report.json", "Bounded vector completion", {
        "isotropic_completion": "C=C_solar+(1-C_solar)A_T",
        "coherence_completion": "C=C_solar+(1-C_solar)A_T(1-Coh)^q",
    }),
    ("results/path_completion_full_test/report.json", "Bounded path completion", {
        "distance_path": "logit C=logit C_solar+integral dr/ell",
        "tidal_path": "logit C=logit C_solar+integral A_T dr/ell",
        "matter_path": "logit C=logit C_solar+integral A_rho dr/ell",
        "hybrid_path": "logit C=logit C_solar+integral A_T A_rho dr/ell",
    }),
    ("results/mass_path_completion_full_test/report.json", "Mass-conditioned path completion", {
        "mass_weighted_path": "d tau/dr=[1+(M*/M_history)^q]^-1/ell",
        "mass_amplified_path": "d tau/dr=(M_history/M*)^q/ell",
        "mass_ceiling_path": "distance recovery capped by [1+(M*/M_history)^q]^-1",
    }),
)
for report_path, family, equations in completion_specs:
    report = load(report_path)
    for key, equation in equations.items():
        model = report["models"][key]
        raw = model.get("raw_lensing", {}).get("heldout", {})
        complete = raw.get("all_roots_converged")
        raw_error = raw.get("exact_radial_RMS_arcsec")
        add(
            family,
            key.replace("_", " "),
            equation,
            galaxy_error=model["SPARC_metrics"]["outer_holdout"]["RMSE_km_s"],
            galaxy_kind="SPARC outer velocity",
            galaxy_test="131 SPARC galaxies, 968 untouched outer points",
            derived_error=model["bridge_metrics"]["cluster"]["equal_system_RMSE_dex"],
            derived_test="20 CLASH clusters, GR/NFW-derived acceleration points",
            raw_error=raw_error,
            raw_test="RXJ2129: 7 held-out images" if raw_error is not None else "",
            raw_complete=complete,
            verdict="failed universal gates" + ("; raw roots incomplete" if complete is False else ""),
            evidence=report_path.removeprefix("results/").removesuffix("/report.json"),
        )

# Refined scalar laws and their four-cluster vector redistribution tests.
multi_raw = load("results/unbounded_running_multicluster_raw/report.json")
refined = (
    ("Curvature power p=2", "E=[1+(T*/T)^2]^epsilon", 14.403, 0.1675, "curvature_power_p2"),
    ("Curvature additive alpha=10", "E=1+10 ln[1+(T*/T)^p]", 16.765, 0.1526, "curvature_additive_alpha10"),
)
for name, equation, galaxy_rmse, cluster_dex, raw_key in refined:
    raw = multi_raw["primary_aggregate"][raw_key]
    add(
        "Refined scalar and spatial lens",
        name,
        equation,
        galaxy_error=galaxy_rmse,
        galaxy_kind="SPARC outer velocity",
        galaxy_test="131 SPARC galaxies, 968 untouched outer points",
        derived_error=cluster_dex,
        derived_test="20 CLASH clusters, GR/NFW-derived acceleration points",
        raw_error=raw["equal_system_radial_RMS_arcsec"],
        raw_test="MACS0329+0429+1115+1931: 11 held-out images",
        raw_complete=raw["all_roots_converged"],
        verdict="balanced phenomenology but fails four-cluster raw lensing",
        evidence="unbounded_running_multicluster_raw",
    )

spatial_rows = (
    ("Curvature p=2 + member vector (GR-linear)", "alpha=alpha_spherical+f Delta alpha_members", 18.688, 14.403),
    ("Curvature p=2 + member vector (running dressed)", "alpha=alpha_spherical+f E(r) Delta alpha_members", 18.724, 14.403),
    ("Additive alpha=10 + member vector (GR-linear)", "alpha=alpha_spherical+f Delta alpha_members", 18.342, 16.765),
    ("Additive alpha=10 + member vector (running dressed)", "alpha=alpha_spherical+f E(r) Delta alpha_members", 18.216, 16.765),
)
for name, equation, raw_error, galaxy_rmse in spatial_rows:
    add(
        "Refined scalar and spatial lens",
        name,
        equation,
        galaxy_error=galaxy_rmse,
        galaxy_kind="SPARC outer velocity",
        galaxy_test="inherits locked scalar parent's 968-point SPARC result",
        raw_error=raw_error,
        raw_test="MACS0329+0429+1115+1931: 11 held-out images",
        raw_complete=True,
        verdict="member-light directions do not improve transfer",
        evidence="unbounded_running_spatial_vector",
    )

# Scalar slip and new member-tensor tests.
add(
    "Metric lens closures",
    "Fixed RAR with scalar metric slip s=5",
    "g_lens=g_bar+(1+s/2)(g_dyn-g_bar)",
    galaxy_error=outer["fixed_RAR:invariant"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 untouched outer points",
    raw_error=18.432239218869125,
    raw_test="MACS1115+1931: 6 held-out images",
    raw_complete=True,
    verdict="scalar slip selected but fails halo-competitiveness gate",
    evidence="metric_slip_raw_lensing",
)
member = load("results/member_tidal_metric/report.json")
add(
    "Metric lens closures",
    "Member tidal-contrast metric",
    "partial_i[(delta_ij+t Qcontrast_ij) partial_j Phi]=source",
    galaxy_error=outer["fixed_RAR:invariant"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="inherits locked fixed-RAR matter law",
    raw_error=member["validation"]["selected_tensor"]["equal_system_radial_RMS_arcsec"],
    raw_test="MACS1115+1931: 6 held-out images",
    raw_complete=True,
    verdict="selected t=0; retired",
    evidence="member_tidal_metric",
)
member_full = load("results/member_full_tidal_metric/report.json")
add(
    "Metric lens closures",
    "Full member tidal metric (new test)",
    "partial_i[(delta_ij+t Qfull_ij) partial_j Phi]=source",
    galaxy_error=outer["fixed_RAR:invariant"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="inherits locked fixed-RAR matter law",
    raw_error=member_full["validation"]["selected_tensor"]["equal_system_radial_RMS_arcsec"],
    raw_test="MACS1115+1931: 6 held-out images",
    raw_complete=True,
    verdict="selected t=0; retired; negative t lost exact roots",
    evidence="member_full_tidal_metric",
)

# Spherical-spacetime and hard-cavity analogies.
spherical = load("results/spherical_spacetime_cavity/galaxy_report.json")
sphere_eq = {
    "closed_global_cluster_safe": "g/g_bar=[(r/L)/sin(r/L)]^2",
    "closed_global_galaxy_only_diagnostic": "same closed-space law, galaxy-only L",
    "local_amplified_screened": "closed-space amplification x local acceleration screen",
}
for key, equation in sphere_eq.items():
    model = spherical["models"][key]
    raw_error = 25.153064278564724 if key == "closed_global_cluster_safe" else None
    add(
        "Spherical spacetime/cavity",
        key.replace("_", " "),
        equation,
        galaxy_error=model["SPARC"]["outer_holdout"]["RMSE_km_s"],
        galaxy_kind="SPARC outer velocity",
        galaxy_test="131 SPARC galaxies, 968 untouched outer points",
        derived_error=model["environment"]["cluster"]["RMSE_dex"],
        derived_test="20 CLASH clusters, derived acceleration points",
        raw_error=raw_error,
        raw_test="MACS1115+1931: 6 held-out images" if raw_error is not None else "",
        raw_complete=True if raw_error is not None else None,
        verdict="failed galaxy and/or cluster-domain gates",
        evidence="spherical_spacetime_cavity",
    )
add(
    "Spherical spacetime/cavity",
    "Hard spherical cavity flow analogy",
    "v_flow/v_inf = 1 + O[(R_body/r)^3] with potential-flow angular factors",
    galaxy_error=spherical["hard_cavity"]["axis_velocity_RMSE_km_s"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, favorable axis upper-bound diagnostic",
    verdict="geometric effect far too small; analytic net force is zero",
    evidence="spherical_spacetime_cavity",
)

# Action-level Sigma variants. Their galaxy column is intentionally blank:
# those particular rows used a synthetic representative galaxy rather than the
# observational SPARC likelihood.
sigma = load("results/sigma_action_exploration/report.json")
sigma_eq = {
    "sigma_refracted_AQUAL": "div[mu(Sigma,|grad Phi|/a0) grad Phi]=4 pi G rho_b",
    "sigma_gated_AQUAL": "AQUAL mu with Sigma activation gate",
    "conformal_symmetron": "Box Sigma=dV_eff/dSigma; matter follows A(Sigma)^2 g_mn",
}
for key, model in sigma["best_joint_rows"].items():
    raw_diag = None
    if key == "sigma_refracted_AQUAL":
        raw_diag = sigma["raw_lensing_diagnostics"]["joint_row"]["scores"]["heldout"]
    add(
        "Action-level Sigma",
        key.replace("_", " "),
        sigma_eq[key],
        derived_error=model["RXJ2129_derived_field_RMSE_dex"],
        derived_test="RXJ2129 radial field derived from a lens model",
        raw_error=raw_diag["exact_radial_RMS_arcsec"] if raw_diag else None,
        raw_test="RXJ2129: 7 held-out images" if raw_diag else "",
        raw_complete=raw_diag["all_roots_converged"] if raw_diag else None,
        verdict="exploratory action; synthetic galaxy test only, no covariant completion",
        evidence="sigma_action_exploration",
    )
cluster_sigma = sigma["raw_lensing_diagnostics"]["cluster_derived_target_only_row"]
add(
    "Action-level Sigma",
    "Sigma refracted AQUAL (cluster-tuned diagnostic)",
    "same AQUAL law with cluster-selected Sigma parameters",
    derived_error=cluster_sigma["parameter_row"]["RXJ2129_derived_field_RMSE_dex"],
    derived_test="RXJ2129 radial field derived from a lens model",
    raw_error=cluster_sigma["scores"]["heldout"]["exact_radial_RMS_arcsec"],
    raw_test="RXJ2129: 7 held-out images",
    raw_complete=cluster_sigma["scores"]["heldout"]["all_roots_converged"],
    verdict="not universal; included only to expose galaxy/cluster tension",
    evidence="sigma_action_exploration",
)
add(
    "Action-level Sigma",
    "Causal catch-up Sigma completion",
    "div(mu grad Phi) - Q(Sigma,y)c^-2 d_t^2 Phi = 4 pi G rho_b",
    galaxy_error=outer["RAR_sharp_coherence_gated_RG:primary"],
    galaxy_kind="SPARC outer velocity",
    galaxy_test="static limit; 131 SPARC galaxies, 968 outer points",
    derived_error=0.13870141328241087,
    derived_test="static CLASH limit; 20 clusters",
    raw_error=2.248177892477939,
    raw_test="RXJ2129: 7 held-out images",
    raw_complete=True,
    verdict="causal and stable, but time term is exactly invisible to static tests",
    evidence="sigma_causal_catchup_all_tests",
)

# Earlier potential, boundary and directly calculated void-tide branches.
for name, equation, rmse, verdict in (
    (
        "P0 baryonic-potential screen",
        "activation S=S(|Phi_bar|/c^2); Delta g/g_bar=A S chi^(-p)",
        23.4094,
        "competitive with but does not beat fixed RAR",
    ),
    (
        "P1 CF4-shifted potential threshold",
        "log chi_t=log chi_t0+zeta V_CF4",
        24.4953,
        "environmental threshold shift worsens prediction",
    ),
    (
        "B1 potential boundary layer",
        "g_B=kappa a_star dS_Phi/d ln R",
        24.4466,
        "boundary coefficient sign is unstable; rejected",
    ),
    (
        "W1 measured void-wall threshold",
        "log chi_t=log chi_t0+zeta_w d_to_measured_void_wall",
        25.6933,
        "measured wall depth worsens prediction",
    ),
):
    add(
        "Potential and boundary screens",
        name,
        equation,
        galaxy_error=rmse,
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        verdict=verdict,
        evidence="NEXT_MODEL_RESULTS",
    )
add(
    "Potential and boundary screens",
    "T0 ordinary CF4 gravity tide",
    "div g_delta=-(3/2) Omega_m H0^2 delta; internal effect from tidal Hessian",
    verdict="median inward tide is 1.33e-5 of the required acceleration; rejected",
    evidence="cf4_tide_test",
)

# Action-derived MOND/Aether controls that followed the algebraic U0 screen.
for report_path, key, name, equation in (
    (
        "results/h7a_cv/report.json",
        "H7a_simple_mu_potential",
        "H7a simple-mu potential-dependent AQUAL",
        "div[mu(g/a_X(Phi_bar)) grad Psi]=4 pi G rho; mu=x/(1+x)",
    ),
    (
        "results/h7s_cv/report.json",
        "H7s_standard_mu_potential",
        "H7s standard-mu potential-dependent AQUAL",
        "div[mu_s(g/a_X(Phi_bar)) grad Psi]=4 pi G rho; mu_s=x/sqrt(1+x^2)",
    ),
):
    metric = load(report_path)["heldout_metrics"][key]
    add(
        "Covariant/Aether attempts",
        name,
        equation,
        galaxy_error=metric["galaxy"]["rms"],
        galaxy_kind="SPARC all-point velocity",
        galaxy_test="131 SPARC galaxies, 3,034 whole-galaxy CV points",
        derived_error=metric["cluster"]["rms"],
        derived_test="20 CLASH clusters, 84 GR/NFW-derived acceleration points",
        verdict="phenomenologically competitive, but it recreates MOND/AQUAL and the environment is noncovariant",
        evidence=report_path.removeprefix("results/").removesuffix("/report.json"),
    )

add(
    "Covariant/Aether attempts",
    "EA-Q0 reciprocal environmental Aether",
    "S~F(Q)R + Aether[K, a_Q(Q)] + beta[(grad Q)^2+Q^2/L_Q^2]",
    verdict="retired before fit: reciprocal Aether source changes Q by orders of magnitude",
    evidence="eaq0_derivation",
)
add(
    "Covariant/Aether attempts",
    "EMOG-Q0 chameleon scalar + Proca vector",
    "S~F(s)R-(grad s)^2-U(s)-B^2/4-mu^2 phi^2/2-phi_a J^a",
    verdict="retired before fit: wrong density ordering, Yukawa shape, and Solar-System conflict",
    evidence="environmental_mog0",
)

# The final measured-density/coherence follow-up is distinct from the earlier
# proxy bridge even though it uses the same algebraic response family.
add(
    "Measured density/coherence",
    "CPR0 measured coherence-partitioned RG",
    "epsilon_mix=w(C)+(1-w)epsilon_RG; nu_src=1+w B0 h(g_bar)",
    galaxy_error=0.09322,
    galaxy_kind="observed log acceleration",
    galaxy_test="44 observed MaNGA BCG dynamical points with measured Lambda_Re",
    derived_error=0.15462,
    derived_test="20 CLASH clusters, 72 ACCEPT-gas+BCG-star derived acceleration points",
    verdict="0.00070-dex gain over RG; fails frozen improvement gate",
    evidence="cpr0_accept_clash_bcg_stellar",
)
add(
    "Measured density/coherence",
    "NBP0 nonlocal scalar permittivity morphology",
    "div[epsilon(X) grad Phi]=4 pi G rho_b; (1-L_X^2 Laplacian)X=rho_b",
    galaxy_error=0.1265,
    galaxy_kind="observed log acceleration",
    galaxy_test="96 SPARC systems; held-out outer residual relative to fixed RAR",
    verdict="morphology worsens RMSE by 7.03%; structural scalar failure",
    evidence="nbp0_sparc_morphology_test",
)

# Structurally tested finite mechanism space. These rows intentionally have no
# proximity percentage: the candidates were rejected or excluded analytically
# before an observational likelihood fit, which is more informative than
# inventing a zero score.
nbm0_mechanisms = (
    ("NBM0 A0 canonical conformal scalar", "g~=exp(2 alpha X) g; canonical massive X", "Weyl-potential contribution cancels"),
    ("NBM0 A1 disformal scalar with prescribed U", "g~=exp(2 alpha X)(g+2 beta X U U)", "no reciprocal equation for preferred direction"),
    ("NBM0 A2 canonical scalar + dynamical Aether", "E(r)=1+A(1+r/L)e^(-r/L)", "positive Yukawa response is never flatter than Keplerian"),
    ("NBM0 A3 massless canonical scalar", "E(r)=1+A", "constant Newtonian rescaling; no flat curve or screening"),
    ("NBM0 A4 positive Yukawa spectrum", "E(r)=1+sum A_i(1+r/L_i)e^(-r/L_i), A_i>=0", "nonnegative spectrum cannot turn gravity on at large radius"),
    ("NBM0 A5 fractional p=3/2 operator", "(-Laplacian)^(3/2) Phi proportional to rho", "flat radial shape but v_flat^4 proportional to M^2"),
    ("NBM0 A6 nonlinear p-Laplacian", "div(|grad Phi| grad Phi) proportional to rho", "unique flat+BTFR limit is already AQUAL/MOND"),
    ("NBM0 A7 smooth external void basin", "X=X0+grad X.r+X2 r^2/2+...", "uniform terms cancel; leading internal force is harmonic"),
    ("NBM0 A8 nonlinear nonlocal basin", "localized nonlinear memory kernel with auxiliary fields", "no healthy non-MOND action survived the closure audit"),
    ("NBM0 A9 self-gravitating basin phase", "G_ab=8 pi G(T_b+T_basin)", "can lens only by adding an independent gravitating energy reservoir"),
)
for name, equation, verdict in nbm0_mechanisms:
    add(
        "Finite mechanism closure",
        name,
        equation,
        verdict=verdict,
        evidence="nbm0_action_space; nbm0_possibility_closure",
    )

add(
    "Action-level Sigma",
    "Sigma complete reciprocal action",
    "S~R/16piG - Z(Sigma)(grad Sigma)^2/2 - V(Sigma) + F(X,Sigma)",
    verdict="reciprocal feedback/stress-energy completion did not resolve the galaxy-cluster tension",
    evidence="sigma_complete_action",
)
add(
    "Action-level Sigma",
    "Sigma covariant weak-field metric closure",
    "Box Sigma=V_eff,Sigma; div[mu(Sigma,X) grad Phi]=4 pi G rho_b",
    verdict="mathematical weak-field closure only; no new independent observational gain",
    evidence="sigma_covariant_weak_field",
)

# The conservative radial derivative is represented by its three physically
# distinct carriers and the memory-plus-diffusion branch. Parameter-grid cells
# within each carrier remain in the consolidated 913-row sensitivity table.
add(
    "Conservative profile diffusion",
    "No-flux fractional-excess diffusion",
    "dX/dtau=d2X/d(ln r)^2; X=F; zero boundary flux",
    galaxy_error=71.79377546552938,
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 fixed-transfer outer points",
    raw_error=18.926798,
    raw_test="MACS0329+0429+1115+1931: 11 held-out images",
    raw_complete=True,
    verdict="measurable but worsens the local-control galaxy/lensing compromise",
    evidence="reopened_hybrid_profile_diffusion_analysis",
)
add(
    "Conservative profile diffusion",
    "No-flux added-acceleration diffusion",
    "dX/dtau=d2X/d(ln r)^2; X=F g_N; ell=0.7, mu=1",
    galaxy_error=78.39799649491455,
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 fixed-transfer outer points",
    raw_error=18.648255,
    raw_test="MACS0329+0429+1115+1931: 11 held-out images",
    raw_complete=True,
    verdict="strongest diffusion raw improvement, paired with an 8.74-km/s galaxy penalty",
    evidence="reopened_hybrid_profile_diffusion_analysis",
)
add(
    "Conservative profile diffusion",
    "No-flux circular-speed-squared diffusion",
    "dX/dtau=d2X/d(ln r)^2; X=F g_N r; ell=0.15, mu=0.5",
    galaxy_error=69.67768338543708,
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 fixed-transfer outer points",
    raw_error=18.857423,
    raw_test="MACS0329+0429+1115+1931: 11 held-out images",
    raw_complete=True,
    verdict="nearly galaxy-neutral but only a 0.089-arcsec development-sample raw gain",
    evidence="reopened_hybrid_profile_diffusion_analysis",
)
add(
    "Conservative profile diffusion",
    "One-sided memory plus no-flux diffusion",
    "F_memory(p=1.927,q=9,ell=0.35) then diffuse F at ell=0.35, mu=1",
    galaxy_error=30.868199813172094,
    galaxy_kind="SPARC outer velocity",
    galaxy_test="131 SPARC galaxies, 968 fixed-transfer outer points",
    raw_error=27.970581,
    raw_test="MACS0329+0429+1115+1931: 11 held-out images",
    raw_complete=False,
    verdict="improves the memory-control galaxy score but loses a held-out lens root",
    evidence="reopened_hybrid_profile_diffusion_analysis",
)


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def error_cell(row: dict, prefix: str) -> str:
    if prefix == "galaxy":
        error = row["galaxy_error"]
        unit = row["galaxy_error_unit"]
        pct = row["galaxy_proximity_percent"]
    elif prefix == "derived":
        error = row["derived_lensing_error_dex"]
        unit = "dex"
        pct = row["derived_lensing_proximity_percent"]
    else:
        error = row["raw_lensing_error_arcsec"]
        unit = "arcsec"
        pct = row["raw_lensing_proximity_percent"]
    if pct is None:
        if prefix == "raw" and error is not None and row["all_raw_roots_complete"] is False:
            return f"not scoreable ({fmt(error, 3)} {unit}; incomplete roots)"
        return "—"
    return f"{pct:.2f}% ({error:.3f} {unit})"


OUT.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0])
with (OUT / "formula_scorecard.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

payload = {
    "report_version": "FORMULA-SCORECARD-0.1.0",
    "formula_rows": len(rows),
    "percentage_definitions": {
        "SPARC_velocity": "100*max(0,1-RMSE/RMS(v_obs))",
        "observed_log_acceleration": "100*10^(-RMSE_dex)",
        "derived_lensing_acceleration": "100*10^(-RMSE_dex)",
        "raw_image_positions": "100*max(0,1-RMS_image/RMS(observed_image_radius))",
        "warning": "Descriptive normalized proximity, not probability, confidence, explained variance, or a common likelihood.",
    },
    "normalizers": {
        "SPARC_outer_observed_velocity_RMS_km_s": SPARC_OUTER_OBS_RMS,
        "SPARC_all_point_observed_velocity_RMS_km_s": SPARC_ALL_OBS_RMS,
        "raw_lensing_observed_radius_RMS_arcsec": RAW_DENOMINATORS,
    },
    "rows": rows,
}
with (OUT / "formula_scorecard.json").open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, allow_nan=False)
    handle.write("\n")

best_galaxy = sorted(
    (r for r in rows if r["galaxy_proximity_percent"] is not None),
    key=lambda r: r["galaxy_proximity_percent"],
    reverse=True,
)[:8]
best_raw = sorted(
    (r for r in rows if r["raw_lensing_proximity_percent"] is not None),
    key=lambda r: r["raw_lensing_proximity_percent"],
    reverse=True,
)[:8]

lines = [
    "# Complete tested-formula scorecard",
    "",
    f"This inventory contains **{len(rows)} scientifically distinct formula/test rows**. It consolidates repeated controls but retains force-law, screen, exponent, path, tensor, metric, and reconstruction variations when they change the tested hypothesis.",
    "",
    "## How to read the percentages",
    "",
    "The percentages are normalized proximity scores created for this audit; the original scientific scores remain beside them. They are **not** probabilities that a theory is true, confidence levels, or interchangeable likelihoods.",
    "",
    "- Galaxy velocity: `100 x max(0, 1 - RMSE / RMS(observed speed))`.",
    "- Observed or derived log acceleration: `100 x 10^(-RMSE_dex)`. A 0.301-dex error is therefore 50% proximity (a typical factor-of-two miss).",
    "- Raw lensing: `100 x max(0, 1 - image-position RMS / RMS(observed image radius))`.",
    "- Raw image positions and GR/NFW-derived acceleration products are kept separate. NFW's 100% derived score is circular by construction, not a prediction.",
    "- A blank means that observable was not tested. Incomplete image roots are not assigned a percentage.",
    "",
    "## Highest descriptive proximity scores",
    "",
    "Galaxy tests are not all the same split, and raw-lens rows can use different clusters; these lists are navigation aids, not a leaderboard.",
    "",
    "### Galaxy",
    "",
    "| Formula | Proximity | Original error | Test |",
    "|---|---:|---:|---|",
]
for row in best_galaxy:
    lines.append(
        f"| {row['formula']} | {row['galaxy_proximity_percent']:.2f}% | {row['galaxy_error']:.3f} {row['galaxy_error_unit']} | {row['galaxy_test']} |"
    )
lines.extend(
    [
        "",
        "### Raw lensing",
        "",
        "| Formula | Proximity | Original error | Test |",
        "|---|---:|---:|---|",
    ]
)
for row in best_raw:
    lines.append(
        f"| {row['formula']} | {row['raw_lensing_proximity_percent']:.2f}% | {row['raw_lensing_error_arcsec']:.3f} arcsec | {row['raw_lensing_test']} |"
    )

lines.extend(
    [
        "",
        "## Full formula table",
        "",
        "| Family | Formula | Schematic equation | Galaxy proximity (error) | Derived-lens proximity (error) | Raw-lens proximity (error) | Verdict |",
        "|---|---|---|---:|---:|---:|---|",
    ]
)
for row in rows:
    eq = row["schematic_equation"].replace("|", "\\|")
    verdict = row["verdict"].replace("|", "\\|")
    lines.append(
        f"| {row['family']} | {row['formula']} | `{eq}` | {error_cell(row, 'galaxy')} | {error_cell(row, 'derived')} | {error_cell(row, 'raw')} | {verdict} |"
    )

lines.extend(
    [
        "",
        "## Bottom-line interpretation",
        "",
        "1. Fixed RAR and simple MOND remain the strongest universal galaxy controls, at about 93.8% velocity proximity on the untouched SPARC outer points, but their two-cluster raw-lens proximity is only about 5–6%.",
        "2. The RAR + squared coherence-gated RG bridge is the only tested project formula that is simultaneously close to RAR on observational SPARC data and very close on its one-cluster raw-lens test. That raw result has not yet transferred to multiple clusters, so it is promising evidence, not a universal solution.",
        "3. On the four-cluster transfer, the best locked modified-gravity scalar law is roughly 29% raw-lens proximity, while the compact-halo comparator is roughly 65%. Adding member-light vectors does not close the gap.",
        "4. The new full member-tidal tensor test selects zero coupling and gives essentially the same two-cluster error as the scalar-slip parent. Strong negative couplings improve a local fitting cost but lose exact image roots, so changing the number does not rescue it.",
        "5. The Solar-screened isothermal tail now has a direct morphology-stratified galaxy test. Its locked cluster value scores 18.60 km/s overall and is especially poor for disk-dominated, dwarf, late-type, flat/rising systems; it is therefore not the missing universal bridge.",
        "6. Conservative radial diffusion confirms that the transported physical quantity matters: added acceleration is lensing-favored, while short-scale circular-speed-squared transport is nearly galaxy-neutral. No diffusion carrier improves the complete-root cross-domain control, and diffusion after the best memory response loses lens roots.",
        "7. No tested formula yet matches both trusted galaxy controls and multi-cluster dark-matter lens reconstructions with one universal setting. The most defensible next test is the existing RAR + coherence/RG candidate on several clusters using complete baryonic maps (gas, BCG, ICL, and member galaxies), with its constants frozen before image scoring.",
        "",
        "Machine-readable versions: `results/formula_scorecard/formula_scorecard.csv` and `results/formula_scorecard/formula_scorecard.json`.",
    ]
)
(ROOT / "docs" / "FORMULA_SCORECARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {len(rows)} rows to {OUT}")
