"""Build the residual-blind partial RX J2129 baryonic input package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.constants import G
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_baryonic_protocol.json"
C_KM_S = 299_792.458
KPC_M = u.kpc.to(u.m)
MSUN_KG = u.Msun.to(u.kg)
G_SI = G.to_value(u.m**3 / (u.kg * u.s**2))


def _resolve(path: str) -> Path:
    return ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_tian_row(path: Path, alias: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 14 or fields[-1] != alias:
            continue
        coordinates = fields[2].split()
        return {
            "name": fields[0],
            "redshift": float(fields[1]),
            "coordinates": " ".join(coordinates),
            "band": fields[3],
            "sersic_n": float(fields[4]),
            "effective_radius_kpc": float(fields[5]),
            "effective_radius_sigma_kpc": float(fields[6]),
            "anchor_radius_kpc": float(fields[7]),
            "stellar_mass_1e11_msun": float(fields[8]),
            "gas_mass_1e11_msun": float(fields[9]),
            "gas_mass_sigma_1e11_msun": float(fields[10]),
            "lensing_mass_1e11_msun_not_used": float(fields[11]),
            "lensing_mass_sigma_1e11_msun_not_used": float(fields[12]),
            "alias": fields[13],
        }
    raise ValueError(f"No Tian table row found for {alias}")


def _load_bin_geometry(path: Path) -> pd.DataFrame:
    columns = ["bin", "semimajor_min_arcsec", "semimajor_max_arcsec"]
    bins = pd.read_csv(path, usecols=columns).sort_values("bin").reset_index(drop=True)
    bins["radius_arcsec"] = 0.5 * (
        bins["semimajor_min_arcsec"] + bins["semimajor_max_arcsec"]
    )
    return bins


def _angular_scale_kpc_per_arcsec(config: dict[str, Any]) -> float:
    values = config["cluster"]["cosmology"]
    cosmology = FlatLambdaCDM(
        H0=values["H0_km_s_Mpc"] * u.km / u.s / u.Mpc,
        Om0=values["Omega_m"],
        Tcmb0=2.7255 * u.K,
    )
    distance = cosmology.angular_diameter_distance(config["cluster"]["redshift"])
    return (distance * (1.0 * u.arcsec).to(u.rad).value).to_value(u.kpc)


def _hernquist_profile(
    radius_kpc: np.ndarray,
    total_mass_msun: float,
    total_mass_sigma_msun: float,
    effective_radius_kpc: float,
    effective_radius_sigma_kpc: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    scale_kpc = 0.551 * effective_radius_kpc
    denominator_kpc = radius_kpc + scale_kpc
    mass_enclosed_msun = total_mass_msun * radius_kpc**2 / denominator_kpc**2
    acceleration_m_s2 = (
        G_SI * total_mass_msun * MSUN_KG / (denominator_kpc * KPC_M) ** 2
    )

    mass_jacobian = np.column_stack(
        [
            mass_enclosed_msun / total_mass_msun,
            -2.0 * 0.551 * mass_enclosed_msun / denominator_kpc,
        ]
    )
    acceleration_jacobian = np.column_stack(
        [
            acceleration_m_s2 / total_mass_msun,
            -2.0 * 0.551 * acceleration_m_s2 / denominator_kpc,
        ]
    )
    nuisance_covariance = np.diag(
        [total_mass_sigma_msun**2, effective_radius_sigma_kpc**2]
    )
    mass_covariance = mass_jacobian @ nuisance_covariance @ mass_jacobian.T
    acceleration_covariance = (
        acceleration_jacobian @ nuisance_covariance @ acceleration_jacobian.T
    )

    profile = pd.DataFrame(
        {
            "radius_kpc": radius_kpc,
            "hernquist_scale_kpc": scale_kpc,
            "bcg_mass_enclosed_msun": mass_enclosed_msun,
            "bcg_mass_enclosed_sigma_msun": np.sqrt(np.diag(mass_covariance)),
            "bcg_acceleration_m_s2": acceleration_m_s2,
            "bcg_acceleration_sigma_m_s2": np.sqrt(np.diag(acceleration_covariance)),
        }
    )
    return profile, mass_covariance, acceleration_covariance


def _read_molino_catalog(path: Path) -> pd.DataFrame:
    header = next(
        line[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# CLASHID")
    )
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header.split(),
        low_memory=False,
    )


def _classify_satellites(
    catalog: pd.DataFrame, config: dict[str, Any], scale_kpc_arcsec: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rule = config["satellite_catalog_rule"]
    cluster = config["cluster"]
    center = SkyCoord(cluster["center_ra_deg"] * u.deg, cluster["center_dec_deg"] * u.deg)
    positions = SkyCoord(
        pd.to_numeric(catalog["RA"], errors="coerce").to_numpy() * u.deg,
        pd.to_numeric(catalog["Dec"], errors="coerce").to_numpy() * u.deg,
    )
    separation_arcsec = center.separation(positions).to_value(u.arcsec)

    specz = pd.to_numeric(catalog["SpeczValue"], errors="coerce").to_numpy()
    spec_quality = pd.to_numeric(catalog["SpeczQual"], errors="coerce").to_numpy()
    photo_low = pd.to_numeric(catalog["zb_Min_1"], errors="coerce").to_numpy()
    photo_high = pd.to_numeric(catalog["zb_Max_1"], errors="coerce").to_numpy()
    log_mass = pd.to_numeric(catalog[rule["stellar_mass_column"]], errors="coerce").to_numpy()

    secure_spec = (
        np.isfinite(specz)
        & (specz >= 0.0)
        & np.isfinite(spec_quality)
        & (spec_quality <= rule["secure_spectroscopic_quality_max"])
    )
    velocity_offset = C_KM_S * (specz - cluster["redshift"]) / (1.0 + cluster["redshift"])
    secure_member = secure_spec & (
        np.abs(velocity_offset) <= rule["secure_spectroscopic_velocity_window_km_s"]
    )
    secure_nonmember = secure_spec & ~secure_member
    photo_possible = (
        ~secure_spec
        & np.isfinite(photo_low)
        & np.isfinite(photo_high)
        & (photo_low <= cluster["redshift"])
        & (photo_high >= cluster["redshift"])
    )
    valid_mass = (
        np.isfinite(log_mass)
        & (log_mass >= rule["valid_log10_stellar_mass_range"][0])
        & (log_mass <= rule["valid_log10_stellar_mass_range"][1])
    )
    bcg_reference = separation_arcsec <= rule["bcg_exclusion_radius_arcsec"]

    classification = np.full(len(catalog), "not_selected", dtype=object)
    classification[secure_nonmember] = "secure_spec_nonmember"
    classification[photo_possible] = "possible_photo_member"
    classification[secure_member] = "secure_spec_member"
    classification[bcg_reference] = "bcg_reference_excluded"

    selected = (secure_member | photo_possible) & valid_mass & ~bcg_reference
    candidates = pd.DataFrame(
        {
            "clash_id": catalog["CLASHID"].astype(str),
            "ra_deg": pd.to_numeric(catalog["RA"], errors="coerce"),
            "dec_deg": pd.to_numeric(catalog["Dec"], errors="coerce"),
            "separation_arcsec": separation_arcsec,
            "projected_separation_kpc": separation_arcsec * scale_kpc_arcsec,
            "membership_class": classification,
            "specz": specz,
            "specz_quality": spec_quality,
            "velocity_offset_km_s": velocity_offset,
            "photoz": pd.to_numeric(catalog["zb_1"], errors="coerce"),
            "photoz_95_low": photo_low,
            "photoz_95_high": photo_high,
            "photoz_odds": pd.to_numeric(catalog["Odds_1"], errors="coerce"),
            "photoz_chi2": pd.to_numeric(catalog["Chi2"], errors="coerce"),
            "log10_stellar_mass_msun": log_mass,
            "stellar_mass_msun": np.power(10.0, log_mass),
            "stellar_mass_upper_0p30dex_msun": np.power(
                10.0, log_mass + rule["provisional_stellar_mass_sigma_dex"]
            ),
        }
    ).loc[selected]
    candidates = candidates.sort_values("separation_arcsec").reset_index(drop=True)

    def _summary(radius: float) -> dict[str, Any]:
        subset = candidates[candidates["separation_arcsec"] <= radius]
        return {
            "radius_arcsec": radius,
            "candidate_count": int(len(subset)),
            "secure_spec_count": int((subset["membership_class"] == "secure_spec_member").sum()),
            "possible_photo_count": int(
                (subset["membership_class"] == "possible_photo_member").sum()
            ),
            "nominal_stellar_mass_sum_msun": float(subset["stellar_mass_msun"].sum()),
            "upper_0p30dex_stellar_mass_sum_msun": float(
                subset["stellar_mass_upper_0p30dex_msun"].sum()
            ),
        }

    summary = {
        "catalog_rows": int(len(catalog)),
        "bcg_reference_rows": int(bcg_reference.sum()),
        "selected_candidate_count_full_footprint": int(len(candidates)),
        "secure_spec_candidate_count_full_footprint": int(
            (candidates["membership_class"] == "secure_spec_member").sum()
        ),
        "possible_photo_candidate_count_full_footprint": int(
            (candidates["membership_class"] == "possible_photo_member").sum()
        ),
        "within_dynamics_support": _summary(5.0),
        "within_inner_diagnostic_radius": _summary(rule["inner_diagnostic_radius_arcsec"]),
        "catalog_footprint_caveat": (
            "The homogeneous catalog footprint is finite and photometric membership is not a "
            "normalized probability. These sums are an inventory, not a complete off-center "
            "stellar-force likelihood or a negligibility proof."
        ),
    }
    return candidates, summary


def _component_inventory(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "BCG stars",
                "observable_input": "Cooke SED mass plus Tian F125W GALFIT Re",
                "numeric_radial_input": True,
                "covariance_or_shared_nuisance": "M_total 10 percent and Re error propagated as shared 4x4 covariance",
                "strict_component_gate_pass": False,
                "blocking_reason": "the empirical PSF gate now passes, but published n=2.70 versus n=4 Hernquist shape, Cooke mass-aperture, and BCG/ICL decomposition systematics are not quantified",
            },
            {
                "component": "intracluster light",
                "observable_input": "F125W and F814W science plus weight mosaics",
                "numeric_radial_input": False,
                "covariance_or_shared_nuisance": "not reconstructed",
                "strict_component_gate_pass": False,
                "blocking_reason": "no frozen PSF-aware BCG plus ICL decomposition or one-percent upper bound",
            },
            {
                "component": "hot gas",
                "observable_input": "one Chandra cumulative-mass anchor at 14.3 kpc",
                "numeric_radial_input": False,
                "covariance_or_shared_nuisance": "single 2.18 +/- 0.07e11 Msun anchor only",
                "strict_component_gate_pass": False,
                "blocking_reason": "Donahue source package contains a plotted profile but no machine-readable radial samples or covariance",
            },
            {
                "component": "satellite stars",
                "observable_input": "Molino ICL-subtracted photometry, BPZ intervals, spectroscopic redshifts, and stellar masses",
                "numeric_radial_input": False,
                "covariance_or_shared_nuisance": "candidate ledger with provisional 0.30-dex per-source width",
                "strict_component_gate_pass": False,
                "blocking_reason": "membership probabilities, per-source mass errors, three-dimensional positions, and off-center light profiles are not jointly constrained",
            },
        ]
    )


def _write_covariance(path: Path, covariance: np.ndarray) -> None:
    labels = [f"bin_{index + 1}" for index in range(covariance.shape[0])]
    frame = pd.DataFrame(covariance, columns=labels)
    frame.insert(0, "row", labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _plot_diagnostics(
    path: Path,
    profile: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    bcg = config["published_bcg_baseline"]
    gas = config["gas_anchor"]
    radius = np.geomspace(0.05, 25.0, 400)
    smooth, _, _ = _hernquist_profile(
        radius,
        bcg["total_stellar_mass_msun"],
        bcg["fractional_mass_sigma"] * bcg["total_stellar_mass_msun"],
        bcg["effective_radius_kpc"],
        bcg["effective_radius_sigma_kpc"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.plot(radius, smooth["bcg_mass_enclosed_msun"] / 1e11, color="tab:blue")
    ax.fill_between(
        radius,
        (smooth["bcg_mass_enclosed_msun"] - smooth["bcg_mass_enclosed_sigma_msun"])
        / 1e11,
        (smooth["bcg_mass_enclosed_msun"] + smooth["bcg_mass_enclosed_sigma_msun"])
        / 1e11,
        color="tab:blue",
        alpha=0.2,
        label="published Hernquist baseline (1 sigma)",
    )
    ax.errorbar(
        gas["radius_kpc"],
        gas["cumulative_mass_msun"] / 1e11,
        yerr=gas["mass_sigma_msun"] / 1e11,
        fmt="s",
        color="tab:red",
        label="only numeric Chandra gas anchor",
    )
    ax.scatter(
        profile["radius_kpc"],
        profile["bcg_mass_enclosed_msun"] / 1e11,
        color="black",
        zorder=3,
        label="four dynamics-bin centers",
    )
    ax.set(xlabel="radius (kpc)", ylabel=r"cumulative mass ($10^{11}\,M_\odot$)")
    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for membership, marker, color in [
        ("secure_spec_member", "o", "tab:green"),
        ("possible_photo_member", "^", "tab:orange"),
    ]:
        subset = candidates[candidates["membership_class"] == membership]
        ax.scatter(
            subset["separation_arcsec"],
            subset["log10_stellar_mass_msun"],
            marker=marker,
            color=color,
            alpha=0.75,
            s=22,
            label=membership.replace("_", " "),
        )
    ax.axvline(5.0, color="black", linestyle="--", linewidth=1, label="dynamics edge")
    ax.axvline(
        config["satellite_catalog_rule"]["inner_diagnostic_radius_arcsec"],
        color="gray",
        linestyle=":",
        linewidth=1,
        label="inner audit radius",
    )
    ax.set(xlabel="projected separation (arcsec)", ylabel=r"catalog $\log_{10} M_\star/M_\odot$")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("RX J2129 residual-blind baryonic availability audit")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reconstruct(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    if config["authorization"]["fit_gravity_response"]:
        raise ValueError("Baryonic protocol must not authorize a gravity response fit")

    tian = _parse_tian_row(
        _resolve(config["inputs"]["tian_table"]), config["cluster"]["catalog_alias"]
    )
    sdss = pd.read_csv(
        _resolve(config["inputs"]["sdss_petrosian_photometry"]), comment="#"
    ).sort_values("distance")
    if sdss.empty:
        raise ValueError("SDSS Petrosian query returned no rows")
    sdss_bcg = sdss.iloc[0]
    sdss_bcg_objid = int(sdss["objID"].iloc[0])
    aperture = config["cooke_mass_aperture"]
    expected = config["published_bcg_baseline"]
    checks = {
        "redshift": np.isclose(tian["redshift"], config["cluster"]["redshift"]),
        "stellar_mass": np.isclose(
            tian["stellar_mass_1e11_msun"] * 1e11,
            expected["total_stellar_mass_msun"],
        ),
        "effective_radius": np.isclose(
            tian["effective_radius_kpc"], expected["effective_radius_kpc"]
        ),
        "gas_anchor_radius": np.isclose(
            tian["anchor_radius_kpc"], config["gas_anchor"]["radius_kpc"]
        ),
        "gas_anchor_mass": np.isclose(
            tian["gas_mass_1e11_msun"] * 1e11,
            config["gas_anchor"]["cumulative_mass_msun"],
        ),
        "sdss_bcg_objid": sdss_bcg_objid == aperture["sdss_objid"],
        "sdss_coordinate_offset": np.isclose(
            float(sdss_bcg["distance"]) * 60.0,
            aperture["coordinate_offset_arcsec"],
        ),
        "sdss_petrosian_radius": np.isclose(
            float(sdss_bcg["petroRad_r"]), aperture["petrosian_radius_r_arcsec"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Tian input validation failed: {checks}")

    bins = _load_bin_geometry(_resolve(config["inputs"]["dynamics_bins"]))
    expected_centers = np.asarray(config["analysis_bins"]["centers_arcsec"], dtype=float)
    if len(bins) != 4 or not np.allclose(bins["radius_arcsec"], expected_centers):
        raise ValueError("Frozen dynamics-bin geometry does not match the baryonic protocol")
    scale = _angular_scale_kpc_per_arcsec(config)
    bins["radius_kpc"] = bins["radius_arcsec"] * scale

    profile, mass_covariance, acceleration_covariance = _hernquist_profile(
        bins["radius_kpc"].to_numpy(),
        expected["total_stellar_mass_msun"],
        expected["fractional_mass_sigma"] * expected["total_stellar_mass_msun"],
        expected["effective_radius_kpc"],
        expected["effective_radius_sigma_kpc"],
    )
    profile = pd.concat([bins, profile.drop(columns="radius_kpc")], axis=1)
    gas = config["gas_anchor"]
    inside_anchor = profile["radius_kpc"] <= gas["radius_kpc"]
    profile["gas_mass_monotonic_lower_msun"] = np.where(
        inside_anchor, 0.0, gas["cumulative_mass_msun"] - gas["mass_sigma_msun"]
    )
    profile["gas_mass_monotonic_upper_msun"] = np.where(
        inside_anchor, gas["cumulative_mass_msun"] + gas["mass_sigma_msun"], np.nan
    )
    profile["gas_profile_numeric_at_bin"] = False

    catalog = _read_molino_catalog(_resolve(config["inputs"]["molino_catalog"]))
    candidates, satellite_summary = _classify_satellites(catalog, config, scale)
    inventory = _component_inventory(config)

    outputs = config["outputs"]
    profile_path = _resolve(outputs["bcg_profile"])
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(profile_path, index=False)
    _write_covariance(_resolve(outputs["bcg_covariance"]), acceleration_covariance)
    candidates.to_csv(_resolve(outputs["satellite_candidates"]), index=False)
    inventory.to_csv(_resolve(outputs["component_inventory"]), index=False)
    _plot_diagnostics(_resolve(outputs["diagnostic"]), profile, candidates, config)

    acceleration_eigenvalues = np.linalg.eigvalsh(acceleration_covariance)
    report = {
        "protocol_version": config["protocol_version"],
        "status": "partial_baryonic_reconstruction_complete_strict_gate_failed",
        "selection_blinding": config["selection_blinding"],
        "gravity_residual_read_or_fit": False,
        "angular_scale_kpc_per_arcsec": scale,
        "tian_table_validation": {key: bool(value) for key, value in checks.items()},
        "source_interpretation": {
            "bcg_mass": expected["source_interpretation"],
            "lensing_mass_used": False,
            "gas_profile_invented_from_single_anchor": False,
        },
        "cooke_mass_aperture": {
            "sdss_objid": sdss_bcg_objid,
            "coordinate_offset_arcsec": float(sdss_bcg["distance"]) * 60.0,
            "petrosian_radius_r_arcsec": float(sdss_bcg["petroRad_r"]),
            "petrosian_flux_aperture_radius_arcsec": 2.0
            * float(sdss_bcg["petroRad_r"]),
            "petro_magnitudes_ugriz": [
                float(sdss_bcg[f"petroMag_{band}"]) for band in "ugriz"
            ],
            "definition": aperture["definition"],
            "remaining_nuisance": aperture["remaining_nuisance"],
        },
        "published_bcg_baseline": {
            "model": expected["model"],
            "total_stellar_mass_msun": expected["total_stellar_mass_msun"],
            "effective_radius_kpc": expected["effective_radius_kpc"],
            "scale_radius_kpc": 0.551 * expected["effective_radius_kpc"],
            "analysis_bin_radii_kpc": profile["radius_kpc"].tolist(),
            "enclosed_mass_msun": profile["bcg_mass_enclosed_msun"].tolist(),
            "acceleration_m_s2": profile["bcg_acceleration_m_s2"].tolist(),
            "acceleration_sigma_m_s2": profile["bcg_acceleration_sigma_m_s2"].tolist(),
            "acceleration_covariance_eigenvalues": acceleration_eigenvalues.tolist(),
            "covariance_positive_semidefinite": bool(
                np.min(acceleration_eigenvalues) >= -1e-35
            ),
            "strict_component_gate_pass": False,
            "blocking_reason": expected["known_shape_obstruction"],
        },
        "hot_gas": {
            "numeric_anchor_count": 1,
            "anchor_radius_kpc": gas["radius_kpc"],
            "anchor_mass_msun": gas["cumulative_mass_msun"],
            "anchor_mass_sigma_msun": gas["mass_sigma_msun"],
            "analysis_bins_with_numeric_profile": 0,
            "strict_component_gate_pass": False,
            "blocking_reason": "No machine-readable radial gas samples or covariance are present in the public source package.",
        },
        "satellites": satellite_summary,
        "component_gates": config["component_gates"],
        "complete_baryonic_forward_inputs": False,
        "strict_r1_ready": False,
        "outputs": outputs,
        "next_actions": [
            "execute the now-authorized PSF-aware F125W/F814W BCG plus ICL decomposition with profile-shape and Cooke mass-aperture covariance",
            "obtain the Donahue JACO gas posterior/profile or reconstruct a Chandra surface-brightness likelihood from public events",
            "turn the satellite candidate ledger into a normalized membership and off-center stellar-mass likelihood",
            "keep gravity fitting and the dynamical/Weyl Jacobian disabled until every baryonic and lens-likelihood gate passes",
        ],
    }
    report_path = _resolve(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    report = reconstruct(arguments.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
