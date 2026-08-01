#!/usr/bin/env python3
"""Construct a residual-blind MACS J0416 baryonic field and sensitivities."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.baryonic_metric import (  # noqa: E402
    build_baryonic_metric_correction_field,
    weighted_morphology,
)
from voidscreen.cluster_baryons import (  # noqa: E402
    block_compress_surface,
    dpie_surface_density_shape,
    dpie_total_mass_msun,
    gaussian_surface_density_shape,
    normalize_surface_mass,
    sersic_surface_density_shape,
    sky_to_lens_offsets,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_member_table(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        return pd.DataFrame(
            {
                "catalog_id": np.asarray(table["ID"]).astype(int),
                "ra_deg": np.asarray(table["ALPHA_J2000_STACK"], dtype=float),
                "dec_deg": np.asarray(table["DELTA_J2000_STACK"], dtype=float),
                "zspec": np.asarray(table["ZSPEC"], dtype=float),
                "zspec_quality": np.asarray(table["ZSPEC_Q"], dtype=float),
                "f160w_flux": np.asarray(table["FLUX_F160W"], dtype=float),
            }
        )


def angle_from_axis(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def build_variant(
    catalog: pd.DataFrame,
    protocol: dict,
    gas_components: list[dict],
    *,
    member_abs_dz: float,
    member_aperture_arcsec: float,
    mass_to_light_exponent: float,
    gas_mass_multiplier: float,
    icl_fraction: float,
    south_to_north_bcg_mass_ratio: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    coords = protocol["coordinate_system"]
    stars = protocol["stellar_model"]
    map_config = protocol["map"]
    scale = float(coords["scale_kpc_per_arcsec_planck18"])
    axis = np.linspace(
        -float(map_config["half_width_arcsec"]),
        float(map_config["half_width_arcsec"]),
        int(map_config["pixels_per_axis"]),
    )
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    x, y = sky_to_lens_offsets(
        catalog.ra_deg,
        catalog.dec_deg,
        reference_ra_deg=coords["reference_ra_deg"],
        reference_dec_deg=coords["reference_dec_deg"],
    )
    candidates = catalog.assign(x_arcsec=x, y_arcsec=y)
    candidates = candidates[
        np.isfinite(candidates.zspec)
        & np.isfinite(candidates.f160w_flux)
        & (candidates.f160w_flux > 0.0)
        & (candidates.zspec_quality >= float(stars["minimum_zspec_quality"]))
        & (np.abs(candidates.zspec - float(coords["redshift"])) <= float(member_abs_dz))
        & (np.hypot(candidates.x_arcsec, candidates.y_arcsec) <= float(member_aperture_arcsec))
    ].copy()

    bcg_rows = []
    bcg_xy: dict[str, tuple[float, float]] = {}
    for key in ("north_bcg", "south_bcg"):
        definition = stars[key]
        bx, by = sky_to_lens_offsets(
            [definition["ra_deg"]],
            [definition["dec_deg"]],
            reference_ra_deg=coords["reference_ra_deg"],
            reference_dec_deg=coords["reference_dec_deg"],
        )
        bcg_xy[key] = (float(bx[0]), float(by[0]))
    near_bcg = np.zeros(len(candidates), dtype=bool)
    for bx, by in bcg_xy.values():
        near_bcg |= np.hypot(candidates.x_arcsec - bx, candidates.y_arcsec - by) < float(
            stars["bcg_exclusion_radius_arcsec"]
        )
    excluded_near_bcg = int(np.count_nonzero(near_bcg))
    candidates = candidates.loc[~near_bcg].copy()
    luminosity_ratio = candidates.f160w_flux / float(stars["f160w_reference_flux_cgs"])
    candidates["mass_msun"] = float(stars["north_bcg_mass_msun"]) * np.power(
        luminosity_ratio, float(mass_to_light_exponent)
    )
    source_rows = [
        {
            "component": "member_star",
            "source_id": str(row.catalog_id),
            "x_arcsec": row.x_arcsec,
            "y_arcsec": row.y_arcsec,
            "mass_msun": row.mass_msun,
        }
        for row in candidates.itertuples()
    ]

    north_mass = float(stars["north_bcg_mass_msun"])
    bcg_masses = {
        "north_bcg": north_mass,
        "south_bcg": north_mass * float(south_to_north_bcg_mass_ratio),
    }
    bcg_map = np.zeros_like(xx)
    effective_radius = float(stars["north_bcg_effective_radius_kpc"]) / scale
    for key, mass in bcg_masses.items():
        bx, by = bcg_xy[key]
        shape = sersic_surface_density_shape(
            xx,
            yy,
            center_x=bx,
            center_y=by,
            effective_radius_arcsec=effective_radius,
            sersic_n=float(stars["north_bcg_sersic_n"]),
        )
        bcg_map += normalize_surface_mass(shape, mass)
        bcg_rows.append((key, bx, by, mass))
    bx, by, bm = block_compress_surface(
        axis, bcg_map, blocks_per_axis=int(map_config["compression_blocks_per_axis"])
    )
    source_rows.extend(
        {
            "component": "bcg_stars",
            "source_id": f"bcg_block_{i:03d}",
            "x_arcsec": px,
            "y_arcsec": py,
            "mass_msun": mass,
        }
        for i, (px, py, mass) in enumerate(zip(bx, by, bm, strict=True))
    )

    member_mass = float(candidates.mass_msun.sum())
    discrete_stellar_mass = member_mass + float(sum(bcg_masses.values()))
    icl_mass = float(icl_fraction) * discrete_stellar_mass
    north_x, north_y = bcg_xy["north_bcg"]
    south_x, south_y = bcg_xy["south_bcg"]
    icl_shape = gaussian_surface_density_shape(
        xx,
        yy,
        center_x=0.5 * (north_x + south_x),
        center_y=0.5 * (north_y + south_y),
        sigma_major_arcsec=float(protocol["icl_model"]["major_sigma_arcsec"]),
        axis_ratio=float(protocol["icl_model"]["axis_ratio"]),
        theta_deg=angle_from_axis(south_x - north_x, south_y - north_y),
    )
    icl_map = normalize_surface_mass(icl_shape, icl_mass)
    if icl_mass > 0.0:
        ix, iy, im = block_compress_surface(
            axis, icl_map, blocks_per_axis=int(map_config["compression_blocks_per_axis"])
        )
        source_rows.extend(
            {
                "component": "icl_nuisance",
                "source_id": f"icl_block_{i:03d}",
                "x_arcsec": px,
                "y_arcsec": py,
                "mass_msun": mass,
            }
            for i, (px, py, mass) in enumerate(zip(ix, iy, im, strict=True))
        )

    gas_map = np.zeros_like(xx)
    gas_component_masses = []
    for component in gas_components:
        shape = dpie_surface_density_shape(
            xx,
            yy,
            center_x=component["x_arcsec"],
            center_y=component["y_arcsec"],
            ellipticity=component["ellipticity"],
            theta_deg=component["theta_deg"],
            r_core_arcsec=component["r_core_arcsec"],
            r_cut_arcsec=component["r_cut_arcsec"],
        )
        component_mass = dpie_total_mass_msun(
            sigma_lt_km_s=component["sigma_LT_km_s"],
            r_core_arcsec=component["r_core_arcsec"],
            r_cut_arcsec=component["r_cut_arcsec"],
            scale_kpc_per_arcsec=scale,
        ) * float(gas_mass_multiplier)
        gas_component_masses.append(component_mass)
        gas_map += normalize_surface_mass(shape, component_mass)
    gx, gy, gm = block_compress_surface(
        axis, gas_map, blocks_per_axis=int(map_config["compression_blocks_per_axis"])
    )
    source_rows.extend(
        {
            "component": "hot_gas",
            "source_id": f"gas_block_{i:03d}",
            "x_arcsec": px,
            "y_arcsec": py,
            "mass_msun": mass,
        }
        for i, (px, py, mass) in enumerate(zip(gx, gy, gm, strict=True))
    )
    sources = pd.DataFrame(source_rows)
    maps = {"axis_arcsec": axis, "bcg_mass": bcg_map, "icl_mass": icl_map, "gas_mass": gas_map}
    audits = {
        "member_count": int(len(candidates)),
        "excluded_near_bcg": excluded_near_bcg,
        "member_mass_msun": member_mass,
        "bcg_mass_msun": float(sum(bcg_masses.values())),
        "icl_mass_msun": icl_mass,
        "gas_mass_msun": float(sum(gas_component_masses)),
        "total_mass_msun": float(sources.mass_msun.sum()),
        "north_bcg_x_arcsec": north_x,
        "north_bcg_y_arcsec": north_y,
        "south_bcg_x_arcsec": south_x,
        "south_bcg_y_arcsec": south_y,
    }
    return sources, maps, audits


def main() -> None:
    protocol_path = ROOT / "configs" / "p0589_macs0416_baryon_field_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    p0588_path = ROOT / "configs" / "p0588_independent_transfer_readiness_protocol.json"
    p0588 = json.loads(p0588_path.read_text(encoding="utf-8"))
    catalog_path = ROOT / protocol["stellar_model"]["member_catalog"]
    catalog = load_member_table(catalog_path)
    gas_components = p0588["macs0416"]["hot_gas_components"]
    output_dir = ROOT / protocol["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    nominal = {
        "member_abs_dz": protocol["stellar_model"]["nominal_abs_dz"],
        "member_aperture_arcsec": protocol["stellar_model"]["nominal_aperture_arcsec"],
        "mass_to_light_exponent": protocol["stellar_model"]["nominal_mass_to_light_exponent"],
        "gas_mass_multiplier": protocol["gas_model"]["nominal_multiplier"],
        "icl_fraction": protocol["icl_model"]["nominal_mass_fraction_of_discrete_stars"],
        "south_to_north_bcg_mass_ratio": protocol["stellar_model"]["south_to_north_mass_ratio"],
    }
    variants = [("nominal", nominal)]
    for parameter, values in protocol["one_at_a_time_variants"].items():
        for value in values:
            if math.isclose(float(value), float(nominal[parameter]), rel_tol=0.0, abs_tol=1e-10):
                continue
            updated = dict(nominal)
            updated[parameter] = value
            variants.append((f"{parameter}_{str(value).replace('.', 'p')}", updated))

    lens = pd.read_csv(ROOT / protocol["lens_geometry_sampling_only"])
    image_x, image_y = sky_to_lens_offsets(
        lens.ra_deg,
        lens.dec_deg,
        reference_ra_deg=protocol["coordinate_system"]["reference_ra_deg"],
        reference_dec_deg=protocol["coordinate_system"]["reference_dec_deg"],
    )
    built: dict[str, tuple[pd.DataFrame, dict, dict, object]] = {}
    rows = []
    for variant_id, parameters in variants:
        sources, maps, audit = build_variant(catalog, protocol, gas_components, **parameters)
        morphology = weighted_morphology(sources.x_arcsec, sources.y_arcsec, sources.mass_msun)
        field = build_baryonic_metric_correction_field(
            sources.x_arcsec,
            sources.y_arcsec,
            sources.mass_msun,
            total_mass_msun=float(sources.mass_msun.sum()),
            scale_kpc_per_arcsec=protocol["coordinate_system"]["scale_kpc_per_arcsec_planck18"],
            minimum_permittivity=protocol["p0586d_field"]["minimum_permittivity"],
            a0_m_s2=protocol["p0586d_field"]["a0_m_s2"],
            gate_power=protocol["p0586d_field"]["gate_power"],
            anisotropy=protocol["p0586d_field"]["anisotropy_tau"],
            smoothing_r80_fraction=protocol["p0586d_field"]["smoothing_r80_fraction"],
            half_width_arcsec=protocol["p0586d_field"]["half_width_arcsec"],
            pixels_per_axis=protocol["p0586d_field"]["pixels_per_axis"],
            point_softening_arcsec=protocol["p0586d_field"]["point_softening_arcsec"],
        )
        built[variant_id] = (sources, maps, audit, field)
        sampled_x, sampled_y = field.alpha_arcsec(image_x, image_y)
        rows.append(
            {
                "variant_id": variant_id,
                **parameters,
                **audit,
                **morphology,
                "asymmetry_gate": field.audit["asymmetry_gate"],
                "field_rms_arcsec": field.audit["correction_RMS_arcsec_at_distance_ratio_one"],
                "image_position_field_rms_arcsec": float(np.sqrt(np.mean(sampled_x**2 + sampled_y**2))),
                "metric_minimum_eigenvalue": field.audit["metric_minimum_eigenvalue"],
                "normalized_curl_rms": field.audit["normalized_curl_RMS"],
            }
        )

    nominal_sources, nominal_maps, nominal_audit, nominal_field = built["nominal"]
    nominal_grid_x = nominal_field.alpha_x_arcsec
    nominal_grid_y = nominal_field.alpha_y_arcsec
    nominal_image_x, nominal_image_y = nominal_field.alpha_arcsec(image_x, image_y)
    table = pd.DataFrame(rows)
    for index, row in table.iterrows():
        field = built[row.variant_id][3]
        difference = np.hypot(
            field.alpha_x_arcsec - nominal_grid_x,
            field.alpha_y_arcsec - nominal_grid_y,
        )
        fx, fy = field.alpha_arcsec(image_x, image_y)
        image_difference = np.hypot(fx - nominal_image_x, fy - nominal_image_y)
        vector_dot = float(np.sum(field.alpha_x_arcsec * nominal_grid_x + field.alpha_y_arcsec * nominal_grid_y))
        norm = math.sqrt(
            float(np.sum(field.alpha_x_arcsec**2 + field.alpha_y_arcsec**2))
            * float(np.sum(nominal_grid_x**2 + nominal_grid_y**2))
        )
        table.loc[index, "field_difference_rms_arcsec"] = float(np.sqrt(np.mean(difference**2)))
        table.loc[index, "image_position_difference_rms_arcsec"] = float(np.sqrt(np.mean(image_difference**2)))
        table.loc[index, "field_vector_cosine_similarity"] = vector_dot / max(norm, np.finfo(float).tiny)

    metrics_path = output_dir / protocol["outputs"]["variant_metrics"]
    table.to_csv(metrics_path, index=False)
    nominal_sources_path = ROOT / protocol["outputs"]["nominal_sources"]
    nominal_sources_path.parent.mkdir(parents=True, exist_ok=True)
    nominal_sources.to_csv(nominal_sources_path, index=False)
    np.savez_compressed(output_dir / protocol["outputs"]["nominal_maps"], **nominal_maps)
    np.savez_compressed(
        output_dir / protocol["outputs"]["nominal_metric_field"],
        axis_arcsec=nominal_field.axis_arcsec,
        alpha_x_arcsec=nominal_field.alpha_x_arcsec,
        alpha_y_arcsec=nominal_field.alpha_y_arcsec,
    )

    impact = table[table.variant_id != "nominal"].copy()
    impact["relative_image_field_change"] = impact.image_position_difference_rms_arcsec / max(
        float(table.loc[table.variant_id == "nominal", "image_position_field_rms_arcsec"].iloc[0]),
        np.finfo(float).tiny,
    )
    impact = impact.sort_values("relative_image_field_change", ascending=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    extent = [nominal_maps["axis_arcsec"][0], nominal_maps["axis_arcsec"][-1]] * 2
    stellar_map = nominal_maps["bcg_mass"] + nominal_maps["icl_mass"]
    axes[0].imshow(np.log10(stellar_map + 1.0), origin="lower", extent=extent, cmap="magma")
    axes[0].scatter(nominal_sources.query("component == 'member_star'").x_arcsec, nominal_sources.query("component == 'member_star'").y_arcsec, s=3, c="cyan", alpha=0.45)
    axes[0].set_title("stellar + bounded ICL")
    axes[1].imshow(np.log10(nominal_maps["gas_mass"] + 1.0), origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title("published gas geometry")
    step = 6
    grid_x, grid_y = np.meshgrid(nominal_field.axis_arcsec, nominal_field.axis_arcsec, indexing="xy")
    axes[2].quiver(grid_x[::step, ::step], grid_y[::step, ::step], nominal_grid_x[::step, ::step], nominal_grid_y[::step, ::step])
    axes[2].scatter(image_x, image_y, s=2, c="tab:red", alpha=0.25)
    axes[2].set_title("frozen P0586D field; images unscored")
    for axis in axes:
        axis.set(xlabel="west offset (arcsec)", ylabel="north offset (arcsec)", xlim=(-150, 150), ylim=(-150, 150))
        axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output_dir / protocol["outputs"]["figure"], dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0589-MACS0416-BARYON-FIELD-RESULTS-0.1.0",
        "status": "complete_residual_blind_baryon_field",
        "protocol": {"path": rel(protocol_path), "sha256": sha256(protocol_path)},
        "input_hashes": {
            "buffalo_catalog": sha256(catalog_path),
            "p0588_protocol": sha256(p0588_path),
            "vizier_bcg_catalog": sha256(ROOT / protocol["stellar_model"]["bcg_coordinate_and_ratio_source"]),
            "lens_geometry": sha256(ROOT / protocol["lens_geometry_sampling_only"]),
        },
        "nominal": {
            **nominal_audit,
            **weighted_morphology(nominal_sources.x_arcsec, nominal_sources.y_arcsec, nominal_sources.mass_msun),
            "compressed_source_points": int(len(nominal_sources)),
            "image_coordinates_sampled_without_residual": int(len(lens)),
            "p0586d_field_rms_arcsec": float(table.loc[table.variant_id == "nominal", "field_rms_arcsec"].iloc[0]),
            "p0586d_image_position_field_rms_arcsec": float(table.loc[table.variant_id == "nominal", "image_position_field_rms_arcsec"].iloc[0]),
        },
        "sensitivity": {
            "variants": int(len(table)),
            "largest_relative_image_field_changes": impact[["variant_id", "relative_image_field_change", "field_vector_cosine_similarity"]].head(6).to_dict("records"),
            "minimum_field_vector_cosine_similarity": float(impact.field_vector_cosine_similarity.min()),
            "maximum_normalized_curl_rms": float(table.normalized_curl_rms.max()),
        },
        "blindness": {
            "kappa_pixels_read": 0,
            "dark_halo_coordinates_read": 0,
            "image_residuals_calculated": 0,
            "formula_parameters_changed": 0,
        },
        "next_stage": "freeze a source-only curved-route family, then compare its predicted arrival density and backtracked origins to several public MACS0416 lens reconstructions",
        "claim_limits": protocol["claim_limits"],
        "outputs": {"variant_metrics": rel(metrics_path), "nominal_sources": rel(nominal_sources_path)},
    }
    report_path = output_dir / protocol["outputs"]["report"]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / protocol["outputs"]["summary"]).write_text(
        "# P0589 MACS J0416 residual-blind baryon field\n\n"
        f"The nominal registered field contains {nominal_audit['member_count']} member galaxies after BCG de-duplication, two explicit BCG profiles, four gas dPIE components, and a bounded (not measured) ICL nuisance. "
        f"Its represented mass is {nominal_audit['total_mass_msun']:.4e} Msun.\n\n"
        f"All {len(table)} baryonic variants were constructed and passed through the unchanged P0586D field before reading any kappa pixel, dark-halo center, or image residual. The largest image-location field sensitivity is {impact.relative_image_field_change.max():.2%}; see variant_metrics.csv for the responsible nuisance.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
