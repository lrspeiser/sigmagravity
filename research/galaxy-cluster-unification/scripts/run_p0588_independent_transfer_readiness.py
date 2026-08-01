#!/usr/bin/env python3
"""Audit independent transfer data and prepare MACS J0416 member candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    protocol_path = ROOT / "configs/p0588_independent_transfer_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_fresh_cluster_formula_score":
        raise RuntimeError("P0588 protocol is not frozen")

    for source in protocol["inputs"].values():
        if not (ROOT / source).is_file():
            raise FileNotFoundError(source)
    macs = protocol["macs0416"]
    for key in ("gas_parameter_source", "lens_position_source", "buffalo_catalog", "buffalo_readme"):
        if not (ROOT / macs[key]).is_file():
            raise FileNotFoundError(macs[key])

    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_report"]).read_text(encoding="utf-8"))
    rxj_light = json.loads((ROOT / protocol["inputs"]["rxj2129_bcg_icl_report"]).read_text(encoding="utf-8"))
    rxj_terminal = json.loads((ROOT / protocol["inputs"]["rxj2129_terminal_report"]).read_text(encoding="utf-8"))
    public_ceiling = json.loads((ROOT / protocol["inputs"]["ten_system_ceiling_report"]).read_text(encoding="utf-8"))

    buffalo_path = ROOT / macs["buffalo_catalog"]
    with fits.open(buffalo_path, memmap=True) as hdul:
        table = hdul[1].data
        catalog_rows = len(table)
        names = set(table.columns.names)
        required = {
            "ID", "ALPHA_J2000_STACK", "DELTA_J2000_STACK", "ZSPEC", "ZSPEC_Q",
            "FLUX_F160W", "FLUXERR_F160W", "ZCHI2", "ZPDF", "ZPDF_LOW", "ZPDF_HIGH",
        }
        if not required.issubset(names):
            raise RuntimeError(f"BUFFALO catalog missing columns: {sorted(required - names)}")
        zspec = np.asarray(table["ZSPEC"], dtype=float)
        zq = np.asarray(table["ZSPEC_Q"], dtype=float)
        f160 = np.asarray(table["FLUX_F160W"], dtype=float)
        member = (
            np.isfinite(zspec)
            & (np.abs(zspec - float(macs["redshift"])) <= float(macs["spectroscopic_member_window_abs_dz"]))
            & (zq >= float(macs["minimum_spectroscopic_quality"]))
            & np.isfinite(f160)
            & (f160 > 0.0)
        )
        member_rows = pd.DataFrame(
            {
                "catalog_id": np.asarray(table["ID"])[member].astype(int),
                "ra_deg": np.asarray(table["ALPHA_J2000_STACK"], dtype=float)[member],
                "dec_deg": np.asarray(table["DELTA_J2000_STACK"], dtype=float)[member],
                "zspec": zspec[member],
                "zspec_quality": zq[member],
                "f160w_flux_catalog_units": f160[member],
                "f160w_flux_error_catalog_units": np.asarray(table["FLUXERR_F160W"], dtype=float)[member],
                "zphot_chi2": np.asarray(table["ZCHI2"], dtype=float)[member],
                "zphot_pdf": np.asarray(table["ZPDF"], dtype=float)[member],
                "zphot_pdf_low": np.asarray(table["ZPDF_LOW"], dtype=float)[member],
                "zphot_pdf_high": np.asarray(table["ZPDF_HIGH"], dtype=float)[member],
            }
        ).sort_values("f160w_flux_catalog_units", ascending=False)
        valid_zspec = int(np.sum(np.isfinite(zspec) & (zspec > 0.0)))

    members_path = ROOT / protocol["outputs"]["macs0416_members"]
    members_path.parent.mkdir(parents=True, exist_ok=True)
    member_rows.to_csv(members_path, index=False)

    lens = pd.read_csv(ROOT / macs["lens_position_source"])
    lens = lens[lens.system.eq("MACS J0416")].copy()
    if len(lens) != int(macs["expected_multiple_images"]):
        raise RuntimeError("MACS J0416 lens row count differs from frozen expectation")

    systems = [row["system_label"] for row in p0559["physical_map_audits"]]
    readiness = []
    for system in systems:
        readiness.append(
            {
                "system": system,
                "formula_untouched_system": False,
                "raw_multiple_image_positions": True,
                "independent_coordinate_covariance": True,
                "gas_surface_mass_with_uncertainty": False,
                "stellar_member_and_icl_mass_with_uncertainty": False,
                "raw_weak_shear_or_magnification": False,
                "strict_fresh_strong_lens_ready": False,
                "readiness_gates_passed_of_5": 2,
                "primary_gap": "spent formula-development lens; gas has diagonal shell errors and stars use one global BCG normalization",
            }
        )
    readiness.extend(
        [
            {
                "system": "RXJ2129",
                "formula_untouched_system": True,
                "raw_multiple_image_positions": True,
                "independent_coordinate_covariance": False,
                "gas_surface_mass_with_uncertainty": False,
                "stellar_member_and_icl_mass_with_uncertainty": False,
                "raw_weak_shear_or_magnification": False,
                "strict_fresh_strong_lens_ready": False,
                "readiness_gates_passed_of_5": 2,
                "primary_gap": "HST centroid covariance failed; BCG/ICL split is non-identifiable; XMM responses exist but gas mass inference was not run",
            },
            {
                "system": "MACS0416",
                "formula_untouched_system": True,
                "raw_multiple_image_positions": True,
                "independent_coordinate_covariance": False,
                "gas_surface_mass_with_uncertainty": False,
                "stellar_member_and_icl_mass_with_uncertainty": False,
                "raw_weak_shear_or_magnification": False,
                "strict_fresh_strong_lens_ready": False,
                "readiness_gates_passed_of_5": 2,
                "primary_gap": "237 image positions are ready, but their error scale is model-rescaled; gas covariance and a retained ICL/member stellar-mass map are missing",
            },
        ]
    )
    readiness_df = pd.DataFrame(readiness)

    evidence = pd.DataFrame(
        [
            {
                "product": "P0559 four-cluster strong-lens positions",
                "classification": "raw_observable_spent",
                "local": True,
                "usable_role": "mechanism development only",
                "not_proven": "fresh transfer",
            },
            {
                "product": "P0559 ACCEPT gas plus Chandra morphology",
                "classification": "baryonic_forward_input_partial",
                "local": True,
                "usable_role": "gas geometry sensitivity",
                "not_proven": "full gas covariance or stellar/ICL separation",
            },
            {
                "product": "CCCP/MENeaCS weak-lensing masses",
                "classification": "model_dependent_lens_summary",
                "local": True,
                "usable_role": "mass-scale comparator",
                "not_proven": "raw shear likelihood under a new metric",
            },
            {
                "product": "RELICS radial kappa profiles and covariance",
                "classification": "model_dependent_lens_reconstruction",
                "local": True,
                "usable_role": "map-shape diagnostic",
                "not_proven": "metric-independent lens response",
            },
            {
                "product": "RXJ2129 nonparametric total-light profile",
                "classification": "baryonic_forward_input_partial",
                "local": True,
                "usable_role": "total-light shape",
                "not_proven": "BCG versus ICL mass separation",
            },
            {
                "product": "MACS0416 Bergamini multiple-image catalog",
                "classification": "raw_observable_fresh",
                "local": True,
                "usable_role": "fresh descriptive strong-lens score",
                "not_proven": "independent image-coordinate covariance",
            },
            {
                "product": "MACS0416 BUFFALO photometric catalog",
                "classification": "raw_observable_fresh",
                "local": True,
                "usable_role": "member-light and redshift reconstruction",
                "not_proven": "retained diffuse ICL mass map",
            },
            {
                "product": "MACS0416 four Chandra-derived gas dPIE components",
                "classification": "baryonic_forward_input_fresh_model",
                "local": True,
                "usable_role": "fresh gas geometry",
                "not_proven": "full gas-parameter covariance",
            },
        ]
    )

    output_dir = ROOT / protocol["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness_df.to_csv(output_dir / protocol["outputs"]["readiness"], index=False)
    evidence.to_csv(output_dir / protocol["outputs"]["evidence_inventory"], index=False)

    input_paths = {"protocol": protocol_path}
    input_paths.update({key: ROOT / value for key, value in protocol["inputs"].items()})
    input_paths.update(
        {
            "macs0416_gas_source": ROOT / macs["gas_parameter_source"],
            "macs0416_lens_positions": ROOT / macs["lens_position_source"],
            "macs0416_buffalo_catalog": buffalo_path,
            "macs0416_buffalo_readme": ROOT / macs["buffalo_readme"],
        }
    )
    report = {
        "report_version": "P0588-INDEPENDENT-TRANSFER-READINESS-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": rel(protocol_path), "sha256": sha256(protocol_path)},
        "input_hashes": {key: {"path": rel(path), "sha256": sha256(path)} for key, path in input_paths.items()},
        "local_inventory": {
            "strict_fresh_strong_lens_ready_systems": int(readiness_df.strict_fresh_strong_lens_ready.sum()),
            "local_raw_weak_shear_or_magnification_likelihoods": 0,
            "public_data_strict_ready_population_systems": int(public_ceiling["audited_public_data_universe"]["current_strict_ready_systems"]),
            "spent_formula_development_clusters": len(systems),
            "fresh_candidates_audited": 2,
        },
        "macs0416_preparation": {
            "selected_as_next_descriptive_transfer_target": True,
            "selection_used_formula_residual": False,
            "buffalo_catalog_rows": catalog_rows,
            "valid_spectroscopic_redshifts": valid_zspec,
            "spectroscopic_member_candidates": len(member_rows),
            "member_redshift_window": [float(macs["redshift"]) - float(macs["spectroscopic_member_window_abs_dz"]), float(macs["redshift"]) + float(macs["spectroscopic_member_window_abs_dz"])],
            "multiple_image_positions": len(lens),
            "source_families_published": int(macs["expected_source_families"]),
            "source_redshift_range": [float(lens.source_redshift.min()), float(lens.source_redshift.max())],
            "published_reference_model_RMS_arcsec": float(macs["published_reference_model_rms_arcsec"]),
            "chandra_derived_gas_components": len(macs["hot_gas_components"]),
            "strict_gates_passed_of_5": 2,
        },
        "rxj2129_checks": {
            "bcg_icl_identifiable": bool(rxj_light["component_identifiability_gate_pass"]),
            "stellar_mass_mapping_authorized": bool(rxj_light["stellar_mass_mapping_authorized"]),
            "xmm_response_products_passed": bool(rxj_terminal["component_outcomes"]["X4"]["status"] == "pass"),
            "strict_ready": bool(rxj_terminal["global_disposition"]["RXJ2129_counts_as_strict_ready_population_system"]),
        },
        "frozen_formula_context": {
            "p0586d_report_sha256": sha256(ROOT / protocol["inputs"]["p0586d_report"]),
            "p0587_report_sha256": sha256(ROOT / protocol["inputs"]["p0587_report"]),
            "no_new_formula_score_run": True,
        },
        "decision": {
            "next_target": "MACS0416",
            "next_stage": "construct_baryon_only_member_plus_gas_field_before_reading_any_formula_residual",
            "minimum_next_products": [
                "convert the 247 spectroscopic member candidates to a registered stellar-mass/light field with a frozen universal M/L treatment",
                "render the four published Chandra-derived gas dPIE components on the same grid and expose normalization/covariance sensitivity",
                "locate or reconstruct the diffuse ICL light removed from the BUFFALO catalog",
                "freeze a descriptive 0.43 arcsec image scale while labeling it model-rescaled rather than independent covariance",
            ],
            "derived_kappa_or_nfw_allowed_as_primary_target": False,
            "strict_validation_authorized": False,
            "descriptive_fresh_transfer_after_baryon_map": True,
        },
        "claim_limits": protocol["claim_limits"],
        "outputs": {
            "readiness": rel(output_dir / protocol["outputs"]["readiness"]),
            "evidence_inventory": rel(output_dir / protocol["outputs"]["evidence_inventory"]),
            "macs0416_members": rel(members_path),
        },
    }
    (output_dir / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / protocol["outputs"]["summary"]).write_text(
        f"# P0588 independent transfer readiness\n\n"
        f"No local system currently passes all five strict fresh strong-lens gates, and no raw weak-shear or magnification likelihood is local. "
        f"MACS J0416 is the best next descriptive transfer target because it is untouched by P0586-P0587 and has {len(lens)} spectroscopic multiple-image positions from {int(macs['expected_source_families'])} sources.\n\n"
        f"The newly downloaded BUFFALO catalog contains {catalog_rows:,} objects and yields {len(member_rows)} spectroscopic member candidates under the frozen redshift/quality cut. "
        f"Four Chandra-derived gas dPIE components are already recoverable from the local paper source. The remaining material gaps are a retained diffuse ICL map, member stellar-mass normalization/covariance, gas covariance, and a position covariance independent of the standard fitted lens model.\n",
        encoding="utf-8",
    )
    print(json.dumps(report["macs0416_preparation"], indent=2))


if __name__ == "__main__":
    main()
