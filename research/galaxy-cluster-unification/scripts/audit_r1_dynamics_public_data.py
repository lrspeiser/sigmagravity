from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_dynamics_public_data_targets.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def build_audit(
    config_path: Path,
    availability_output: Path,
    product_output: Path,
    report_output: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    availability_rows: list[dict] = []
    product_rows: list[dict] = []

    for system in config["systems"]:
        publication = system["publication"]
        profile = system["bcg_profile"]
        source_dir = ROOT / publication["local_source_directory"]
        required = [source_dir / name for name in publication["required_local_files"]]
        missing_required = [_display_path(path) for path in required if not path.is_file()]
        local_products = 0

        for product in system["archive"]["products"]:
            cutout = product["cutout"]
            local_path = ROOT / cutout["local_path"]
            local = local_path.is_file()
            local_products += int(local)
            product_rows.append(
                {
                    "system": system["system"],
                    "dp_id": product["dp_id"],
                    "proposal_ids": ";".join(system["archive"]["proposal_ids"]),
                    "target_name": product["target_name"],
                    "calibration_level": product["calibration_level"],
                    "public_release": product["public_release"],
                    "exposure_seconds": product["exposure_seconds"],
                    "full_cube_content_length_bytes": product["content_length_bytes"],
                    "download_url": product["download_url"],
                    "datalink_url": product["datalink_url"],
                    "cutout_center_ra_deg": cutout["center_ra_deg"],
                    "cutout_center_dec_deg": cutout["center_dec_deg"],
                    "cutout_radius_deg": cutout["radius_deg"],
                    "cutout_wavelength_min_m": cutout["wavelength_min_m"],
                    "cutout_wavelength_max_m": cutout["wavelength_max_m"],
                    "local_path": cutout["local_path"],
                    "local_file_present": local,
                    "local_size_bytes": local_path.stat().st_size if local else 0,
                    "local_sha256": _sha256(local_path) if local else "",
                }
            )

        public_level3 = all(
            product["public_release"] and product["calibration_level"] == 3
            for product in system["archive"]["products"]
        )
        all_products_local = local_products == len(system["archive"]["products"])
        published_ready = all(
            [
                profile["published_numerical_values_table"],
                profile["published_measurement_covariance"],
                profile["published_likelihood_or_posterior"],
            ]
        )
        raw_inputs_local = public_level3 and all_products_local
        raw_ready = raw_inputs_local and system["extraction_protocol_frozen"]
        availability_rows.append(
            {
                "system": system["system"],
                "redshift": system["redshift"],
                "profile_points": profile["profile_points"],
                "radial_support_kpc": profile["radial_support_kpc"],
                "source_package_present": source_dir.is_dir() and not missing_required,
                "missing_required_source_files": ";".join(missing_required),
                "published_numerical_profile_table": profile[
                    "published_numerical_values_table"
                ],
                "published_measurement_covariance": profile[
                    "published_measurement_covariance"
                ],
                "published_likelihood_or_posterior": profile[
                    "published_likelihood_or_posterior"
                ],
                "publication_package_state": profile["publication_package_state"],
                "public_level3_cube_products": len(system["archive"]["products"]),
                "all_required_level3_cubes_public": public_level3,
                "local_cutout_products": local_products,
                "all_required_cutouts_local": all_products_local,
                "raw_inputs_local": raw_inputs_local,
                "extraction_protocol_frozen": system["extraction_protocol_frozen"],
                "published_likelihood_ready": published_ready,
                "raw_reconstruction_ready": raw_ready,
                "full_r1_ready": False,
                "readiness": system["readiness"],
                "remaining_reconstruction_requirements": ";".join(
                    system["reconstruction_requirements"]
                ),
            }
        )

    availability = pd.DataFrame(availability_rows)
    products = pd.DataFrame(product_rows)
    for path in (availability_output, product_output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(availability_output, index=False)
    products.to_csv(product_output, index=False)

    report = {
        "audit_version": config["audit_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": config["selection_rule"],
        "summary": {
            "systems_audited": len(availability),
            "systems_with_published_numerical_profile_table": int(
                availability["published_numerical_profile_table"].sum()
            ),
            "systems_with_published_measurement_covariance": int(
                availability["published_measurement_covariance"].sum()
            ),
            "systems_with_public_level3_raw_cubes": int(
                availability["all_required_level3_cubes_public"].sum()
            ),
            "systems_with_all_cutouts_local": int(
                availability["all_required_cutouts_local"].sum()
            ),
            "systems_with_raw_inputs_local": int(
                availability["raw_inputs_local"].sum()
            ),
            "systems_published_likelihood_ready": int(
                availability["published_likelihood_ready"].sum()
            ),
            "systems_raw_reconstruction_ready": int(
                availability["raw_reconstruction_ready"].sum()
            ),
            "systems_full_r1_ready": int(availability["full_r1_ready"].sum()),
        },
        "decision": {
            "public_data_shortfall": "No numerical BCG dispersion table, measurement covariance, or likelihood/posterior was found in either publication package.",
            "raw_data_path": "Both systems have public level-3 ESO MUSE cubes, so an independent pPXF reconstruction is possible after the unreported masks and nuisance choices are frozen.",
            "figure_digitization": config["decision_rule"]["figure_digitization"],
            "next_action": (
                "Freeze and test the MACS J1206 mask, pPXF setup, and covariance protocol; in parallel continue the residual-blind search for the second non-disturbed structural promotion."
                if "MACS J1206"
                in availability.loc[availability["raw_inputs_local"], "system"].tolist()
                else "Ingest the MACS J1206 BCG cutout and continue the residual-blind search for the second non-disturbed structural promotion."
            ),
        },
        "outputs": {
            "availability": _display_path(availability_output),
            "archive_products": _display_path(product_output),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--availability-output",
        type=Path,
        default=ROOT / "data/derived/r1_dynamics_public_data_availability.csv",
    )
    parser.add_argument(
        "--product-output",
        type=Path,
        default=ROOT / "data/derived/r1_dynamics_archive_products.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results/r1_dynamics_public_data_audit/report.json",
    )
    args = parser.parse_args()
    report = build_audit(
        args.config,
        args.availability_output,
        args.product_output,
        args.report_output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
