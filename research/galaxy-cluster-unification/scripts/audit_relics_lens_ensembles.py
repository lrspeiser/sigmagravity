from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "relics_lens_models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def audit_ensembles(*, input_dir: Path, manifest_output: Path, report_output: Path) -> dict:
    systems = {
        "A2537": {"archive_slug": "abell2537", "version": "v1"},
        "MACS_J0417": {"archive_slug": "macs0417m11", "version": "v2"},
        "MACS_J0949": {"archive_slug": "rxc0949p17", "version": "v1"},
    }
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for system, expected in systems.items():
        system_dir = input_dir / system
        best = list(system_dir.glob("*_kappa.fits"))
        samples = sorted((system_dir / "range").glob("*_kappa.fits"))
        if len(best) != 1 or len(samples) != 100:
            raise ValueError(
                f"{system}: expected one best map and 100 range maps; "
                f"found {len(best)} and {len(samples)}"
            )
        paths = best + samples
        range_wcs: tuple | None = None
        for path in paths:
            header = fits.getheader(path)
            data = np.asarray(fits.getdata(path), dtype=float)
            if data.ndim != 2:
                raise ValueError(f"{path}: expected 2D FITS image")
            wcs_key = tuple(
                header.get(key)
                for key in (
                    "NAXIS1",
                    "NAXIS2",
                    "CRVAL1",
                    "CRVAL2",
                    "CDELT1",
                    "CDELT2",
                )
            )
            sample_match = re.search(r"-map(\d{3})_", path.name)
            if sample_match:
                if range_wcs is None:
                    range_wcs = wcs_key
                elif wcs_key != range_wcs:
                    raise ValueError(f"{system}: inconsistent range-map WCS in {path.name}")
            rows.append(
                {
                    "system": system.replace("_", " "),
                    "archive_slug": expected["archive_slug"],
                    "version": expected["version"],
                    "map_kind": "mcmc_range" if sample_match else "best",
                    "sample_index": int(sample_match.group(1)) if sample_match else pd.NA,
                    "path": _display_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "naxis1": int(header["NAXIS1"]),
                    "naxis2": int(header["NAXIS2"]),
                    "crval1_deg": float(header["CRVAL1"]),
                    "crval2_deg": float(header["CRVAL2"]),
                    "pixel_scale_x_arcsec": abs(float(header["CDELT1"])) * 3600.0,
                    "pixel_scale_y_arcsec": abs(float(header["CDELT2"])) * 3600.0,
                    "finite_fraction": float(np.isfinite(data).mean()),
                    "kappa_min": float(np.nanmin(data)),
                    "kappa_max": float(np.nanmax(data)),
                }
            )
        summaries[system.replace("_", " ")] = {
            "best_maps": len(best),
            "mcmc_range_maps": len(samples),
            "range_map_shape": [int(range_wcs[1]), int(range_wcs[0])],
            "range_map_pixel_scale_arcsec": abs(float(range_wcs[4])) * 3600.0,
            "radial_covariance_derivable_from_ensemble": True,
        }

    manifest = pd.DataFrame(rows).sort_values(
        ["system", "map_kind", "sample_index"], na_position="first", kind="stable"
    )
    if not np.allclose(manifest["finite_fraction"], 1.0):
        raise ValueError("one or more RELICS kappa maps contains non-finite pixels")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_output, index=False)

    readme = input_dir / "hlsp_relics_hst_multi_lens-models_multi_v3_readme.pdf"
    report = {
        "audit_version": "RELICS-lensing-ensemble-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "name": "MAST RELICS HLSP",
            "doi": "10.17909/T9SP45",
            "landing_page": "https://archive.stsci.edu/hlsp/relics",
            "readme": {
                "path": _display_path(readme),
                "sha256": _sha256(readme),
            },
            "readme_interpretation": (
                "The archive documents one best model plus a range of models from MCMC "
                "sampling of the lens-model uncertainties. Kappa is convergence scaled to D_ls/D_s=1."
            ),
        },
        "systems": summaries,
        "totals": {
            "systems": len(summaries),
            "best_maps": int((manifest["map_kind"] == "best").sum()),
            "mcmc_range_maps": int((manifest["map_kind"] == "mcmc_range").sum()),
            "files_in_manifest": len(manifest),
            "bytes_in_manifest": int(manifest["bytes"].sum()),
        },
        "classification": {
            "observable_level_likelihood": False,
            "alternative_metric_forward_model_ready": False,
            "rerunnable_lenstool_inputs_local": False,
            "model_dependent": "Lenstool strong-lensing reconstructions under the standard lens equation",
            "covariance_status": (
                "Projected radial kappa covariance can be estimated from the 100 MCMC range maps; "
                "this is not a joint covariance with the BCG kinematics or baryonic-profile inference."
            ),
            "archive_input_audit": (
                "The exact A2537, MACS J0417 v2, and RXC J0949 Lenstool directories expose map "
                "products and MCMC range maps but no Lenstool .par file, full member catalog, "
                "image-position input file, or posterior chain."
            ),
        },
        "output": _display_path(manifest_output),
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "data" / "derived" / "relics_lens_ensemble_manifest.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "relics_lens_ensemble_audit" / "report.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit_ensembles(
                input_dir=args.input_dir,
                manifest_output=args.manifest_output,
                report_output=args.report_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
