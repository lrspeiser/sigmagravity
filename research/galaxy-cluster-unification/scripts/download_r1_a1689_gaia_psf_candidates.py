#!/usr/bin/env python3
"""Query the frozen Gaia DR3 PSF-candidate cone for A1689."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from io import StringIO


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_hst_photometry_astrometry_protocol.json"
TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    query = cfg["inputs"]["gaia_dr3_psf_query"]
    adql = f"""SELECT source_id,ra,dec,phot_g_mean_mag,bp_rp,ruwe,astrometric_params_solved
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{query['center_ra_deg']},{query['center_dec_deg']},{query['radius_deg']}))
AND phot_g_mean_mag<{query['maximum_g_mag']}
AND ruwe<{query['maximum_ruwe']}
AND astrometric_params_solved={query['required_astrometric_params_solved']}
ORDER BY phot_g_mean_mag ASC"""
    response = requests.get(TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, timeout=180)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    output = ROOT / query["output"]
    provenance = ROOT / query["provenance"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest().upper()
    provenance.write_text(json.dumps({
        "provenance_version": cfg["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "service": TAP,
        "adql": adql,
        "rows": int(len(frame)),
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "sha256": digest,
        "hst_science_pixels_inspected": False,
        "lens_or_gravity_residual_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(frame), "output": query["output"], "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
