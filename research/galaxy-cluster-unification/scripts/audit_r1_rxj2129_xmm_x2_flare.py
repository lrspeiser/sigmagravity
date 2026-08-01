#!/usr/bin/env python3
"""Audit the RX J2129 X2a flare-filter products and update the stage report."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis"
)
MANIFEST_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_reduction_manifest.json"
DIAGNOSTICS_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_x2/flare_diagnostics.json"
REPORT_PATH = PROJECT / "results/r1_rxj2129_xmm_event_processing/report.json"

INSTRUMENTS = {
    "MOS1": {
        "clean": "MOS1_clean_events.ds",
        "instrument": "EMOS1",
        "expidstr": "S001",
        "ceiling": 0.35,
    },
    "MOS2": {
        "clean": "MOS2_clean_events.ds",
        "instrument": "EMOS2",
        "expidstr": "S002",
        "ceiling": 0.35,
    },
    "pn": {
        "clean": "pn_clean_events.ds",
        "instrument": "EPN",
        "expidstr": "S003",
        "ceiling": 0.40,
    },
}


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text())
    products: dict[str, object] = {}

    for label, spec in INSTRUMENTS.items():
        path = ANALYSIS / str(spec["clean"])
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
            events = hdus["EVENTS"]
            header = events.header
            rows = int(header["NAXIS2"])
            cleaned_ontime = float(header["ONTIME"])
            cleaned_livetime = float(header["LIVETIME"])
            identity_gate = (
                header.get("INSTRUME") == spec["instrument"]
                and header.get("OBS_ID") == "0093030201"
                and header.get("EXPIDSTR") == spec["expidstr"]
                and header.get("DATAMODE") == "IMAGING"
                and header.get("FILTER") == "Medium"
            )

        raw_livetime = float(manifest["products"][label]["header"]["LIVETIME"])
        ratio = cleaned_livetime / raw_livetime
        flare = diagnostics["instruments"][label]
        convergence_gate = (
            1 <= len(flare["iterations"]) <= 10
            and flare["iterations"][-1]["input_bins"]
            == flare["iterations"][-1]["retained_bins"]
            and float(flare["final_rate_limit_counts_per_second"]) <= float(spec["ceiling"])
        )
        exposure_gate = cleaned_livetime >= 15000 and ratio >= 0.25
        products[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "event_rows": rows,
            "cleaned_ontime_seconds": cleaned_ontime,
            "cleaned_livetime_seconds": cleaned_livetime,
            "raw_livetime_seconds": raw_livetime,
            "cleaned_to_raw_livetime_fraction": ratio,
            "final_rate_limit_counts_per_second": flare[
                "final_rate_limit_counts_per_second"
            ],
            "retained_rate_bins": flare["retained_bins"],
            "rejected_rate_bins": flare["rejected_bins"],
            "identity_gate_passed": identity_gate,
            "convergence_gate_passed": convergence_gate,
            "exposure_gate_passed": exposure_gate,
            "gate_passed": identity_gate
            and convergence_gate
            and exposure_gate
            and rows > 0
            and math.isfinite(cleaned_livetime),
        }

    log_names = [
        "x2a_MOS1_rate.log",
        "x2a_MOS2_rate.log",
        "x2a_pn_rate.log",
        "x2a_MOS1_clean.log",
        "x2a_MOS2_clean.log",
        "x2a_pn_clean.log",
    ]
    error_pattern = re.compile(r"^\*\* .*: error \(", re.MULTILINE)
    logs = {}
    for name in log_names:
        path = ANALYSIS / name
        text = path.read_text(errors="replace") if path.is_file() else ""
        logs[name] = {
            "exists": path.is_file(),
            "sas_error_records": len(error_pattern.findall(text)),
            "task_end_record_present": "ended:" in text,
        }
    logs_gate = all(
        item["exists"]
        and item["sas_error_records"] == 0
        and item["task_end_record_present"]
        for item in logs.values()
    )
    passing_instruments = [
        label for label, item in products.items() if item["gate_passed"]
    ]
    x2a_pass = len(passing_instruments) >= 2 and logs_gate
    generated = datetime.now(timezone.utc).isoformat()

    manifest["manifest_version"] = "R1B3-RXJ2129-XMM-X2a-0.1"
    manifest["generated_utc"] = generated
    manifest["event_arrays_read_during_X2a_flare_filter"] = True
    manifest["X2a_flare"] = {
        "diagnostics": str(DIAGNOSTICS_PATH.relative_to(PROJECT)),
        "products": products,
        "logs": logs,
        "passing_instruments": passing_instruments,
        "nonfatal_metadata_warning": (
            "The first generated GTIs lacked optional CREATOR/DATE keywords. SAS emitted "
            "metadata warnings but applied the GTIs and exited normally; the generator now "
            "writes those keywords for future executions."
        ),
        "fine_timing_science_authorized": False,
    }
    manifest["gates"]["R1B3_XMM_X2a_flare_exposure_gate_passed"] = x2a_pass
    manifest["gates"]["R1B3_XMM_X2_flare_background_gate_passed"] = False
    manifest["gates"]["R1B3_XMM_X3_gas_likelihood_gate_passed"] = False
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    report = {
        "report_version": "R1B3-RXJ2129-XMM-event-processing-0.4",
        "generated_utc": generated,
        "stage": "X2a_flare_exposure",
        "status": "pass" if x2a_pass else "fail",
        "outcome": (
            f"X2a passed with {len(passing_instruments)} instruments: "
            + ", ".join(passing_instruments)
            if x2a_pass
            else "X2a failed; do not proceed to background characterization."
        ),
        "instruments": products,
        "gates": manifest["gates"],
        "authorization": {
            "run_frozen_X2_background_mask_and_scale_audit": x2a_pass,
            "claim_full_X2_pass": False,
            "fit_gas_profile": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not x2a_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
