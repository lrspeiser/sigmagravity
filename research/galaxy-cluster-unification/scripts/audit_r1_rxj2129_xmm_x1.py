#!/usr/bin/env python3
"""Audit the frozen RX J2129 XMM X1 calibration outputs without reading event arrays."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis"
)
MANIFEST_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_reduction_manifest.json"
REPORT_PATH = PROJECT / "results/r1_rxj2129_xmm_event_processing/report.json"

PRODUCTS = {
    "MOS1": {
        "name": "0529_0093030201_EMOS1_S001_ImagingEvts.ds",
        "instrument": "EMOS1",
        "expidstr": "S001",
        "minimum_gti_extensions": 7,
        "kind": "normal",
    },
    "MOS2": {
        "name": "0529_0093030201_EMOS2_S002_ImagingEvts.ds",
        "instrument": "EMOS2",
        "expidstr": "S002",
        "minimum_gti_extensions": 7,
        "kind": "normal",
    },
    "pn": {
        "name": "0529_0093030201_EPN_S003_ImagingEvts.ds",
        "instrument": "EPN",
        "expidstr": "S003",
        "minimum_gti_extensions": 12,
        "kind": "normal",
    },
    "pn_OOT": {
        "name": "0529_0093030201_EPN_S003_OutOfTimeEvts.ds",
        "instrument": "EPN",
        "expidstr": "S003",
        "minimum_gti_extensions": 12,
        "kind": "out_of_time_simulation",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_product(spec: dict[str, object]) -> dict[str, object]:
    path = ANALYSIS / str(spec["name"])
    result: dict[str, object] = {
        "path": str(path),
        "exists": path.is_file(),
        "kind": spec["kind"],
    }
    if not path.is_file():
        result["gate_passed"] = False
        return result

    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        events = hdus["EVENTS"]
        header = events.header
        values = {
            key: header.get(key)
            for key in (
                "TELESCOP",
                "INSTRUME",
                "OBS_ID",
                "EXP_ID",
                "EXPIDSTR",
                "DATAMODE",
                "SUBMODE",
                "FILTER",
                "ONTIME",
                "LIVETIME",
                "TSTART",
                "TSTOP",
                "RA_PNT",
                "DEC_PNT",
            )
        }
        gti_extensions = sum(hdu.name.startswith("STDGTI") for hdu in hdus)
        rows = int(header["NAXIS2"])

    finite_exposure = all(
        isinstance(values[key], (int, float))
        and math.isfinite(float(values[key]))
        and float(values[key]) > 0
        for key in ("ONTIME", "LIVETIME")
    )
    identity_gate = (
        values["TELESCOP"] == "XMM"
        and values["INSTRUME"] == spec["instrument"]
        and values["OBS_ID"] == "0093030201"
        and values["EXPIDSTR"] == spec["expidstr"]
        and values["DATAMODE"] == "IMAGING"
        and values["FILTER"] == "Medium"
    )
    result.update(
        {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "event_rows": rows,
            "gti_extensions": gti_extensions,
            "header": values,
            "identity_gate_passed": identity_gate,
            "finite_positive_exposure_gate_passed": finite_exposure,
            "gti_gate_passed": gti_extensions >= int(spec["minimum_gti_extensions"]),
            "gate_passed": identity_gate
            and finite_exposure
            and rows > 0
            and gti_extensions >= int(spec["minimum_gti_extensions"]),
        }
    )
    return result


def audit_logs() -> dict[str, object]:
    log_names = [
        "cifbuild.log",
        "odfingest.log",
        "emproc.log",
        "epproc.log",
        "epproc_normal.log",
    ]
    warning_pattern = re.compile(r"warning \(([^)]+)\)")
    error_pattern = re.compile(r"^\*\* .*: error \(", re.MULTILINE)
    result: dict[str, object] = {}
    for name in log_names:
        path = ANALYSIS / name
        text = path.read_text(errors="replace") if path.is_file() else ""
        counts = Counter(warning_pattern.findall(text))
        result[name] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sas_error_records": len(error_pattern.findall(text)),
            "warning_records_by_code": dict(sorted(counts.items())),
            "task_end_record_present": (
                name in {"cifbuild.log", "odfingest.log"}
                or "ended:" in text
            ),
        }
    return result


def main() -> None:
    products = {label: audit_product(spec) for label, spec in PRODUCTS.items()}
    logs = audit_logs()
    ccf_path = ANALYSIS / "ccf.cif"
    summary_path = ANALYSIS / "0529_0093030201_SCX00000SUM.SAS"

    with fits.open(ccf_path, memmap=True, lazy_load_hdus=True) as hdus:
        ccf = {
            "path": str(ccf_path),
            "bytes": ccf_path.stat().st_size,
            "sha256": sha256(ccf_path),
            "created_utc": hdus[0].header.get("DATE"),
            "creator": hdus[0].header.get("CREATOR"),
            "calindex_rows": int(hdus["CALINDEX"].header["NAXIS2"]),
        }

    marker_names = [
        ".cifbuild_complete",
        ".odfingest_complete",
        ".emproc_complete",
        ".epproc_oot_complete",
        ".epproc_normal_complete",
    ]
    markers = {name: (ANALYSIS / name).is_file() for name in marker_names}
    all_logs_clean = all(
        item["exists"]
        and item["sas_error_records"] == 0
        and item["task_end_record_present"]
        for item in logs.values()
    )
    products_distinct = products["pn"]["sha256"] != products["pn_OOT"]["sha256"]
    x1_pass = (
        all(item["gate_passed"] for item in products.values())
        and all(markers.values())
        and all_logs_clean
        and ccf["calindex_rows"] > 0
        and summary_path.is_file()
        and products_distinct
    )

    warnings_requiring_x2_check = {
        "codes": ["UnidentifiedTimeGaps", "InvalidObtValue"],
        "interpretation": (
            "These pn warnings can compromise fine timing products. X2 must verify the "
            "100-second bin live-time accounting and cleaned exposure; they do not alter "
            "the frozen thresholds and do not by themselves reject spatial spectroscopy."
        ),
        "fine_timing_science_authorized": False,
    }
    manifest = {
        "manifest_version": "R1B3-RXJ2129-XMM-X1-0.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "configs/r1_rxj2129_xmm_event_processing_protocol.json",
        "obsid": "0093030201",
        "analysis_root": str(ANALYSIS),
        "event_arrays_read_during_X1_audit": False,
        "ccf": ccf,
        "summary_file": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size if summary_path.is_file() else 0,
            "exists": summary_path.is_file(),
        },
        "products": products,
        "markers": markers,
        "logs": logs,
        "warnings_requiring_X2_livetime_check": warnings_requiring_x2_check,
        "gates": {
            "all_product_identity_exposure_and_GTI_gates_passed": all(
                item["gate_passed"] for item in products.values()
            ),
            "pn_normal_and_OOT_are_distinct": products_distinct,
            "all_SAS_tasks_exited_without_error_records": all_logs_clean,
            "R1B3_XMM_X1_calibration_gate_passed": x1_pass,
            "R1B3_XMM_X2_flare_background_gate_passed": False,
            "R1B3_XMM_X3_gas_likelihood_gate_passed": False,
        },
    }
    report = {
        "report_version": "R1B3-RXJ2129-XMM-event-processing-0.2",
        "generated_utc": manifest["generated_utc"],
        "stage": "X1_calibration_identity",
        "status": "pass" if x1_pass else "fail",
        "outcome": (
            "X1 passed: exact MOS1 S001, MOS2 S002, pn S003 normal, and pn S003 OOT "
            "products are calibrated, distinct, finite, and GTI-bearing."
            if x1_pass
            else "X1 failed; do not run flare filtering."
        ),
        "raw_livetime_seconds": {
            label: item["header"]["LIVETIME"]
            for label, item in products.items()
            if label != "pn_OOT"
        },
        "event_rows": {
            label: item["event_rows"] for label, item in products.items()
        },
        "warnings_requiring_X2_livetime_check": warnings_requiring_x2_check,
        "gates": manifest["gates"],
        "authorization": {
            "run_frozen_X2_flare_filter_and_livetime_audit": x1_pass,
            "fit_gas_profile": False,
            "infer_dynamical_or_Weyl_response": False,
            "fit_new_force_or_action": False,
        },
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not x1_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
