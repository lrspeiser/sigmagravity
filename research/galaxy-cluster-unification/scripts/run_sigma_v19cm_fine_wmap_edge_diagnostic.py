#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_sigma_v19r_response_commissioning as v19r

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cm_fine_wmap_edge_diagnostic.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict[str, Any]:
    config = load_json(DEFAULT_CONFIG)
    parent_path = ROOT / config["parent"]["path"]
    parent = load_json(parent_path)
    source = Path(config["input"]["source"])
    background = Path(config["input"]["background"])
    workspace = Path(config["diagnostic"]["workspace"])
    prefix = Path(config["environment"]["prefix"])
    parent_log = ROOT / "results" / "sigma_v19ck_single_rmf_diagnostic" / "mkacisrmf_direct_verbose.log"
    preflight = {
        "parent_hash_and_decision_exact": sha256(parent_path) == config["parent"]["sha256"] and parent["decision"] == config["parent"]["required_decision"],
        "exact_diagnostic_error_bound": sha256(parent_log) == config["parent"]["direct_log_sha256"] and config["parent"]["required_error"] in parent_log.read_text(encoding="utf-8"),
        "input_hashes_exact": sha256(source) == config["input"]["source_sha256"] and sha256(background) == config["input"]["background_sha256"],
        "specextract_hash_exact": sha256(prefix / "bin" / "specextract") == config["environment"]["specextract_sha256"],
        "workspace_absent": not workspace.exists(),
        "no_final_or_scientific_authorization": not config["authorization"]["modify_recovery_archive"] and not config["authorization"]["admit_diagnostic_products"] and not config["authorization"]["authorize_final_retry"] and not config["authorization"]["open_target_or_change_gravity_physics"],
    }
    if not all(preflight.values()):
        raise RuntimeError(f"V19CM preflight failed: {preflight}")

    event_dir, products, temp, pfiles = (workspace / name for name in ("e", "products", "tmp", "pfiles"))
    for path in (event_dir, products, temp, pfiles):
        path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, event_dir / "s.fits")
    shutil.copy2(background, event_dir / "b.fits")
    settings = config["diagnostic"]["unchanged_settings"]
    outroot = products / config["input"]["cell"]
    command = [
        str(prefix / "bin" / "specextract"),
        f"infile={event_dir / 's.fits'}[sky=region({settings['fov']})]", f"outroot={outroot}",
        f"bkgfile={event_dir / 'b.fits'}[sky=region({settings['fov']})]", f"asp=@{settings['aspect']}",
        f"mskfile={settings['mask']}", f"badpixfile={settings['badpix']}", "dafile=CALDB", "bkgresp=no",
        "weight=yes", "weight_rmf=yes", "resp_pos=CENTROID", f"refcoord={settings['refcoord']}",
        "correctpsf=no", "combine=no", "grouptype=NONE", "binspec=NONE", "bkg_grouptype=NONE", "bkg_binspec=NONE",
        f"energy={settings['energy']}", f"energy_wmap={settings['energy_wmap']}", "binwmap=det=1", "binarfwmap=1",
        "parallel=no", "nproc=1", f"tmpdir={temp}", "clobber=no", "verbose=5", "mode=h",
    ]
    env = os.environ.copy()
    env["PFILES"] = f"{pfiles};{prefix / 'param'}"
    env["ASCDS_WORK_PATH"] = str(temp)
    env["ASCDS_TMP"] = str(temp)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    log_path = ROOT / config["outputs"]["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr, encoding="utf-8")
    product_paths = {
        "source_pha": Path(f"{outroot}.pi"),
        "background_pha": Path(f"{outroot}_bkg.pi"),
        "arf": Path(f"{outroot}.arf"),
        "rmf": Path(f"{outroot}.rmf"),
    }
    nonempty = all(path.is_file() and path.stat().st_size > 0 for path in product_paths.values())
    response = v19r.response_audit(product_paths["arf"], product_paths["rmf"]) if nonempty else None
    links = v19r.pha_links(product_paths["source_pha"], env) if nonempty else None
    post_hashes = {"source": sha256(source), "background": sha256(background), "v19ck_log": sha256(parent_log)}
    gates = {
        "specextract_returncode_zero": completed.returncode == 0,
        "all_four_products_nonempty": nonempty,
        "arf_finite_with_positive_bins": bool(response and response["arf_finite"] and response["arf_positive_bins"] > 0),
        "rmf_finite_with_nonzero_elements": bool(response and response["rmf_finite"] and response["rmf_nonzero_elements"] > 0),
        "source_pha_links_all_four_products": bool(links and links["BACKFILE"] == product_paths["background_pha"].name and links["ANCRFILE"] == product_paths["arf"].name and links["RESPFILE"] == product_paths["rmf"].name),
        "recovery_inputs_and_logs_unchanged": post_hashes == {"source": config["input"]["source_sha256"], "background": config["input"]["background_sha256"], "v19ck_log": config["parent"]["direct_log_sha256"]},
        "success_authorizes_only_separate_final_recovery_preregistration": not config["authorization"]["authorize_final_retry"],
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": "completed_fine_wmap_edge_diagnostic",
        "decision": "fine_wmap_physically_equivalent_recovery_candidate_passed_separate_final_protocol_required" if all(gates.values()) else "fine_wmap_edge_diagnostic_failed_no_fallback_authorized",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "preflight": preflight,
        "command": command,
        "returncode": completed.returncode,
        "log": config["outputs"]["log"],
        "log_sha256": sha256(log_path),
        "products": {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for key, path in product_paths.items() if path.is_file()},
        "response_audit": response,
        "source_pha_links": links,
        "gate_results": gates,
        "recovery_inputs_after": post_hashes,
        "authorization_boundary": {"final_retry_authorized": False, "diagnostic_products_admitted": False, "target_or_gravity_accessed": False},
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run()
    except Exception as exc:
        report = {"protocol_version": config["protocol_version"], "status": "fine_wmap_edge_diagnostic_failed_closed", "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(), "authorization_boundary": {"final_retry_authorized": False, "target_or_gravity_accessed": False}, "claim_boundary": config["claim_boundary"]}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "completed_fine_wmap_edge_diagnostic":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
