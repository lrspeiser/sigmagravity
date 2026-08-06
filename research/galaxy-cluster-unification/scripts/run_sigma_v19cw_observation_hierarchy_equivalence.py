#!/usr/bin/env python3
"""Commission an observation-hierarchical CIAO response combination on Abell 2146."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v17c_integrated_spectra as inherited_spectra

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cw_observation_hierarchy_equivalence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_file(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or path.stat().st_size != int(item.get("bytes", path.stat().st_size)):
        raise RuntimeError(f"missing or resized frozen input: {path}")
    if sha256(path) != item["sha256"]:
        raise RuntimeError(f"frozen input hash changed: {path}")
    return path


def validate_preflight(config: dict[str, Any]) -> dict[str, Any]:
    implementation = config["implementation"]
    runner = ROOT / implementation["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19CW config names another runner")
    if sha256(runner) != implementation["runner_sha256"]:
        raise RuntimeError("V19CW runner changed after freeze")
    parents = config["parents"]
    checked: dict[str, Any] = {}
    for key in ("v19x2_config", "v19x2_runner", "validated_cell_index", "unified_product_index"):
        checked[key] = str(validate_file(parents[key]))
    x2_report_path = validate_file(parents["v19x2_terminal_report"])
    x2_report = load_json(x2_report_path)
    required_x2 = parents["v19x2_terminal_report"]
    if x2_report.get("status") != required_x2["required_status"] or x2_report.get("execution_exception") != required_x2["required_exception"]:
        raise RuntimeError("V19CW parent V19X2 failure disposition changed")
    cu_report_path = validate_file(parents["v19cu_terminal_report"])
    if load_json(cu_report_path).get("status") != parents["v19cu_terminal_report"]["required_status"]:
        raise RuntimeError("V19CW parent V19CU disposition changed")
    failure_log = validate_file(parents["bullet_failure_log"])
    failure_text = failure_log.read_text(encoding="utf-8", errors="replace")
    missing_fragments = [fragment for fragment in parents["bullet_failure_log"]["required_fragments"] if fragment not in failure_text]
    if missing_fragments:
        raise RuntimeError(f"V19CW failure log lacks frozen fragments: {missing_fragments}")
    direct_paths = {role: validate_file(item) for role, item in config["direct_reference"]["products"].items()}
    authorization = config["authorization"]
    sealed = (
        authorization["run_abell2146_equivalence_commissioning"]
        and not authorization["run_bullet_hierarchy"]
        and not authorization["fit_temperature_or_abundance"]
        and not authorization["change_cells_grouping_background_rule_or_response_weights"]
        and not authorization["change_gravity_formula_parameter_source_state_or_lensing_target"]
        and not authorization["run_v19bq_v19bs_or_derive_action"]
    )
    if not sealed:
        raise RuntimeError("V19CW authorization boundary is not sealed")
    return {
        "inputs_exact": True,
        "failure_fragments_exact": True,
        "authorization_sealed": True,
        "direct_products": {key: str(value) for key, value in direct_paths.items()},
    }


def load_abell_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    index_path = ROOT / config["parents"]["unified_product_index"]["path"]
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = int(config["parents"]["unified_product_index"]["rows"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"unified product index has {len(rows)} rows, expected {expected_rows}")
    cluster = config["direct_reference"]["cluster"]
    selected = [row for row in rows if row["cluster"] == cluster]
    if len(selected) != int(config["direct_reference"]["cells"]):
        raise RuntimeError("Abell 2146 product-index membership changed")
    cells: list[dict[str, Any]] = []
    for row in selected:
        product_root = Path(row["cell_directory"]) / "products"
        paths = {
            "source": product_root / row["source_pha_name"],
            "background": product_root / row["background_pha_name"],
            "arf": product_root / row["arf_name"],
            "rmf": product_root / row["rmf_name"],
        }
        expected = {
            "source": (int(row["source_pha_bytes"]), row["source_pha_sha256"]),
            "background": (int(row["background_pha_bytes"]), row["background_pha_sha256"]),
            "arf": (int(row["arf_bytes"]), row["arf_sha256"]),
            "rmf": (int(row["rmf_bytes"]), row["rmf_sha256"]),
        }
        for role, path in paths.items():
            size, digest = expected[role]
            if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
                raise RuntimeError(f"V19CW changed {role} input: {path}")
        cells.append({"obsid": int(row["obsid"]), "cell_name": row["cell_name"], **paths})
    cells.sort(key=lambda row: (row["obsid"], row["cell_name"]))
    observed_groups: dict[str, int] = defaultdict(int)
    for row in cells:
        observed_groups[str(row["obsid"])] += 1
    if dict(observed_groups) != config["direct_reference"]["observation_groups"]:
        raise RuntimeError(f"V19CW observation partition changed: {dict(observed_groups)}")
    return cells


def write_stack(path: Path, values: list[Path]) -> dict[str, Any]:
    content = "\n".join(str(value) for value in values) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"existing V19CW stack changed: {path}")
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "rows": len(values), "sha256": sha256(path)}


def pha_only_combine(label: str, sources: list[Path], work: Path, config: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    stack = work / f"{label}_source.lis"
    stack_record = write_stack(stack, sources)
    outroot = work / label
    source = work / f"{label}_src.pi"
    background = work / f"{label}_bkg.pi"
    hierarchy = config["hierarchy"]
    command = [
        "combine_spectra", f"src_spectra=@{stack}", f"outroot={outroot}",
        "src_arfs=NONE", "src_rmfs=NONE", "bkg_arfs=NONE", "bkg_rmfs=NONE",
        f"method={hierarchy['first_level_method']}", f"bscale_method={hierarchy['bscale_method']}",
        f"exp_origin={hierarchy['exp_origin']}", "clobber=no", "verbose=1", "mode=h",
    ]
    step = inherited_spectra.run_step(command, work / "logs" / f"{label}_pha.log", [source, background], env)
    return {"source": source, "background": background, "stack": stack_record, "step": step}


def add_response(label: str, phas: list[Path], arfs: list[Path], rmfs: list[Path], work: Path, threshold: float, env: dict[str, str]) -> dict[str, Any]:
    pha_stack = write_stack(work / f"{label}_response_pha.lis", phas)
    arf_stack = write_stack(work / f"{label}_response_arf.lis", arfs)
    rmf_stack = write_stack(work / f"{label}_response_rmf.lis", rmfs)
    arf = work / f"{label}_src.arf"
    rmf = work / f"{label}_src.rmf"
    command = [
        "addresp", f"infile=@{rmf_stack['path']}", f"arffile=@{arf_stack['path']}",
        f"phafile=@{pha_stack['path']}", f"outfile={rmf}", f"outarf={arf}",
        "type=rmf", "method=sum", f"thresh={threshold:.17g}", "clobber=no", "verbose=1", "mode=h",
    ]
    step = inherited_spectra.run_step(command, work / "logs" / f"{label}_addresp.log", [arf, rmf], env)
    return {"arf": arf, "rmf": rmf, "pha_stack": pha_stack, "arf_stack": arf_stack, "rmf_stack": rmf_stack, "step": step}


def set_links(source: Path, background: Path, arf: Path, rmf: Path) -> dict[str, str]:
    values = {"BACKFILE": background.name, "ANCRFILE": arf.name, "RESPFILE": rmf.name}
    with fits.open(source, mode="update", memmap=False) as hdus:
        header = hdus["SPECTRUM"].header
        for key, value in values.items():
            header[key] = value
        hdus.flush(output_verify="exception")
    return values


def group_source(source: Path, destination: Path, work: Path, env: dict[str, str]) -> dict[str, Any]:
    command = [
        "dmgroup", f"infile={source}", f"outfile={destination}", "grouptype=NUM_CTS",
        "grouptypeval=25", "binspec=", "xcolumn=CHANNEL", "ycolumn=COUNTS", "tabspec=",
        "tabcolumn=", "stopspec=", "stopcolumn=", "clobber=no", "verbose=1", "mode=h",
    ]
    return inherited_spectra.run_step(command, work / "logs" / "final_dmgroup.log", [destination], env)


def relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def pha_comparison(reference: Path, candidate: Path, *, grouped: bool) -> dict[str, Any]:
    with fits.open(reference, memmap=False) as left, fits.open(candidate, memmap=False) as right:
        a = left["SPECTRUM"]
        b = right["SPECTRUM"]
        counts_a = np.asarray(a.data["COUNTS"], dtype=float)
        counts_b = np.asarray(b.data["COUNTS"], dtype=float)
        difference = counts_b - counts_a
        result = {
            "counts_exact": bool(np.array_equal(counts_a, counts_b)),
            "counts_max_absolute_difference": float(np.max(np.abs(difference))),
            "counts_relative_l1_difference": float(np.sum(np.abs(difference)) / max(np.sum(np.abs(counts_a)), np.finfo(float).tiny)),
            "exposure_reference": float(a.header["EXPOSURE"]),
            "exposure_candidate": float(b.header["EXPOSURE"]),
            "exposure_relative_difference": relative_difference(float(b.header["EXPOSURE"]), float(a.header["EXPOSURE"])),
            "backscal_reference": float(a.header["BACKSCAL"]),
            "backscal_candidate": float(b.header["BACKSCAL"]),
            "backscal_relative_difference": relative_difference(float(b.header["BACKSCAL"]), float(a.header["BACKSCAL"])),
        }
        if grouped:
            result["grouping_exact"] = bool(np.array_equal(a.data["GROUPING"], b.data["GROUPING"]))
            result["quality_exact"] = bool(np.array_equal(a.data["QUALITY"], b.data["QUALITY"]))
        return result


def arf_comparison(reference: Path, candidate: Path) -> dict[str, Any]:
    with fits.open(reference, memmap=False) as left, fits.open(candidate, memmap=False) as right:
        a = left["SPECRESP"].data
        b = right["SPECRESP"].data
        grid_exact = bool(np.array_equal(a["ENERG_LO"], b["ENERG_LO"]) and np.array_equal(a["ENERG_HI"], b["ENERG_HI"]))
        av = np.asarray(a["SPECRESP"], dtype=float)
        bv = np.asarray(b["SPECRESP"], dtype=float)
        diff = bv - av
        mask = np.abs(av) > np.finfo(np.float32).tiny
        return {
            "energy_grid_exact": grid_exact,
            "relative_l2_difference": float(np.linalg.norm(diff) / max(np.linalg.norm(av), np.finfo(float).tiny)),
            "max_relative_difference": float(np.max(np.abs(diff[mask] / av[mask]))),
            "max_absolute_difference": float(np.max(np.abs(diff))),
        }


def dense_rmf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdus:
        matrix = hdus["MATRIX"].data
        ebounds = hdus["EBOUNDS"].data
        channels = np.asarray(ebounds["CHANNEL"], dtype=int)
        channel_index = {int(value): index for index, value in enumerate(channels)}
        dense = np.zeros((len(matrix), len(channels)), dtype=float)
        for row_index, row in enumerate(matrix):
            groups = int(row["N_GRP"])
            starts = np.atleast_1d(row["F_CHAN"])[:groups]
            lengths = np.atleast_1d(row["N_CHAN"])[:groups]
            values = np.asarray(row["MATRIX"], dtype=float)
            offset = 0
            for start, length in zip(starts, lengths, strict=True):
                length_int = int(length)
                target = channel_index[int(start)]
                dense[row_index, target : target + length_int] = values[offset : offset + length_int]
                offset += length_int
            if offset != len(values):
                raise RuntimeError(f"RMF row packing mismatch in {path}: row {row_index}")
        energies = np.column_stack((np.asarray(matrix["ENERG_LO"]), np.asarray(matrix["ENERG_HI"])))
        bounds = np.column_stack((np.asarray(ebounds["E_MIN"]), np.asarray(ebounds["E_MAX"])))
        return energies, channels, bounds, dense


def rmf_comparison(reference: Path, candidate: Path) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    ea, ca, ba, a = dense_rmf(reference)
    eb, cb, bb, b = dense_rmf(candidate)
    grids_exact = bool(np.array_equal(ea, eb) and np.array_equal(ca, cb) and np.array_equal(ba, bb))
    if a.shape != b.shape:
        raise RuntimeError(f"RMF shape changed: {a.shape} != {b.shape}")
    diff = b - a
    return ({
        "energy_and_channel_grids_exact": grids_exact,
        "dense_shape": list(a.shape),
        "dense_relative_frobenius_difference": float(np.linalg.norm(diff) / max(np.linalg.norm(a), np.finfo(float).tiny)),
        "dense_max_absolute_difference": float(np.max(np.abs(diff))),
    }, (a, b))


def folded_comparison(reference_arf: Path, candidate_arf: Path, reference_rmf: np.ndarray, candidate_rmf: np.ndarray) -> dict[str, Any]:
    with fits.open(reference_arf, memmap=False) as left, fits.open(candidate_arf, memmap=False) as right:
        table_a = left["SPECRESP"].data
        table_b = right["SPECRESP"].data
        energy = 0.5 * (np.asarray(table_a["ENERG_LO"], dtype=float) + np.asarray(table_a["ENERG_HI"], dtype=float))
        width = np.asarray(table_a["ENERG_HI"], dtype=float) - np.asarray(table_a["ENERG_LO"], dtype=float)
        area_a = np.asarray(table_a["SPECRESP"], dtype=float)
        area_b = np.asarray(table_b["SPECRESP"], dtype=float)
    profiles = {
        "flat": np.ones_like(energy),
        "powerlaw_e_minus_2": np.power(np.maximum(energy, 1e-6), -2.0),
        "thermal_8kev_proxy": np.exp(-energy / 8.0) / np.sqrt(np.maximum(energy, 1e-6)),
        "line_6p7kev_proxy": np.exp(-0.5 * np.square((energy - 6.7) / 0.08)),
    }
    results: dict[str, Any] = {}
    for name, profile in profiles.items():
        folded_a = (profile * width * area_a) @ reference_rmf
        folded_b = (profile * width * area_b) @ candidate_rmf
        diff = folded_b - folded_a
        scale = max(float(np.sum(np.abs(folded_a))), np.finfo(float).tiny)
        results[name] = {
            "relative_l1_difference": float(np.sum(np.abs(diff)) / scale),
            "max_channel_fraction_difference": float(np.max(np.abs(diff)) / scale),
        }
    return results


def execute(config: dict[str, Any]) -> dict[str, Any]:
    preflight = validate_preflight(config)
    cells = load_abell_cells(config)
    scratch = Path(config["implementation"]["scratch_root"])
    env = inherited_spectra.isolated_environment(os.environ, scratch / "pfiles", scratch / "tmp")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[int(cell["obsid"])].append(cell)
    first_level: list[dict[str, Any]] = []
    for obsid in sorted(grouped):
        group = grouped[obsid]
        work = scratch / f"obs{obsid}"
        pha = pha_only_combine(f"ABELL2146_obs{obsid}", [row["source"] for row in group], work, config, env)
        response = add_response(
            f"ABELL2146_obs{obsid}", [row["source"] for row in group], [row["arf"] for row in group],
            [row["rmf"] for row in group], work, float(config["hierarchy"]["intermediate_rmf_threshold"]), env,
        )
        first_level.append({"obsid": obsid, "cells": len(group), "pha": pha, "response": response})
    final_work = scratch / "final"
    final_pha = pha_only_combine(
        "ABELL2146_integrated", [row["pha"]["source"] for row in first_level], final_work, config, env
    )
    final_response = add_response(
        "ABELL2146_integrated", [row["pha"]["source"] for row in first_level],
        [row["response"]["arf"] for row in first_level], [row["response"]["rmf"] for row in first_level],
        final_work, float(config["hierarchy"]["final_rmf_threshold"]), env,
    )
    links = set_links(final_pha["source"], final_pha["background"], final_response["arf"], final_response["rmf"])
    grouped_source = final_work / "ABELL2146_integrated_src_grp.pi"
    grouping_step = group_source(final_pha["source"], grouped_source, final_work, env)
    direct = {role: ROOT / item["path"] for role, item in config["direct_reference"]["products"].items()}
    source_cmp = pha_comparison(direct["source_grouped"], grouped_source, grouped=True)
    background_cmp = pha_comparison(direct["background"], final_pha["background"], grouped=False)
    arf_cmp = arf_comparison(direct["arf"], final_response["arf"])
    rmf_cmp, matrices = rmf_comparison(direct["rmf"], final_response["rmf"])
    folded_cmp = folded_comparison(direct["arf"], final_response["arf"], matrices[0], matrices[1])
    limits = config["equivalence_gates"]
    gates = {
        "preflight_exact": all(value is True or isinstance(value, dict) for value in preflight.values()),
        "ten_observation_groups_exact": len(first_level) == 10 and sum(row["cells"] for row in first_level) == 1270,
        "source_counts_and_grouping_exact": source_cmp["counts_exact"] and source_cmp["grouping_exact"] and source_cmp["quality_exact"],
        "source_exposure_within_tolerance": source_cmp["exposure_relative_difference"] <= float(limits["source_exposure_relative_difference_at_most"]),
        "background_counts_within_tolerance": background_cmp["counts_max_absolute_difference"] <= float(limits["background_counts_max_absolute_difference_at_most"]) and background_cmp["counts_relative_l1_difference"] <= float(limits["background_counts_relative_l1_difference_at_most"]),
        "background_headers_within_tolerance": max(background_cmp["exposure_relative_difference"], background_cmp["backscal_relative_difference"]) <= float(limits["background_exposure_and_backscal_relative_difference_at_most"]),
        "arf_within_tolerance": arf_cmp["energy_grid_exact"] and arf_cmp["relative_l2_difference"] <= float(limits["arf_specrsp_relative_l2_difference_at_most"]) and arf_cmp["max_relative_difference"] <= float(limits["arf_specrsp_max_relative_difference_at_most"]),
        "rmf_within_tolerance": rmf_cmp["energy_and_channel_grids_exact"] and rmf_cmp["dense_relative_frobenius_difference"] <= float(limits["rmf_dense_relative_frobenius_difference_at_most"]) and rmf_cmp["dense_max_absolute_difference"] <= float(limits["rmf_dense_max_absolute_difference_at_most"]),
        "forward_folds_within_tolerance": all(row["relative_l1_difference"] <= float(limits["forward_folded_relative_l1_difference_at_most"]) and row["max_channel_fraction_difference"] <= float(limits["forward_folded_max_channel_fraction_difference_at_most"]) for row in folded_cmp.values()),
        "bullet_remained_sealed": not config["authorization"]["run_bullet_hierarchy"],
        "gravity_and_downstream_stages_remained_sealed": not config["authorization"]["change_gravity_formula_parameter_source_state_or_lensing_target"] and not config["authorization"]["run_v19bq_v19bs_or_derive_action"],
    }
    passed = all(gates.values())
    products = {
        "source_grouped": grouped_source,
        "background": final_pha["background"],
        "arf": final_response["arf"],
        "rmf": final_response["rmf"],
    }
    return {
        "status": "observation_hierarchy_equivalent_and_bullet_recovery_may_be_frozen" if passed else "observation_hierarchy_equivalence_failed_closed",
        "decision": "freeze_bullet_hierarchical_recovery" if passed else "retire_observation_hierarchy_remediation",
        "preflight": preflight,
        "first_level": [{"obsid": row["obsid"], "cells": row["cells"], "pha_steps": row["pha"]["step"], "response_step": row["response"]["step"]} for row in first_level],
        "final_steps": {"pha": final_pha["step"], "response": final_response["step"], "grouping": grouping_step, "links": links},
        "candidate_products": {role: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for role, path in products.items()},
        "comparisons": {"source": source_cmp, "background": background_cmp, "arf": arf_cmp, "rmf": rmf_cmp, "forward_folds": folded_cmp},
        "gates": gates,
        "bullet_hierarchical_execution_authorized": passed,
        "gravity_formula_or_parameter_changed": False,
        "source_state_or_lensing_target_opened": False,
        "v19bq_or_v19bs_run": False,
        "action_derived": False,
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["implementation"]["result"]
    try:
        result = execute(config)
    except Exception as exc:  # noqa: BLE001 - preserve exact terminal disposition
        result = {
            "status": "observation_hierarchy_equivalence_execution_failed_closed",
            "decision": "diagnose_without_running_bullet",
            "exception": f"{type(exc).__name__}: {exc}",
            "bullet_hierarchical_execution_authorized": False,
            "gravity_formula_or_parameter_changed": False,
            "source_state_or_lensing_target_opened": False,
            "v19bq_or_v19bs_run": False,
            "action_derived": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "observation_hierarchy_equivalent_and_bullet_recovery_may_be_frozen":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
