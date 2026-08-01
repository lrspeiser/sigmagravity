from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


SYSTEMS = {
    "MACS J1206": {
        "redshift": 0.4398,
        "bcg_ra_deg": 181.550625,
        "bcg_dec_deg": -8.8009444444,
        "kpc_per_arcsec": 340.0 / 60.0,
        "dynamics_bins": 6,
        "dynamics_support_kpc": 50.0,
        "dynamics_source": "Biviano et al. 2023, arXiv:2307.06804",
        "lens_source": "Caminha et al. 2017, arXiv:1707.00690",
        "lens_path": ROOT
        / "data/raw/r1_replacement_search_sources/caminha2017_macs_j1206_lensing/main.tex",
        "table_label": "tab:multiple_images",
        "disturbed": False,
        "disturbance_note": "The lens source describes the cluster as dynamically relaxed, while recording an asymmetric smooth component.",
        "baryonic_profile_availability": "parametric BCG light plus satellite-star and hot-gas profiles described; numerical profile arrays not published in the source package",
        "dynamics_values_availability": "six annulus edges are numerical; dispersion values and errors are figure-only",
    },
    "Abell S1063": {
        "redshift": 0.3458,
        "bcg_ra_deg": 342.1832916667,
        "bcg_dec_deg": -44.5308277778,
        "kpc_per_arcsec": 296.0 / 60.0,
        "dynamics_bins": 9,
        "dynamics_support_kpc": 40.0,
        "dynamics_source": "Sartoris et al. 2020, arXiv:2003.08475",
        "lens_source": "Caminha et al. 2016, arXiv:1512.04555",
        "lens_path": ROOT
        / "data/raw/r1_replacement_search_sources/caminha2016_abell_s1063_lensing/sec02.tex",
        "table_label": "tab:families",
        "disturbed": True,
        "disturbance_note": "The dynamics source reports evidence for a recent off-axis merger, so this cannot be a non-disturbed promotion.",
        "baryonic_profile_availability": "parametric BCG plus satellite-star and hot-gas profiles described; numerical profile arrays not published in the source package",
        "dynamics_values_availability": "resolved NE/SW profile extends to 40 kpc; dispersion values and errors are figure-only",
    },
}


def _table_text(text: str, label: str) -> str:
    label_at = text.index("\\label{" + label + "}")
    begin = text.rfind("\\begin{", 0, label_at)
    table_kind = "longtable" if "longtable" in text[begin:label_at] else "tabular"
    data_begin = text.find("\\begin{" + table_kind, label_at)
    if data_begin < 0:
        data_begin = text.rfind("\\begin{" + table_kind, 0, label_at)
    data_end = text.index("\\end{" + table_kind + "}", data_begin)
    return text[data_begin:data_end]


def _number(cell: str) -> float | None:
    matches = re.findall(r"\d+(?:\.\d+)?", cell)
    if not matches:
        return None
    value = float(matches[0])
    if "$-" in cell or cell.lstrip().startswith("-"):
        value *= -1
    return value


def _family(image_id: str) -> str:
    match = re.match(r"(\d+)", image_id)
    if not match:
        raise ValueError(f"cannot identify family for {image_id}")
    return match.group(1)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _parse_images(system: str, meta: dict) -> pd.DataFrame:
    text = meta["lens_path"].read_text(encoding="utf-8")
    table = _table_text(text, meta["table_label"])
    rows = []
    for line in table.splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        parts = [part.strip() for part in line.split("&")]
        raw_id = parts[0]
        id_match = re.match(r"\(?([0-9]+[a-z])\)?", raw_id, re.IGNORECASE)
        if not id_match:
            continue
        image_id = id_match.group(1)
        ra = _number(parts[1])
        dec = _number(parts[2])
        if ra is None or dec is None:
            continue
        predicted = raw_id.lstrip().startswith("(")
        source_excluded = "ast" in raw_id or "^*" in raw_id
        spec_redshift_here = _number(parts[3])
        rows.append(
            {
                "system": system,
                "image_id": image_id,
                "family_id": _family(image_id),
                "ra_deg": ra,
                "dec_deg": dec,
                "spec_redshift_here": spec_redshift_here,
                "model_predicted_not_observed": predicted,
                "source_marked_excluded": source_excluded,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no image rows parsed for {system}")
    family_z = (
        frame.dropna(subset=["spec_redshift_here"])
        .groupby("family_id")["spec_redshift_here"]
        .first()
        .to_dict()
    )
    frame["source_redshift"] = frame["family_id"].map(family_z)
    frame["observable_position"] = ~(
        frame["model_predicted_not_observed"] | frame["source_marked_excluded"]
    )
    frame["strict_position_redshift_input"] = (
        frame["observable_position"] & frame["source_redshift"].notna()
    )
    cos_dec = math.cos(math.radians(meta["bcg_dec_deg"]))
    frame["bcg_radius_arcsec"] = (
        ((frame["ra_deg"] - meta["bcg_ra_deg"]) * cos_dec * 3600.0) ** 2
        + ((frame["dec_deg"] - meta["bcg_dec_deg"]) * 3600.0) ** 2
    ) ** 0.5
    frame["bcg_radius_kpc"] = frame["bcg_radius_arcsec"] * meta["kpc_per_arcsec"]
    frame["inside_dynamics_support"] = (
        frame["bcg_radius_kpc"] <= meta["dynamics_support_kpc"]
    )
    frame["strict_inner_input"] = (
        frame["strict_position_redshift_input"] & frame["inside_dynamics_support"]
    )
    return frame


def build_audit(image_output: Path, ledger_output: Path, report_output: Path) -> dict:
    all_images = []
    ledger = []
    for system, meta in SYSTEMS.items():
        images = _parse_images(system, meta)
        all_images.append(images)
        strict = images.loc[images["strict_position_redshift_input"]]
        inner = images.loc[images["strict_inner_input"]]
        inner_families = set(inner["family_id"])
        associated = strict.loc[strict["family_id"].isin(inner_families)]
        family_dof = 2 * len(associated) - 2 * len(inner_families)
        radial_rank_bound = len(inner)
        structural_pass = (
            meta["dynamics_bins"] >= 3
            and radial_rank_bound >= 3
            and family_dof >= 4
        )
        ledger.append(
            {
                "system": system,
                "redshift": meta["redshift"],
                "residual_blind_disturbed_flag": meta["disturbed"],
                "disturbance_note": meta["disturbance_note"],
                "dynamics_source": meta["dynamics_source"],
                "dynamics_bins": meta["dynamics_bins"],
                "dynamics_support_kpc": meta["dynamics_support_kpc"],
                "dynamics_values_availability": meta["dynamics_values_availability"],
                "baryonic_profile_availability": meta["baryonic_profile_availability"],
                "lens_source": meta["lens_source"],
                "published_image_rows": len(images),
                "strict_position_redshift_inputs": len(strict),
                "strict_inner_image_positions": len(inner),
                "inner_source_families": len(inner_families),
                "family_wide_strict_positions": len(associated),
                "family_wide_position_dof_after_source_coordinates": family_dof,
                "structural_radial_rank_upper_bound": radial_rank_bound,
                "structural_promotion_pass": structural_pass,
                "non_disturbed_structural_promotion": structural_pass
                and not meta["disturbed"],
                "full_r1_ready": False,
                "remaining_blocker": (
                    "numerical BCG dispersion values/covariance, numerical baryonic arrays, "
                    "and complete rerunnable lens nuisance posterior"
                    if structural_pass
                    else "fewer than three strict inner images on dynamics support"
                ),
            }
        )

    image_frame = pd.concat(all_images, ignore_index=True)
    ledger_frame = pd.DataFrame(ledger)
    for path in (image_output, ledger_output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    image_frame.to_csv(image_output, index=False)
    ledger_frame.to_csv(ledger_output, index=False)
    promoted = ledger_frame.loc[ledger_frame["non_disturbed_structural_promotion"]]
    report = {
        "audit_version": "R1A2-replacement-cycle1-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Observable coverage and data quality only; no gravity residual or model preference was inspected.",
        "summary": {
            "new_systems_audited": len(ledger_frame),
            "structural_promotions": int(ledger_frame["structural_promotion_pass"].sum()),
            "non_disturbed_structural_promotions": len(promoted),
            "non_disturbed_promotion_systems": promoted["system"].tolist(),
            "full_r1_ready_systems": int(ledger_frame["full_r1_ready"].sum()),
            "required_non_disturbed_promotions": 2,
            "promotion_gap": max(0, 2 - len(promoted)),
        },
        "decision": {
            "R1A2_advance_to_two_system_jacobian": len(promoted) >= 2,
            "MACS_J1206": "promote to nuisance-input acquisition; do not fit a gravity response",
            "Abell_S1063": "retain as a disturbed/low-inner-rank control; do not count toward the non-disturbed promotion target",
            "next_action": "Find at least one more non-disturbed structural pass and recover numerical MUSE BCG dispersion tables plus complete lens nuisance products for MACS J1206.",
        },
        "outputs": {
            "image_support": _display_path(image_output),
            "candidate_ledger": _display_path(ledger_output),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-output",
        type=Path,
        default=ROOT / "data/derived/r1_replacement_cycle1_image_support.csv",
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        default=ROOT / "data/derived/r1_replacement_cycle1_candidate_ledger.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results/r1_replacement_search_cycle1/report.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_audit(args.image_output, args.ledger_output, args.report_output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
