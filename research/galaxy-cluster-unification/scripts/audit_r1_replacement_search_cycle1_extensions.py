from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAND_SOURCE = (
    ROOT
    / "data/raw/r1_replacement_search_sources"
    / "sand2004_six_cluster_full_analysis/ms.tex"
)
J0416_SOURCE = (
    ROOT
    / "data/raw/r1_replacement_search_sources"
    / "bergamini2023_macs_j0416_lensing/paper.tex"
)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _section(text: str, start: str, end: str, start_at: int = 0) -> str:
    begin = text.index(start, start_at)
    finish = text.index(end, begin)
    return text[begin:finish]


def _parse_j0416_images(text: str) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"^\s*([0-9][0-9.]*(?:[a-z]))(?:\\tablefootmark\{[a-z,]+\})?\s*"
        r"&\s*([0-9.]+)\s*&\s*(-[0-9.]+)\s*&\s*([0-9.]+)\\\\",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "system": "MACS J0416",
                "image_id": match.group(1),
                "ra_deg": float(match.group(2)),
                "dec_deg": float(match.group(3)),
                "source_redshift": float(match.group(4)),
                "observable_position": True,
                "spectroscopic_source_redshift": True,
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_id", "ra_deg", "dec_deg"])
    if len(frame) != 237:
        raise ValueError(f"expected 237 MACS J0416 images, parsed {len(frame)}")
    return frame.reset_index(drop=True)


def _parse_sand_dynamics(text: str) -> dict[str, list[dict]]:
    table_start = text.index("%{\\large{\\bf Velocity Disperion Profiles}}")
    table = _section(text, "\\begin{tabular}{lcc}", "\\end{tabular}", table_start)
    current = ""
    rows: dict[str, list[dict]] = {"RXJ 1133": [], "Abell 1201": []}
    for line in table.splitlines():
        if "&" not in line or "\\pm" not in line:
            continue
        parts = [part.strip() for part in line.split("&")]
        first = parts[0]
        if first in {"MACS 1206", "RXJ 1133", "Abell 1201", "Abell 383", "Abell 963"}:
            current = first
        if current not in rows:
            continue
        spatial = parts[1]
        sigma_match = re.search(r"(\d+)\\pm(\d+)", parts[2])
        bounds = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", spatial)]
        if not sigma_match or len(bounds) != 2:
            continue
        rows[current].append(
            {
                "spatial_bin_arcsec": spatial,
                "sigma_km_s": int(sigma_match.group(1)),
                "sigma_error_km_s": int(sigma_match.group(2)),
            }
        )
    if len(rows["RXJ 1133"]) != 3 or len(rows["Abell 1201"]) != 8:
        raise ValueError(f"unexpected Sand dynamics counts: {rows}")
    return rows


def _parse_sand_arcs(text: str) -> dict[str, dict]:
    label_at = text.index("Gravitational Arc Properties")
    table = _section(text, "\\begin{tabular}{lcccc}", "\\end{tabular}", label_at)
    output = {}
    for system in ("RXJ 1133", "Abell 1201"):
        match = re.search(rf"^{re.escape(system)}&(.*)$", table, re.MULTILINE)
        if not match:
            raise ValueError(f"missing Sand arc row for {system}")
        cells = [cell.strip() for cell in match.group(1).split("&")]
        radius_values = []
        for cell in cells[:2]:
            found = re.search(r"(\d+(?:\.\d+)?)\\pm(\d+(?:\.\d+)?)", cell)
            if found:
                radius_values.append(
                    {"radius_arcsec": float(found.group(1)), "error_arcsec": float(found.group(2))}
                )
        output[system] = {
            "critical_radii": radius_values,
            "critical_radius_constraints": len(radius_values),
        }
    return output


def build_audit(image_output: Path, ledger_output: Path, report_output: Path) -> dict:
    sand_text = SAND_SOURCE.read_text(encoding="utf-8")
    j0416_text = J0416_SOURCE.read_text(encoding="utf-8")
    images = _parse_j0416_images(j0416_text)
    dynamics = _parse_sand_dynamics(sand_text)
    arcs = _parse_sand_arcs(sand_text)

    ledger = [
        {
            "system": "MACS J0416",
            "redshift": 0.396,
            "residual_blind_disturbed_flag": True,
            "disturbance_note": "Bergamini et al. 2023 describes the highly elongated geometry as typical of merging clusters.",
            "dynamics_source": "Bergamini et al. 2023, arXiv:2208.14020",
            "resolved_bcg_dynamics_bins": 0,
            "other_stellar_kinematic_measurements": 64,
            "dynamics_note": "64 inner stellar dispersions are separate cluster-member aperture measurements, not a radial BCG profile.",
            "lens_source": "Bergamini et al. 2023, arXiv:2208.14020",
            "lens_observable_kind": "spectroscopic multiple-image positions",
            "observable_lens_positions": len(images),
            "critical_radius_constraints": 0,
            "structural_radial_rank_upper_bound": 0,
            "structural_promotion_pass": False,
            "exclusion_reason": "no resolved BCG radial dynamics; independently merger-like",
        }
    ]
    for system, redshift in (("RXJ 1133", 0.394), ("Abell 1201", 0.169)):
        lens_constraints = arcs[system]["critical_radius_constraints"]
        ledger.append(
            {
                "system": system,
                "redshift": redshift,
                "residual_blind_disturbed_flag": False,
                "disturbance_note": "Sand et al. 2004 selected the sample to have no obvious strong ellipticity or bimodality.",
                "dynamics_source": "Sand et al. 2004, arXiv:astro-ph/0309465, Table 5",
                "resolved_bcg_dynamics_bins": len(dynamics[system]),
                "other_stellar_kinematic_measurements": 0,
                "dynamics_note": "numerical slit-bin dispersions and one-sigma errors are published",
                "lens_source": "Sand et al. 2004, arXiv:astro-ph/0309465, Table 3",
                "lens_observable_kind": "visually inferred one-dimensional critical radii",
                "observable_lens_positions": 0,
                "critical_radius_constraints": lens_constraints,
                "structural_radial_rank_upper_bound": lens_constraints,
                "structural_promotion_pass": False,
                "exclusion_reason": "no image-level lens positions and fewer than three radial lens constraints",
            }
        )

    ledger_frame = pd.DataFrame(ledger)
    for path in (image_output, ledger_output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    images.to_csv(image_output, index=False)
    ledger_frame.to_csv(ledger_output, index=False)
    report = {
        "audit_version": "R1A2-replacement-cycle1-extensions-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Observable coverage and disturbance state only; no gravity residual or model preference was inspected.",
        "summary": {
            "new_unique_hosts_source_screened": 3,
            "cumulative_unique_hosts_source_screened": 16,
            "remaining_hosts_to_30_target": 14,
            "new_structural_promotions": 0,
            "cumulative_non_disturbed_structural_promotions": 1,
            "remaining_non_disturbed_promotions": 1,
            "named_sample_exhausted": "Sand et al. 2004 six-cluster resolved-BCG sample",
        },
        "decision": {
            "cycle_1_complete": True,
            "cycle_1_progress_outcome": "The Sand six-cluster sample is exhausted and the numerical-publication shortfall for the one promotion was tested; start a new independent sample without lowering the gate.",
            "MACS_J0416": "exclude: lens-rich but lacks resolved BCG dynamics and is merger-like",
            "RXJ_1133": "exclude: three dynamics bins but only two critical-radius constraints and no image-position likelihood",
            "Abell_1201": "exclude: eight slit measurements but only one critical-radius constraint and no image-position likelihood",
            "next_action": "Start cycle 2 with an independent resolved-BCG/ICL kinematics sample and cross-match it against modern HST/JWST spectroscopic strong-lens catalogs.",
        },
        "outputs": {
            "macs_j0416_images": _display_path(image_output),
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
        default=ROOT / "data/derived/r1_replacement_cycle1_extension_images.csv",
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        default=ROOT / "data/derived/r1_replacement_cycle1_extension_ledger.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results/r1_replacement_search_cycle1_extensions/report.json",
    )
    args = parser.parse_args()
    print(json.dumps(build_audit(args.image_output, args.ledger_output, args.report_output), indent=2))


if __name__ == "__main__":
    main()
