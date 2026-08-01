from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data/raw/r1_replacement_search_sources"
    / "bolamperti2024_sdss_j0100_group_lens/main.tex"
)
RXJ_SOURCE = (
    ROOT
    / "data/raw/r1_replacement_search_sources"
    / "jauzac2021_three_muse_cluster_lenses/muse_clusters.tex"
)
RXJ_CENTER = (322.41651, 0.08923)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _family(image_id: str) -> str:
    return image_id[0]


def _parse_images(text: str) -> pd.DataFrame:
    begin = text.index("\\label{tab:multiple_images}")
    table_begin = text.rfind("\\begin{tabular}", 0, begin)
    table_end = text.index("\\end{tabular}", table_begin)
    table = text[table_begin:table_end]
    pattern = re.compile(
        r"^\s*([ABCEF]\d)\s*&\s*([^&]+)\s*&\s*([^&]+)\s*&\s*"
        r"([0-9.]+)\s*&\s*([0-9.]+)\s*\\\\",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(table):
        image_id = match.group(1)
        family = _family(image_id)
        rows.append(
            {
                "system": "SDSS J0100+1818",
                "image_id": image_id,
                "family_id": family,
                "ra_hms": match.group(2).strip(),
                "dec_dms": match.group(3).strip(),
                "source_redshift": float(match.group(4)),
                "bcg_radius_arcsec": float(match.group(5)),
                "position_error_arcsec": 0.066 if family in {"A", "B", "C"} else 0.15,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 18:
        raise ValueError(f"expected 18 image rows, parsed {len(frame)}")
    frame["inside_dynamics_support"] = frame["bcg_radius_arcsec"] <= 3.0
    frame["spectroscopic_redshift"] = True
    return frame


def _parse_rxj2129_images(text: str) -> pd.DataFrame:
    begin = text.index("1.1 & 322.42038")
    end = text.index("\\end{tabular}", begin)
    table = text[begin:end]
    pattern = re.compile(
        r"^\s*([0-9]+\.[0-9]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*"
        r"([^&]+)\s*&",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(table):
        image_id, ra_text, dec_text, redshift_text = match.groups()
        ra = float(ra_text)
        dec = float(dec_text)
        spectroscopic = "ast" not in redshift_text
        redshift_match = re.search(r"[0-9]+(?:\.[0-9]+)?", redshift_text)
        if redshift_match is None:
            raise ValueError(f"missing RX J2129 redshift for {image_id}")
        radius = 3600 * math.hypot(
            (ra - RXJ_CENTER[0]) * math.cos(math.radians(RXJ_CENTER[1])),
            dec - RXJ_CENTER[1],
        )
        rows.append(
            {
                "system": "RX J2129",
                "image_id": image_id,
                "family_id": image_id.split(".")[0],
                "ra_deg": ra,
                "dec_deg": dec,
                "source_redshift": float(redshift_match.group()),
                "bcg_radius_arcsec": radius,
                "position_error_arcsec": 0.5,
                "spectroscopic_redshift": spectroscopic,
                "inside_dynamics_support": spectroscopic and radius <= 5.0,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 25 or frame["spectroscopic_redshift"].sum() != 21:
        raise ValueError("RX J2129 image table did not yield 25 rows and 21 spectra")
    return frame


def build_audit(image_output: Path, ledger_output: Path, report_output: Path) -> dict:
    text = SOURCE.read_text(encoding="utf-8")
    group_images = _parse_images(text)
    rxj_images = _parse_rxj2129_images(RXJ_SOURCE.read_text(encoding="utf-8"))
    images = pd.concat([group_images, rxj_images], ignore_index=True, sort=False)
    inner = group_images.loc[group_images["inside_dynamics_support"]]
    inner_families = set(inner["family_id"])
    associated = images.loc[images["family_id"].isin(inner_families)]
    family_dof = 2 * len(associated) - 2 * len(inner_families)
    rank_bound = len(inner)
    structural_pass = rank_bound >= 3 and family_dof >= 4

    rxj_inner = rxj_images.loc[rxj_images["inside_dynamics_support"]]
    rxj_inner_families = set(rxj_inner["family_id"])
    rxj_associated = rxj_images.loc[
        rxj_images["family_id"].isin(rxj_inner_families)
    ]
    rxj_family_dof = 2 * len(rxj_associated) - 2 * len(rxj_inner_families)
    rxj_rank_bound = len(rxj_inner)
    rxj_structural_pass = rxj_rank_bound >= 3 and rxj_family_dof >= 4

    ledger = pd.DataFrame(
        [
            {
                "system": "SDSS J0100+1818",
                "host_class": "brightest group galaxy / group-scale bridge",
                "redshift": 0.581,
                "disturbance_state": "unknown_not_countable_as_non_disturbed",
                "disturbance_note": "The source calls it a candidate fossil system but states that deep X-ray data are needed to confirm a dynamically evolved, undisturbed halo.",
                "dynamics_source": "Bolamperti et al. 2024, arXiv:2411.07289",
                "resolved_bgg_dynamics_bins": 6,
                "dynamics_support_arcsec": 3.0,
                "dynamics_support_kpc": 20.0,
                "dynamics_values_availability": "profile values and correlations are figure-only; text gives peak and outer summaries",
                "lens_source": "Bolamperti et al. 2024, arXiv:2411.07289",
                "spectroscopic_multiple_image_positions": len(group_images),
                "strict_inner_image_positions": len(inner),
                "inner_source_families": len(inner_families),
                "family_wide_positions": len(associated),
                "family_wide_position_dof_after_source_coordinates": family_dof,
                "structural_radial_rank_upper_bound": rank_bound,
                "structural_promotion_pass": structural_pass,
                "non_disturbed_structural_promotion": False,
                "full_r1_ready": False,
                "exclusion_reason": "only one spectroscopic image position overlaps the 3-arcsec kinematic support; disturbance state is also unconfirmed",
            },
            {
                "system": "RX J2129",
                "host_class": "relaxed cluster BCG / raw-MUSE reconstruction",
                "redshift": 0.234,
                "disturbance_state": "relaxed_with_reported_sloshing",
                "disturbance_note": "Jimenez-Teja et al. 2024 confirm a relaxed state from ICL and X-ray evidence while reporting residual sloshing from an old merger.",
                "dynamics_source": "project pPXF reconstruction from ESO ADP.2017-12-14T12:30:03.217 under resolution-corrected frozen protocols R1A2-RXJ2129-pPXF-0.2 and R1B0-RXJ2129-covariance-0.2",
                "resolved_bgg_dynamics_bins": 4,
                "dynamics_support_arcsec": 5.0,
                "dynamics_support_kpc": 18.71,
                "dynamics_values_availability": "four numerical bins and a positive-definite 4x4 covariance reconstructed from the public MUSE cube; all 100 XSL block bootstraps and the frozen resolution/mask grid pass",
                "lens_source": "Jauzac et al. 2021, arXiv:2006.10700",
                "spectroscopic_multiple_image_positions": int(rxj_images["spectroscopic_redshift"].sum()),
                "strict_inner_image_positions": len(rxj_inner),
                "inner_source_families": len(rxj_inner_families),
                "family_wide_positions": len(rxj_associated),
                "family_wide_position_dof_after_source_coordinates": rxj_family_dof,
                "structural_radial_rank_upper_bound": rxj_rank_bound,
                "structural_promotion_pass": rxj_structural_pass,
                "non_disturbed_structural_promotion": rxj_structural_pass,
                "full_r1_ready": False,
                "exclusion_reason": "not excluded structurally or kinematically; the published BCG baseline, empirical PSF, and nonparametric two-band light profile are reconstructed, but the BCG/ICL identifiability and stellar-mass mapping, radial gas, satellite, and rerunnable lens nuisance likelihoods remain required",
            },
        ]
    )
    for path in (image_output, ledger_output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    images.to_csv(image_output, index=False)
    ledger.to_csv(ledger_output, index=False)
    report = {
        "audit_version": "R1A2-replacement-cycle2-0.5",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Both Cycle 2 candidates were selected from same-object dynamics, lens coverage, disturbance evidence, and public-data availability before inspecting any gravity residual.",
        "summary": {
            "cycle_2_new_hosts_screened": 2,
            "cumulative_unique_hosts_source_screened": 18,
            "remaining_hosts_to_30_target": 12,
            "cycle_2_new_non_disturbed_promotions": 1,
            "cumulative_non_disturbed_structural_promotions": 2,
            "remaining_non_disturbed_promotions": 0,
        },
        "decision": {
            "SDSS_J0100_1818": "retain as a group-scale bridge and low-overlap control; do not promote",
            "RX_J2129": "promote structurally after the frozen four-bin public-MUSE reconstruction passes every baseline internal-consistency check",
            "reason": "RX J2129 supplies four covariance-complete dynamics bins to 5 arcsec and three spectroscopic inner images from three families, with 12 family-wide positional degrees of freedom. Its published Hernquist BCG baseline, empirical two-band PSF, and nonparametric HST light profile with joint covariance are reconstructed, while BCG/ICL identifiability and mass mapping, radial gas, satellite, and lens-nuisance likelihoods still prevent strict R1 readiness.",
            "cycle_2_status": "structural_promotion_threshold_met_inventory_in_progress",
            "next_action": "Continue the 30-host or hard-shortfall inventory while fitting the frozen one- versus two-component RX J2129 light models, acquiring a radial gas likelihood, and building rerunnable lens nuisance information without fitting gravity.",
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
        default=ROOT / "data/derived/r1_replacement_cycle2_image_support.csv",
    )
    parser.add_argument(
        "--ledger-output",
        type=Path,
        default=ROOT / "data/derived/r1_replacement_cycle2_candidate_ledger.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results/r1_replacement_search_cycle2/report.json",
    )
    args = parser.parse_args()
    print(json.dumps(build_audit(args.image_output, args.ledger_output, args.report_output), indent=2))


if __name__ == "__main__":
    main()
