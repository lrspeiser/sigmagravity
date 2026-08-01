"""Audit MUSE-to-Molino labels for an RX J2129 satellite likelihood."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_satellite_membership_protocol.json"


def _resolve(path: str) -> Path:
    return ROOT / path


def _read_molino(path: Path) -> pd.DataFrame:
    header = next(
        line[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# CLASHID")
    )
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=header.split(),
        low_memory=False,
    )


def _parse_muse_redshifts(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    label = text.index(r"\label{rxj2129_z}")
    start = text.index(r"\begin{supertabular}", label)
    end = text.index(r"\end{supertabular}", start)
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(.+?)\s*&\s*([0-9]{3}\.[0-9]+)\s*&\s*"
        r"([+-]?[0-9.]+)\s*&\s*([0-9.]+)\s*&\s*"
        r"([0-9.]+)\s*&\s*([23])\\\\"
    )
    for raw_line in text[start:end].splitlines():
        if raw_line.lstrip().startswith("%"):
            continue
        match = pattern.match(raw_line)
        if match is None:
            continue
        raw_id, ra, dec, redshift, error_1e4, quality = match.groups()
        clean_id = re.sub(r"[^A-Za-z0-9_]", "", raw_id.replace(r"\textbf", ""))
        rows.append(
            {
                "muse_id": clean_id,
                "muse_ra_deg": float(ra),
                "muse_dec_deg": float(dec),
                "muse_redshift": float(redshift),
                "muse_redshift_error_1e4": float(error_1e4),
                "muse_quality": int(quality),
            }
        )
    return pd.DataFrame(rows)


def _crossmatch(
    muse: pd.DataFrame, molino: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    muse_coordinates = SkyCoord(
        muse["muse_ra_deg"].to_numpy() * u.deg,
        muse["muse_dec_deg"].to_numpy() * u.deg,
    )
    molino_coordinates = SkyCoord(
        pd.to_numeric(molino["RA"], errors="coerce").to_numpy() * u.deg,
        pd.to_numeric(molino["Dec"], errors="coerce").to_numpy() * u.deg,
    )
    indices, separations, _ = muse_coordinates.match_to_catalog_sky(molino_coordinates)
    provisional = muse.copy()
    provisional["molino_index"] = indices
    provisional["match_separation_arcsec"] = separations.to_value(u.arcsec)
    provisional = provisional[
        provisional["match_separation_arcsec"]
        <= config["crossmatch"]["maximum_separation_arcsec"]
    ].copy()
    before_unique = len(provisional)
    provisional = (
        provisional.sort_values("match_separation_arcsec")
        .drop_duplicates("molino_index", keep="first")
        .reset_index(drop=True)
    )
    photo = molino.iloc[provisional["molino_index"].to_numpy()].reset_index(drop=True)
    matched = pd.concat([provisional.reset_index(drop=True), photo.add_prefix("molino_")], axis=1)
    if config["labels"]["exclude_molino_point_sources"]:
        matched = matched[
            pd.to_numeric(matched["molino_PointS"], errors="coerce") == 0
        ].copy()
    lower, upper = config["labels"]["cluster_member_redshift_interval_inclusive"]
    matched["is_cluster_member"] = matched["muse_redshift"].between(
        lower, upper, inclusive="both"
    )
    center = SkyCoord(
        config["crossmatch"]["center_ra_deg"] * u.deg,
        config["crossmatch"]["center_dec_deg"] * u.deg,
    )
    matched_coordinates = SkyCoord(
        matched["muse_ra_deg"].to_numpy() * u.deg,
        matched["muse_dec_deg"].to_numpy() * u.deg,
    )
    matched["radius_from_bcg_arcsec"] = center.separation(matched_coordinates).to_value(
        u.arcsec
    )
    matched["inside_30arcsec"] = (
        matched["radius_from_bcg_arcsec"]
        <= config["crossmatch"]["inner_diagnostic_radius_arcsec"]
    )
    matched = matched.sort_values("radius_from_bcg_arcsec").reset_index(drop=True)
    audit = {
        "provisional_matches": int(before_unique),
        "duplicate_molino_matches_removed": int(before_unique - len(provisional)),
        "unique_match_fraction": float(len(provisional) / max(before_unique, 1)),
    }
    return matched, audit


def _plot(matched: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, group in matched.groupby("is_cluster_member"):
        axes[0].scatter(
            group["molino_zb_1"],
            group["muse_redshift"],
            s=18,
            alpha=0.75,
            label="member" if label else "nonmember",
        )
        axes[1].hist(
            group["match_separation_arcsec"],
            bins=np.linspace(0, 0.5, 16),
            alpha=0.55,
            label="member" if label else "nonmember",
        )
    axes[0].set(xlabel="Molino BPZ first-peak z", ylabel="MUSE spectroscopic z")
    axes[1].set(xlabel="MUSE-Molino separation (arcsec)", ylabel="matches")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("RX J2129 MUSE-to-Molino satellite training audit")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    if authorization["membership_classifier_fit"] or authorization["satellite_mass_likelihood"]:
        raise ValueError("crossmatch audit cannot authorize a satellite likelihood")
    if authorization["lens_residual_read"] or authorization["gravity_response_fit"]:
        raise ValueError("crossmatch audit cannot read a residual")
    muse = _parse_muse_redshifts(_resolve(config["inputs"]["jauzac_source"]))
    expected_rows = config["labels"]["expected_jauzac_active_table_redshifts"]
    if len(muse) != expected_rows:
        raise ValueError(
            f"parsed {len(muse)} MUSE redshifts; expected "
            f"{expected_rows} active table rows"
        )
    molino = _read_molino(_resolve(config["inputs"]["molino_catalog"]))
    matched, crossmatch_audit = _crossmatch(muse, molino, config)
    inner = matched[matched["inside_30arcsec"]]
    metrics = {
        "parsed_muse_redshifts": int(len(muse)),
        "molino_catalog_rows": int(len(molino)),
        "unique_extended_matches": int(len(matched)),
        "member_matches": int(matched["is_cluster_member"].sum()),
        "nonmember_matches": int((~matched["is_cluster_member"]).sum()),
        "unique_match_fraction": crossmatch_audit["unique_match_fraction"],
        "labeled_matches_inside_30arcsec": int(len(inner)),
        "member_matches_inside_30arcsec": int(inner["is_cluster_member"].sum()),
    }
    thresholds = config["training_viability_thresholds"]
    checks = {
        "unique_extended_matches_minimum": metrics["unique_extended_matches"]
        >= thresholds["unique_extended_matches_minimum"],
        "member_matches_minimum": metrics["member_matches"]
        >= thresholds["member_matches_minimum"],
        "nonmember_matches_minimum": metrics["nonmember_matches"]
        >= thresholds["nonmember_matches_minimum"],
        "unique_match_fraction_minimum": metrics["unique_match_fraction"]
        >= thresholds["unique_match_fraction_minimum"],
        "labeled_matches_inside_30arcsec_minimum": metrics[
            "labeled_matches_inside_30arcsec"
        ]
        >= thresholds["labeled_matches_inside_30arcsec_minimum"],
        "member_matches_inside_30arcsec_minimum": metrics[
            "member_matches_inside_30arcsec"
        ]
        >= thresholds["member_matches_inside_30arcsec_minimum"],
    }
    gate_pass = all(checks.values())
    output_path = _resolve(config["outputs"]["matched_training_ledger"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_path, index=False)
    _plot(matched, _resolve(config["outputs"]["diagnostic"]))
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "spectroscopic_training_set_viable_classifier_protocol_pending"
            if gate_pass
            else "spectroscopic_training_set_viability_gate_failed"
        ),
        "gravity_or_lens_residual_read": False,
        "metrics": metrics,
        "source_count_audit": {
            "published_prose_count": config["labels"][
                "published_prose_redshift_count"
            ],
            "active_table_rows": int(len(muse)),
            "commented_source_rows": config["labels"]["commented_source_rows"],
            "interpretation": config["labels"]["source_count_inconsistency"],
        },
        "crossmatch_audit": crossmatch_audit,
        "checks": checks,
        "training_viability_gate_pass": gate_pass,
        "membership_classifier_fit": False,
        "strict_r1_ready": False,
        "outputs": config["outputs"],
        "next_action": (
            "Freeze a calibrated, spatially grouped membership-probability model with "
            "bootstrap uncertainty and held-out Brier/log-loss gates. Do not convert "
            "photo-z interval overlap directly into probability."
            if gate_pass
            else "Record the labeled-sample shortfall; do not invent normalized membership probabilities."
        ),
    }
    report_path = _resolve(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(audit(args.config), indent=2))


if __name__ == "__main__":
    main()
