"""Extract the frozen RX J2129 image-position likelihood inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/r1_rxj2129_lens_protocol.json"


def _resolve(path: str) -> Path:
    return ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_table(path: Path, center_ra: float, center_dec: float) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    label = text.index(r"\label{mul_rxj2129}")
    start = text.index(r"\begin{tabular}", label)
    end = text.index(r"\end{tabular}", start)
    pattern = re.compile(
        r"^\s*([1-8]\.[1-4])\s*&\s*"
        r"([0-9.]+)\s*&\s*([+-]?[0-9.]+)\s*&\s*"
        r"([^&]+?)\s*&\s*([^&]+?)\s*&\s*([0-9.]+)\s*\\\\"
    )
    rows: list[dict[str, Any]] = []
    cosine = float(np.cos(np.deg2rad(center_dec)))
    for raw_line in text[start:end].splitlines():
        if raw_line.lstrip().startswith("%"):
            continue
        match = pattern.match(raw_line)
        if match is None:
            continue
        image_id, ra_text, dec_text, zspec_text, zmodel_text, _published_rms = (
            match.groups()
        )
        family = int(image_id.split(".", 1)[0])
        ra = float(ra_text)
        dec = float(dec_text)
        cleaned_redshift = re.sub(r"[^0-9.]", "", zspec_text)
        redshift = float(cleaned_redshift)
        spectroscopic = family != 2
        east = (ra - center_ra) * cosine * 3600.0
        north = (dec - center_dec) * 3600.0
        radius = float(np.hypot(east, north))
        rows.append(
            {
                "system": "RX J2129",
                "image_id": image_id,
                "source_family": family,
                "ra_deg": ra,
                "dec_deg": dec,
                "delta_ra_east_arcsec": east,
                "delta_dec_north_arcsec": north,
                "radius_arcsec": radius,
                "source_redshift": redshift,
                "redshift_kind": (
                    "spectroscopic" if spectroscopic else "fixed_photometric"
                ),
                "likelihood_included": spectroscopic,
                "exclusion_reason": (
                    ""
                    if spectroscopic
                    else "photometric-redshift off-center galaxy-galaxy lens system"
                ),
                "image_position_sigma_arcsec": 0.5,
                "inside_dynamics_support": radius <= 5.0,
                "source_table": "Jauzac2021 mul_rxj2129",
                "published_model_rms_ingested": False,
            }
        )
    frame = pd.DataFrame(rows)
    frame["image_sort"] = frame["image_id"].map(
        lambda value: 10 * int(value.split(".")[0]) + int(value.split(".")[1])
    )
    return frame.sort_values("image_sort").drop(columns="image_sort").reset_index(drop=True)


def _covariance(frame: pd.DataFrame, variance: float) -> pd.DataFrame:
    selected = frame[frame["likelihood_included"]].reset_index(drop=True)
    labels = [
        f"{image_id}_{axis}"
        for image_id in selected["image_id"]
        for axis in ("east", "north")
    ]
    matrix = np.eye(len(labels), dtype=float) * variance
    output = pd.DataFrame(matrix, columns=labels)
    output.insert(0, "row", labels)
    return output


def _plot(frame: pd.DataFrame, path: Path, support_radius: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    excluded = frame[~frame["likelihood_included"]]
    if len(excluded):
        ax.scatter(
            excluded["delta_ra_east_arcsec"],
            excluded["delta_dec_north_arcsec"],
            marker="x",
            color="0.55",
            label="excluded photometric system 2",
        )
    selected = frame[frame["likelihood_included"]]
    for family, group in selected.groupby("source_family"):
        ax.scatter(
            group["delta_ra_east_arcsec"],
            group["delta_dec_north_arcsec"],
            label=f"spectroscopic family {family}",
        )
        for row in group.itertuples():
            ax.annotate(
                row.image_id,
                (row.delta_ra_east_arcsec, row.delta_dec_north_arcsec),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.add_patch(
        plt.Circle((0.0, 0.0), support_radius, fill=False, color="black", linestyle="--")
    )
    ax.scatter([0], [0], marker="+", s=100, color="black", label="BCG/dynamics center")
    ax.set(
        xlabel="east offset from BCG (arcsec)",
        ylabel="north offset from BCG (arcsec)",
        title="RX J2129 frozen image-position observables",
        aspect="equal",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def extract(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    if authorization["independent_lens_model_fit"]:
        raise ValueError("observable protocol cannot authorize a lens-model fit")
    if authorization["gravity_response_fit"] or authorization["published_gr_mass_map_use"]:
        raise ValueError("observable protocol cannot read a gravity result or GR mass map")
    source_path = _resolve(config["source"]["paper_source"])
    archive_path = _resolve(config["source"]["source_archive"])
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != config["source"]["source_archive_sha256"]:
        raise ValueError("Jauzac source archive SHA-256 mismatch")
    system = config["system"]
    frame = _parse_table(
        source_path, system["center_ra_deg"], system["center_dec_deg"]
    )
    frame["inside_dynamics_support"] = (
        frame["radius_arcsec"] <= system["dynamics_support_radius_arcsec"]
    )
    covariance = _covariance(
        frame, config["covariance"]["variance_arcsec2_per_coordinate"]
    )
    selected = frame[frame["likelihood_included"]]
    inner = selected[selected["inside_dynamics_support"]]
    numeric_covariance = covariance.drop(columns="row").to_numpy()
    eigenvalues = np.linalg.eigvalsh(numeric_covariance)
    observed = {
        "listed_images": int(len(frame)),
        "spectroscopic_likelihood_images": int(len(selected)),
        "spectroscopic_source_families": int(selected["source_family"].nunique()),
        "excluded_photometric_images": int((~frame["likelihood_included"]).sum()),
        "strict_images_inside_dynamics_support": int(len(inner)),
        "strict_inner_source_families": int(inner["source_family"].nunique()),
    }
    thresholds = config["advance_thresholds"]
    checks = {
        key: observed[key] == thresholds[key]
        for key in observed
    }
    checks.update(
        {
            "all_likelihood_coordinates_finite": bool(
                np.isfinite(
                    selected[["delta_ra_east_arcsec", "delta_dec_north_arcsec"]]
                ).all().all()
            ),
            "coordinate_covariance_symmetric_positive_semidefinite": bool(
                np.allclose(numeric_covariance, numeric_covariance.T)
                and eigenvalues.min() >= -1e-12
            ),
            "published_model_residual_used": False,
        }
    )
    gate_pass = all(
        bool(checks[key])
        for key in thresholds
        if key != "published_model_residual_used"
    ) and (
        checks["published_model_residual_used"]
        == thresholds["published_model_residual_used"]
    )
    image_path = _resolve(config["outputs"]["image_ledger"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(image_path, index=False)
    covariance_path = _resolve(config["outputs"]["coordinate_covariance"])
    covariance.to_csv(covariance_path, index=False)
    _plot(
        frame,
        _resolve(config["outputs"]["diagnostic"]),
        system["dynamics_support_radius_arcsec"],
    )
    report = {
        "protocol_version": config["protocol_version"],
        "status": (
            "observable_likelihood_inputs_complete_model_nuisance_protocol_pending"
            if gate_pass
            else "observable_likelihood_input_gate_failed"
        ),
        "source_archive_sha256_verified": True,
        "gravity_or_published_gr_mass_map_read": False,
        "independent_lens_residual_evaluated": False,
        "observed_counts": observed,
        "inner_likelihood_images": inner[
            ["image_id", "source_family", "radius_arcsec", "source_redshift"]
        ].to_dict(orient="records"),
        "coordinate_covariance_shape": list(numeric_covariance.shape),
        "coordinate_covariance_eigenvalue_range_arcsec2": [
            float(eigenvalues.min()),
            float(eigenvalues.max()),
        ],
        "checks": checks,
        "observable_likelihood_gate_pass": gate_pass,
        "strict_r1_ready": False,
        "outputs": config["outputs"],
        "next_action": (
            "Freeze the independent lens generative model and nuisance priors for the "
            "21 spectroscopic images. Marginalize seven source positions, the smooth "
            "cluster halo, BCG mass, member scaling, gas, and ICL; do not import the "
            "published GR convergence map or tune after inspecting residuals."
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
    print(json.dumps(extract(args.config), indent=2))


if __name__ == "__main__":
    main()
