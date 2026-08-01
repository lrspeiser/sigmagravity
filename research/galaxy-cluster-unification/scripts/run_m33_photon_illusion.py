#!/usr/bin/env python3
"""Compare M33's directly measured angular rotation with H I spectroscopy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KMS_PER_MASYR_KPC = 4.74047


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "m33_photon_illusion_protocol.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_M33_uncertainty_propagation":
        raise RuntimeError("protocol was not frozen before M33 scoring")
    published = protocol["published_inputs"]
    rng = np.random.default_rng(int(protocol["uncertainty"]["seed"]))
    draws = int(protocol["uncertainty"]["monte_carlo_draws"])
    proper_motion = rng.normal(
        float(published["relative_right_ascension_proper_motion_microarcsec_per_year"]),
        float(published["proper_motion_standard_error_microarcsec_per_year"]),
        draws,
    )
    distance = rng.normal(
        float(published["independent_TRGB_distance_kpc"]),
        float(published["TRGB_distance_standard_error_kpc"]),
        draws,
    )
    hi_velocity = rng.normal(
        float(published["HI_model_relative_right_ascension_velocity_km_s"]),
        float(published["HI_model_systematic_standard_error_km_s"]),
        draws,
    )
    astrometric_velocity = (
        KMS_PER_MASYR_KPC * (proper_motion / 1000.0) * distance
    )
    photon_excess = hi_velocity - astrometric_velocity
    fractional_excess = photon_excess / hi_velocity
    geometric_distance = hi_velocity / (
        KMS_PER_MASYR_KPC * (proper_motion / 1000.0)
    )

    central_astrometric = (
        KMS_PER_MASYR_KPC
        * (
            float(
                published[
                    "relative_right_ascension_proper_motion_microarcsec_per_year"
                ]
            )
            / 1000.0
        )
        * float(published["independent_TRGB_distance_kpc"])
    )
    targets = {
        f"probability_excess_at_least_{target:g}_km_s": float(
            np.mean(photon_excess >= float(target))
        )
        for target in protocol["comparison_targets_km_s"]
    }
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed M33 angular-vs-spectroscopic rotation test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "source": {
            **protocol["source"],
            "pdf_sha256": sha256(ROOT / protocol["source"]["local_pdf"]),
            "provenance_sha256": sha256(ROOT / protocol["source"]["provenance"]),
        },
        "published_inputs": published,
        "results": {
            "astrometric_velocity_central_km_s": central_astrometric,
            "astrometric_velocity_95_interval_km_s": interval(
                astrometric_velocity
            ),
            "HI_minus_astrometric_velocity_central_km_s": float(
                published["HI_model_relative_right_ascension_velocity_km_s"]
                - central_astrometric
            ),
            "HI_minus_astrometric_velocity_95_interval_km_s": interval(
                photon_excess
            ),
            "probability_spectroscopy_overstates_motion": float(
                np.mean(photon_excess > 0.0)
            ),
            "fractional_frequency_illusion_median": float(
                np.median(fractional_excess)
            ),
            "fractional_frequency_illusion_95_interval": interval(
                fractional_excess
            ),
            "implied_geometric_distance_median_kpc": float(
                np.median(geometric_distance)
            ),
            "implied_geometric_distance_95_interval_kpc": interval(
                geometric_distance
            ),
            **targets,
        },
        "interpretation": {
            "simple_frequency_illusion": "not supported; direct angular rotation agrees with the H I rotation model",
            "new_physics_branch": "A coordinated optical metric affecting Doppler frequency, angular astrometry, and electromagnetic distance indicators remains logically open but is more constrained and less minimal.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }

    quantiles = pd.DataFrame(
        [
            {
                "quantity": name,
                "p025": np.quantile(values, 0.025),
                "median": np.median(values),
                "p975": np.quantile(values, 0.975),
            }
            for name, values in {
                "astrometric_velocity_km_s": astrometric_velocity,
                "HI_minus_astrometric_velocity_km_s": photon_excess,
                "fractional_frequency_illusion": fractional_excess,
                "implied_geometric_distance_kpc": geometric_distance,
            }.items()
        ]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    quantiles.to_csv(ROOT / protocol["outputs"]["samples"], index=False)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(astrometric_velocity, bins=100, density=True, alpha=0.7)
    axes[0].axvline(
        float(published["HI_model_relative_right_ascension_velocity_km_s"]),
        color="#D95F02",
        label="H I model central value",
    )
    axes[0].set(
        xlabel="direct astrometric relative velocity (km/s)",
        ylabel="probability density",
        title="M33 rotation from angular motion",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(photon_excess, bins=100, density=True, alpha=0.7)
    axes[1].axvline(0.0, color="black", linewidth=1)
    for target in protocol["comparison_targets_km_s"]:
        axes[1].axvline(
            float(target),
            linestyle=":",
            label=f"{target:g} km/s target",
        )
    axes[1].set(
        xlabel="H I minus astrometric motion (km/s)",
        ylabel="probability density",
        title="Allowed photon-frequency excess",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(ROOT / protocol["outputs"]["figure"], dpi=190)
    plt.close(figure)

    excess_interval = report["results"][
        "HI_minus_astrometric_velocity_95_interval_km_s"
    ]
    lines = [
        "# M33 photon-illusion cross-check",
        "",
        f"Direct astrometric relative velocity: **{central_astrometric:.2f} km/s**.",
        "",
        f"H I model: **{published['HI_model_relative_right_ascension_velocity_km_s']:.1f} +/- "
        f"{published['HI_model_systematic_standard_error_km_s']:.1f} km/s**.",
        "",
        f"Allowed H I-minus-astrometric excess: **{report['results']['HI_minus_astrometric_velocity_central_km_s']:.2f} km/s** "
        f"(95% {excess_interval[0]:.2f} to {excess_interval[1]:.2f}).",
        "",
        f"Probability of an excess at least 61.4 km/s: **{targets['probability_excess_at_least_61.4_km_s']:.5f}**.",
        "",
        "The simple spectroscopic-illusion interpretation is not supported.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
