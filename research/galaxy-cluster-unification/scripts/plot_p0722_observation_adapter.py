#!/usr/bin/env python3
"""Render the deterministic P0722 DDO101 curve comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0722_massive_tracer_observation_adapter"


def main() -> None:
    values = np.genfromtxt(RESULTS / "ddo101_curve.csv", delimiter=",", names=True)
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.errorbar(
        values["radius_kpc"],
        values["observed_speed_km_s"],
        yerr=values["uncertainty_km_s"],
        fmt="o",
        color="#172a3a",
        capsize=3,
        label="Published DDO101 circular speed",
    )
    axis.plot(
        values["radius_kpc"],
        values["api_newtonian_speed_km_s"],
        "-o",
        color="#c44e52",
        label="Generic API Newtonian field",
    )
    axis.plot(
        values["radius_kpc"],
        values["frozen_newtonian_speed_km_s"],
        "--",
        color="#4c72b0",
        label="Earlier frozen Newtonian fixture",
    )
    axis.set(
        xlabel="Radius (kpc)",
        ylabel="Circular speed (km/s)",
        title="DDO101: observation adapter commissioning",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.text(
        0.01,
        0.01,
        "25x25x9 commissioning grid; massive tracers only; no photon lensing",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(RESULTS / "ddo101_curve.png", dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


if __name__ == "__main__":
    main()
