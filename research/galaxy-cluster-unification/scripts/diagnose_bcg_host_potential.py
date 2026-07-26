from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import KPC_M
from voidscreen.unified import A0_M_S2, C_M_S, load_clash_acceleration_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post hoc scale check for the BCG host-potential completion of U0."
    )
    parser.add_argument(
        "--external-report",
        type=Path,
        default=ROOT / "results" / "external_bcg" / "report.json",
    )
    parser.add_argument(
        "--bcg-predictions",
        type=Path,
        default=ROOT / "results" / "external_bcg" / "bcg_predictions.csv",
    )
    parser.add_argument(
        "--clash",
        type=Path,
        default=ROOT / "data" / "raw" / "clash_tian2020" / "fig2.dat",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "bcg_host_potential_diagnostic"
    )
    args = parser.parse_args()

    external = json.loads(args.external_report.read_text(encoding="utf-8"))
    parameters = external["development_fit"]["parameters"]
    bcg = pd.read_csv(args.bcg_predictions)
    bcg = bcg[bcg["model"] == "U0_emond_like"].copy()
    gbar = np.power(10.0, bcg["log_gbar"].to_numpy())
    gobs = np.power(10.0, bcg["log_gobs"].to_numpy())
    newtonian_consistent = gobs > gbar
    inferred_a_eff = np.full(len(bcg), np.nan)
    inferred_a_eff[newtonian_consistent] = gbar[newtonian_consistent] / (
        -np.log1p(-gbar[newtonian_consistent] / gobs[newtonian_consistent])
    ) ** 2
    activation = np.log(inferred_a_eff / A0_M_S2) / np.log(parameters["F"])
    feasible = newtonian_consistent & (activation > 0.0) & (activation < 1.0)
    required_chi = np.full(len(bcg), np.nan)
    required_chi[feasible] = parameters["chi_t"] * np.power(
        10.0,
        parameters["w_dex"]
        * np.log(activation[feasible] / (1.0 - activation[feasible])),
    )
    required_host_chi = np.maximum(required_chi - bcg["chi"].to_numpy(), 0.0)
    required_speed = np.sqrt(required_host_chi * C_M_S**2) / 1000.0
    bcg["inferred_a_eff_m_s2"] = inferred_a_eff
    bcg["required_U0_activation"] = activation
    bcg["required_chi"] = required_chi
    bcg["required_host_delta_chi"] = required_host_chi
    bcg["required_host_potential_speed_km_s"] = required_speed
    bcg["frozen_U0_can_match_with_finite_host_potential"] = feasible

    clash = load_clash_acceleration_frame(args.clash)
    central = clash.sort_values("radius_kpc").groupby("system", as_index=False).first()
    local_tail_chi = (
        central["gbar_m_s2"].to_numpy()
        * central["radius_kpc"].to_numpy()
        * KPC_M
        / C_M_S**2
    )
    central["extended_baryon_chi"] = np.maximum(
        central["chi"].to_numpy() - local_tail_chi, 0.0
    )
    central["extended_baryon_potential_speed_km_s"] = np.sqrt(
        central["extended_baryon_chi"] * C_M_S**2
    ) / 1000.0

    valid_speed = bcg.loc[feasible, "required_host_potential_speed_km_s"].to_numpy()
    clash_speed = central["extended_baryon_potential_speed_km_s"].to_numpy()
    clash_10, clash_90 = np.quantile(clash_speed, [0.1, 0.9])
    report = {
        "status": "post hoc physical-scale diagnostic; not a fitted model",
        "frozen_U0_parameters": parameters,
        "bcgs": len(bcg),
        "finite_U0_host_solution": int(feasible.sum()),
        "required_host_delta_chi_median": float(np.nanmedian(required_host_chi)),
        "required_host_potential_speed_km_s": {
            "median": float(np.median(valid_speed)),
            "p10": float(np.quantile(valid_speed, 0.1)),
            "p90": float(np.quantile(valid_speed, 0.9)),
        },
        "CLASH_extended_baryon_potential_speed_km_s": {
            "clusters": len(central),
            "median": float(np.median(clash_speed)),
            "p10": float(clash_10),
            "p90": float(clash_90),
        },
        "fraction_of_finite_BCG_requirements_within_CLASH_p10_p90": float(
            np.mean((valid_speed >= clash_10) & (valid_speed <= clash_90))
        ),
        "interpretation": (
            "The inferred host term is a scale check only. It was solved from each BCG's "
            "observed acceleration and is not a prediction. A valid next test needs independently "
            "measured host gas+stellar potential profiles for the same MaNGA BCGs."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    bcg.to_csv(args.output / "bcg_required_host_potential.csv", index=False)
    central.to_csv(args.output / "clash_extended_baryon_potential.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    bins = np.linspace(0.0, max(valid_speed.max(), clash_speed.max()) * 1.05, 18)
    axis.hist(valid_speed, bins=bins, alpha=0.65, label="host potential required by BCG dynamics")
    axis.hist(clash_speed, bins=bins, alpha=0.65, label="extended baryonic potential in CLASH")
    axis.set(
        xlabel=r"equivalent potential speed $\sqrt{|\Phi|}$ (km/s)",
        ylabel="systems",
        title="Is the missing external potential astrophysically plausible?",
    )
    axis.legend(fontsize=8)
    figure.savefig(args.output / "host_potential_scale_check.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
