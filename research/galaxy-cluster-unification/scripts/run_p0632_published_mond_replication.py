#!/usr/bin/env python3
"""Reproduce Li et al. (2018) and run strict simulator MOND controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.data import load_curves  # noqa: E402
from voidscreen.galaxy_replica import (  # noqa: E402
    load_replica_seed,
    render_observed_replica,
    render_replica,
    valid_rotation_mask,
)
from voidscreen.mond_benchmark import (  # noqa: E402
    catalog_curve,
    parse_li2018_table,
    precision_mask,
    published_fit_curve,
    reduced_chi_square,
)


LAW_LABELS = {
    "baryons": "Newtonian baryons",
    "li2018_rar_mond": "Li 2018 RAR/MOND",
    "simple_mond": "simple μ MOND",
    "standard_mond": "standard μ MOND",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_P0632_full_catalog_score":
        raise RuntimeError("P0632 protocol is not frozen")
    return protocol


def _metric(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "rms_about_zero": float(np.sqrt(np.mean(np.square(array)))),
        "median_absolute": float(np.median(np.abs(array))),
    }


def plot_replica_comparison(seed, mond_evaluation, published_evaluation, output: Path) -> None:
    mask = mond_evaluation["valid"]
    observed = render_observed_replica(seed, pixels=501)
    mond = render_replica(
        seed,
        mond_evaluation["radius_kpc"][mask],
        mond_evaluation["predicted_velocity_km_s"][mask],
        pixels=501,
    )
    coordinate = observed.x_kpc[0]
    extent = [coordinate[0], coordinate[-1], coordinate[0], coordinate[-1]]
    vmax = max(
        float(np.nanmax(np.abs(observed.line_of_sight_velocity_km_s))),
        float(np.nanmax(np.abs(mond.line_of_sight_velocity_km_s))),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    first = axes[0, 0].imshow(
        observed.line_of_sight_velocity_km_s,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=norm,
    )
    axes[0, 0].set_title("Replica from observed rotation")
    axes[0, 1].imshow(
        mond.line_of_sight_velocity_km_s,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=norm,
    )
    axes[0, 1].set_title("Blind fixed RAR/MOND prediction")
    figure.colorbar(first, ax=axes[0, :], label="line-of-sight km/s", shrink=0.85)
    for axis in axes[0, :]:
        axis.set_xlabel("x [kpc]")
        axis.set_ylabel("y [kpc]")

    curve_mask = valid_rotation_mask(seed)
    radius = seed.rotation.radius_kpc[curve_mask]
    axes[1, 0].errorbar(
        radius,
        seed.rotation.velocity_observed_kms[curve_mask],
        yerr=seed.rotation.velocity_error_kms[curve_mask],
        fmt="o",
        ms=3,
        color="black",
        label="SPARC",
    )
    axes[1, 0].plot(
        mond_evaluation["radius_kpc"][mask],
        mond_evaluation["predicted_velocity_km_s"][mask],
        color="#d95f02",
        lw=2,
        label="fixed RAR/MOND",
    )
    published_mask = published_evaluation["valid"]
    axes[1, 0].plot(
        published_evaluation["radius_kpc"][published_mask],
        published_evaluation["predicted_velocity_km_s"][published_mask],
        color="#1b9e77",
        lw=1.7,
        ls="--",
        label="published per-galaxy refit",
    )
    axes[1, 0].set(
        xlabel="radius [kpc]", ylabel="circular speed [km/s]", title="Rotation curve"
    )
    axes[1, 0].legend(fontsize=8)

    residual = (
        mond_evaluation["predicted_velocity_km_s"][mask]
        - mond_evaluation["observed_velocity_km_s"][mask]
    )
    axes[1, 1].axhline(0.0, color="0.5", lw=1)
    axes[1, 1].plot(
        mond_evaluation["radius_kpc"][mask], residual, "o-", ms=3, color="#7570b3"
    )
    axes[1, 1].set(
        xlabel="radius [kpc]",
        ylabel="predicted − observed [km/s]",
        title=f"Strict RMSE = {np.sqrt(np.mean(np.square(residual))):.2f} km/s",
    )
    figure.suptitle(
        f"{seed.name}: observation replica versus fixed published RAR/MOND",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "p0632_published_mond_replication_protocol.json",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    outputs = protocol["outputs"]
    output = ROOT / outputs["directory"]
    representative_output = output / "representatives"
    output.mkdir(parents=True, exist_ok=True)
    representative_output.mkdir(parents=True, exist_ok=True)

    inputs = protocol["inputs"]
    fits = parse_li2018_table(ROOT / inputs["li2018_source_table"])
    curves = load_curves(ROOT / inputs["sparc_directory"])
    curve_map = {curve.metadata.name: curve for curve in curves}
    if set(fits) != set(curve_map):
        raise RuntimeError("Published fit table and SPARC snapshot names differ")
    ledger = json.loads((ROOT / inputs["p0630_split_ledger"]).read_text(encoding="utf-8"))
    split_by_name = {}
    for split, names in ledger["systems"]["galaxy"].items():
        for name in names:
            split_by_name[name] = split

    parameter_rows = []
    published_system_rows = []
    published_point_rows = []
    strict_point_rows = []
    strict_system_rows = []
    laws = list(protocol["laws"])
    sample = protocol["sample_reproduction"]

    for curve in curves:
        name = curve.metadata.name
        fit = fits[name]
        parameter_rows.append(
            {
                "sparc_id": fit.sparc_id,
                "galaxy": name,
                "log_luminosity_lsun": fit.log_luminosity_lsun,
                "disk_mass_to_light": fit.disk_mass_to_light,
                "disk_mass_to_light_error": fit.disk_mass_to_light_error,
                "bulge_mass_to_light": fit.bulge_mass_to_light,
                "bulge_mass_to_light_error": fit.bulge_mass_to_light_error,
                "distance_mpc": fit.distance_mpc,
                "distance_error_mpc": fit.distance_error_mpc,
                "distance_ratio": fit.distance_ratio,
                "inclination_deg": fit.inclination_deg,
                "inclination_error_deg": fit.inclination_error_deg,
                "inclination_ratio": fit.inclination_ratio,
                "published_reduced_chi_square": fit.reduced_chi_square,
            }
        )
        replay = published_fit_curve(curve, fit)
        fitted_parameters = 3 + int(fit.bulge_mass_to_light is not None)
        calculated_chi = reduced_chi_square(
            replay, fitted_parameters=fitted_parameters
        )
        quality_selected = (
            curve.metadata.quality <= int(sample["quality_max"])
            and curve.metadata.inclination_deg
            >= float(sample["minimum_inclination_deg"])
        )
        published_system_rows.append(
            {
                "galaxy": name,
                "quality_selected": quality_selected,
                "points": int(replay["valid"].sum()),
                "precision_points": int(precision_mask(replay).sum()),
                "published_reduced_chi_square": fit.reduced_chi_square,
                "calculated_reduced_chi_square": calculated_chi,
                "absolute_chi_square_difference": abs(
                    calculated_chi - fit.reduced_chi_square
                ),
            }
        )
        if quality_selected:
            selected = precision_mask(
                replay,
                fractional_error_max=float(
                    sample["maximum_fractional_velocity_error_strictly_less_than"]
                ),
            )
            for index in np.flatnonzero(selected):
                published_point_rows.append(
                    {
                        "galaxy": name,
                        "radius_kpc": replay["radius_kpc"][index],
                        "gbar_m_s2": replay["gbar_m_s2"][index],
                        "gobs_m_s2": replay["gobs_m_s2"][index],
                        "predicted_acceleration_m_s2": replay[
                            "predicted_acceleration_m_s2"
                        ][index],
                        "log_residual_dex": math.log10(
                            replay["gobs_m_s2"][index]
                            / replay["predicted_acceleration_m_s2"][index]
                        ),
                    }
                )

        if not quality_selected:
            continue
        for law in laws:
            evaluation = catalog_curve(curve, law)
            selected = precision_mask(
                evaluation,
                fractional_error_max=float(
                    sample["maximum_fractional_velocity_error_strictly_less_than"]
                ),
            )
            residual_velocity = (
                evaluation["predicted_velocity_km_s"][selected]
                - evaluation["observed_velocity_km_s"][selected]
            )
            residual_log = np.log10(
                evaluation["gobs_m_s2"][selected]
                / evaluation["predicted_acceleration_m_s2"][selected]
            )
            if selected.any():
                velocity_rmse = float(np.sqrt(np.mean(np.square(residual_velocity))))
                log_rmse = float(np.sqrt(np.mean(np.square(residual_log))))
                mean_log_residual = float(np.mean(residual_log))
            else:
                velocity_rmse = math.nan
                log_rmse = math.nan
                mean_log_residual = math.nan
            strict_system_rows.append(
                {
                    "galaxy": name,
                    "split": split_by_name.get(name, "outside_P0630_split"),
                    "law": law,
                    "points": int(selected.sum()),
                    "velocity_RMSE_km_s": velocity_rmse,
                    "log_acceleration_RMSE_dex": log_rmse,
                    "mean_log_residual_dex": mean_log_residual,
                }
            )
            for index in np.flatnonzero(selected):
                strict_point_rows.append(
                    {
                        "galaxy": name,
                        "split": split_by_name.get(name, "outside_P0630_split"),
                        "law": law,
                        "radius_kpc": evaluation["radius_kpc"][index],
                        "observed_velocity_km_s": evaluation[
                            "observed_velocity_km_s"
                        ][index],
                        "predicted_velocity_km_s": evaluation[
                            "predicted_velocity_km_s"
                        ][index],
                        "gbar_m_s2": evaluation["gbar_m_s2"][index],
                        "gobs_m_s2": evaluation["gobs_m_s2"][index],
                        "predicted_acceleration_m_s2": evaluation[
                            "predicted_acceleration_m_s2"
                        ][index],
                        "log_residual_dex": math.log10(
                            evaluation["gobs_m_s2"][index]
                            / evaluation["predicted_acceleration_m_s2"][index]
                        ),
                    }
                )

    parameters = pd.DataFrame(parameter_rows).sort_values("sparc_id")
    published_systems = pd.DataFrame(published_system_rows).sort_values("galaxy")
    published_points = pd.DataFrame(published_point_rows)
    strict_points = pd.DataFrame(strict_point_rows)
    strict_systems = pd.DataFrame(strict_system_rows)
    parameters.to_csv(output / outputs["published_parameters"], index=False)
    published_systems.to_csv(
        output / outputs["published_replay_system_scores"], index=False
    )
    published_points.to_csv(output / outputs["published_replay_points"], index=False)
    strict_points.to_csv(output / outputs["strict_points"], index=False)
    strict_systems.to_csv(output / outputs["strict_system_scores"], index=False)

    replay_residual = published_points.log_residual_dex.to_numpy(float)
    calculated_chi = published_systems.calculated_reduced_chi_square.to_numpy(float)
    published_chi = published_systems.published_reduced_chi_square.to_numpy(float)
    paper = protocol["published_benchmark"]
    reproduction = {
        "published_table_entries": int(len(parameters)),
        "scatter_sample_galaxies": int(
            published_systems.quality_selected.astype(bool).sum()
        ),
        "scatter_points": int(len(published_points)),
        "replayed_scatter_dex": float(np.std(replay_residual)),
        "published_scatter_dex": float(paper["published_refit_scatter_dex"]),
        "absolute_scatter_difference_dex": abs(
            float(np.std(replay_residual))
            - float(paper["published_refit_scatter_dex"])
        ),
        "reduced_chi_square_correlation": float(
            np.corrcoef(calculated_chi, published_chi)[0, 1]
        ),
        "reduced_chi_square_median_absolute_difference": float(
            np.median(np.abs(calculated_chi - published_chi))
        ),
        "reduced_chi_square_RMSE": float(
            np.sqrt(np.mean(np.square(calculated_chi - published_chi)))
        ),
    }

    strict_summary = {}
    for law in laws:
        subset = strict_points.loc[strict_points.law.eq(law)]
        velocity_residual = (
            subset.predicted_velocity_km_s.to_numpy(float)
            - subset.observed_velocity_km_s.to_numpy(float)
        )
        per_system_rmse = strict_systems.loc[
            strict_systems.law.eq(law), "velocity_RMSE_km_s"
        ].dropna().to_numpy(float)
        strict_summary[law] = {
            "points": int(len(subset)),
            "log_acceleration_residual": _metric(subset.log_residual_dex),
            "point_weighted_velocity_RMSE_km_s": float(
                np.sqrt(np.mean(np.square(velocity_residual)))
            ),
            "equal_galaxy_velocity_RMSE_km_s": float(
                np.sqrt(np.mean(np.square(per_system_rmse)))
            ),
            "galaxies_with_precision_points": int(len(per_system_rmse)),
        }

    holdout_rows = []
    holdout_names = set(ledger["systems"]["galaxy"]["holdout"])
    for law in laws:
        system_mse = []
        system_log_mse = []
        points = 0
        for name in sorted(holdout_names):
            evaluation = catalog_curve(curve_map[name], law)
            valid = evaluation["valid"]
            velocity_residual = (
                evaluation["predicted_velocity_km_s"][valid]
                - evaluation["observed_velocity_km_s"][valid]
            )
            log_residual = np.log10(
                evaluation["gobs_m_s2"][valid]
                / evaluation["predicted_acceleration_m_s2"][valid]
            )
            system_mse.append(float(np.mean(np.square(velocity_residual))))
            system_log_mse.append(float(np.mean(np.square(log_residual))))
            points += int(valid.sum())
        holdout_rows.append(
            {
                "law": law,
                "galaxies": len(holdout_names),
                "points": points,
                "equal_galaxy_velocity_RMSE_km_s": float(np.sqrt(np.mean(system_mse))),
                "equal_galaxy_log_acceleration_RMSE_dex": float(
                    np.sqrt(np.mean(system_log_mse))
                ),
            }
        )
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(output / outputs["holdout_scores"], index=False)

    gates = protocol["replication_gates"]
    fixed_scatter = strict_summary["li2018_rar_mond"][
        "log_acceleration_residual"
    ]["standard_deviation"]
    gate_results = {
        "published_table_entries": reproduction["published_table_entries"]
        == int(gates["published_table_entries_exact"]),
        "scatter_sample_galaxies": reproduction["scatter_sample_galaxies"]
        == int(gates["scatter_sample_galaxies_exact"]),
        "scatter_points": reproduction["scatter_points"]
        == int(gates["scatter_points_exact"]),
        "refit_scatter": reproduction["absolute_scatter_difference_dex"]
        <= float(gates["absolute_refit_scatter_difference_dex_max"]),
        "per_galaxy_chi_correlation": reproduction[
            "reduced_chi_square_correlation"
        ]
        >= float(gates["published_reduced_chi2_correlation_min"]),
        "per_galaxy_chi_absolute_difference": reproduction[
            "reduced_chi_square_median_absolute_difference"
        ]
        <= float(gates["published_reduced_chi2_median_absolute_difference_max"]),
        "fixed_nuisance_scatter": abs(
            fixed_scatter - float(paper["published_fixed_nuisance_scatter_dex"])
        )
        <= float(gates["absolute_fixed_nuisance_scatter_difference_dex_max"]),
    }

    p0631_protocol = json.loads(
        (ROOT / inputs["p0631_replica_protocol"]).read_text(encoding="utf-8")
    )
    p0631_inputs = p0631_protocol["inputs"]
    representative_metrics = []
    for name in protocol["representative_galaxies"]:
        seed = load_replica_seed(
            name,
            ROOT / p0631_inputs["sparc_directory"],
            ROOT / p0631_inputs["photometric_profiles"],
            ROOT / p0631_inputs["bulge_disk_decompositions"],
        )
        evaluation = catalog_curve(curve_map[name], "li2018_rar_mond")
        published_evaluation = published_fit_curve(curve_map[name], fits[name])
        valid = evaluation["valid"]
        residual = (
            evaluation["predicted_velocity_km_s"][valid]
            - evaluation["observed_velocity_km_s"][valid]
        )
        representative_metrics.append(
            {
                "galaxy": name,
                "strict_velocity_RMSE_km_s": float(
                    np.sqrt(np.mean(np.square(residual)))
                ),
                "published_refit_velocity_RMSE_km_s": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                published_evaluation["predicted_velocity_km_s"][
                                    published_evaluation["valid"]
                                ]
                                - published_evaluation["observed_velocity_km_s"][
                                    published_evaluation["valid"]
                                ]
                            )
                        )
                    )
                ),
            }
        )
        plot_replica_comparison(
            seed,
            evaluation,
            published_evaluation,
            representative_output / f"{name}_mond_comparison.png",
        )

    report = {
        "protocol_id": protocol["protocol_id"],
        "published_replication_pass": bool(all(gate_results.values())),
        "gate_results": gate_results,
        "published_reproduction": reproduction,
        "strict_no_nuisance": strict_summary,
        "whole_galaxy_holdout": holdout.to_dict(orient="records"),
        "representatives": representative_metrics,
        "claim_boundary": {
            "matched": "Li et al. 2018 algebraic RAR/MOND circular-orbit benchmark and its SPARC nuisance transformations.",
            "not_matched": "A full AQUAL/QUMOND field solution, external-field effect, relativistic lensing theory, or galaxy formation history.",
            "per_galaxy_published_refit": "Publication replication only; it uses each galaxy's rotation data to fit mass-to-light ratio, distance, and inclination.",
            "strict_test": "Uses fixed M/L=(0.5,0.7), catalog distance/inclination, one published acceleration scale, and no per-galaxy fitted gravity or nuisance setting.",
        },
        "provenance": {
            "protocol_sha256": sha256(args.protocol),
            "li2018_table_sha256": sha256(ROOT / inputs["li2018_source_table"]),
            "li2018_provenance_sha256": sha256(ROOT / inputs["li2018_provenance"]),
            "p0630_split_ledger_sha256": sha256(ROOT / inputs["p0630_split_ledger"]),
        },
    }
    (output / outputs["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    fixed = strict_points.loc[strict_points.law.eq("li2018_rar_mond")]
    axes[0, 0].hexbin(
        np.log10(fixed.gbar_m_s2),
        np.log10(fixed.gobs_m_s2),
        gridsize=55,
        mincnt=1,
        cmap="viridis",
        norm=LogNorm(),
    )
    order = np.argsort(fixed.gbar_m_s2.to_numpy(float))
    axes[0, 0].plot(
        np.log10(fixed.gbar_m_s2.to_numpy(float)[order]),
        np.log10(fixed.predicted_acceleration_m_s2.to_numpy(float)[order]),
        color="#d95f02",
        lw=2,
        label="published equation",
    )
    axes[0, 0].set(
        xlabel=r"$\log_{10} g_{bar}$ [m/s²]",
        ylabel=r"$\log_{10} g_{obs}$ [m/s²]",
        title=f"Fixed inputs: scatter {fixed_scatter:.3f} dex",
    )
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(published_chi, calculated_chi, s=12, alpha=0.7)
    maximum = max(float(np.max(published_chi)), float(np.max(calculated_chi)))
    axes[0, 1].plot([0, maximum], [0, maximum], color="black", lw=1)
    axes[0, 1].set(
        xlabel="published reduced χ²",
        ylabel="simulator replay reduced χ²",
        title=f"175 galaxies: r={reproduction['reduced_chi_square_correlation']:.6f}",
    )

    labels = [LAW_LABELS[law] for law in laws]
    scatters = [
        strict_summary[law]["log_acceleration_residual"]["standard_deviation"]
        for law in laws
    ]
    axes[1, 0].barh(labels, scatters, color=["0.6", "#1b9e77", "#7570b3", "#e7298a"])
    axes[1, 0].axvline(0.13, color="#d95f02", ls="--", label="published fixed-input 0.13 dex")
    axes[1, 0].set(xlabel="SPARC log-acceleration scatter [dex]", title="No per-galaxy refit")
    axes[1, 0].legend(fontsize=8)

    holdout_values = dict(
        zip(holdout.law, holdout.equal_galaxy_velocity_RMSE_km_s, strict=True)
    )
    axes[1, 1].barh(
        labels,
        [holdout_values[law] for law in laws],
        color=["0.6", "#1b9e77", "#7570b3", "#e7298a"],
    )
    axes[1, 1].set(
        xlabel="equal-galaxy velocity RMSE [km/s]",
        title="23 whole-galaxy holdouts",
    )
    figure.suptitle("P0632 published MOND/RAR simulator replication", fontsize=14)
    figure.savefig(output / outputs["figure"], dpi=180)
    plt.close(figure)

    mond_holdout = holdout.loc[holdout.law.eq("li2018_rar_mond")].iloc[0]
    summary = [
        "# P0632 published MOND/RAR simulator replication",
        "",
        f"**Published benchmark replication: {'PASS' if report['published_replication_pass'] else 'FAIL'}**",
        "",
        f"- Recovered the published sample exactly: {reproduction['scatter_sample_galaxies']} galaxies and {reproduction['scatter_points']} points.",
        f"- Replayed Li et al. nuisance-fit scatter: {reproduction['replayed_scatter_dex']:.6f} dex versus 0.057 dex published.",
        f"- Recalculated versus published per-galaxy reduced chi-square correlation: {reproduction['reduced_chi_square_correlation']:.6f}.",
        f"- Fixed-input RAR/MOND scatter: {fixed_scatter:.6f} dex versus approximately 0.13 dex published.",
        f"- Strict 23-galaxy holdout velocity RMSE: {mond_holdout.equal_galaxy_velocity_RMSE_km_s:.3f} km/s.",
        "",
        "The first two comparisons reproduce the paper. The whole-galaxy holdout is our stricter simulator diagnostic: it fixes the published acceleration scale, stellar mass-to-light ratios, catalog distance, and catalog inclination, with no per-galaxy fit.",
        "",
        "This validates the algebraic circular-orbit MOND/RAR plugin. It does not validate a complete AQUAL/QUMOND field solver, external-field effect, relativistic MOND lensing theory, or galaxy formation simulation.",
    ]
    (output / outputs["summary"]).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
