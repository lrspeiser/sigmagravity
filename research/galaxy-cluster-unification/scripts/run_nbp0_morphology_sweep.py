from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from voidscreen.axisymmetric_permittivity import AxisymmetricGrid
from voidscreen.permittivity_morphology import (
    MorphologyParameters,
    solve_morphology_response,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray | pd.Series) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    labels = ["minimum", "p05", "median", "p95", "maximum"]
    return {
        label: float(value)
        for label, value in zip(
            labels, np.quantile(finite, [0.0, 0.05, 0.5, 0.95, 1.0]), strict=True
        )
    }


def log_uniform(unit_value: float, lower: float, upper: float) -> float:
    return float(10.0 ** (math.log10(lower) + unit_value * math.log10(upper / lower)))


def map_unit_parameters(unit: np.ndarray, *, include_bulge_fraction: bool) -> dict[str, float]:
    offset = 0
    values: dict[str, float] = {}
    if include_bulge_fraction:
        values["stellar_bulge_fraction"] = float(unit[0])
        offset = 1
    values.update(
        {
            "disk_vertical_scale_over_Rdisk": log_uniform(unit[offset], 0.0625, 0.8),
            "bulge_scale_over_Rdisk": log_uniform(unit[offset + 1], 0.05, 0.8),
            "gas_fraction": float(0.7 * unit[offset + 2]),
            "gas_radial_scale_over_Rdisk": log_uniform(unit[offset + 3], 1.5, 4.0),
            "gas_vertical_scale_over_Rdisk": log_uniform(unit[offset + 4], 0.0625, 0.3),
            "minimum_permittivity": log_uniform(unit[offset + 5], 0.03, 1.0),
            "log10_critical_density_dimensionless": float(-5.0 + 6.0 * unit[offset + 6]),
            "sharpness": log_uniform(unit[offset + 7], 0.5, 8.0),
            "smoothing_length_over_Rdisk": (
                0.0
                if unit[offset + 8] < 0.1
                else log_uniform((unit[offset + 8] - 0.1) / 0.9, 0.03, 3.0)
            ),
        }
    )
    return values


def build_cases(protocol: dict[str, object]) -> list[dict[str, object]]:
    synthetic = protocol["synthetic_sources"]
    baseline = dict(synthetic["baseline"])
    cases: list[dict[str, object]] = [
        {"case_family": "baseline", "case_id": "baseline", "parameters": baseline}
    ]
    for key, values in synthetic["one_at_a_time_values"].items():
        for index, value in enumerate(values):
            parameters = dict(baseline)
            parameters[key] = value
            cases.append(
                {
                    "case_family": "one_at_a_time",
                    "case_id": f"oat_{key}_{index:02d}",
                    "varied_parameter": key,
                    "parameters": parameters,
                }
            )

    factorial = synthetic["morphology_factorial"]
    factorial_keys = list(factorial)
    for index, combination in enumerate(
        itertools.product(*(factorial[key] for key in factorial_keys))
    ):
        parameters = dict(baseline)
        parameters.update(dict(zip(factorial_keys, combination, strict=True)))
        cases.append(
            {
                "case_family": "morphology_factorial",
                "case_id": f"factorial_{index:03d}",
                "parameters": parameters,
            }
        )

    seed = int(synthetic["sobol_seed"])
    sobol_count = int(synthetic["sobol_cases"])
    sobol = qmc.Sobol(d=10, scramble=True, seed=seed)
    sobol_units = sobol.random_base2(int(round(math.log2(sobol_count))))
    if len(sobol_units) != sobol_count:
        raise ValueError("sobol_cases must be a power of two")
    for index, unit in enumerate(sobol_units):
        cases.append(
            {
                "case_family": "sobol",
                "case_id": f"sobol_{index:04d}",
                "parameters": map_unit_parameters(unit, include_bulge_fraction=True),
            }
        )

    pair_count = int(synthetic["paired_environment_cases"])
    pair_sampler = qmc.Sobol(d=9, scramble=True, seed=seed + 1)
    pair_units = pair_sampler.random_base2(int(round(math.log2(pair_count))))
    if len(pair_units) != pair_count:
        raise ValueError("paired_environment_cases must be a power of two")
    for pair_index, unit in enumerate(pair_units):
        environment = map_unit_parameters(unit, include_bulge_fraction=False)
        for label, fraction in (("disk", 0.0), ("bulge", 1.0)):
            parameters = {"stellar_bulge_fraction": fraction, **environment}
            cases.append(
                {
                    "case_family": "paired_environment",
                    "case_id": f"pair_{pair_index:03d}_{label}",
                    "pair_id": pair_index,
                    "morphology_member": label,
                    "parameters": parameters,
                }
            )
    return cases


def flatten_response(case: dict[str, object], response: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "case_family": case["case_family"],
        "case_id": case["case_id"],
        "varied_parameter": case.get("varied_parameter", ""),
        "pair_id": case.get("pair_id", math.nan),
        "morphology_member": case.get("morphology_member", ""),
        **response["parameters"],
        **{f"mass_{key}": value for key, value in response["component_masses"].items()},
        "newtonian_speed_log_slope": response["newtonian_speed_log_slope"],
        "modified_speed_log_slope": response["modified_speed_log_slope"],
        "outer_speed_slope_change": response["outer_speed_slope_change"],
        "epsilon_minimum_realized": response["epsilon_minimum_realized"],
        "epsilon_maximum_realized": response["epsilon_maximum_realized"],
    }
    for radius, enhancement, geometry, epsilon in zip(
        response["response_radii_over_Rdisk"],
        response["midplane_acceleration_enhancement"],
        response["geometry_only_enhancement"],
        response["epsilon_midplane_at_response_radii"],
        strict=True,
    ):
        label = str(radius).replace(".", "p")
        row[f"enhancement_R{label}"] = enhancement
        row[f"geometry_enhancement_R{label}"] = geometry
        row[f"epsilon_midplane_R{label}"] = epsilon
    probe = response["above_plane_probe"]
    row.update(
        {
            "probe_R": probe["radius_over_Rdisk"],
            "probe_z": probe["height_over_Rdisk"],
            "probe_inward_radial_acceleration": probe["inward_radial_acceleration"],
            "probe_toward_plane_acceleration": probe["toward_plane_vertical_acceleration"],
            "probe_vertical_to_radial_ratio": probe["absolute_vertical_to_radial_ratio"],
            "probe_newtonian_vertical_to_radial_ratio": probe[
                "newtonian_absolute_vertical_to_radial_ratio"
            ],
            "probe_constitutive_direction_ratio_change": probe[
                "constitutive_direction_ratio_change"
            ],
            "probe_radial_acceleration_enhancement": probe[
                "radial_acceleration_enhancement"
            ],
            "probe_vertical_acceleration_enhancement": probe[
                "vertical_acceleration_enhancement"
            ],
        }
    )
    return row


def paired_summary(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    paired = frame.loc[frame["case_family"] == "paired_environment"].pivot(
        index="pair_id", columns="morphology_member", values=metric
    )
    difference = paired["disk"] - paired["bulge"]
    return {
        "metric": metric,
        "difference_is_disk_minus_bulge": True,
        "difference_quantiles": quantiles(difference),
        "fraction_disk_greater": float(np.mean(difference > 0.0)),
        "fraction_disk_at_least_one_percent_greater": float(
            np.mean(difference > 0.01 * np.abs(paired["bulge"]))
        ),
        "fraction_tied_within_one_percent": float(
            np.mean(np.abs(difference) <= 0.01 * np.abs(paired["bulge"]))
        ),
    }


def rank_correlations(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    parameter_names = list(MorphologyParameters.__dataclass_fields__)
    result: dict[str, float] = {}
    for parameter in parameter_names:
        correlation = spearmanr(
            frame[parameter].to_numpy(dtype=float),
            frame[metric].to_numpy(dtype=float),
            nan_policy="omit",
        ).statistic
        result[parameter] = float(correlation)
    return result


def convergence_cases(baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    def changed(**updates: float) -> dict[str, float]:
        result = dict(baseline)
        result.update(updates)
        return result

    return {
        "baseline": dict(baseline),
        "newtonian_limit": changed(minimum_permittivity=1.0),
        "thin_disk_sharp_transition": changed(
            stellar_bulge_fraction=0.0,
            disk_vertical_scale_over_Rdisk=0.0625,
            minimum_permittivity=0.03,
            sharpness=8.0,
            smoothing_length_over_Rdisk=0.0,
        ),
        "thick_disk": changed(
            stellar_bulge_fraction=0.0, disk_vertical_scale_over_Rdisk=0.8
        ),
        "compact_bulge": changed(
            stellar_bulge_fraction=1.0, bulge_scale_over_Rdisk=0.05
        ),
        "extended_bulge": changed(
            stellar_bulge_fraction=1.0, bulge_scale_over_Rdisk=0.8
        ),
        "long_smoothing": changed(smoothing_length_over_Rdisk=3.0),
    }


def run_convergence(
    protocol: dict[str, object], response_radii: np.ndarray
) -> dict[str, object]:
    solver = protocol["solver"]
    grids = {}
    for label in ("sweep_grid", "reference_grid"):
        settings = solver[label]
        grids[label] = AxisymmetricGrid(
            int(settings["radial_cells"]),
            int(settings["vertical_cells"]),
            float(settings["radial_max_Rdisk"]),
            float(settings["vertical_max_Rdisk"]),
        )
    rows = []
    baseline = protocol["synthetic_sources"]["baseline"]
    for case_id, values in convergence_cases(baseline).items():
        parameters = MorphologyParameters(**values)
        responses = {
            label: solve_morphology_response(
                grid,
                parameters,
                response_radii=response_radii,
                outer_slope_interval=tuple(
                    protocol["synthetic_sources"]["outer_slope_interval_over_Rdisk"]
                ),
            )
            for label, grid in grids.items()
        }
        for radius, sweep_value, reference_value in zip(
            response_radii,
            responses["sweep_grid"]["midplane_acceleration_enhancement"],
            responses["reference_grid"]["midplane_acceleration_enhancement"],
            strict=True,
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "radius_over_Rdisk": float(radius),
                    "sweep_enhancement": float(sweep_value),
                    "reference_enhancement": float(reference_value),
                    "relative_change": float(
                        abs(sweep_value - reference_value) / abs(reference_value)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    gates = solver["validation_gates"]
    median_change = float(frame["relative_change"].median())
    maximum_change = float(frame["relative_change"].max())
    return {
        "cases": rows,
        "median_relative_change": median_change,
        "maximum_relative_change": maximum_change,
        "median_gate": float(gates["selected_case_median_enhancement_resolution_change_max"]),
        "maximum_gate": float(gates["selected_case_maximum_enhancement_resolution_change_max"]),
        "passes_median_gate": bool(
            median_change
            <= gates["selected_case_median_enhancement_resolution_change_max"]
        ),
        "passes_maximum_gate": bool(
            maximum_change
            <= gates["selected_case_maximum_enhancement_resolution_change_max"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbp0_morphology_protocol.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "results" / "nbp0_morphology_sweep",
    )
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    synthetic = protocol["synthetic_sources"]
    grid_settings = protocol["solver"]["sweep_grid"]
    grid = AxisymmetricGrid(
        int(grid_settings["radial_cells"]),
        int(grid_settings["vertical_cells"]),
        float(grid_settings["radial_max_Rdisk"]),
        float(grid_settings["vertical_max_Rdisk"]),
    )
    response_radii = np.asarray(synthetic["response_radii_over_Rdisk"], dtype=float)
    slope_interval = tuple(synthetic["outer_slope_interval_over_Rdisk"])
    cases = build_cases(protocol)
    rows = []
    for index, case in enumerate(cases, start=1):
        response = solve_morphology_response(
            grid,
            MorphologyParameters(**case["parameters"]),
            response_radii=response_radii,
            outer_slope_interval=slope_interval,
        )
        rows.append(flatten_response(case, response))
        if index % 100 == 0 or index == len(cases):
            print(f"completed {index}/{len(cases)} synthetic cases", flush=True)
    frame = pd.DataFrame(rows)

    radius_labels = [str(value).replace(".", "p") for value in response_radii]
    paired_metrics = [f"enhancement_R{label}" for label in radius_labels]
    paired_metrics += [
        f"geometry_enhancement_R{label}" for label in radius_labels
    ]
    paired_metrics += [
        "outer_speed_slope_change",
        "probe_vertical_to_radial_ratio",
        "probe_newtonian_vertical_to_radial_ratio",
        "probe_constitutive_direction_ratio_change",
        "probe_radial_acceleration_enhancement",
        "probe_vertical_acceleration_enhancement",
    ]
    paired_results = {metric: paired_summary(frame, metric) for metric in paired_metrics}
    predicted_outer_metrics = ["enhancement_R4p0", "enhancement_R6p0", "enhancement_R8p0"]
    pair_wide = {
        metric: frame.loc[frame["case_family"] == "paired_environment"].pivot(
            index="pair_id", columns="morphology_member", values=metric
        )
        for metric in predicted_outer_metrics
    }
    all_outer_predicted = np.logical_and.reduce(
        [values["disk"] > values["bulge"] for values in pair_wide.values()]
    )
    synthetic_sign_fraction = float(np.mean(all_outer_predicted))

    sobol_frame = frame.loc[frame["case_family"] == "sobol"]
    factorial_frame = frame.loc[frame["case_family"] == "morphology_factorial"]
    oat_frame = frame.loc[frame["case_family"] == "one_at_a_time"]
    oat_ranges: dict[str, dict[str, dict[str, float]]] = {}
    for parameter, subset in oat_frame.groupby("varied_parameter"):
        oat_ranges[str(parameter)] = {
            metric: quantiles(subset[metric])
            for metric in predicted_outer_metrics + ["outer_speed_slope_change"]
        }

    factorial_group_signs = []
    for _, subset in factorial_frame.groupby(
        ["disk_vertical_scale_over_Rdisk", "bulge_scale_over_Rdisk"]
    ):
        for metric in predicted_outer_metrics:
            factorial_group_signs.append(
                {
                    "metric": metric,
                    "spearman_bulge_fraction": float(
                        spearmanr(subset["stellar_bulge_fraction"], subset[metric]).statistic
                    ),
                }
            )

    convergence = run_convergence(protocol, response_radii)
    gate = float(
        protocol["empirical_morphology_test"]["advance_gates"][
            "synthetic_parameter_fraction_with_predicted_sign_min"
        ]
    )
    report = {
        "report_version": "NBP0-M1-axisymmetric-morphology-sweep-0.1",
        "status": "completed predeclared synthetic morphology sweep",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "elapsed_seconds": float(time.monotonic() - started),
        "case_counts": {
            str(key): int(value)
            for key, value in frame["case_family"].value_counts().sort_index().items()
        },
        "total_cases": int(len(frame)),
        "response_distributions": {
            metric: quantiles(frame[metric])
            for metric in predicted_outer_metrics + ["outer_speed_slope_change"]
        },
        "paired_disk_minus_bulge": paired_results,
        "synthetic_predicted_sign": {
            "definition": "disk enhancement exceeds matched bulge enhancement at R/Rdisk=4, 6, and 8 simultaneously",
            "fraction": synthetic_sign_fraction,
            "required_fraction": gate,
            "passes_gate": bool(synthetic_sign_fraction >= gate),
        },
        "sobol_spearman_correlations": {
            metric: rank_correlations(sobol_frame, metric)
            for metric in predicted_outer_metrics + ["outer_speed_slope_change"]
        },
        "morphology_factorial": {
            "group_count": int(len(factorial_group_signs) / len(predicted_outer_metrics)),
            "fraction_negative_bulge_fraction_correlation": {
                metric: float(
                    np.mean(
                        [
                            row["spearman_bulge_fraction"] < 0.0
                            for row in factorial_group_signs
                            if row["metric"] == metric
                        ]
                    )
                )
                for metric in predicted_outer_metrics
            },
        },
        "one_at_a_time_response_ranges": oat_ranges,
        "resolution_convergence": convergence,
        "interpretation_limits": [
            "The sweep is dimensionless and tests the morphology response of the weak-field constitutive equations, not a calibrated physical density scale.",
            "A synthetic morphology sign does not establish a covariant completion or a lensing law.",
            "Parameters are varied globally; no observed galaxy rotation speed is used or fitted here.",
        ],
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_directory / "responses.csv", index=False)
    (args.output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
