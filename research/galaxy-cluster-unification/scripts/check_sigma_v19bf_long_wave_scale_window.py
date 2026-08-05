#!/usr/bin/env python3
"""Derive a target-blind scale window for the admitted long-wave premise."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bf_long_wave_scale_window.json"

AU_M = 149_597_870_700.0
PARSEC_M = 3.0856775814913673e16
SPEED_OF_LIGHT_M_S = 299_792_458.0
JULIAN_YEAR_S = 31_557_600.0
AU_TO_KPC = AU_M / (1_000.0 * PARSEC_M)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def low_pass_activation(radius_kpc: float, length_kpc: float) -> float:
    """Return 1-(1+x)exp(-x), using a stable series when x is tiny."""

    if radius_kpc < 0.0 or length_kpc <= 0.0:
        raise ValueError("radius must be nonnegative and length must be positive")
    x = radius_kpc / length_kpc
    if x < 1.0e-3:
        return (
            0.5 * x**2
            - x**3 / 3.0
            + x**4 / 8.0
            - x**5 / 30.0
            + x**6 / 144.0
        )
    return 1.0 - (1.0 + x) * math.exp(-x)


def literal_tidal_factor(radius_kpc: float, length_kpc: float) -> float:
    return (radius_kpc / (2.0 * math.pi * length_kpc)) ** 2


def x_at_activation(target: float) -> float:
    """Invert the monotone low-pass activation for 0 < target < 1."""

    if not 0.0 < target < 1.0:
        raise ValueError("activation target must lie strictly between zero and one")
    lower = 0.0
    upper = 1.0
    while low_pass_activation(upper, 1.0) < target:
        upper *= 2.0
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        if low_pass_activation(midpoint, 1.0) < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def baseline_kpc(item: dict[str, Any]) -> float:
    value = float(item["value"])
    unit = item["unit"]
    if unit == "au":
        return value * AU_TO_KPC
    if unit == "pc":
        return value / 1_000.0
    if unit == "kpc":
        return value
    raise ValueError(f"unsupported baseline unit: {unit}")


def verify_parent_hashes(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Path]]:
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"parent hash mismatch for {name}: {actual} != {spec['sha256']}")
        hashes[name] = actual
        paths[name] = path
    return hashes, paths


def constraint_bounds(config: dict[str, Any]) -> dict[str, Any]:
    c = config["preselection_constraints"]
    solar_kpc = float(c["planetary_baseline_au"]) * AU_TO_KPC

    lower_bounds = {
        "planetary_low_pass": solar_kpc
        / x_at_activation(float(c["maximum_planetary_low_pass_activation"])),
        "planetary_literal_tidal": solar_kpc
        / (2.0 * math.pi * math.sqrt(float(c["maximum_planetary_literal_tidal_factor"]))),
        "inner_galaxy_maximum": float(c["inner_galaxy_radius_kpc"])
        / x_at_activation(float(c["maximum_inner_galaxy_activation"])),
        "transition_maximum": float(c["transition_radius_kpc"])
        / x_at_activation(float(c["maximum_transition_activation"])),
    }
    upper_bounds = {
        "transition_minimum": float(c["transition_radius_kpc"])
        / x_at_activation(float(c["minimum_transition_activation"])),
        "outer_galaxy_minimum": float(c["outer_galaxy_radius_kpc"])
        / x_at_activation(float(c["minimum_outer_galaxy_activation"])),
    }
    effective_lower = max(lower_bounds.values())
    effective_upper = min(upper_bounds.values())
    return {
        "lower_bounds_kpc": lower_bounds,
        "upper_bounds_kpc": upper_bounds,
        "effective_lower_kpc": effective_lower,
        "effective_upper_kpc": effective_upper,
        "nonempty": effective_lower <= effective_upper,
        "active_lower_constraint": max(lower_bounds, key=lower_bounds.get),
        "active_upper_constraint": min(upper_bounds, key=upper_bounds.get),
    }


def logarithmic_grid(minimum: float, maximum: float, count: int) -> list[float]:
    if minimum <= 0.0 or maximum <= minimum or count < 2:
        raise ValueError("invalid logarithmic grid")
    lo = math.log10(minimum)
    hi = math.log10(maximum)
    return [10.0 ** (lo + (hi - lo) * index / (count - 1)) for index in range(count)]


def representative_diagnostics(config: dict[str, Any], length_kpc: float) -> list[dict[str, Any]]:
    amplitudes = [float(value) for value in config["scale_model"]["unselected_amplitude_controls"]]
    rows: list[dict[str, Any]] = []
    for item in config["reported_baselines"]:
        radius = baseline_kpc(item)
        activation = low_pass_activation(radius, length_kpc)
        rows.append(
            {
                "name": item["name"],
                "radius_kpc": radius,
                "low_pass_activation": activation,
                "literal_tidal_factor": literal_tidal_factor(radius, length_kpc),
                "amplitude_control_products": {
                    f"A={amplitude:g}": amplitude * activation for amplitude in amplitudes
                },
            }
        )
    return rows


def write_grid(config: dict[str, Any], output: Path, bounds: dict[str, Any]) -> None:
    spec = config["scale_model"]["correlation_length_grid_kpc"]
    lengths = logarithmic_grid(float(spec["minimum"]), float(spec["maximum"]), int(spec["count"]))
    c = config["preselection_constraints"]
    solar = float(c["planetary_baseline_au"]) * AU_TO_KPC
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "length_kpc",
            "wavelength_kpc",
            "planetary_low_pass_activation",
            "planetary_literal_tidal_factor",
            "inner_1kpc_activation",
            "transition_10kpc_activation",
            "outer_30kpc_activation",
            "cluster_100kpc_activation",
            "inside_derived_window",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for length in lengths:
            writer.writerow(
                {
                    "length_kpc": f"{length:.12g}",
                    "wavelength_kpc": f"{2.0 * math.pi * length:.12g}",
                    "planetary_low_pass_activation": f"{low_pass_activation(solar, length):.12g}",
                    "planetary_literal_tidal_factor": f"{literal_tidal_factor(solar, length):.12g}",
                    "inner_1kpc_activation": f"{low_pass_activation(1.0, length):.12g}",
                    "transition_10kpc_activation": f"{low_pass_activation(10.0, length):.12g}",
                    "outer_30kpc_activation": f"{low_pass_activation(30.0, length):.12g}",
                    "cluster_100kpc_activation": f"{low_pass_activation(100.0, length):.12g}",
                    "inside_derived_window": str(
                        bounds["effective_lower_kpc"] <= length <= bounds["effective_upper_kpc"]
                    ).lower(),
                }
            )


def svg_path(length_kpc: float, x_min: float, x_max: float, width: float, height: float) -> str:
    points: list[str] = []
    for index in range(240):
        fraction = index / 239.0
        radius = 10.0 ** (math.log10(x_min) + fraction * math.log10(x_max / x_min))
        activation = low_pass_activation(radius, length_kpc)
        x = 72.0 + fraction * width
        y = 24.0 + (1.0 - activation) * height
        points.append(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_svg(output: Path, bounds: dict[str, Any]) -> None:
    plot_w = 710.0
    plot_h = 270.0
    x_min = 100.0 * AU_TO_KPC
    x_max = 300.0
    lower = float(bounds["effective_lower_kpc"])
    upper = float(bounds["effective_upper_kpc"])
    middle = math.sqrt(lower * upper)
    curves = [
        (lower, "#2563eb", "lower edge"),
        (middle, "#7c3aed", "geometric midpoint"),
        (upper, "#dc2626", "upper edge"),
    ]
    grid_lines: list[str] = []
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = 24.0 + (1.0 - value) * plot_h
        grid_lines.append(
            f'<line x1="72" y1="{y:.2f}" x2="782" y2="{y:.2f}" stroke="#d1d5db" stroke-width="1"/>'
            f'<text x="62" y="{y + 4:.2f}" text-anchor="end" font-size="12">{value:.2g}</text>'
        )
    ticks: list[str] = []
    for radius, label in ((x_min, "100 AU"), (0.001, "1 pc"), (1.0, "1 kpc"), (10.0, "10 kpc"), (100.0, "100 kpc")):
        fraction = math.log10(radius / x_min) / math.log10(x_max / x_min)
        x = 72.0 + fraction * plot_w
        ticks.append(
            f'<line x1="{x:.2f}" y1="294" x2="{x:.2f}" y2="300" stroke="#111827"/>'
            f'<text x="{x:.2f}" y="318" text-anchor="middle" font-size="12">{html.escape(label)}</text>'
        )
    paths = "".join(
        f'<path d="{svg_path(length, x_min, x_max, plot_w, plot_h)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        for length, color, _ in curves
    )
    legend = "".join(
        f'<line x1="{96 + index * 225}" y1="344" x2="{122 + index * 225}" y2="344" stroke="{color}" stroke-width="3"/>'
        f'<text x="128" y="348" transform="translate({index * 225},0)" font-size="12">{html.escape(label)} L={length:.3f} kpc</text>'
        for index, (length, color, label) in enumerate(curves)
    )
    document = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="370" viewBox="0 0 820 370" '
        'role="img" aria-labelledby="title description">'
        '<title id="title">Long-wave activation across physical baselines</title>'
        '<desc id="description">The exact low-pass activation for the lower, middle, and upper edges of the derived universal correlation-length interval.</desc>'
        '<rect width="820" height="370" fill="#ffffff"/>'
        '<text x="20" y="18" font-family="sans-serif" font-size="14" font-weight="600">Dimensionless long-wave activation</text>'
        f'<g font-family="sans-serif" fill="#111827">{"".join(grid_lines)}{"".join(ticks)}{paths}{legend}'
        '<line x1="72" y1="24" x2="72" y2="294" stroke="#111827"/>'
        '<line x1="72" y1="294" x2="782" y2="294" stroke="#111827"/>'
        '<text x="427" y="334" text-anchor="middle" font-size="12">physical baseline (log scale)</text>'
        '<text x="14" y="160" text-anchor="middle" font-size="12" transform="rotate(-90,14,160)">fraction of asymptotic response</text>'
        '</g></svg>\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runner = (ROOT / config["implementation"]["runner"]).resolve()
    if runner != Path(__file__).resolve():
        raise RuntimeError("frozen runner path does not identify this implementation")
    runner_hash = sha256(runner)
    if runner_hash != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen runner hash mismatch")
    parent_hashes, parent_paths = verify_parent_hashes(config)
    v19be = json.loads(parent_paths["v19be_report"].read_text(encoding="utf-8"))
    bounds = constraint_bounds(config)
    lower = float(bounds["effective_lower_kpc"])
    upper = float(bounds["effective_upper_kpc"])
    control = float(config["scale_model"]["illustrative_control_length_kpc"])
    c = config["preselection_constraints"]
    solar = float(c["planetary_baseline_au"]) * AU_TO_KPC

    endpoint_checks = {
        "planetary_low_pass_at_lower": low_pass_activation(solar, lower),
        "planetary_literal_tidal_at_lower": literal_tidal_factor(solar, lower),
        "inner_activation_at_lower": low_pass_activation(float(c["inner_galaxy_radius_kpc"]), lower),
        "transition_activation_at_lower": low_pass_activation(float(c["transition_radius_kpc"]), lower),
        "transition_activation_at_upper": low_pass_activation(float(c["transition_radius_kpc"]), upper),
        "outer_activation_at_upper": low_pass_activation(float(c["outer_galaxy_radius_kpc"]), upper),
    }
    authorization = config["authorization"]
    gates = {
        "all_parent_hashes_exact": True,
        "v19be_passed_without_action_selection": (
            v19be["decision"] == "passed_action_admission_requirements"
            and not v19be["theory_state"]["covariant_action_selected"]
            and not v19be["theory_state"]["universal_constants_selected"]
        ),
        "nonempty_universal_length_window": bool(bounds["nonempty"]),
        "illustrative_control_inside_window": lower <= control <= upper,
        "planetary_constraints_pass_at_entire_window": (
            endpoint_checks["planetary_low_pass_at_lower"]
            <= float(c["maximum_planetary_low_pass_activation"]) * (1.0 + 1.0e-9)
            and endpoint_checks["planetary_literal_tidal_at_lower"]
            <= float(c["maximum_planetary_literal_tidal_factor"]) * (1.0 + 1.0e-9)
        ),
        "inner_transition_outer_constraints_pass_at_entire_window": (
            endpoint_checks["inner_activation_at_lower"]
            <= float(c["maximum_inner_galaxy_activation"]) * (1.0 + 1.0e-9)
            and endpoint_checks["transition_activation_at_lower"]
            <= float(c["maximum_transition_activation"]) * (1.0 + 1.0e-9)
            and endpoint_checks["transition_activation_at_upper"]
            >= float(c["minimum_transition_activation"]) * (1.0 - 1.0e-9)
            and endpoint_checks["outer_activation_at_upper"]
            >= float(c["minimum_outer_galaxy_activation"]) * (1.0 - 1.0e-9)
        ),
        "no_action_constant_or_payload_selected": (
            not authorization["read_v19w_or_v19x_gas_result"]
            and not authorization["select_candidate_action"]
            and not authorization["select_universal_length_or_amplitude"]
            and not authorization["read_lensing_or_halo_payload"]
            and not authorization["open_holdout"]
            and not authorization["change_gravity_physics"]
        ),
    }
    gates = {name: bool(value) for name, value in gates.items()}
    required = config["required_gates"]
    if set(gates) != set(required) or not all(required.values()):
        raise RuntimeError("frozen required-gate schema changed")

    grid_path = ROOT / config["outputs"]["grid"]
    plot_path = ROOT / config["outputs"]["plot"]
    write_grid(config, grid_path, bounds)
    write_svg(plot_path, bounds)

    wavelength_bounds = {
        "lower_kpc": 2.0 * math.pi * lower,
        "upper_kpc": 2.0 * math.pi * upper,
    }
    crossing_time_bounds = {
        "lower_years": wavelength_bounds["lower_kpc"] * 1_000.0 * PARSEC_M / SPEED_OF_LIGHT_M_S / JULIAN_YEAR_S,
        "upper_years": wavelength_bounds["upper_kpc"] * 1_000.0 * PARSEC_M / SPEED_OF_LIGHT_M_S / JULIAN_YEAR_S,
    }
    decision = "passed_dimensionless_scale_window" if all(gates.values()) else "failed_closed"
    report = {
        "protocol_version": config["protocol_version"],
        "decision": decision,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "implementation": {
            "runner": config["implementation"]["runner"],
            "runner_sha256": runner_hash,
        },
        "input_hashes": parent_hashes,
        "derived_correlation_length_window_kpc": {
            "lower": lower,
            "upper": upper,
            "width_ratio": upper / lower,
            "active_lower_constraint": bounds["active_lower_constraint"],
            "active_upper_constraint": bounds["active_upper_constraint"],
        },
        "derived_literal_wavelength_window_kpc": wavelength_bounds,
        "literal_wave_crossing_time_window_years": crossing_time_bounds,
        "all_constraint_bounds": bounds,
        "endpoint_checks": endpoint_checks,
        "illustrative_control": {
            "correlation_length_kpc": control,
            "literal_wavelength_kpc": 2.0 * math.pi * control,
            "literal_wave_crossing_time_years": 2.0 * math.pi * control * 1_000.0 * PARSEC_M / SPEED_OF_LIGHT_M_S / JULIAN_YEAR_S,
            "baselines": representative_diagnostics(config, control),
            "selected_as_constant": False,
        },
        "gate_results": gates,
        "outputs": {
            "grid": config["outputs"]["grid"],
            "grid_sha256": sha256(grid_path),
            "plot": config["outputs"]["plot"],
            "plot_sha256": sha256(plot_path),
        },
        "theory_state": {
            "physical_postulate_recorded": True,
            "dimensionless_scale_window_derived": True,
            "covariant_action_selected": False,
            "euler_lagrange_equations_derived": False,
            "weak_field_metric_derived": False,
            "universal_constants_selected": False,
            "gas_source_state_available": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision == "failed_closed":
        raise RuntimeError(f"V19BF failed closed: {gates}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
