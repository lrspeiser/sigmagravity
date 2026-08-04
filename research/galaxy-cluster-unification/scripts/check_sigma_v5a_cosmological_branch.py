from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_causal_polarization import (
    signed_trace_bandpass,
    transition_bandpass_y_derivative,
)
from voidscreen.sigma_nonmetricity import (
    trace_action_derivative,
    trace_action_primitive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the v5A FLRW action domain.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v5a_cosmological_branch_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v5a_cosmological_branch_audit",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    probe = config["perturbation_probe"]
    positive = np.asarray(probe["positive_Y_values"], dtype=float)
    derivative = trace_action_derivative(positive)
    derivative_growth = float(abs(derivative[-1] / derivative[0]))

    negative_is_real = True
    negative_error = None
    try:
        value = trace_action_primitive(np.asarray([probe["negative_Y_probe"]]))
        negative_is_real = bool(np.all(np.isfinite(value)))
    except ValueError as error:
        negative_is_real = False
        negative_error = str(error)

    signed = np.linspace(-2.0, 2.0, 10001)
    source = signed_trace_bandpass(signed)
    evenness = float(np.max(np.abs(source - signed_trace_bandpass(-signed))))
    derivative_at_zero = float(abs(transition_bandpass_y_derivative(0.0)))
    gates_config = config["gates"]
    gates = {
        "v5a_real_open_background_domain": negative_is_real,
        "v5a_finite_background_derivative": derivative_growth
        <= float(gates_config["maximum_background_derivative_growth_ratio"]),
        "v5b_signed_source_evenness": evenness
        <= float(gates_config["maximum_v5b_signed_source_evenness_error"]),
        "v5b_source_smooth_at_flrw": derivative_at_zero
        <= float(gates_config["maximum_v5b_source_derivative_at_zero"]),
    }
    report = {
        "status": "completed Sigma v5A cosmological branch audit",
        "observational_data_accessed": False,
        "flrw_contractions": config["flrw_coincident_gauge"],
        "checks": {
            "positive_Y_action_derivatives": derivative.tolist(),
            "derivative_growth_smallest_over_largest_probe": derivative_growth,
            "negative_Y_primitive_is_real": negative_is_real,
            "negative_Y_error": negative_error,
            "v5b_signed_source_evenness_error": evenness,
            "v5b_source_derivative_at_zero": derivative_at_zero,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_v5a_gates_pass": bool(all(gates.values())),
        "decision": "retire_exact_v5a_base_select_v5b_stegr_polarization",
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
