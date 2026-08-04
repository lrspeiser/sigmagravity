from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10c_hyperbolic_aether_tidal import audit_v10c_selection


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Sigma v10C selection gates.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10c_hyperbolic_aether_tidal_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10c_hyperbolic_aether_tidal_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_selection_values"]
    parameters = config["physical_parameters"]
    audit = audit_v10c_selection(
        maximum_sourced_base_speed_squared=float(
            fixed["maximum_sourced_base_speed_squared"]
        ),
        static_mixing_fraction=float(fixed["static_mixing_fraction"]),
        k_b=float(fixed["K_B"]),
        existing_cluster_amplification_target=float(
            fixed["existing_spent_cluster_amplification_target"]
        ),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
    )
    gates = dict(audit["selection_gates"])
    gates["one_minimally_coupled_matter_metric"] = True
    gates["no_object_labels"] = True
    gates["no_object_specific_gravity_parameters"] = (
        int(parameters["object_specific_gravity_parameters"]) == 0
    )
    gates["no_lensing_only_parameters"] = int(parameters["lensing_only_parameters"]) == 0
    all_selection = bool(all(gates.values()))
    report = {
        "status": "completed Sigma v10C hyperbolic action selection",
        "candidate": config["candidate"],
        "action": config["action"],
        "definitions": config["definitions"],
        "derived_coefficient_protocol": config["derived_coefficients"],
        "physical_parameters": parameters,
        "fixed_selection_values": fixed,
        "coefficients": audit["coefficients"],
        "static_channels": audit["static_channels"],
        "hyperbolic_channels": audit["hyperbolic_channels"],
        "cone_margins": audit["cone_margins"],
        "response": audit["response"],
        "geometry": audit["geometry"],
        "convexity": audit["convexity"],
        "retarded_source_structure": audit["retarded_source_structure"],
        "linear_metric_structure": audit["linear_metric_structure"],
        "selection_gates": gates,
        "all_selection_gates_pass": all_selection,
        "unresolved_mandatory_gates": audit["unresolved_mandatory_gates"],
        "all_mandatory_theory_gates_pass": False,
        "decision": "advance_to_full_covariant_variation_ADM_and_background_cones_only"
        if all_selection
        else "retire_before_full_variation",
        "reason": "The derived coefficients c_P^2=3/11 and beta^2/K_B=2/11 preserve the v10B threefold static capacity while closing the worst sourced flat cone exactly at c^2=1. The second longitudinal root is 9/44, transverse roots are below one, unmixed carrier modes have c^2=3/11, and the physical TT mode remains unsourced and luminal. The hyperbolic time kinetic term replaces v10B's equal-time elliptic tail with a retarded finite-front equation. Static convexity and zero-boundary data uniquely fix an object's stationary carrier state.",
        "scope": "This is a theory-only selection pass, not a viable field theory or observational result. Full covariant equations, nonlinear ADM constraints, tilted/inhomogeneous characteristics, stress-energy, PPN, Solar screening, FLRW stability, and numerical convergence remain mandatory. No observational array, map, object catalog, or holdout was opened.",
        "next_gate": "Vary the complete action and construct the nonlinear ADM kinetic/constraint system. Evaluate the full reduced principal symbol on tilted aether, static Hessian, separated-source, and FLRW backgrounds. Reject exact v10C for any rank change, negative mode, superluminal cone, c_T shift, or nonunique no-incoming branch before data.",
        "prior_art_scope": config["prior_art_scope"],
        "novelty_claim": False,
        "new_observational_product_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
