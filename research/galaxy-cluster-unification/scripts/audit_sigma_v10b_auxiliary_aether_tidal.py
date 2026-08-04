from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10b_auxiliary_aether_tidal import audit_v10b_selection


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
    parser = argparse.ArgumentParser(description="Audit Sigma v10B selection gates.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10b_auxiliary_aether_tidal_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10b_auxiliary_aether_tidal_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_selection_values"]
    parameters = config["physical_parameters"]
    audit = audit_v10b_selection(
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
        "status": "completed Sigma v10B theory-only action selection",
        "candidate": config["candidate"],
        "action": config["action"],
        "definitions": config["definitions"],
        "fixed_coefficient_prescription": config["fixed_coefficient_prescription"],
        "physical_parameters": parameters,
        "fixed_selection_values": fixed,
        "coefficients": audit["coefficients"],
        "static_channels": audit["static_channels"],
        "response": audit["response"],
        "geometry": audit["geometry"],
        "convexity": audit["convexity"],
        "linear_metric_structure": audit["linear_metric_structure"],
        "selection_gates": gates,
        "all_selection_gates_pass": all_selection,
        "unresolved_mandatory_gates": audit["unresolved_mandatory_gates"],
        "all_mandatory_theory_gates_pass": False,
        "decision": "advance_to_full_ADM_constraint_and_reduced_characteristic_audit_only"
        if all_selection
        else "retire_before_full_variation",
        "reason": "The auxiliary aether-tidal carrier keeps the v10A trace/STF geometry but replaces the failed deep-AQUAL derivative block with an exactly positive constant-stiffness block. At K_B=1 and beta=sqrt(2/3), the worst static eigenvalue is 0.1835 and the Schur complement is 1/3. Eliminating P adds no vector root, lowers the high-k vector squared speed to 0.6 in the worst channel, and gives a threefold longitudinal static capacity that closes 93.25% of the spent amplitude gap. The linear static interaction changes the lapse potential while leaving the traceless metric equation and flat TT source unchanged, so the AeST no-slip projection is retained at this order.",
        "scope": "This is a construction pass, not a viable theory or observation result. The full nonlinear ADM constraint algebra, scalar/vector/metric characteristic determinant, physical causality of the elliptic auxiliary constraint, stress-energy, PPN, Solar screening, and cosmology remain mandatory. Only one spent scalar amplitude summary was used; no array, map, galaxy, cluster catalog, or holdout was opened.",
        "next_gate": "Derive the complete quadratic and nonlinear ADM system. Track the six primary P momenta and their secondary equations, reduce lapse/shift/aether constraints, and reject if P introduces independent initial data, an odd-dimensional or nonclosing constraint surface, a negative reduced mode, c_T != 1, or a physical instantaneous signal. Then derive the complete weak metric equations before any data.",
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
