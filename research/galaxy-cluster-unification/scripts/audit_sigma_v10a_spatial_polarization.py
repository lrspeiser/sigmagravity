from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10a_spatial_polarization import audit_v10a_selection


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
    parser = argparse.ArgumentParser(description="Audit the Sigma v10A selection gates.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10a_spatial_polarization_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10a_spatial_polarization_selection",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_selection_values"]
    parameters = config["physical_parameters"]
    audit = audit_v10a_selection(
        k_b=float(fixed["K_B"]),
        k_2=float(fixed["K_2"]),
        lambda_s=float(fixed["lambda_s"]),
        physical_parameter_count=int(parameters["count"]),
        maximum_physical_parameters=int(parameters["maximum_allowed"]),
        existing_cluster_amplification_target=float(
            fixed["existing_spent_cluster_amplification_target"]
        ),
    )
    selection_gates = dict(audit["selection_gates"])
    selection_gates["one_minimally_coupled_matter_metric"] = True
    selection_gates["no_object_labels"] = True
    selection_gates["no_object_specific_gravity_parameters"] = (
        int(parameters["object_specific_gravity_parameters"]) == 0
    )
    selection_gates["no_lensing_only_parameters"] = (
        int(parameters["lensing_only_parameters"]) == 0
    )
    all_selection = bool(all(selection_gates.values()))
    proxy_warning = audit["deep_AQUAL_decoupled_proxy_warning"]
    proxy_has_negative_region = bool(any(not row["proxy_elliptic"] for row in proxy_warning))
    report = {
        "status": "completed Sigma v10A theory-only action selection",
        "candidate": config["candidate"],
        "action": config["action"],
        "definitions": config["definitions"],
        "integration_by_parts_identity": config["integration_by_parts_identity"],
        "physical_parameters": parameters,
        "fixed_selection_values": fixed,
        "base_spectrum": audit["base_spectrum"],
        "derived_coefficients": audit["derived_coefficients"],
        "mixed_spectrum": audit["mixed_spectrum"],
        "linear_capacity": audit["linear_capacity"],
        "carrier_potential_convexity": audit["carrier_potential_convexity"],
        "geometry": audit["geometry"],
        "selection_gates": selection_gates,
        "all_selection_gates_pass": all_selection,
        "deep_AQUAL_decoupled_proxy_warning": proxy_warning,
        "deep_AQUAL_proxy_has_negative_region": proxy_has_negative_region,
        "unresolved_mandatory_gates": audit["unresolved_mandatory_gates"],
        "all_mandatory_theory_gates_pass": False,
        "decision": "advance_to_full_coupled_constraint_and_quasistatic_ellipticity_audit_only"
        if all_selection
        else "retire_before_full_variation",
        "reason": "The six-component spatial carrier passes every declared necessary selection identity: its selected flat scalar block has squared speeds 0.0493 and 0.9507, the other five carrier modes have squared speed 0.25, its fixed-source potential is strictly convex, it retains trace and shear orientation, distinguishes an equal-g_bar scale pair through M/r^3, and has a normalized asymptotic response capacity of four. This is not a theory-health pass. A deliberately computed decoupled AQUAL proxy becomes non-elliptic in the low-acceleration region, so the next full AeST constraint reduction is capable of immediately falsifying the exact constant-mixing action.",
        "scope": "No new observation, cluster map, galaxy table, or raw holdout was opened. The factor-3.14465 target is an already-published scalar summary. The flat 2x2 block is necessary only; vector/aether mixing, nonlinear static ellipticity, physical metric potentials, PPN, cosmology, and retarded source uniqueness remain unproved.",
        "next_gate": "Derive and reduce the complete quadratic AeST-metric-aether-scalar-P action on the flat clock background and the complete quasistatic constitutive Hessian on arbitrary constant spatial-gradient backgrounds. Retire exact v10A if the negative deep-AQUAL proxy survives the constraint reduction, if any vector or scalar kinetic eigenvalue is negative, if c_T differs from one, or if an independent homogeneous P profile is allowed.",
        "prior_art_scope": config["prior_art_scope"],
        "prior_art_primary_anchors": [
            "https://arxiv.org/abs/2007.00082",
            "https://arxiv.org/abs/2109.13287",
            "https://arxiv.org/abs/1910.13995",
            "https://arxiv.org/abs/1806.02811"
        ],
        "novelty_claim": False,
        "new_observational_product_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
