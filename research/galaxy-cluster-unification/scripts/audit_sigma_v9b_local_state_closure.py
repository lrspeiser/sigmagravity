from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v9b_local_state_closure import (
    audit_v9b_local_state_closure,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v9B spherical local-state closure."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v9b_local_state_closure.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v9b_local_state_closure",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    audit = audit_v9b_local_state_closure(
        sparc_predictions_path=ROOT / inputs["SPARC_point_predictions"],
        cluster_sample_path=ROOT / inputs["cluster_sample"],
        nearest_neighbors=int(inputs["nearest_neighbors"]),
    )
    report = {
        "status": "completed Sigma v9B local first-gradient state closure",
        "protocol_version": config["protocol_version"],
        "question": config["question"],
        "scope": config["scope"],
        **audit,
        "reason": (
            "In spherical symmetry, every regular shift-symmetric local "
            "first-gradient equation integrates to a constitutive flux fixed by "
            "G M_b(<r)/r^2. With a unique inverse and universal boundary condition, "
            "the physical enhancement is therefore one function of g_bar. In the "
            "spent development products, all 72 cluster points lie inside the SPARC "
            "outer g_bar range; their median nearest acceleration separation is about "
            "0.00145 dex, but their median required enhancement is about 0.509 dex "
            "larger. The declared local-state conflict gate passes."
        ),
        "claim_boundary": (
            "This closes only the stated regular static F(Y,Z,U) mechanism lane. "
            "The cluster accelerations are NFW-deprojected development targets, not "
            "a raw lensing likelihood, and the point match ignores covariance. The "
            "result does not rule out curvature, density, potential, matter-kinematic, "
            "nonlocal-memory, or additional uniquely baryon-forced carrier theories."
        ),
        "next_gate": (
            "Select a carrier whose unique baryon-forced state depends on a tidal or "
            "finite-environment invariant, so equal-g_bar spheres need not be equal, "
            "and whose traceless response predicts shear orientation. Prove its "
            "constraint/characteristic health before observational use."
        ),
        "requirements": config["requirements"],
        "decision_rule": config["decision_rule"],
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
