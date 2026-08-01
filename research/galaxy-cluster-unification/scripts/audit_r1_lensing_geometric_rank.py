from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def audit(
    *,
    targets_path: Path,
    sample_config_path: Path,
    images_path: Path,
    support_path: Path,
    dynamics_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict:
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    sample_records = pd.DataFrame(
        json.loads(sample_config_path.read_text(encoding="utf-8"))["records"]
    )
    selected_sources = (
        sample_records.sort_values("source_priority", ascending=False)
        .drop_duplicates("system", keep="first")
        .set_index("system")["source_sample"]
        .to_dict()
    )
    pilot_meta = {row["system"]: row for row in targets["pilot_order"]}
    pilot_rank = {row["system"]: rank for rank, row in enumerate(targets["pilot_order"], 1)}
    images = pd.read_csv(images_path, dtype={"source_family": str, "image_id": str})
    support = pd.read_csv(support_path, dtype={"source_family": str, "image_id": str})
    dynamics = pd.read_csv(dynamics_path)
    screen = targets["geometric_pre_screen"]

    rows = []
    for system, system_images in images.groupby("system", sort=True):
        selected_source = selected_sources[system]
        dynamics_bins = dynamics.loc[
            (dynamics["system"] == system) & (dynamics["source_sample"] == selected_source)
        ]
        system_support = support.loc[support["system"] == system]
        inner = system_support.loc[
            system_support["inside_dynamics_support"].astype(bool)
            & system_support["alternative_metric_likelihood_ready"].astype(bool)
        ]
        inner_families = set(inner["source_family"])
        associated = system_images.loc[
            system_images["source_family"].isin(inner_families)
            & system_images["alternative_metric_likelihood_ready"].astype(bool)
        ]
        family_counts = associated.groupby("source_family").size()
        family_wide_dof = int(sum(max(0, 2 * int(count) - 2) for count in family_counts))
        inner_images = len(inner)
        structural_radial_rank_upper_bound = min(
            inner_images, targets["full_jacobian_gate"]["inner_weyl_response_nodes"]
        )
        outer_counterimages = len(associated) - inner_images
        radial_auditable = bool(system_support["bcg_centering_verified"].all())
        passes = bool(
            len(dynamics_bins) >= screen["minimum_selected_dynamics_bins"]
            and inner_images >= screen["minimum_observable_images_inside_dynamics_support"]
            and structural_radial_rank_upper_bound
            >= screen["minimum_structural_radial_rank_upper_bound"]
            and family_wide_dof
            >= screen["minimum_family_wide_position_degrees_of_freedom_after_source_position"]
            and radial_auditable
        )
        meta = pilot_meta.get(system, {})
        rows.append(
            {
                "system": system,
                "selected_dynamics_source": selected_source,
                "selected_dynamics_bins": len(dynamics_bins),
                "dynamics_r_max_arcsec": float(dynamics_bins["bin_max_arcsec"].max()),
                "radial_support_auditable": radial_auditable,
                "strict_inner_image_positions": inner_images,
                "strict_inner_source_families": len(inner_families),
                "structural_radial_rank_upper_bound": structural_radial_rank_upper_bound,
                "family_wide_strict_image_positions": len(associated),
                "outer_counterimages_for_inner_families": outer_counterimages,
                "family_wide_position_dof_after_source_coordinates": family_wide_dof,
                "geometric_prescreen_pass": passes,
                "pilot_priority": pilot_rank.get(system),
                "pilot_role": meta.get("role"),
                "disturbed_control": meta.get("disturbed", False),
                "full_marginalized_jacobian_status": (
                    "required_next" if passes else "not_authorized_structural_rank_below_three"
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["geometric_prescreen_pass", "pilot_priority", "system"],
        ascending=[False, True, True],
        na_position="last",
    )
    qualified = result.loc[result["geometric_prescreen_pass"]]
    non_disturbed = qualified.loc[~qualified["disturbed_control"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report = {
        "audit_version": targets["audit_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": targets["stage"],
        "selection_rule": targets["selection_rule"],
        "summary": {
            "systems_audited": len(result),
            "geometric_prescreen_passes": len(qualified),
            "geometric_prescreen_systems": qualified["system"].tolist(),
            "non_disturbed_prescreen_passes": len(non_disturbed),
            "non_disturbed_prescreen_systems": non_disturbed["system"].tolist(),
            "full_marginalized_jacobians_completed": 0,
            "systems_passing_three_mode_gate": 0,
        },
        "interpretation": {
            "what_pass_means": (
                "At least three strict inner images provide a structural radial-rank upper bound "
                "of three and their source families retain enough family-wide position degrees "
                "of freedom to anchor the unknown source coordinates."
            ),
            "what_pass_does_not_mean": (
                "It does not establish three independent radial lensing modes; nuisance-marginalized "
                "Jacobian rank and precision remain to be calculated."
            ),
            "next_action": (
                "Do not build the full Jacobian for the current candidates. Search residual-blind "
                "replacement systems with longer BCG kinematic support or at least three central "
                "image radii, then apply this same structural gate."
            ),
        },
        "full_jacobian_gate": targets["full_jacobian_gate"],
        "progress_rule": targets["progress_rule"],
        "output": _display_path(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "configs" / "r1_identifiability_targets.json",
    )
    parser.add_argument(
        "--sample-config",
        type=Path,
        default=ROOT / "configs" / "r1_replacement_sample_gate.json",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_strong_lens_image_observables.csv",
    )
    parser.add_argument(
        "--support",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_strong_lens_radial_support.csv",
    )
    parser.add_argument(
        "--dynamics",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_velocity_profiles.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_lensing_geometric_rank.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "r1_lensing_geometric_rank" / "report.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                targets_path=args.targets,
                sample_config_path=args.sample_config,
                images_path=args.images,
                support_path=args.support,
                dynamics_path=args.dynamics,
                output_path=args.output,
                report_path=args.report,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
