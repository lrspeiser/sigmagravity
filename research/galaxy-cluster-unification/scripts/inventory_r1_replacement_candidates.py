from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _number_pair(cell: str, separator: str) -> tuple[float, float]:
    cleaned = cell.replace("$", "").replace("~", "").strip()
    match = re.search(rf"([0-9.]+)\s*{separator}\s*([0-9.]+)", cleaned)
    if not match:
        raise ValueError(f"could not parse pair from {cell!r}")
    return float(match.group(1)), float(match.group(2))


def _scalar(cell: str) -> float:
    cleaned = cell.replace("$", "").replace("~", "").strip()
    match = re.search(r"[-+]?[0-9.]+", cleaned)
    if not match:
        raise ValueError(f"could not parse scalar from {cell!r}")
    return float(match.group(0))


def _parse_newman_velocity(text: str) -> pd.DataFrame:
    start = text.index("\\tablecaption{Velocity dispersion profiles}")
    end = text.index("\\tablecomments{Line-of-sight velocity dispersions", start)
    block = text[start:end]
    rows: list[dict] = []
    current = [None, None]
    for line in block.splitlines():
        if "&" not in line or "\\pm" not in line:
            continue
        parts = [part.strip().removesuffix("\\").strip() for part in line.split("&")]
        if len(parts) != 6:
            continue
        for side, offset in enumerate((0, 3)):
            name, radius, sigma = parts[offset : offset + 3]
            if not radius or not sigma:
                continue
            if name != r"\ldots":
                current[side] = name
            if current[side] is None:
                raise ValueError("continuation row before Newman cluster name")
            radius_min, radius_max = _number_pair(radius, "-")
            value, error = _number_pair(sigma, r"\\pm")
            rows.append(
                {
                    "source_sample": "Newman2013",
                    "system": current[side],
                    "bin_min_arcsec": radius_min,
                    "bin_max_arcsec": radius_max,
                    "bin_min_kpc": float("nan"),
                    "bin_max_kpc": float("nan"),
                    "sigma_km_s": value,
                    "sigma_error_km_s": error,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 35:
        raise ValueError(f"expected 35 Newman velocity bins, parsed {len(frame)}")
    return frame


def _parse_newman_lensing_dof(text: str) -> dict[str, int]:
    start = text.index("Fit Quality to Strong Lensing")
    end = text.index("\\tablecomments{", start)
    result: dict[str, int] = {}
    for line in text[start:end].splitlines():
        match = re.match(r"\s*([A-Z][A-Za-z0-9]+)\s*&.*?&\s*[^&]*/(\d+)\s*&", line)
        if match:
            result[match.group(1)] = int(match.group(2))
    expected = {"MS2137", "A963", "A383", "A611", "A2537", "A2667", "A2390"}
    if set(result) != expected:
        raise ValueError(f"unexpected Newman lensing systems: {sorted(result)}")
    return result


def _parse_newman_photometry(text: str) -> pd.DataFrame:
    start = text.index(r"\tablecaption{\emph{HST} surface photometry of BCGs")
    end = text.index(r"\enddata", start)
    rows = []
    for line in text[start:end].splitlines():
        if "&" not in line or "\\pm" not in line:
            continue
        parts = [part.strip().removesuffix("\\").strip() for part in line.split("&")]
        if len(parts) != 9:
            continue
        system, filter_name, r_cut, r_core, axis_ratio, pa, magnitude, luminosity, proposal = parts
        r_cut_value, r_cut_error = _number_pair(r_cut, r"\\pm")
        rows.append(
            {
                "source_sample": "Newman2013",
                "system": system,
                "filter": filter_name,
                "axis_ratio_b_over_a": _scalar(axis_ratio),
                "position_angle_deg": _scalar(pa),
                "magnitude": _scalar(magnitude),
                "r_core_kpc": _scalar(r_core),
                "r_cut_kpc": r_cut_value,
                "r_cut_error_kpc": r_cut_error,
                "rest_v_luminosity_1e11_lsun": _scalar(luminosity),
                "hst_proposal_id": int(_scalar(proposal)),
                "profile_kind": "parametric_dPIE_starlight_fit",
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 7:
        raise ValueError(f"expected 7 Newman photometry fits, parsed {len(frame)}")
    return frame


def _parse_newman_sps(text: str) -> pd.DataFrame:
    start = text.index(r"\tablecaption{Stellar Population Synthesis Fits to BCGs")
    end = text.index(r"\enddata", start)
    rows = []
    for line in text[start:end].splitlines():
        if "&" not in line:
            continue
        parts = [part.strip().removesuffix("\\").strip() for part in line.split("&")]
        if len(parts) != 4 or not re.fullmatch(r"[A-Z][A-Za-z0-9]+", parts[0]):
            continue
        system, mass_to_light, filter_count, source = parts
        rows.append(
            {
                "system": system,
                "stellar_m_to_l_v_sps": _scalar(mass_to_light),
                "sps_filter_count": int(_scalar(filter_count)),
                "sps_photometry_source": source,
                "sps_imf": "Chabrier",
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 7:
        raise ValueError(f"expected 7 Newman SPS fits, parsed {len(frame)}")
    return frame


def _parse_newman_redshifts(text: str) -> pd.DataFrame:
    start = text.index(r"\label{tab:sample}")
    table_start = text.rfind(r"\startdata", 0, start)
    table_end = text.index(r"\enddata", table_start)
    aliases = {"MS2137.3-2353": "MS2137"}
    rows = []
    for line in text[table_start:table_end].splitlines():
        if "&" not in line:
            continue
        parts = [part.strip().removesuffix("\\").strip() for part in line.split("&")]
        if len(parts) != 10 or not parts[0]:
            continue
        name = aliases.get(parts[0], parts[0])
        if not re.fullmatch(r"[A-Z][A-Za-z0-9.]+", name):
            continue
        rows.append({"system": name, "cluster_redshift": _scalar(parts[1])})
    frame = pd.DataFrame(rows)
    if len(frame) != 7:
        raise ValueError(f"expected 7 Newman redshifts, parsed {len(frame)}")
    return frame


def _parse_kaleidoscope_velocity(text: str) -> pd.DataFrame:
    start = text.index("\\label{tab.bcgsigma}")
    end = text.index("\\end{tabular}", start)
    aliases = {
        r"\mstwo": "MACS J0326",
        r"\msthree": "MACS J1427",
        "MACSJ0949": "MACS J0949",
        "MACSJ0417": "MACS J0417",
    }
    rows: list[dict] = []
    current: str | None = None
    for line in text[start:end].splitlines():
        if "&" not in line or "\\pm" not in line:
            continue
        parts = [part.strip().removesuffix("\\").strip() for part in line.split("&")]
        if len(parts) < 4:
            continue
        name, arcsec, kpc, sigma = parts[:4]
        if name:
            current = aliases.get(name, name)
        if current is None:
            raise ValueError("continuation row before Kaleidoscope cluster name")
        arc_min, arc_max = _number_pair(arcsec, "-")
        kpc_min, kpc_max = _number_pair(kpc, "-")
        value, error = _number_pair(sigma, r"\\pm")
        rows.append(
            {
                "source_sample": "Kaleidoscope2025",
                "system": current,
                "bin_min_arcsec": arc_min,
                "bin_max_arcsec": arc_max,
                "bin_min_kpc": kpc_min,
                "bin_max_kpc": kpc_max,
                "sigma_km_s": value,
                "sigma_error_km_s": error,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 35:
        raise ValueError(f"expected 35 Kaleidoscope velocity bins, parsed {len(frame)}")
    return frame


def _parse_kaleidoscope_arc_counts(text: str) -> dict[str, int]:
    labels = {
        "A383": "tab.a383arcs",
        "MS2137": "tab.ms2137arcs",
        "MACS J0326": "tab.ms0326arcs",
        "MACS J1427": "tab.ms1427arcs",
    }
    counts: dict[str, int] = {}
    for system, label in labels.items():
        start = text.index(f"\\label{{{label}}}")
        end = text.index("\\end{tabular}", start)
        ids = re.findall(r"^\s*([0-9]+\.[0-9]+\*?)\s*&", text[start:end], flags=re.MULTILINE)
        counts[system] = sum(not value.endswith("*") for value in ids)
    return counts


def _parse_kaleidoscope_photometry(text: str) -> pd.DataFrame:
    start = text.index(r"\label{tab.bcgphot}")
    end = text.index(r"\end{tabular}", start)
    aliases = {r"\mstwo": "MACS J0326", r"\msthree": "MACS J1427"}
    rows = []
    for line in text[start:end].splitlines():
        if "&" not in line or "\\pm" not in line:
            continue
        parts = [part.strip().removesuffix("\\").strip() for part in line.split("&")]
        if len(parts) != 7:
            continue
        system, filter_name, axis_ratio, pa, magnitude, r_core, r_cut = parts
        r_cut_value, r_cut_error = _number_pair(r_cut, r"\\pm")
        rows.append(
            {
                "source_sample": "Kaleidoscope2025",
                "system": aliases.get(system, system),
                "filter": filter_name,
                "axis_ratio_b_over_a": _scalar(axis_ratio),
                "position_angle_deg": _scalar(pa),
                "magnitude": _scalar(magnitude),
                "r_core_kpc": _scalar(r_core),
                "r_cut_kpc": r_cut_value,
                "r_cut_error_kpc": r_cut_error,
                "rest_v_luminosity_1e11_lsun": float("nan"),
                "hst_proposal_id": float("nan"),
                "profile_kind": "parametric_dPIE_starlight_fit",
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 6:
        raise ValueError(f"expected 6 Kaleidoscope photometry fits, parsed {len(frame)}")
    return frame


def build_inventory(
    *,
    config_path: Path,
    lens_observables_path: Path,
    candidate_output: Path,
    queue_output: Path,
    dynamics_output: Path,
    photometry_output: Path,
    report_output: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_paths = {
        name: ROOT / source["local_source"] for name, source in config["sources"].items()
    }
    source_text = {name: path.read_text(encoding="utf-8") for name, path in source_paths.items()}

    newman_dynamics = _parse_newman_velocity(source_text["Newman2013"])
    kaleidoscope_dynamics = _parse_kaleidoscope_velocity(source_text["Kaleidoscope2025"])
    dynamics = pd.concat([newman_dynamics, kaleidoscope_dynamics], ignore_index=True)
    newman_photometry = _parse_newman_photometry(source_text["Newman2013"])
    newman_sps = _parse_newman_sps(source_text["Newman2013"])
    newman_redshifts = _parse_newman_redshifts(source_text["Newman2013"])
    newman_photometry = newman_photometry.merge(
        newman_sps, on="system", how="left", validate="one_to_one"
    ).merge(newman_redshifts, on="system", how="left", validate="one_to_one")
    newman_photometry["stellar_mass_sps_1e11_msun"] = (
        newman_photometry["rest_v_luminosity_1e11_lsun"]
        * newman_photometry["stellar_m_to_l_v_sps"]
    )
    photometry = pd.concat(
        [
            newman_photometry,
            _parse_kaleidoscope_photometry(source_text["Kaleidoscope2025"]),
        ],
        ignore_index=True,
    )
    newman_lensing = _parse_newman_lensing_dof(source_text["Newman2013"])
    kaleidoscope_lensing = _parse_kaleidoscope_arc_counts(source_text["Kaleidoscope2025"])
    lens_observables = pd.read_csv(lens_observables_path)
    observable_lens_systems = set(
        lens_observables.loc[
            lens_observables["observable_level_image_position"].astype(bool), "system"
        ]
    )
    likelihood_input_systems = set(
        lens_observables.loc[
            lens_observables["alternative_metric_likelihood_ready"].astype(bool), "system"
        ]
    )

    candidates = pd.DataFrame(config["records"])
    candidates["local_bcg_stellar_component_profile"] = candidates["system"].isin(
        set(newman_photometry["system"])
    )
    candidates["local_observable_level_lens_positions"] = candidates["system"].isin(
        observable_lens_systems
    )
    candidates["local_position_redshift_likelihood_inputs"] = candidates["system"].isin(
        likelihood_input_systems
    )
    # A complete alternative-metric rerun also needs the member-galaxy catalogue,
    # baryonic components, nuisance priors, and either a chain or full covariance.
    candidates["alternative_metric_forward_model_lensing_ready"] = False
    parsed_dynamics = dynamics.groupby(["source_sample", "system"]).size().to_dict()
    for row in candidates.itertuples(index=False):
        key = (row.source_sample, row.system)
        if parsed_dynamics.get(key) != row.expected_dynamics_points:
            raise ValueError(
                f"dynamics count mismatch for {key}: "
                f"config={row.expected_dynamics_points}, parsed={parsed_dynamics.get(key)}"
            )
        if row.source_sample == "Newman2013":
            parsed_lensing = newman_lensing[row.system]
        elif row.system in kaleidoscope_lensing:
            parsed_lensing = kaleidoscope_lensing[row.system]
        else:
            parsed_lensing = row.published_lensing_constraint_points
        if parsed_lensing != row.published_lensing_constraint_points:
            raise ValueError(
                f"lensing count mismatch for {key}: "
                f"config={row.published_lensing_constraint_points}, parsed={parsed_lensing}"
            )

    gate = config["gate"]
    candidates["published_coverage_candidate"] = (
        (candidates["expected_dynamics_points"] >= gate["minimum_dynamics_radial_points"])
        & (
            candidates["published_lensing_constraint_points"]
            >= gate["minimum_lensing_constraint_points"]
        )
        & candidates["published_baryonic_profile"]
    )
    candidates["analysis_ready"] = (
        candidates["published_coverage_candidate"]
        & candidates["local_dynamics_table"]
        & candidates["local_lensing_constraints"]
        & candidates["local_forward_model_baryonic_profile"]
        & candidates["alternative_metric_forward_model_lensing_ready"]
        & candidates["full_covariance_local"]
        & candidates["exact_radial_overlap_verified"]
    )
    def blocking_reason(row: pd.Series) -> str:
        if not row["published_coverage_candidate"]:
            return "published coverage below 3+3 gate"
        blockers = []
        if not row["local_forward_model_baryonic_profile"]:
            if row["local_bcg_stellar_component_profile"]:
                blockers.append("remaining baryonic components and uncertainties")
            else:
                blockers.append("numerical baryonic profile")
        if not row["alternative_metric_forward_model_lensing_ready"]:
            if row["local_observable_level_lens_positions"]:
                blockers.append("complete positional/redshift likelihood and rerunnable lens model")
            else:
                blockers.append("observable-level lensing inputs")
        if not row["full_covariance_local"]:
            blockers.append("joint observable covariance")
        if not row["exact_radial_overlap_verified"]:
            blockers.append("exact radial-overlap audit")
        return "need " + ", ".join(blockers)

    candidates["blocking_reason"] = candidates.apply(blocking_reason, axis=1)
    candidates = candidates.sort_values(
        ["system", "source_priority", "source_sample"], ascending=[True, False, True]
    ).reset_index(drop=True)

    queue = (
        candidates.sort_values(["system", "source_priority"], ascending=[True, False])
        .drop_duplicates("system", keep="first")
        .sort_values(["published_coverage_candidate", "system"], ascending=[False, True])
        .reset_index(drop=True)
    )
    queue["acquisition_priority"] = range(1, len(queue) + 1)
    queue["residual_blind_selection"] = True

    for output in (
        candidate_output,
        queue_output,
        dynamics_output,
        photometry_output,
        report_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidate_output, index=False)
    queue.to_csv(queue_output, index=False)
    dynamics.to_csv(dynamics_output, index=False)
    photometry.to_csv(photometry_output, index=False)

    coverage_count = int(queue["published_coverage_candidate"].sum())
    ready_count = int(queue["analysis_ready"].sum())
    report = {
        "audit_version": config["audit_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_rule": config["freeze_rule"],
        "gate": gate,
        "inputs": {
            **{
                name: {"path": _display_path(path), "sha256": _sha256(path)}
                for name, path in source_paths.items()
            },
            "R1NormalizedLensObservables": {
                "path": _display_path(lens_observables_path),
                "sha256": _sha256(lens_observables_path),
            },
        },
        "parsed_observables": {
            "newman_velocity_bins": len(newman_dynamics),
            "kaleidoscope_velocity_bins": len(kaleidoscope_dynamics),
            "total_velocity_bins": len(dynamics),
            "parametric_bcg_starlight_fits": len(photometry),
            "newman_sps_mass_normalizations": len(newman_sps),
            "source_rows": len(candidates),
            "unique_systems": int(queue["system"].nunique()),
        },
        "candidate_gate": {
            "published_coverage_candidates": coverage_count,
            "candidate_systems": queue.loc[
                queue["published_coverage_candidate"], "system"
            ].tolist(),
            "analysis_ready_systems": ready_count,
            "lensing_mcmc_ensemble_systems": int(
                queue["lensing_likelihood_or_chain_local"].sum()
            ),
            "systems_with_local_bcg_stellar_component": int(
                queue["local_bcg_stellar_component_profile"].sum()
            ),
            "systems_with_observable_level_lens_positions": int(
                queue["local_observable_level_lens_positions"].sum()
            ),
            "systems_with_position_redshift_likelihood_inputs": int(
                queue["local_position_redshift_likelihood_inputs"].sum()
            ),
            "alternative_metric_forward_model_lensing_ready_systems": int(
                queue["alternative_metric_forward_model_lensing_ready"].sum()
            ),
            "published_count_gate_passes": coverage_count >= gate["minimum_systems"],
            "strict_R1_gate_passes": ready_count >= gate["minimum_systems"],
        },
        "outputs": {
            "source_inventory": _display_path(candidate_output),
            "acquisition_queue": _display_path(queue_output),
            "velocity_profiles": _display_path(dynamics_output),
            "bcg_photometric_fits": _display_path(photometry_output),
        },
        "stage_decision": {
            "R1_candidate_inventory": "complete",
            "R1A1_identifiability_pilot": "completed_failed_structural_rank_gate",
            "R1A2_replacement_qualification": "authorized",
            "R1_sample_freeze": "not_authorized",
            "R2_two_potential_reconstruction": "not_authorized",
            "reason": (
                "Ten unique systems pass the published-count screen and three now have RELICS "
                "MCMC lens-map ensembles, and all ten have observable-level image positions "
                "plus at least one strict position/redshift likelihood input. Seven have a "
                "normalized BCG stellar component, but zero "
                "have the complete baryonic profile, joint "
                "observable covariance, alternative-metric-ready lens likelihood, and verified "
                "overlapping radial support required by R1."
            ),
            "next_action": (
                "Do not build a full Jacobian for the current ten systems: their structural radial-"
                "rank upper bounds are below three. Screen residual-blind replacement BCG hosts "
                "with longer kinematic support or at least three central image radii, then rerun "
                "the frozen structural gate before any response fit."
            ),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "r1_replacement_sample_gate.json"
    )
    parser.add_argument(
        "--lens-observables",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_strong_lens_image_observables.csv",
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_replacement_source_inventory.csv",
    )
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_replacement_acquisition_queue.csv",
    )
    parser.add_argument(
        "--dynamics-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_velocity_profiles.csv",
    )
    parser.add_argument(
        "--photometry-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_photometric_fits.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "r1_replacement_inventory" / "report.json",
    )
    args = parser.parse_args()
    report = build_inventory(
        config_path=args.config,
        lens_observables_path=args.lens_observables,
        candidate_output=args.candidate_output,
        queue_output=args.queue_output,
        dynamics_output=args.dynamics_output,
        photometry_output=args.photometry_output,
        report_output=args.report_output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
