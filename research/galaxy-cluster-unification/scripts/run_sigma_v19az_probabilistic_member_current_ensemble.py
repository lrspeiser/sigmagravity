#!/usr/bin/env python3
"""Build the exact V19AZ member-position posterior and sampled current ensemble."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19az_probabilistic_member_current_ensemble.json"
SPEED_OF_LIGHT_KM_S = 299792.458


@dataclass(frozen=True)
class State:
    member_id: str
    state_type: str
    candidate_id: str
    ra_deg: float
    dec_deg: float
    local_probability: float


@dataclass
class ComponentPosterior:
    component_id: str
    members: tuple[str, ...]
    assignments: list[tuple[int, ...]]
    probabilities: np.ndarray
    log_partition: float
    cartesian_state_count: int
    valid_state_count: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def numeric_member_key(member_id: str) -> tuple[int, str]:
    try:
        return int(member_id), member_id
    except ValueError:
        return 10**9, member_id


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized probabilities and log(sum(exp(log_weights)))."""
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be a nonempty finite one-dimensional array")
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    total = float(np.sum(shifted))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("log-weight normalization failed")
    return shifted / total, maximum + math.log(total)


def exact_component_posterior(
    component_id: str,
    members: tuple[str, ...],
    states_by_member: dict[str, list[State]],
) -> ComponentPosterior:
    """Enumerate a conflict component with one-to-one real candidate usage."""
    choices = [range(len(states_by_member[member])) for member in members]
    cartesian_count = math.prod(len(states_by_member[member]) for member in members)
    assignments: list[tuple[int, ...]] = []
    log_weights: list[float] = []
    for assignment in itertools.product(*choices):
        candidate_ids = [
            states_by_member[member][state_index].candidate_id
            for member, state_index in zip(members, assignment, strict=True)
            if states_by_member[member][state_index].state_type != "null"
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            continue
        probabilities = [
            states_by_member[member][state_index].local_probability
            for member, state_index in zip(members, assignment, strict=True)
        ]
        if any(probability <= 0.0 for probability in probabilities):
            continue
        assignments.append(tuple(int(value) for value in assignment))
        log_weights.append(sum(math.log(probability) for probability in probabilities))
    if not assignments:
        raise ValueError(f"component {component_id} has no valid joint assignment")
    probabilities, log_partition = normalize_log_weights(np.asarray(log_weights))
    return ComponentPosterior(
        component_id=component_id,
        members=members,
        assignments=assignments,
        probabilities=probabilities,
        log_partition=log_partition,
        cartesian_state_count=cartesian_count,
        valid_state_count=len(assignments),
    )


def component_marginals(
    component: ComponentPosterior,
    states_by_member: dict[str, list[State]],
) -> dict[str, np.ndarray]:
    result = {
        member: np.zeros(len(states_by_member[member]), dtype=float) for member in component.members
    }
    for assignment, probability in zip(component.assignments, component.probabilities, strict=True):
        for member, state_index in zip(component.members, assignment, strict=True):
            result[member][state_index] += float(probability)
    return result


def build_conflict_components(
    states_by_member: dict[str, list[State]],
) -> tuple[list[tuple[str, ...]], dict[str, set[str]]]:
    members = sorted(states_by_member, key=numeric_member_key)
    candidate_members: dict[str, set[str]] = defaultdict(set)
    for member in members:
        for state in states_by_member[member]:
            if state.state_type != "null":
                candidate_members[state.candidate_id].add(member)

    parent = {member: member for member in members}

    def find(member: str) -> str:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(first: str, second: str) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            if numeric_member_key(root_first) <= numeric_member_key(root_second):
                parent[root_second] = root_first
            else:
                parent[root_first] = root_second

    for linked_members in candidate_members.values():
        ordered = sorted(linked_members, key=numeric_member_key)
        for other in ordered[1:]:
            union(ordered[0], other)

    grouped: dict[str, list[str]] = defaultdict(list)
    for member in members:
        grouped[find(member)].append(member)
    components = [tuple(sorted(group, key=numeric_member_key)) for group in grouped.values()]
    components.sort(key=lambda group: numeric_member_key(group[0]))
    return components, candidate_members


def null_position(
    rng: np.random.Generator, paper_ra_deg: float, paper_dec_deg: float
) -> tuple[float, float]:
    east_half_width = 7.5 * math.cos(math.radians(paper_dec_deg))
    east_arcsec = rng.uniform(-east_half_width, east_half_width)
    north_arcsec = rng.uniform(-0.5, 0.5)
    ra_deg = paper_ra_deg + east_arcsec / (3600.0 * math.cos(math.radians(paper_dec_deg)))
    dec_deg = paper_dec_deg + north_arcsec / 3600.0
    return ra_deg, dec_deg


def local_offsets_arcsec(
    ra_deg: float, dec_deg: float, origin_ra_deg: float, origin_dec_deg: float
) -> tuple[float, float]:
    east = (ra_deg - origin_ra_deg) * 3600.0 * math.cos(math.radians(origin_dec_deg))
    north = (dec_deg - origin_dec_deg) * 3600.0
    return east, north


def verify_parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        value = sha256(path)
        if value != spec["sha256"]:
            raise ValueError(f"parent hash mismatch for {name}: {value} != {spec['sha256']}")
        actual[name] = value
    return actual


def make_figure(
    path: Path,
    member_summaries: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = [row for row in member_summaries if row["position_class"] != "missing_photometry"]
    center_ra = float(np.median([float(row["expected_ra_deg"]) for row in finite]))
    center_dec = float(np.median([float(row["expected_dec_deg"]) for row in finite]))
    velocity_limit = max(abs(float(row["nominal_v_los_rest_km_s"])) for row in finite)

    fig, axis = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
    for position_class, marker, label in (
        ("fixed_anchor", "s", "15 validated anchors"),
        ("probabilistic", "o", "57 exact positional posteriors"),
    ):
        rows = [row for row in finite if row["position_class"] == position_class]
        x = np.asarray(
            [
                (float(row["expected_ra_deg"]) - center_ra)
                * 60.0
                * math.cos(math.radians(center_dec))
                for row in rows
            ]
        )
        y = np.asarray([(float(row["expected_dec_deg"]) - center_dec) * 60.0 for row in rows])
        xerr = np.asarray([float(row["east_position_std_arcsec"]) / 60.0 for row in rows])
        yerr = np.asarray([float(row["north_position_std_arcsec"]) / 60.0 for row in rows])
        sizes = np.asarray(
            [35.0 + 55.0 * math.sqrt(float(row["relative_i_luminosity"])) for row in rows]
        )
        velocities = np.asarray([float(row["nominal_v_los_rest_km_s"]) for row in rows])
        axis.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.65", alpha=0.7, zorder=1)
        scatter = axis.scatter(
            x,
            y,
            s=np.clip(sizes, 25.0, 260.0),
            c=velocities,
            cmap="coolwarm",
            vmin=-velocity_limit,
            vmax=velocity_limit,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=label,
            zorder=2,
        )

    missing = [row for row in inventory if row["photometry_status"] == "missing_bri"]
    if missing:
        x = [
            (float(row["paper_ra_deg"]) - center_ra) * 60.0 * math.cos(math.radians(center_dec))
            for row in missing
        ]
        y = [(float(row["paper_dec_deg"]) - center_dec) * 60.0 for row in missing]
        axis.scatter(x, y, marker="x", s=60, c="0.2", label="6 explicit missing-BRI members")

    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("nominal rest-frame line-of-sight velocity (km/s)")
    axis.set_xlabel(r"$\Delta$RA cos(Dec) (arcmin)")
    axis.set_ylabel(r"$\Delta$Dec (arcmin)")
    axis.set_title("Bullet Cluster: probabilistic baryonic member positions and LOS current proxy")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    axis.text(
        0.01,
        0.01,
        "marker area follows relative Bessel-I luminosity; error bars are posterior 1-sigma",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    input_hashes = verify_parent_hashes(config)
    parent_paths = {name: ROOT / spec["path"] for name, spec in config["parents"].items()}

    hypotheses = read_csv(parent_paths["candidate_hypotheses"])
    bri_rows = read_csv(parent_paths["published_bri"])
    member_rows = read_csv(parent_paths["spectroscopic_members"])
    anchor_rows = read_csv(parent_paths["anchor_manifest"])
    association_rows = read_csv(parent_paths["member_associations"])
    failed_flux_report = json.loads(
        parent_paths["failed_signed_flux_likelihood_report"].read_text()
    )
    if failed_flux_report["decision"] != "failed_closed":
        raise ValueError("V19AY must remain failed closed before V19AZ")
    if failed_flux_report["ambiguous_candidate_scoring_performed"]:
        raise ValueError("V19AY unexpectedly scored ambiguous candidates")

    members = {row["object_id"]: row for row in member_rows}
    bri = {row["object_id"]: row for row in bri_rows}
    associations = {row["object_id"]: row for row in association_rows if row["cluster"] == "BULLET"}
    if len(members) != config["population"]["expected_spectroscopic_members"]:
        raise ValueError("unexpected spectroscopic member count")
    if set(members) != set(bri) or set(members) != set(associations):
        raise ValueError("member, BRI, and association inventories differ")

    finite_bri_ids = {
        member_id
        for member_id, row in bri.items()
        if row["published_bri_available"].lower() == "true"
        and all(row[field] != "" for field in ("b_bessel_mag", "r_bessel_mag", "i_bessel_mag"))
    }
    missing_bri_ids = set(members) - finite_bri_ids

    anchors_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        if row["cluster"] == "BULLET":
            anchors_grouped[row["member_id"]].append(row)
    anchors: dict[str, dict[str, Any]] = {}
    for member_id, rows in anchors_grouped.items():
        unique_values = {(row["nsc_id"], row["ra_deg"], row["dec_deg"]) for row in rows}
        if len(unique_values) != 1:
            raise ValueError(f"anchor {member_id} has inconsistent catalog coordinates")
        nsc_id, ra_deg, dec_deg = next(iter(unique_values))
        anchors[member_id] = {
            "candidate_id": f"NSC:{nsc_id}",
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
        }

    hypotheses_by_member: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidate_coordinates: dict[str, tuple[float, float]] = {}
    for row in hypotheses:
        member_id = row["member_id"]
        hypotheses_by_member[member_id].append(row)
        coordinates = (float(row["candidate_ra_deg"]), float(row["candidate_dec_deg"]))
        if row["candidate_id"] in candidate_coordinates:
            previous = candidate_coordinates[row["candidate_id"]]
            if max(abs(a - b) for a, b in zip(previous, coordinates, strict=True)) > 1e-10:
                raise ValueError(f"candidate {row['candidate_id']} has inconsistent coordinates")
        candidate_coordinates[row["candidate_id"]] = coordinates

    probabilistic_ids = set(hypotheses_by_member)
    if finite_bri_ids != set(anchors) | probabilistic_ids:
        raise ValueError("finite-BRI members are not exactly anchors plus probabilistic members")
    if set(anchors) & probabilistic_ids:
        raise ValueError("anchor and probabilistic member sets overlap")
    anchor_candidate_ids = {row["candidate_id"] for row in anchors.values()}
    if anchor_candidate_ids & set(candidate_coordinates):
        raise ValueError("a probabilistic candidate conflicts with a fixed anchor")

    expected = config["population"]
    if len(finite_bri_ids) != expected["expected_members_with_finite_published_bri"]:
        raise ValueError("unexpected finite-BRI member count")
    if len(anchors) != expected["expected_fixed_anchor_members"]:
        raise ValueError("unexpected anchor count")
    if len(probabilistic_ids) != expected["expected_probabilistic_members"]:
        raise ValueError("unexpected probabilistic member count")
    if len(missing_bri_ids) != expected["expected_missing_bri_members"]:
        raise ValueError("unexpected missing-BRI member count")

    local_normalization_errors: list[float] = []
    null_crosscheck_errors: list[float] = []
    states_by_member: dict[str, list[State]] = {}
    for member_id in sorted(probabilistic_ids, key=numeric_member_key):
        member_states: list[State] = []
        for row in sorted(hypotheses_by_member[member_id], key=lambda item: item["candidate_id"]):
            probability = float(row["positional_posterior_q_0_90"])
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError(f"invalid local probability for {member_id}")
            member_states.append(
                State(
                    member_id=member_id,
                    state_type="candidate",
                    candidate_id=row["candidate_id"],
                    ra_deg=float(row["candidate_ra_deg"]),
                    dec_deg=float(row["candidate_dec_deg"]),
                    local_probability=probability,
                )
            )
        candidate_sum = math.fsum(state.local_probability for state in member_states)
        null_probability = 1.0 - candidate_sum
        if null_probability < -1e-14:
            raise ValueError(f"negative null probability for {member_id}")
        null_probability = max(0.0, null_probability)
        paper = members[member_id]
        member_states.append(
            State(
                member_id=member_id,
                state_type="null",
                candidate_id="",
                ra_deg=float(paper["ra_deg"]),
                dec_deg=float(paper["dec_deg"]),
                local_probability=null_probability,
            )
        )
        local_normalization_errors.append(
            abs(math.fsum(state.local_probability for state in member_states) - 1.0)
        )
        null_crosscheck_errors.append(
            abs(null_probability - float(associations[member_id]["null_posterior_q_0.90"]))
        )
        states_by_member[member_id] = member_states

    components_raw, candidate_members = build_conflict_components(states_by_member)
    components: list[ComponentPosterior] = []
    component_for_member: dict[str, ComponentPosterior] = {}
    exact_marginals: dict[str, np.ndarray] = {}
    component_rows: list[dict[str, Any]] = []
    component_normalization_errors: list[float] = []
    for index, component_members in enumerate(components_raw, start=1):
        component = exact_component_posterior(f"C{index:03d}", component_members, states_by_member)
        components.append(component)
        component_normalization_errors.append(abs(float(np.sum(component.probabilities)) - 1.0))
        marginals = component_marginals(component, states_by_member)
        exact_marginals.update(marginals)
        for member_id in component_members:
            component_for_member[member_id] = component
        component_candidate_ids = {
            state.candidate_id
            for member_id in component_members
            for state in states_by_member[member_id]
            if state.state_type != "null"
        }
        shared_count = sum(
            1
            for candidate_id in component_candidate_ids
            if len(candidate_members[candidate_id]) > 1
        )
        component_rows.append(
            {
                "component_id": component.component_id,
                "member_ids": "|".join(component.members),
                "member_count": len(component.members),
                "unique_candidate_count": len(component_candidate_ids),
                "shared_candidate_count": shared_count,
                "cartesian_state_count": component.cartesian_state_count,
                "valid_state_count": component.valid_state_count,
                "conditioning_acceptance_probability": math.exp(component.log_partition),
                "log_partition": component.log_partition,
                "probability_normalization_error": abs(
                    float(np.sum(component.probabilities)) - 1.0
                ),
            }
        )

    state_rows: list[dict[str, Any]] = []
    member_marginal_errors: list[float] = []
    for member_id in sorted(probabilistic_ids, key=numeric_member_key):
        component = component_for_member[member_id]
        marginals = exact_marginals[member_id]
        member_marginal_errors.append(abs(float(np.sum(marginals)) - 1.0))
        paper = members[member_id]
        east_half_width = 7.5 * math.cos(math.radians(float(paper["dec_deg"])))
        for state, joint_probability in zip(states_by_member[member_id], marginals, strict=True):
            state_rows.append(
                {
                    "cluster": "BULLET",
                    "member_id": member_id,
                    "component_id": component.component_id,
                    "component_member_count": len(component.members),
                    "state_type": state.state_type,
                    "candidate_id": state.candidate_id,
                    "ra_deg_or_null_center": state.ra_deg,
                    "dec_deg_or_null_center": state.dec_deg,
                    "null_east_half_width_arcsec": east_half_width
                    if state.state_type == "null"
                    else "",
                    "null_north_half_width_arcsec": 0.5 if state.state_type == "null" else "",
                    "local_probability_q_0_90": state.local_probability,
                    "joint_one_to_one_probability": float(joint_probability),
                    "joint_minus_local_probability": float(joint_probability)
                    - state.local_probability,
                }
            )
    for member_id in sorted(anchors, key=numeric_member_key):
        anchor = anchors[member_id]
        state_rows.append(
            {
                "cluster": "BULLET",
                "member_id": member_id,
                "component_id": "FIXED",
                "component_member_count": 1,
                "state_type": "fixed_anchor",
                "candidate_id": anchor["candidate_id"],
                "ra_deg_or_null_center": anchor["ra_deg"],
                "dec_deg_or_null_center": anchor["dec_deg"],
                "null_east_half_width_arcsec": "",
                "null_north_half_width_arcsec": "",
                "local_probability_q_0_90": 1.0,
                "joint_one_to_one_probability": 1.0,
                "joint_minus_local_probability": 0.0,
            }
        )
        member_marginal_errors.append(0.0)

    occupancy_rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_members):
        member_probabilities = []
        for member_id in sorted(candidate_members[candidate_id], key=numeric_member_key):
            state_index = next(
                index
                for index, state in enumerate(states_by_member[member_id])
                if state.candidate_id == candidate_id and state.state_type == "candidate"
            )
            member_probabilities.append((member_id, float(exact_marginals[member_id][state_index])))
        occupancy = math.fsum(probability for _, probability in member_probabilities)
        occupancy_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_ra_deg": candidate_coordinates[candidate_id][0],
                "candidate_dec_deg": candidate_coordinates[candidate_id][1],
                "eligible_member_count": len(member_probabilities),
                "eligible_member_ids": "|".join(member for member, _ in member_probabilities),
                "member_assignment_probabilities": "|".join(
                    f"{member}:{probability:.17g}" for member, probability in member_probabilities
                ),
                "total_assignment_probability": occupancy,
                "unassigned_probability": 1.0 - occupancy,
            }
        )

    nominal_cz = np.asarray([float(members[member]["heliocentric_cz_km_s"]) for member in members])
    nominal_systemic_cz = float(np.median(nominal_cz))
    bri_values: dict[str, dict[str, float]] = {}
    for member_id in finite_bri_ids:
        row = bri[member_id]
        i_mag = float(row["i_bessel_mag"])
        bri_values[member_id] = {
            "b": float(row["b_bessel_mag"]),
            "r": float(row["r_bessel_mag"]),
            "i": i_mag,
            "luminosity": 10.0 ** (-0.4 * (i_mag - 20.0)),
        }

    member_summary_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for member_id in sorted(members, key=numeric_member_key):
        paper = members[member_id]
        has_bri = member_id in finite_bri_ids
        if member_id in anchors:
            position_class = "fixed_anchor"
        elif member_id in probabilistic_ids:
            position_class = "probabilistic"
        else:
            position_class = "missing_photometry"
        inventory_rows.append(
            {
                "cluster": "BULLET",
                "member_id": member_id,
                "paper_ra_deg": paper["ra_deg"],
                "paper_dec_deg": paper["dec_deg"],
                "heliocentric_cz_km_s": paper["heliocentric_cz_km_s"],
                "cz_uncertainty_km_s": paper["cz_uncertainty_km_s"],
                "photometry_status": "finite_bri" if has_bri else "missing_bri",
                "position_status": position_class,
                "transverse_velocity_status": "unmeasured_not_imputed",
                "included_in_ensemble": has_bri,
            }
        )
        if not has_bri:
            continue

        paper_ra = float(paper["ra_deg"])
        paper_dec = float(paper["dec_deg"])
        if position_class == "fixed_anchor":
            anchor = anchors[member_id]
            expected_east, expected_north = local_offsets_arcsec(
                anchor["ra_deg"], anchor["dec_deg"], paper_ra, paper_dec
            )
            east_variance = 0.0
            north_variance = 0.0
            local_null = 0.0
            joint_null = 0.0
            top_state_type = "fixed_anchor"
            top_candidate_id = anchor["candidate_id"]
            top_probability = 1.0
            entropy = 0.0
            candidate_count = 1
            component_id = "FIXED"
            component_size = 1
        else:
            states = states_by_member[member_id]
            marginals = exact_marginals[member_id]
            east_means = []
            north_means = []
            east_second = []
            north_second = []
            east_null_variance = (7.5 * math.cos(math.radians(paper_dec))) ** 2 / 3.0
            north_null_variance = 0.5**2 / 3.0
            for state in states:
                east, north = local_offsets_arcsec(state.ra_deg, state.dec_deg, paper_ra, paper_dec)
                east_means.append(east)
                north_means.append(north)
                east_second.append(
                    east * east + (east_null_variance if state.state_type == "null" else 0.0)
                )
                north_second.append(
                    north * north + (north_null_variance if state.state_type == "null" else 0.0)
                )
            expected_east = float(marginals @ np.asarray(east_means))
            expected_north = float(marginals @ np.asarray(north_means))
            east_variance = max(0.0, float(marginals @ np.asarray(east_second)) - expected_east**2)
            north_variance = max(
                0.0, float(marginals @ np.asarray(north_second)) - expected_north**2
            )
            null_index = next(
                index for index, state in enumerate(states) if state.state_type == "null"
            )
            local_null = states[null_index].local_probability
            joint_null = float(marginals[null_index])
            top_index = int(np.argmax(marginals))
            top_state = states[top_index]
            top_state_type = top_state.state_type
            top_candidate_id = top_state.candidate_id
            top_probability = float(marginals[top_index])
            positive = marginals[marginals > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)))
            candidate_count = len(states) - 1
            component = component_for_member[member_id]
            component_id = component.component_id
            component_size = len(component.members)

        expected_ra = paper_ra + expected_east / (3600.0 * math.cos(math.radians(paper_dec)))
        expected_dec = paper_dec + expected_north / 3600.0
        luminosity = bri_values[member_id]["luminosity"]
        nominal_v_los = (float(paper["heliocentric_cz_km_s"]) - nominal_systemic_cz) / (
            1.0 + nominal_systemic_cz / SPEED_OF_LIGHT_KM_S
        )
        member_summary_rows.append(
            {
                "cluster": "BULLET",
                "member_id": member_id,
                "position_class": position_class,
                "candidate_count": candidate_count,
                "component_id": component_id,
                "component_member_count": component_size,
                "local_null_probability": local_null,
                "joint_null_probability": joint_null,
                "top_joint_state_type": top_state_type,
                "top_joint_candidate_id": top_candidate_id,
                "top_joint_probability": top_probability,
                "position_entropy_nats": entropy,
                "expected_ra_deg": expected_ra,
                "expected_dec_deg": expected_dec,
                "east_position_std_arcsec": math.sqrt(east_variance),
                "north_position_std_arcsec": math.sqrt(north_variance),
                "radial_position_rms_arcsec": math.sqrt(east_variance + north_variance),
                "b_bessel_mag": bri_values[member_id]["b"],
                "r_bessel_mag": bri_values[member_id]["r"],
                "i_bessel_mag": bri_values[member_id]["i"],
                "relative_i_luminosity": luminosity,
                "heliocentric_cz_km_s": float(paper["heliocentric_cz_km_s"]),
                "cz_uncertainty_km_s": float(paper["cz_uncertainty_km_s"]),
                "nominal_cluster_median_cz_km_s": nominal_systemic_cz,
                "nominal_v_los_rest_km_s": nominal_v_los,
                "nominal_los_current_proxy": luminosity * nominal_v_los,
                "transverse_velocity_state": "unmeasured_not_imputed",
            }
        )

    output_specs = config["outputs"]
    inventory_path = ROOT / output_specs["membership_inventory"]
    state_path = ROOT / output_specs["state_marginals"]
    occupancy_path = ROOT / output_specs["candidate_occupancy"]
    component_path = ROOT / output_specs["component_partitions"]
    summary_path = ROOT / output_specs["member_summary"]
    ensemble_path = ROOT / output_specs["ensemble"]
    figure_path = ROOT / output_specs["figure"]
    report_path = ROOT / output_specs["report"]

    write_csv(inventory_path, inventory_rows, list(inventory_rows[0]))
    write_csv(state_path, state_rows, list(state_rows[0]))
    write_csv(occupancy_path, occupancy_rows, list(occupancy_rows[0]))
    write_csv(component_path, component_rows, list(component_rows[0]))
    write_csv(summary_path, member_summary_rows, list(member_summary_rows[0]))

    rng = np.random.Generator(np.random.PCG64(config["ensemble"]["seed"]))
    draws = int(config["ensemble"]["draws"])
    state_frequency: Counter[tuple[str, str, str]] = Counter()
    ordered_member_ids = sorted(finite_bri_ids, key=numeric_member_key)
    all_member_ids = sorted(members, key=numeric_member_key)
    ensemble_fields = [
        "sample_id",
        "cluster",
        "member_id",
        "position_state_type",
        "selected_candidate_id",
        "ra_deg",
        "dec_deg",
        "b_bessel_mag",
        "r_bessel_mag",
        "i_bessel_mag",
        "relative_i_luminosity",
        "cz_draw_km_s",
        "cluster_median_cz_draw_km_s",
        "v_los_rest_km_s",
        "los_current_proxy",
        "cz_uncertainty_km_s",
        "v_east_km_s",
        "v_north_km_s",
        "transverse_velocity_state",
    ]
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    all_samples_one_to_one = True
    all_samples_have_72 = True
    with gzip.open(ensemble_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ensemble_fields, extrasaction="raise")
        writer.writeheader()
        for sample_id in range(draws):
            selected: dict[str, State] = {}
            for component in components:
                assignment_index = int(
                    rng.choice(len(component.assignments), p=component.probabilities)
                )
                assignment = component.assignments[assignment_index]
                for member_id, state_index in zip(component.members, assignment, strict=True):
                    selected[member_id] = states_by_member[member_id][state_index]
            real_ids = [
                state.candidate_id for state in selected.values() if state.state_type != "null"
            ]
            if len(real_ids) != len(set(real_ids)):
                all_samples_one_to_one = False
            if set(selected) != probabilistic_ids:
                all_samples_have_72 = False

            cz_draws = {
                member_id: float(members[member_id]["heliocentric_cz_km_s"])
                + rng.normal(0.0, float(members[member_id]["cz_uncertainty_km_s"]))
                for member_id in all_member_ids
            }
            systemic_draw = float(np.median(list(cz_draws.values())))
            for member_id in ordered_member_ids:
                if member_id in anchors:
                    state_type = "fixed_anchor"
                    candidate_id = anchors[member_id]["candidate_id"]
                    ra_deg = anchors[member_id]["ra_deg"]
                    dec_deg = anchors[member_id]["dec_deg"]
                else:
                    state = selected[member_id]
                    state_type = state.state_type
                    candidate_id = state.candidate_id
                    if state.state_type == "null":
                        ra_deg, dec_deg = null_position(
                            rng,
                            float(members[member_id]["ra_deg"]),
                            float(members[member_id]["dec_deg"]),
                        )
                    else:
                        ra_deg, dec_deg = state.ra_deg, state.dec_deg
                state_frequency[(member_id, state_type, candidate_id)] += 1
                v_los = (cz_draws[member_id] - systemic_draw) / (
                    1.0 + systemic_draw / SPEED_OF_LIGHT_KM_S
                )
                values = bri_values[member_id]
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "cluster": "BULLET",
                        "member_id": member_id,
                        "position_state_type": state_type,
                        "selected_candidate_id": candidate_id,
                        "ra_deg": ra_deg,
                        "dec_deg": dec_deg,
                        "b_bessel_mag": values["b"],
                        "r_bessel_mag": values["r"],
                        "i_bessel_mag": values["i"],
                        "relative_i_luminosity": values["luminosity"],
                        "cz_draw_km_s": cz_draws[member_id],
                        "cluster_median_cz_draw_km_s": systemic_draw,
                        "v_los_rest_km_s": v_los,
                        "los_current_proxy": values["luminosity"] * v_los,
                        "cz_uncertainty_km_s": float(members[member_id]["cz_uncertainty_km_s"]),
                        "v_east_km_s": "",
                        "v_north_km_s": "",
                        "transverse_velocity_state": "unmeasured_not_imputed",
                    }
                )

    sampled_differences: list[float] = []
    for row in state_rows:
        if row["state_type"] == "fixed_anchor":
            sampled = (
                state_frequency[(row["member_id"], "fixed_anchor", row["candidate_id"])] / draws
            )
        else:
            sampled = (
                state_frequency[(row["member_id"], row["state_type"], row["candidate_id"])] / draws
            )
        sampled_differences.append(abs(sampled - float(row["joint_one_to_one_probability"])))

    make_figure(figure_path, member_summary_rows, inventory_rows)

    gates_spec = config["gates"]
    shared_candidates = [
        candidate for candidate, linked in candidate_members.items() if len(linked) > 1
    ]
    max_component_members = max(len(component.members) for component in components)
    max_local_norm = max(local_normalization_errors)
    max_component_norm = max(component_normalization_errors)
    max_member_norm = max(member_marginal_errors)
    max_occupancy = max(float(row["total_assignment_probability"]) for row in occupancy_rows)
    max_sample_difference = max(sampled_differences)
    gate_results = {
        "input_hashes_match": True,
        "exact_candidate_hypothesis_rows": len(hypotheses)
        == gates_spec["exact_candidate_hypothesis_rows"],
        "exact_unique_probabilistic_candidates": len(candidate_members)
        == gates_spec["exact_unique_probabilistic_candidates"],
        "exact_shared_candidate_count": len(shared_candidates)
        == gates_spec["exact_shared_candidate_count"],
        "maximum_conflict_component_members": max_component_members
        <= gates_spec["maximum_conflict_component_members"],
        "local_probability_normalization": max_local_norm
        <= gates_spec["local_probability_normalization_tolerance"],
        "null_probability_crosscheck": max(null_crosscheck_errors)
        <= gates_spec["local_probability_normalization_tolerance"],
        "component_probability_normalization": max_component_norm
        <= gates_spec["component_probability_normalization_tolerance"],
        "member_marginal_normalization": max_member_norm
        <= gates_spec["member_marginal_normalization_tolerance"],
        "candidate_occupancy_ceiling": max_occupancy
        <= 1.0 + gates_spec["candidate_occupancy_ceiling_tolerance"],
        "sampled_vs_exact_state_marginal": max_sample_difference
        <= gates_spec["sampled_vs_exact_state_marginal_max_absolute_difference"],
        "every_sample_one_to_one": all_samples_one_to_one,
        "all_72_bri_members_in_every_draw": all_samples_have_72,
        "all_6_missing_bri_members_explicit": sum(
            row["photometry_status"] == "missing_bri" for row in inventory_rows
        )
        == expected["expected_missing_bri_members"],
        "no_lensing_halo_or_gravity_payload": True,
    }
    decision = "passed" if all(gate_results.values()) else "failed_closed"
    output_paths = {
        "membership_inventory": inventory_path,
        "state_marginals": state_path,
        "candidate_occupancy": occupancy_path,
        "component_partitions": component_path,
        "member_summary": summary_path,
        "ensemble": ensemble_path,
        "figure": figure_path,
    }
    local_to_joint_changes = [
        abs(float(row["joint_minus_local_probability"])) for row in state_rows
    ]
    overall_log_acceptance = math.fsum(component.log_partition for component in components)
    report: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": config_hash,
        "input_hashes": input_hashes,
        "population": {
            "spectroscopic_members": len(members),
            "finite_bri_members": len(finite_bri_ids),
            "fixed_anchor_members": len(anchors),
            "probabilistic_members": len(probabilistic_ids),
            "missing_bri_members": len(missing_bri_ids),
            "missing_bri_member_ids": sorted(missing_bri_ids, key=numeric_member_key),
        },
        "candidate_graph": {
            "hypothesis_rows": len(hypotheses),
            "unique_candidates": len(candidate_members),
            "shared_candidates": len(shared_candidates),
            "connected_components": len(components),
            "coupled_components": sum(len(component.members) > 1 for component in components),
            "maximum_component_members": max_component_members,
            "total_cartesian_states": sum(
                component.cartesian_state_count for component in components
            ),
            "total_valid_states": sum(component.valid_state_count for component in components),
            "independent_draw_probability_of_no_candidate_collision": math.exp(
                overall_log_acceptance
            ),
        },
        "exact_posterior": {
            "solver": "complete connected-component enumeration with log-sum-exp normalization",
            "approximation": "none",
            "maximum_local_normalization_error": max_local_norm,
            "maximum_null_crosscheck_error": max(null_crosscheck_errors),
            "maximum_component_normalization_error": max_component_norm,
            "maximum_member_marginal_normalization_error": max_member_norm,
            "maximum_candidate_occupancy": max_occupancy,
            "maximum_absolute_local_to_joint_state_probability_change": max(local_to_joint_changes),
            "member_state_rows_including_anchors": len(state_rows),
        },
        "ensemble": {
            "draws": draws,
            "rows": draws * len(finite_bri_ids),
            "seed": config["ensemble"]["seed"],
            "maximum_absolute_sampled_vs_exact_state_probability_difference": max_sample_difference,
            "transverse_velocity_state": "unmeasured_not_imputed",
        },
        "kinematic_and_luminosity_proxies": {
            "nominal_cluster_median_cz_km_s": nominal_systemic_cz,
            "relative_i_luminosity_min": min(value["luminosity"] for value in bri_values.values()),
            "relative_i_luminosity_median": float(
                np.median([value["luminosity"] for value in bri_values.values()])
            ),
            "relative_i_luminosity_max": max(value["luminosity"] for value in bri_values.values()),
            "absolute_mass_inferred": False,
            "three_dimensional_current_inferred": False,
        },
        "gate_results": gate_results,
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in output_paths.items()
        },
        "claim_boundary": config["claim_boundary"],
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision != "passed":
        raise RuntimeError(f"V19AZ failed closed: {gate_results}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
