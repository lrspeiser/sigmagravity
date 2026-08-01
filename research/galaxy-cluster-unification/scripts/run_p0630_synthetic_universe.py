#!/usr/bin/env python3
"""Build P0630 seeded baryonic scenes and run locked cross-domain holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import RawLens, near_bound, score  # noqa: E402
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.data import load_curves  # noqa: E402
from voidscreen.raw_lensing import (  # noqa: E402
    RadialDeflectionField,
    spherical_deflection_radians,
)
from voidscreen.synthetic_universe import (  # noqa: E402
    A0_M_S2,
    G_SI,
    KPC_M,
    M_SUN_KG,
    ClusterSeed,
    GalaxySeed,
    RadialBaryonProfile,
    TRANSPORT_PARAMETER_BOUNDS,
    TRANSPORT_PARAMETER_NAMES,
    generate_cluster_scene,
    generate_galaxy_scene,
    parameter_mapping,
    predict_acceleration,
    radial_particle_acceleration_m_s2,
    rotation_velocity_km_s,
    sobol_galaxy_population,
    stable_hash_partition,
    stable_seed,
    transport_acceleration_from_features,
)


@dataclass(frozen=True)
class SystemRecord:
    name: str
    domain: str
    split: str
    seed: GalaxySeed | ClusterSeed
    target_g_m_s2: np.ndarray
    target_velocity_km_s: np.ndarray | None = None

    @property
    def profile(self) -> RadialBaryonProfile:
        return self.seed.profile


@dataclass(frozen=True)
class PackedTargets:
    gbar_m_s2: np.ndarray
    radius_kpc: np.ndarray
    target_g_m_s2: np.ndarray
    surface_density_msun_pc2: np.ndarray
    r80_kpc: np.ndarray
    reference_gbar_m_s2: np.ndarray
    system_index: np.ndarray
    system_domains: np.ndarray
    system_names: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_P0630_target_score":
        raise RuntimeError("P0630 protocol was not frozen before target scoring")
    if tuple(protocol["gravity_laws"]["universal_parameter_names"]) != (
        TRANSPORT_PARAMETER_NAMES
    ):
        raise RuntimeError("protocol and implementation parameter names differ")
    if not np.allclose(
        np.asarray(protocol["gravity_laws"]["universal_parameter_bounds"], dtype=float),
        np.asarray(TRANSPORT_PARAMETER_BOUNDS, dtype=float),
    ):
        raise RuntimeError("protocol and implementation bounds differ")
    return protocol


def _valid_galaxy_points(curve, disk_ml: float, bulge_ml: float) -> tuple[np.ndarray, np.ndarray]:
    baryonic_v2 = (
        np.sign(curve.velocity_gas_kms) * curve.velocity_gas_kms**2
        + disk_ml * curve.velocity_disk_unit_ml_kms**2
        + bulge_ml * curve.velocity_bulge_unit_ml_kms**2
    )
    valid = (
        np.isfinite(curve.radius_kpc)
        & np.isfinite(baryonic_v2)
        & np.isfinite(curve.velocity_observed_kms)
        & np.isfinite(curve.velocity_error_kms)
        & (curve.radius_kpc > 0.0)
        & (baryonic_v2 > 0.0)
        & (curve.velocity_observed_kms > 0.0)
        & (curve.velocity_error_kms > 0.0)
    )
    return valid, baryonic_v2


def load_galaxy_records(protocol: dict) -> list[SystemRecord]:
    settings = protocol["galaxy_sample"]
    seed_policy = protocol["seed_policy"]
    disk_ml = float(seed_policy["stellar_mass_to_light"]["disk"])
    bulge_ml = float(seed_policy["stellar_mass_to_light"]["bulge"])
    curves = load_curves(ROOT / settings["directory"])
    morphology_columns = [
        "galaxy",
        "hubble_type",
        "disk_mass_solar",
        "bulge_mass_solar",
        "gas_mass_solar",
        "disk_scale_kpc",
        "bulge_scale_fit_kpc",
        "HI_radius_kpc",
        "gas_fraction",
    ]
    morphology = pd.read_csv(ROOT / settings["morphology"], usecols=morphology_columns)
    morphology = morphology.set_index("galaxy")

    retained = []
    for curve in curves:
        metadata = curve.metadata
        valid, baryonic_v2 = _valid_galaxy_points(curve, disk_ml, bulge_ml)
        if metadata.quality > int(settings["quality_max"]):
            continue
        if metadata.inclination_deg < float(settings["minimum_inclination_deg"]):
            continue
        if int(valid.sum()) < int(settings["minimum_points"]):
            continue
        retained.append((curve, valid, baryonic_v2))
    labels = [curve.metadata.name for curve, _, _ in retained]
    if len(labels) != int(settings["expected_systems"]):
        raise RuntimeError(f"expected {settings['expected_systems']} galaxies, found {len(labels)}")
    partitions = stable_hash_partition(
        labels,
        salt=settings["split_salt"],
        train_fraction=float(settings["train_fraction"]),
        development_fraction=float(settings["development_fraction"]),
    )
    counts = Counter(partitions.values())
    if counts != Counter(settings["expected_split_counts"]):
        raise RuntimeError(f"galaxy split changed: {dict(counts)}")

    records = []
    for curve, valid, baryonic_v2 in retained:
        name = curve.metadata.name
        radius = curve.radius_kpc[valid]
        gbar = baryonic_v2[valid] * 1.0e6 / (radius * KPC_M)
        target_velocity = curve.velocity_observed_kms[valid]
        target_g = target_velocity**2 * 1.0e6 / (radius * KPC_M)
        profile = RadialBaryonProfile(name, radius, gbar)
        row = morphology.loc[name]
        component = np.maximum(
            np.asarray(
                [row.disk_mass_solar, row.bulge_mass_solar, row.gas_mass_solar],
                dtype=float,
            ),
            0.0,
        )
        if not np.isfinite(component).all() or component.sum() <= 0.0:
            component = np.array([0.6, 0.05, 0.35])
        component = profile.total_mass_msun * component / component.sum()
        disk_scale = max(float(row.disk_scale_kpc), 0.05)
        bulge_scale = float(row.bulge_scale_fit_kpc)
        if not np.isfinite(bulge_scale) or bulge_scale <= 0.0:
            bulge_scale = 0.2 * disk_scale
        hi_radius = float(row.HI_radius_kpc)
        gas_scale = hi_radius / 3.2 if np.isfinite(hi_radius) and hi_radius > 0.0 else 2.0 * disk_scale
        hubble = float(row.hubble_type)
        gas_fraction = float(row.gas_fraction)
        seed = GalaxySeed(
            name=name,
            profile=profile,
            disk_mass_msun=float(component[0]),
            bulge_mass_msun=float(component[1]),
            gas_mass_msun=float(component[2]),
            disk_scale_kpc=disk_scale,
            bulge_scale_kpc=max(bulge_scale, 0.02),
            gas_scale_kpc=max(gas_scale, 0.05),
            bar_strength=float(np.clip((5.0 - hubble) / 12.0, 0.0, 0.45)),
            spiral_strength=float(np.clip((hubble + 1.0) / 18.0, 0.0, 0.55)),
            clumpiness=float(np.clip(gas_fraction, 0.0, 1.0)),
            inclination_deg=float(curve.metadata.inclination_deg),
            random_seed=stable_seed(name),
        )
        records.append(
            SystemRecord(
                name=name,
                domain="galaxy",
                split=partitions[name],
                seed=seed,
                target_g_m_s2=target_g,
                target_velocity_km_s=target_velocity,
            )
        )
    return records


def load_cluster_records(protocol: dict) -> list[SystemRecord]:
    settings = protocol["cluster_sample"]
    table = pd.read_csv(
        ROOT / settings["radial_profile"],
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_gbar",
            "log_gobs",
            "err_log_gbar",
            "err_log_gobs",
        ],
    )
    split_lookup = {
        name: split for split in ("train", "development", "holdout") for name in settings[split]
    }
    if set(split_lookup) != set(table.system.unique()):
        missing = set(table.system.unique()) - set(split_lookup)
        extra = set(split_lookup) - set(table.system.unique())
        raise RuntimeError(f"cluster ledger mismatch; missing={missing}, extra={extra}")
    records = []
    for name, frame in table.groupby("system", sort=True):
        frame = frame.sort_values("radius_kpc")
        profile = RadialBaryonProfile(
            name,
            frame.radius_kpc.to_numpy(float),
            np.power(10.0, frame.log_gbar.to_numpy(float)),
        )
        seed = ClusterSeed(name=name, profile=profile, random_seed=stable_seed(name))
        records.append(
            SystemRecord(
                name=name,
                domain="cluster",
                split=split_lookup[name],
                seed=seed,
                target_g_m_s2=np.power(10.0, frame.log_gobs.to_numpy(float)),
            )
        )
    return records


def split_ledger(records: Sequence[SystemRecord], protocol_path: Path, protocol: dict) -> dict:
    inputs = {
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "SPARC_table": {
            "path": f"{protocol['galaxy_sample']['directory']}/table1.dat",
            "sha256": sha256(ROOT / protocol["galaxy_sample"]["directory"] / "table1.dat"),
        },
        "SPARC_morphology": {
            "path": protocol["galaxy_sample"]["morphology"],
            "sha256": sha256(ROOT / protocol["galaxy_sample"]["morphology"]),
        },
        "cluster_radial": {
            "path": protocol["cluster_sample"]["radial_profile"],
            "sha256": sha256(ROOT / protocol["cluster_sample"]["radial_profile"]),
        },
        "cluster_images": {
            "path": protocol["cluster_sample"]["raw_image_catalog"],
            "sha256": sha256(ROOT / protocol["cluster_sample"]["raw_image_catalog"]),
        },
    }
    by_domain = {}
    for domain in ("galaxy", "cluster"):
        domain_records = [record for record in records if record.domain == domain]
        by_domain[domain] = {
            split: sorted(record.name for record in domain_records if record.split == split)
            for split in ("train", "development", "holdout")
        }
    return {
        "protocol_version": protocol["protocol_version"],
        "status": "locked before P0630 optimizer or holdout scoring",
        "inputs": inputs,
        "seed_policy": protocol["seed_policy"],
        "systems": by_domain,
        "raw_image_holdouts": protocol["cluster_sample"]["raw_image_holdouts"],
        "claim_status": protocol["claim_status"],
    }


def pack_targets(records: Sequence[SystemRecord]) -> PackedTargets:
    gbar, radius, target, surface, r80, reference, indices = [], [], [], [], [], [], []
    names, domains = [], []
    for index, record in enumerate(records):
        profile = record.profile
        context = profile.context()
        count = len(profile.radius_kpc)
        gbar.append(profile.gbar_m_s2)
        radius.append(profile.radius_kpc)
        target.append(record.target_g_m_s2)
        surface.append(np.full(count, context["mean_surface_density_msun_pc2"]))
        r80.append(np.full(count, context["r80_kpc"]))
        reference.append(np.full(count, context["reference_gbar_m_s2"]))
        indices.append(np.full(count, index, dtype=int))
        names.append(record.name)
        domains.append(record.domain)
    return PackedTargets(
        gbar_m_s2=np.concatenate(gbar),
        radius_kpc=np.concatenate(radius),
        target_g_m_s2=np.concatenate(target),
        surface_density_msun_pc2=np.concatenate(surface),
        r80_kpc=np.concatenate(r80),
        reference_gbar_m_s2=np.concatenate(reference),
        system_index=np.concatenate(indices),
        system_domains=np.asarray(domains),
        system_names=tuple(names),
    )


def packed_objective(vector: np.ndarray, packed: PackedTargets, weights: dict[str, float]) -> float:
    prediction = transport_acceleration_from_features(
        packed.gbar_m_s2,
        packed.radius_kpc,
        surface_density_msun_pc2=packed.surface_density_msun_pc2,
        r80_kpc=packed.r80_kpc,
        reference_gbar_m_s2=packed.reference_gbar_m_s2,
        parameters=vector,
    )
    squared = np.square(np.log10(prediction) - np.log10(packed.target_g_m_s2))
    sums = np.bincount(packed.system_index, weights=squared, minlength=len(packed.system_names))
    counts = np.bincount(packed.system_index, minlength=len(packed.system_names))
    per_system = sums / np.maximum(counts, 1)
    value = 0.0
    total_weight = 0.0
    for domain, weight in weights.items():
        selected = packed.system_domains == domain
        if np.any(selected):
            value += float(weight) * float(per_system[selected].mean())
            total_weight += float(weight)
    return value / max(total_weight, 1.0e-30)


def fit_transport(records: Sequence[SystemRecord], protocol: dict, *, seed_offset: int = 0) -> dict:
    packed = pack_targets(records)
    settings = protocol["fit"]
    weights = {key: float(value) for key, value in settings["train_domain_weights"].items()}
    result = differential_evolution(
        lambda vector: packed_objective(vector, packed, weights),
        bounds=TRANSPORT_PARAMETER_BOUNDS,
        seed=int(settings["random_seed"]) + seed_offset,
        popsize=int(settings["population_size_multiplier"]),
        maxiter=int(settings["maximum_iterations"]),
        tol=float(settings["tolerance"]),
        workers=int(settings["workers"]),
        updating="immediate",
        polish=False,
    )
    polished = minimize(
        lambda vector: packed_objective(vector, packed, weights),
        result.x,
        method="L-BFGS-B",
        bounds=TRANSPORT_PARAMETER_BOUNDS,
        options={"maxiter": 2000, "ftol": 1.0e-14, "gtol": 1.0e-9},
    )
    winner = polished if polished.fun <= result.fun else result
    return {
        "parameters": np.asarray(winner.x, dtype=float),
        "parameter_mapping": parameter_mapping(winner.x),
        "objective": float(winner.fun),
        "success": bool(winner.success),
        "message": str(winner.message),
        "function_evaluations": int(result.nfev + getattr(polished, "nfev", 0)),
        "systems": len(records),
        "points": len(packed.gbar_m_s2),
        "domain_counts": dict(Counter(record.domain for record in records)),
    }


def prediction_tables(
    records: Sequence[SystemRecord], parameters: Sequence[float], laws: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows, score_rows = [], []
    for record in records:
        profile = record.profile
        target_g = record.target_g_m_s2
        for law in laws:
            predicted_g = predict_acceleration(
                law,
                profile,
                parameters=parameters if law == "transport" else None,
            )
            log_residual = np.log10(predicted_g) - np.log10(target_g)
            predicted_v = rotation_velocity_km_s(predicted_g, profile.radius_kpc)
            if record.target_velocity_km_s is None:
                target_v = rotation_velocity_km_s(target_g, profile.radius_kpc)
            else:
                target_v = record.target_velocity_km_s
            velocity_residual = predicted_v - target_v
            fractional = predicted_v / target_v - 1.0
            for index in range(len(profile.radius_kpc)):
                point_rows.append(
                    {
                        "domain": record.domain,
                        "system": record.name,
                        "split": record.split,
                        "law": law,
                        "radius_kpc": profile.radius_kpc[index],
                        "gbar_m_s2": profile.gbar_m_s2[index],
                        "target_g_m_s2": target_g[index],
                        "predicted_g_m_s2": predicted_g[index],
                        "target_velocity_km_s": target_v[index],
                        "predicted_velocity_km_s": predicted_v[index],
                        "fractional_velocity_error": fractional[index],
                    }
                )
            score_rows.append(
                {
                    "domain": record.domain,
                    "system": record.name,
                    "split": record.split,
                    "law": law,
                    "points": len(profile.radius_kpc),
                    "log_acceleration_RMSE_dex": float(np.sqrt(np.mean(log_residual**2))),
                    "velocity_RMSE_km_s": float(np.sqrt(np.mean(velocity_residual**2))),
                    "mean_velocity_residual_km_s": float(np.mean(velocity_residual)),
                    "median_absolute_fractional_velocity_error": float(np.median(np.abs(fractional))),
                    "point_fraction_within_10_percent": float(np.mean(np.abs(fractional) <= 0.10)),
                    "point_fraction_within_20_percent": float(np.mean(np.abs(fractional) <= 0.20)),
                }
            )
    return pd.DataFrame(point_rows), pd.DataFrame(score_rows)


def aggregate_scores(scores: pd.DataFrame, domain: str, split: str) -> list[dict]:
    selected = scores[(scores.domain == domain) & (scores.split == split)]
    rows = []
    for law, frame in selected.groupby("law", sort=True):
        rows.append(
            {
                "domain": domain,
                "split": split,
                "law": law,
                "systems": int(len(frame)),
                "equal_system_log_acceleration_RMSE_dex": float(
                    np.sqrt(np.mean(np.square(frame.log_acceleration_RMSE_dex)))
                ),
                "equal_system_velocity_RMSE_km_s": float(
                    np.sqrt(np.mean(np.square(frame.velocity_RMSE_km_s)))
                ),
                "mean_system_point_fraction_within_10_percent": float(
                    frame.point_fraction_within_10_percent.mean()
                ),
                "mean_system_point_fraction_within_20_percent": float(
                    frame.point_fraction_within_20_percent.mean()
                ),
                "median_system_absolute_fractional_velocity_error": float(
                    frame.median_absolute_fractional_velocity_error.median()
                ),
            }
        )
    return rows


def injected_truth_test(protocol: dict) -> dict:
    settings = protocol["injected_truth"]
    system_count = int(settings["systems"])
    point_count = int(settings["points_per_system"])
    population = sobol_galaxy_population(system_count, seed=int(settings["random_seed"]))
    true_parameters = np.asarray(settings["parameters"], dtype=float)
    rng = np.random.default_rng(int(settings["random_seed"]) + 1)
    records = []
    for index in range(system_count):
        is_cluster = index >= 2 * system_count // 3
        mass = 10.0 ** population["log10_mass_msun"][index]
        scale = population["disk_scale_kpc"][index]
        if is_cluster:
            mass *= 2.0e3
            scale *= 30.0
        radius = np.geomspace(max(0.1 * scale, 0.05), 12.0 * scale, point_count)
        enclosed = mass * (1.0 - np.exp(-radius / scale) * (1.0 + radius / scale))
        gbar = G_SI * enclosed * M_SUN_KG / np.square(radius * KPC_M)
        profile = RadialBaryonProfile(f"injected_{index:03d}", radius, gbar)
        target = predict_acceleration("transport", profile, parameters=true_parameters)
        target *= np.power(10.0, rng.normal(0.0, float(settings["noise_dex"]), point_count))
        if is_cluster:
            seed = ClusterSeed(profile.system, profile, random_seed=index)
            domain = "cluster"
        else:
            seed = GalaxySeed(
                profile.system,
                profile,
                0.55 * mass,
                0.10 * mass,
                0.35 * mass,
                scale,
                0.2 * scale,
                2.0 * scale,
                random_seed=index,
            )
            domain = "galaxy"
        records.append(
            SystemRecord(
                name=profile.system,
                domain=domain,
                split="train" if index % 4 else "holdout",
                seed=seed,
                target_g_m_s2=target,
            )
        )
    fit = fit_transport([record for record in records if record.split == "train"], protocol, seed_offset=1000)
    holdout = [record for record in records if record.split == "holdout"]
    _, scores = prediction_tables(holdout, fit["parameters"], ["baryons", "transport"])
    aggregate = aggregate_scores(scores, "galaxy", "holdout") + aggregate_scores(
        scores, "cluster", "holdout"
    )
    transport_scores = scores[scores.law == "transport"]
    heldout_rmse = float(
        np.sqrt(np.mean(np.square(transport_scores.log_acceleration_RMSE_dex)))
    )
    return {
        "known_parameters": parameter_mapping(true_parameters),
        "recovered_parameters": fit["parameter_mapping"],
        "recovered_objective": fit["objective"],
        "heldout_equal_system_log_RMSE_dex": heldout_rmse,
        "gate_max_dex": float(settings["heldout_log_RMSE_dex_max"]),
        "injected_law_recovered_on_holdout": heldout_rmse
        <= float(settings["heldout_log_RMSE_dex_max"]),
        "aggregate": aggregate,
        "identifiability_warning": (
            "Passing predictive recovery does not require every correlated coefficient to be "
            "numerically recovered; it validates the end-to-end forward prediction."
        ),
    }


def particle_checks(records: Sequence[SystemRecord], protocol: dict) -> pd.DataFrame:
    settings = protocol["particle_checks"]
    wanted = set(settings["representative_galaxies"]) | set(
        settings["representative_clusters"]
    )
    rows = []
    for record in records:
        if record.name not in wanted:
            continue
        if record.domain == "galaxy":
            scene = generate_galaxy_scene(
                record.seed, n_particles=int(settings["galaxy_particles"])
            )
            softening = max(0.03 * record.seed.disk_scale_kpc, 0.02)
        else:
            scene = generate_cluster_scene(
                record.seed, n_particles=int(settings["cluster_particles"])
            )
            softening = max(0.02 * record.profile.r80_kpc, 0.2)
        radii = np.geomspace(
            max(record.profile.radius_kpc[0], 2.0 * softening),
            record.profile.radius_kpc[-1],
            12,
        )
        direct = radial_particle_acceleration_m_s2(scene, radii, softening_kpc=softening)
        anchored = record.profile.interpolate(radii)
        finite = (direct > 0.0) & np.isfinite(direct)
        log_rmse = (
            float(np.sqrt(np.mean(np.square(np.log10(direct[finite] / anchored[finite])))))
            if finite.any()
            else math.nan
        )
        rows.append(
            {
                "domain": record.domain,
                "system": record.name,
                "particles": len(scene.positions_kpc),
                "seed_mass_msun": scene.total_mass_msun,
                "profile_force_equivalent_mass_msun": record.profile.total_mass_msun,
                "relative_mass_error": scene.total_mass_msun / record.profile.total_mass_msun - 1.0,
                "direct_force_finite_fraction": float(finite.mean()),
                "direct_vs_radial_anchor_log_RMSE_dex": log_rmse,
                "seed_fingerprint": scene.seed_fingerprint,
                "interpretation": "geometry realization check; radial force knots remain the fidelity path for real scoring",
            }
        )
    if set(row["system"] for row in rows) != wanted:
        raise RuntimeError("a representative particle-scene seed was not generated")
    return pd.DataFrame(rows)


def million_sweep(parameters: Sequence[float], protocol: dict) -> tuple[pd.DataFrame, dict]:
    settings = protocol["million_scale_sweep"]
    population = sobol_galaxy_population(
        int(settings["synthetic_systems"]), seed=int(settings["sobol_seed"])
    )
    predicted = transport_acceleration_from_features(
        population["gbar_m_s2"],
        population["radius_kpc"],
        surface_density_msun_pc2=population["surface_density_msun_pc2"],
        r80_kpc=population["r80_kpc"],
        reference_gbar_m_s2=population["reference_gbar_m_s2"],
        parameters=parameters,
    )
    boost = predicted / population["gbar_m_s2"]
    mass_labels = np.select(
        [population["log10_mass_msun"] < 9.0, population["log10_mass_msun"] < 10.5],
        ["dwarf_logM_lt9", "intermediate_logM_9_10p5"],
        default="giant_logM_ge10p5",
    )
    surface_labels = np.select(
        [population["surface_density_msun_pc2"] < 10.0, population["surface_density_msun_pc2"] < 100.0],
        ["diffuse_Sigma_lt10", "middle_Sigma_10_100"],
        default="dense_Sigma_ge100",
    )
    gas_labels = np.where(population["gas_fraction"] >= 0.5, "gas_rich", "gas_poor")
    bulge_labels = np.where(population["bulge_fraction"] >= 0.2, "bulge_rich", "disk_dominated")
    frame = pd.DataFrame(
        {
            "mass_regime": mass_labels,
            "surface_regime": surface_labels,
            "gas_regime": gas_labels,
            "bulge_regime": bulge_labels,
            "boost": boost,
        }
    )
    rows = []
    for dimensions in (
        ["mass_regime"],
        ["surface_regime"],
        ["gas_regime"],
        ["bulge_regime"],
        ["mass_regime", "surface_regime"],
    ):
        for keys, group in frame.groupby(dimensions, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {name: value for name, value in zip(dimensions, keys, strict=True)}
            row.update(
                {
                    "grouping": "+".join(dimensions),
                    "synthetic_systems": len(group),
                    "mean_acceleration_boost": float(group.boost.mean()),
                    "median_acceleration_boost": float(group.boost.median()),
                    "p05_acceleration_boost": float(group.boost.quantile(0.05)),
                    "p95_acceleration_boost": float(group.boost.quantile(0.95)),
                }
            )
            rows.append(row)
    correlations = {}
    for feature in (
        "log10_mass_msun",
        "gas_fraction",
        "bulge_fraction",
        "surface_density_msun_pc2",
        "concentration",
        "clumpiness",
        "gbar_m_s2",
    ):
        correlations[feature] = float(
            pd.Series(population[feature]).corr(pd.Series(boost), method="spearman")
        )
    return pd.DataFrame(rows), {
        "synthetic_systems": len(boost),
        "boost_distribution": {
            "minimum": float(np.min(boost)),
            "p01": float(np.quantile(boost, 0.01)),
            "median": float(np.median(boost)),
            "p99": float(np.quantile(boost, 0.99)),
            "maximum": float(np.max(boost)),
        },
        "spearman_sensitivity": correlations,
    }


def radial_field(
    law: str,
    profile: RadialBaryonProfile,
    local_protocol: dict,
    settings: dict,
    parameters: Sequence[float],
) -> RadialDeflectionField:
    maximum = float(settings["maximum_radius_kpc"])
    radius_grid = np.geomspace(0.1, maximum, int(settings["radial_grid_points"]))
    acceleration = predict_acceleration(
        law,
        profile,
        radius_grid,
        parameters=parameters if law == "transport" else None,
    )

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(acceleration)))

    impact_arcsec = np.geomspace(0.05, 500.0, int(settings["impact_grid_points"]))
    scale = float(local_protocol["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    physical_alpha = spherical_deflection_radians(
        impact_arcsec * scale,
        lookup,
        maximum_radius_kpc=maximum,
        integration_points=int(settings["integration_points"]),
    )
    return RadialDeflectionField(impact_arcsec, physical_alpha)


def raw_lens_holdouts(
    records: Sequence[SystemRecord], parameters: Sequence[float], protocol: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    settings = protocol["raw_lensing"]
    cluster_settings = protocol["cluster_sample"]
    base_protocol = json.loads((ROOT / cluster_settings["raw_protocol"]).read_text(encoding="utf-8"))
    systems = {item["label"]: item for item in base_protocol["systems"]}
    catalog = pd.read_csv(ROOT / cluster_settings["raw_image_catalog"])
    profiles = {record.name: record.profile for record in records if record.domain == "cluster"}
    score_rows, prediction_tables = [], []
    fit_audit = {}
    laws = ["baryons", "rar", "simple_mond", "transport"]
    for system_index, label in enumerate(cluster_settings["raw_image_holdouts"]):
        system = systems[label]
        local = system_protocol(base_protocol, system)
        local["optimization"]["maximum_function_evaluations"] = int(
            settings["maximum_function_evaluations"]
        )
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        fields = {
            law: radial_field(law, profiles[label], local, settings, parameters) for law in laws
        }
        fields["baryons_GR"] = fields["baryons"]
        lens = RawLens(local, fields)
        model_names = [*laws, "GR_plus_cluster_halo"]
        for model_index, model in enumerate(model_names):
            fitted = lens.fit(
                model,
                training,
                starts=int(settings["starts"]),
                seed=int(settings["random_seed"]) + 1000 * system_index + model_index,
            )
            training_prediction = lens.exact_predictions(
                model, fitted["result"].x, fitted["sources"], training, stage="training"
            )
            heldout_prediction = lens.exact_predictions(
                model, fitted["result"].x, fitted["sources"], heldout, stage="holdout"
            )
            for table in (training_prediction, heldout_prediction):
                table.insert(0, "system_label", label)
                if "model" not in table.columns:
                    table.insert(1, "model", model)
                prediction_tables.append(table)
            training_score = score(training_prediction, lens.sigma, free_parameters=len(fitted["result"].x))
            heldout_score = score(heldout_prediction, lens.sigma)
            score_rows.append(
                {
                    "system": label,
                    "model": model,
                    "gravity_parameters_per_object": 2 if model == "GR_plus_cluster_halo" else 0,
                    "geometry_parameters_per_object": len(fitted["result"].x),
                    "training_images": len(training),
                    "heldout_images": len(heldout),
                    "training_exact_RMS_arcsec": training_score.get("exact_radial_RMS_arcsec"),
                    "training_all_roots_converged": training_score.get("all_roots_converged"),
                    "heldout_exact_RMS_arcsec": heldout_score.get("exact_radial_RMS_arcsec"),
                    "heldout_all_roots_converged": heldout_score.get("all_roots_converged"),
                    "heldout_maximum_residual_arcsec": heldout_score.get("maximum_radial_residual_arcsec"),
                    "geometry_at_boundary": bool(any(near_bound(model, fitted["result"].x).values())),
                }
            )
            fit_audit[f"{label}:{model}"] = {
                "success": bool(fitted["result"].success),
                "message": str(fitted["result"].message),
                "function_evaluations": int(fitted["result"].nfev),
                "parameters": fitted["result"].x,
                "geometry_at_boundary": near_bound(model, fitted["result"].x),
            }
    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    return scores, predictions, fit_audit


def raw_aggregate(scores: pd.DataFrame) -> list[dict]:
    rows = []
    for model, frame in scores.groupby("model", sort=True):
        finite = frame.heldout_exact_RMS_arcsec.notna()
        rows.append(
            {
                "model": model,
                "clusters": len(frame),
                "heldout_images": int(frame.heldout_images.sum()),
                "clusters_training_all_roots": int(
                    (frame.training_all_roots_converged == True).sum()
                ),
                "clusters_all_roots": int((frame.heldout_all_roots_converged == True).sum()),
                "all_training_roots_converged": bool(
                    (frame.training_all_roots_converged == True).all()
                ),
                "all_roots_converged": bool((frame.heldout_all_roots_converged == True).all()),
                "equal_cluster_heldout_RMS_arcsec": (
                    float(np.sqrt(np.mean(np.square(frame.loc[finite, "heldout_exact_RMS_arcsec"]))))
                    if finite.any()
                    else None
                ),
                "gravity_parameters_per_object": int(frame.gravity_parameters_per_object.max()),
            }
        )
    return rows


def make_figure(
    galaxy_aggregate: list[dict],
    cluster_aggregate: list[dict],
    raw_summary: list[dict],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    g = pd.DataFrame(galaxy_aggregate)
    c = pd.DataFrame(cluster_aggregate)
    r = pd.DataFrame(raw_summary)
    order = ["baryons", "rar", "simple_mond", "transport"]
    g = g.set_index("law").reindex(order).dropna(how="all").reset_index()
    c = c.set_index("law").reindex(order).dropna(how="all").reset_index()
    axes[0].bar(g.law, g.equal_system_velocity_RMSE_km_s)
    axes[0].set(title="Held-back whole galaxies", ylabel="equal-galaxy velocity RMSE (km/s)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(c.law, c.equal_system_log_acceleration_RMSE_dex)
    axes[1].set(title="Held-back cluster radial profiles", ylabel="equal-cluster log RMSE (dex)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    raw_order = ["baryons", "rar", "simple_mond", "transport", "GR_plus_cluster_halo"]
    r = r.set_index("model").reindex(raw_order).dropna(how="all").reset_index()
    axes[2].bar(r.model, r.equal_cluster_heldout_RMS_arcsec)
    axes[2].set(title="Held-back raw cluster images", ylabel="equal-cluster image RMS (arcsec)")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_actuals_figure(
    point_predictions: pd.DataFrame,
    records: Sequence[SystemRecord],
    output: Path,
) -> list[str]:
    """Descriptive, target-independent selection of representative holdouts."""
    galaxy = [record for record in records if record.domain == "galaxy" and record.split == "holdout"]
    galaxy = sorted(galaxy, key=lambda record: record.profile.total_mass_msun)
    indices = np.linspace(0, len(galaxy) - 1, 4).round().astype(int)
    galaxy_names = [galaxy[index].name for index in indices]
    cluster_names = sorted(
        record.name
        for record in records
        if record.domain == "cluster" and record.split == "holdout"
    )
    selected_names = [*galaxy_names, *cluster_names]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    colors = {
        "baryons": "0.55",
        "rar": "#2b8cbe",
        "simple_mond": "#756bb1",
        "transport": "#d95f0e",
    }
    for column, name in enumerate(galaxy_names):
        ax = axes[0, column]
        block = point_predictions[
            (point_predictions.system == name) & (point_predictions.domain == "galaxy")
        ]
        target = block[block.law == "transport"]
        ax.errorbar(
            target.radius_kpc,
            target.target_velocity_km_s,
            fmt="o",
            markersize=3.5,
            color="black",
            label="actual",
        )
        for law in colors:
            local = block[block.law == law]
            ax.plot(
                local.radius_kpc,
                local.predicted_velocity_km_s,
                color=colors[law],
                linewidth=1.5,
                label=law,
            )
        ax.set_xscale("log")
        ax.set(title=name, xlabel="radius (kpc)", ylabel="speed (km/s)")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=7)
    for column, name in enumerate(cluster_names):
        ax = axes[1, column]
        block = point_predictions[
            (point_predictions.system == name) & (point_predictions.domain == "cluster")
        ]
        target = block[block.law == "transport"]
        ax.plot(
            target.radius_kpc,
            np.log10(target.target_g_m_s2),
            "o",
            markersize=4,
            color="black",
            label="actual radial target",
        )
        for law in colors:
            local = block[block.law == law]
            ax.plot(
                local.radius_kpc,
                np.log10(local.predicted_g_m_s2),
                color=colors[law],
                linewidth=1.5,
                label=law,
            )
        ax.set_xscale("log")
        ax.set(title=name, xlabel="radius (kpc)", ylabel="log10 acceleration (m/s2)")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=7)
    figure.suptitle(
        "P0630 whole-system holdouts (galaxies chosen only by baryonic mass quantile)",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return selected_names


def find_row(rows: list[dict], key: str, value: str) -> dict:
    return next(row for row in rows if row[key] == value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p0630_synthetic_universe_protocol.json")
    parser.add_argument("--skip-raw-lensing", action="store_true")
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    config_path = ROOT / arguments.config
    protocol = load_protocol(config_path)
    if arguments.quick:
        protocol = json.loads(json.dumps(protocol))
        protocol["fit"]["maximum_iterations"] = 8
        protocol["fit"]["population_size_multiplier"] = 5
        protocol["million_scale_sweep"]["synthetic_systems"] = 4096
        protocol["raw_lensing"]["starts"] = 1
        protocol["raw_lensing"]["integration_points"] = 200
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    print("loading baryon-only galaxy and cluster seeds", flush=True)
    galaxy_records = load_galaxy_records(protocol)
    cluster_records = load_cluster_records(protocol)
    records = [*galaxy_records, *cluster_records]
    ledger = split_ledger(records, config_path, protocol)
    (output / protocol["outputs"]["split_ledger"]).write_text(
        json.dumps(json_safe(ledger), indent=2) + "\n", encoding="utf-8"
    )

    print("running injected-law recovery", flush=True)
    injection = injected_truth_test(protocol)
    (output / protocol["outputs"]["injection_report"]).write_text(
        json.dumps(json_safe(injection), indent=2) + "\n", encoding="utf-8"
    )

    train = [record for record in records if record.split == "train"]
    development = [record for record in records if record.split == "development"]
    holdout = [record for record in records if record.split == "holdout"]
    print(f"fitting development law on {len(train)} systems", flush=True)
    development_fit = fit_transport(train, protocol)
    _, development_scores = prediction_tables(
        development,
        development_fit["parameters"],
        ["baryons", "rar", "simple_mond", "transport"],
    )
    print(f"locking final law on {len(train) + len(development)} non-holdout systems", flush=True)
    final_fit = fit_transport([*train, *development], protocol, seed_offset=10)

    print("opening whole-system galaxy and radial-cluster holdouts", flush=True)
    point_predictions, system_scores = prediction_tables(
        records,
        final_fit["parameters"],
        ["baryons", "rar", "simple_mond", "transport"],
    )
    galaxy_points = point_predictions[point_predictions.domain == "galaxy"].copy()
    cluster_points = point_predictions[point_predictions.domain == "cluster"].copy()
    galaxy_scores = system_scores[system_scores.domain == "galaxy"].copy()
    cluster_scores = system_scores[system_scores.domain == "cluster"].copy()
    galaxy_points.to_csv(output / protocol["outputs"]["galaxy_predictions"], index=False)
    cluster_points.to_csv(output / protocol["outputs"]["cluster_predictions"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_scores"], index=False)
    cluster_scores.to_csv(output / protocol["outputs"]["cluster_scores"], index=False)

    print("verifying representative particle scenes", flush=True)
    particles = particle_checks(records, protocol)
    particles.to_csv(output / protocol["outputs"]["particle_checks"], index=False)
    print("running million-scale structured counterfactual sweep", flush=True)
    sweep, sweep_report = million_sweep(final_fit["parameters"], protocol)
    sweep.to_csv(output / protocol["outputs"]["million_sweep"], index=False)

    raw_scores, raw_predictions, raw_audit = pd.DataFrame(), pd.DataFrame(), {}
    if not arguments.skip_raw_lensing:
        print("opening P0630 raw image-plane cluster holdouts", flush=True)
        raw_scores, raw_predictions, raw_audit = raw_lens_holdouts(
            records, final_fit["parameters"], protocol
        )
        raw_scores.to_csv(output / protocol["outputs"]["raw_lens_scores"], index=False)
        raw_predictions.to_csv(
            output / protocol["outputs"]["raw_lens_predictions"], index=False
        )

    galaxy_holdout = aggregate_scores(system_scores, "galaxy", "holdout")
    cluster_holdout = aggregate_scores(system_scores, "cluster", "holdout")
    development_summary = (
        aggregate_scores(development_scores, "galaxy", "development")
        + aggregate_scores(development_scores, "cluster", "development")
    )
    raw_summary = raw_aggregate(raw_scores) if len(raw_scores) else []
    actuals_figure_path = output / "heldout_actuals.png"
    actuals_figure_systems = make_actuals_figure(
        point_predictions, records, actuals_figure_path
    )
    candidate_galaxy = find_row(galaxy_holdout, "law", "transport")
    rar_galaxy = find_row(galaxy_holdout, "law", "rar")
    candidate_cluster = find_row(cluster_holdout, "law", "transport")
    rar_cluster = find_row(cluster_holdout, "law", "rar")
    if raw_summary:
        candidate_raw = find_row(raw_summary, "model", "transport")
        halo_raw = find_row(raw_summary, "model", "GR_plus_cluster_halo")
    else:
        candidate_raw = halo_raw = None

    universal_fit = {
        "development_fit": development_fit,
        "final_fit_before_holdout_open": final_fit,
        "universal_parameter_count": len(TRANSPORT_PARAMETER_NAMES),
        "per_object_gravity_parameters": 0,
        "formula": protocol["gravity_laws"]["candidate_equation"],
    }
    (output / protocol["outputs"]["universal_fit"]).write_text(
        json.dumps(json_safe(universal_fit), indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed P0630 seeded simulator and project-spent cross-domain holdouts",
        "protocol": ledger["inputs"]["protocol"],
        "split_counts": {
            domain: {split: len(names) for split, names in splits.items()}
            for domain, splits in ledger["systems"].items()
        },
        "simulator": {
            "seed_target_separation": protocol["seed_policy"],
            "particle_scene_checks": particles.to_dict(orient="records"),
            "gravity_plugins": ["baryons", "rar", "simple_mond", "transport"],
            "universal_fit": universal_fit,
            "injected_truth": injection,
            "million_scale_sweep": sweep_report,
            "descriptive_heldout_actuals_figure": {
                "path": str(actuals_figure_path.relative_to(ROOT)).replace("\\", "/"),
                "systems": actuals_figure_systems,
                "selection": "galaxy baryonic-mass quantiles plus every radial-cluster holdout; no target residual used",
            },
        },
        "development_replay": development_summary,
        "heldout": {
            "galaxy": galaxy_holdout,
            "cluster_radial": cluster_holdout,
            "raw_cluster_images": raw_summary,
            "raw_fit_audit": raw_audit,
        },
        "comparison": {
            "transport_to_RAR_galaxy_RMSE_ratio": candidate_galaxy[
                "equal_system_velocity_RMSE_km_s"
            ]
            / rar_galaxy["equal_system_velocity_RMSE_km_s"],
            "transport_to_RAR_cluster_radial_RMSE_ratio": candidate_cluster[
                "equal_system_log_acceleration_RMSE_dex"
            ]
            / rar_cluster["equal_system_log_acceleration_RMSE_dex"],
            "transport_to_per_cluster_halo_raw_RMS_ratio": (
                candidate_raw["equal_cluster_heldout_RMS_arcsec"]
                / halo_raw["equal_cluster_heldout_RMS_arcsec"]
                if candidate_raw and halo_raw
                else None
            ),
        },
        "verdict": {
            "end_to_end_injected_law_recovery_pass": injection[
                "injected_law_recovered_on_holdout"
            ],
            "candidate_generates_heldout_galaxy_actuals_better_than_RAR": candidate_galaxy[
                "equal_system_velocity_RMSE_km_s"
            ]
            < rar_galaxy["equal_system_velocity_RMSE_km_s"],
            "candidate_generates_heldout_cluster_radial_actuals_better_than_RAR": candidate_cluster[
                "equal_system_log_acceleration_RMSE_dex"
            ]
            < rar_cluster["equal_system_log_acceleration_RMSE_dex"],
            "candidate_matches_per_cluster_halo_raw_images": (
                candidate_raw["all_training_roots_converged"]
                and candidate_raw["all_roots_converged"]
                and candidate_raw["equal_cluster_heldout_RMS_arcsec"]
                <= halo_raw["equal_cluster_heldout_RMS_arcsec"]
                if candidate_raw and halo_raw
                else None
            ),
            "validated_new_field_theory": False,
        },
        "claim_limits": [
            protocol["claim_status"],
            protocol["cluster_sample"]["disclosure"],
            "Particle scenes test geometry and direct vector summation; real-system scores use higher-fidelity measured baryonic force knots rather than claiming a full hydrodynamic formation history.",
            "The Tian cluster total-acceleration targets are model-derived radial summaries; raw image positions are the stronger cluster observable check.",
            "The compact halo raw comparator fits object-specific halo parameters and is a performance ceiling, not a parameter-matched universal law.",
            "The transport formula is phenomenological and has not been derived from a covariant action or conservation law.",
        ],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    if raw_summary:
        make_figure(
            galaxy_holdout,
            cluster_holdout,
            raw_summary,
            output / protocol["outputs"]["figure"],
        )
    lines = [
        "# P0630 seeded synthetic-universe result",
        "",
        f"Injected-law predictive recovery: **{injection['injected_law_recovered_on_holdout']}** "
        f"({injection['heldout_equal_system_log_RMSE_dex']:.5f} dex).",
        "",
        f"Universal transport parameters: **{len(TRANSPORT_PARAMETER_NAMES)}**; per-object gravity parameters: **0**.",
        f"Held-back galaxies: transport **{candidate_galaxy['equal_system_velocity_RMSE_km_s']:.3f} km/s**, "
        f"RAR **{rar_galaxy['equal_system_velocity_RMSE_km_s']:.3f} km/s**.",
        f"Held-back cluster radial profiles: transport **{candidate_cluster['equal_system_log_acceleration_RMSE_dex']:.4f} dex**, "
        f"RAR **{rar_cluster['equal_system_log_acceleration_RMSE_dex']:.4f} dex**.",
    ]
    if candidate_raw and halo_raw:
        lines.append(
            f"Held-back raw cluster images: transport **{candidate_raw['equal_cluster_heldout_RMS_arcsec']:.3f} arcsec**, "
            f"per-cluster compact halo **{halo_raw['equal_cluster_heldout_RMS_arcsec']:.3f} arcsec**."
        )
        lines.append(
            f"Transport exact training roots all converged: **{candidate_raw['all_training_roots_converged']}**; "
            f"held-out roots all converged: **{candidate_raw['all_roots_converged']}**."
        )
    lines.extend(
        [
            "",
            "These are project-spent replay holdouts, not untouched external validation.",
        ]
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
