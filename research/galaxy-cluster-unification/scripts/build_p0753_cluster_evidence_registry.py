from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import rfc8785

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "p0753_cluster_evidence_registry" / "registry.json"
DEFAULT_HOSTED_OUTPUT = ROOT / "hosted-simulator" / "data" / "resolved-cluster-evidence-v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def parse_comparator_role(role: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        r"(?P<method>.+)_v(?P<version>\d+)_(?P<component>kappa|x-arcsec-deflect|y-arcsec-deflect)",
        role,
    )
    if not match:
        raise ValueError(f"unknown comparator role: {role}")
    return match["method"], f"v{match['version']}", match["component"]


def load_comparators(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["domain"] != "cluster_comparator":
                continue
            method, version, component = parse_comparator_role(row["role"])
            grouped[(row["system"], method, version)].append(
                {
                    "component": component,
                    "bytes": int(row["bytes"]),
                    "sha256": row["sha256"],
                    "sourceUrl": row["url"],
                }
            )

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_order = {"kappa": 0, "x-arcsec-deflect": 1, "y-arcsec-deflect": 2}
    for (system, method, version), components in sorted(grouped.items()):
        components.sort(key=lambda item: component_order[item["component"]])
        result[system].append(
            {
                "method": method,
                "version": version,
                "scientificRole": "model_derived_discovery_target",
                "components": components,
                "bundleSha256": canonical_sha256(components),
                "mayBeUsedFor": ["inverse_hypothesis_generation", "published_model_comparator"],
                "mayNotBeUsedAs": ["baryonic_input", "raw_observation", "prospective_holdout"],
            }
        )
    return result


def build_registry() -> dict[str, Any]:
    p0633_path = ROOT / "configs" / "p0633_external_validation_preregistration.json"
    p0640_provenance_path = (
        ROOT / "results" / "p0640_relics_input_acquisition" / "provenance.json"
    )
    p0641_path = ROOT / "results" / "p0641_registered_cluster_baryon_maps" / "report.json"
    p0710_path = ROOT / "results" / "p0710_external_target_acquisition" / "provenance.csv"
    p0710_report_path = ROOT / "results" / "p0710_external_target_acquisition" / "report.json"
    p0713_path = ROOT / "results" / "p0713_external_cluster_readiness_audit" / "report.json"
    p0735_path = ROOT / "results" / "p0735_raw_multiple_image_adapter" / "report.json"

    p0633 = read_json(p0633_path)
    p0640 = read_json(p0640_provenance_path)
    p0641 = read_json(p0641_path)
    p0710_report = read_json(p0710_report_path)
    p0713 = read_json(p0713_path)
    p0735 = read_json(p0735_path)

    selected = {item["id"]: item for item in p0633["cluster_validation"]["systems"]}
    baryon_maps = {item["system"]: item for item in p0641["systems"]}
    readiness = {item["cluster"]: item for item in p0713["cluster_rows"]}
    adapters = p0735["catalogImportAudit"]["clusters"]
    comparators = load_comparators(p0710_path)

    open_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_containers: dict[str, dict[str, Any]] = {}
    for record in p0640["records"]:
        if record["kind"] == "open_baryon":
            open_inputs[record["system"]].append(record)
        elif record["kind"] == "sealed_constraint_container":
            public_record = {
                "role": "raw_observation_source_container",
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "sourceUrl": record["url"],
                "originalAcquisitionState": "sealed_and_hash_only",
                "currentSampleState": "opened_and_spent_by_P0710_to_P0714",
            }
            for system in record["system"].split(","):
                source_containers[system] = public_record

    systems: list[dict[str, Any]] = []
    for system_id in sorted(selected):
        meta = selected[system_id]
        mass = baryon_maps[system_id]
        raw = readiness[system_id]
        source_rows = open_inputs[system_id]
        role_counts: dict[str, int] = defaultdict(int)
        role_bytes: dict[str, int] = defaultdict(int)
        for row in source_rows:
            role_counts[row["role"]] += 1
            role_bytes[row["role"]] += int(row["bytes"])

        raw_blockers = ["spent_sample_not_prospective_holdout"]
        if not raw["family_gate_passed"]:
            raw_blockers.append("fewer_than_three_secure_source_families")
        if not raw["spectroscopic_gate_passed"]:
            raw_blockers.append("no_secure_spectroscopic_source_family")
        if not raw["image_gate_passed"]:
            raw_blockers.append("fewer_than_eight_secure_images")
        raw_blockers.append("published_per_image_position_uncertainties_not_registered")

        adapter = adapters.get(system_id)
        comparator_rows = comparators[system_id]
        systems.append(
            {
                "id": system_id,
                "label": meta["label"],
                "archiveSlug": meta["slug"],
                "redshift": meta["redshift"],
                "sampleState": "spent_development",
                "baryonicEvidence": {
                    "scientificRole": "baryonic_input",
                    "state": "registered_projected_mass_map_with_uncertainty_brackets",
                    "ready": True,
                    "grid": {
                        "dimensions": 2,
                        "shape": [mass["grid_size"], mass["grid_size"]],
                        "cellKpc": mass["cell_kpc"],
                    },
                    "selectedStellarMembers": mass["selected_members"],
                    "stellarMassMsolar": mass["stellar_mass_msun"],
                    "gasMassMsolar": mass["gas_mass_msun"],
                    "gasMassSigmaMsolar": mass["gas_mass_sigma_msun"],
                    "totalBaryonMassMsolar": mass["baryon_mass_msun"],
                    "gasStellarCentroidOffsetKpc": mass["gas_stellar_centroid_offset_kpc"],
                    "mapSha256": mass["map_sha256"],
                    "sourceProducts": [
                        {
                            "role": role,
                            "scientificMeaning": {
                                "f160w_image": "stellar_light_morphology_proxy",
                                "member_catalog": "photometric_membership_and_flux_input",
                                "segmentation": "member_pixel_assignment",
                                "chandra_center_image": "xray_gas_morphology_proxy_not_gas_mass",
                            }[role],
                            "count": role_counts[role],
                            "bytes": role_bytes[role],
                        }
                        for role in sorted(role_counts)
                    ],
                    "sharedUncertaintyPolicy": {
                        "stellarMassToLightSolar": [0.5, 0.8, 1.1],
                        "xrayBrightnessToDensityExponent": [0.4, 0.5, 0.6],
                        "perClusterGravityParameters": 0,
                    },
                    "limitations": [
                        "projected_2d_mass_map_not_unique_3d_reconstruction",
                        "single_band_stellar_population_calibration",
                        "xray_gas_depth_and_emissivity_approximation",
                        "no_independently_calibrated_intracluster_light_product",
                    ],
                },
                "modelDerivedLensingEvidence": {
                    "scientificRole": "model_derived_discovery_target",
                    "readyForInverseHypothesisGeneration": len(comparator_rows) >= 2,
                    "independentPublishedMethods": len(comparator_rows),
                    "models": comparator_rows,
                    "warning": "These maps include lens-model assumptions and dark-matter-like components. They are discovery targets and comparators, not raw observations.",
                },
                "rawLensingEvidence": {
                    "scientificRole": "raw_observation",
                    "sourceContainer": source_containers[system_id],
                    "secureFamilies": raw["secure_families"],
                    "spectroscopicFamilies": raw["spectroscopic_families"],
                    "secureImages": raw["secure_images"],
                    "frozenReadinessGatePassed": raw["ready"],
                    "coordinateSerializationValidated": adapter is not None,
                    "scoreReadyNow": False,
                    "prospectiveHoldoutEligible": False,
                    "blockers": raw_blockers,
                },
                "readiness": {
                    "baryonicFieldInput": True,
                    "inverseHaloShapeDiscovery": len(comparator_rows) >= 2,
                    "rawForwardLensingScore": False,
                    "blindTheoryValidation": False,
                },
            }
        )

    core = {
        "schemaVersion": "sigma-resolved-cluster-evidence/1",
        "registryVersion": "P0753-1.0.0",
        "evidenceClass": "spent_cluster_development_registry",
        "purpose": "Expose exactly which resolved cluster products can support baryonic field solves, inverse halo-response hypothesis generation, and raw forward lensing tests without confusing their scientific roles.",
        "sample": {
            "survey": "RELICS",
            "systemCount": len(systems),
            "sampleState": "spent",
            "prospectiveHoldoutSystems": 0,
            "registeredBaryonMaps": sum(item["readiness"]["baryonicFieldInput"] for item in systems),
            "inverseDiscoverySystems": sum(
                item["readiness"]["inverseHaloShapeDiscovery"] for item in systems
            ),
            "rawCatalogReadinessGateSystems": sum(
                item["rawLensingEvidence"]["frozenReadinessGatePassed"] for item in systems
            ),
            "rawForwardScoreReadySystems": sum(
                item["readiness"]["rawForwardLensingScore"] for item in systems
            ),
        },
        "roleContract": {
            "baryonic_input": "May source a forward gravity solve; must remain independent of the gravity formula and lensing target.",
            "model_derived_discovery_target": "May reveal candidate response geometry in an inverse analysis; cannot validate a theory or substitute for raw lensing observations.",
            "raw_observation": "May score a frozen forward prediction only when provenance, uncertainty, selection, coordinates, and nuisance policy are complete.",
        },
        "systems": systems,
        "platformUse": {
            "availableNow": [
                "Select any of four content-addressed baryonic mass maps for a local 2D field solve.",
                "Use two published lens-model methods per cluster as explicitly model-derived inverse targets.",
                "Run five deterministic null families before interpreting a learned baryon-to-response kernel.",
                "Serialize 65 secure raw image coordinates in 18 families for AS295 and PLCKG287 through the local adapter.",
            ],
            "notEstablished": [
                "No system in this registry is a new blind holdout.",
                "No raw cluster target is score-ready with registered per-image positional uncertainties.",
                "A reconstructed halo-like map is not evidence that gravity physically flowed along the inferred kernel.",
                "The projected baryon maps do not uniquely determine line-of-sight three-dimensional structure.",
                "The Vercel deployment does not execute heavy field or lensing jobs.",
            ],
        },
        "requirementsForUsefulSigmaOrInverseHaloResearch": [
            {
                "priority": 1,
                "deliverable": "prospective_cluster_release",
                "acceptance": "At least four untouched clusters selected before outcomes are opened, each with at least three secure source families, one spectroscopic family, eight images, registered astrometric uncertainties, and a fixed detectability policy.",
            },
            {
                "priority": 2,
                "deliverable": "uncertainty_aware_complete_baryons",
                "acceptance": "Per-cluster ensembles include multiband stellar mass, calibrated hot-gas mass and deprojection, intracluster light or an explicit bound, line-of-sight structure, WCS, PSF, masks, covariance, provenance, and license metadata.",
            },
            {
                "priority": 3,
                "deliverable": "inverse_to_forward_freeze",
                "acceptance": "Learn only on spent model-derived targets, beat every declared null, freeze one compact universal response law and all parameters, then remove halo targets before predicting raw held-out images.",
            },
            {
                "priority": 4,
                "deliverable": "joint_cross_domain_report",
                "acceptance": "One deterministic report compares galaxies, cluster image positions and topology, Solar-System limits, dark-matter comparators, MOND/RAR, uncertainty, numerical convergence, and parameter counts without cross-domain averaging rescuing a failed gate.",
            },
            {
                "priority": 5,
                "deliverable": "production_scientific_execution",
                "acceptance": "Durable queue, isolated workers, metadata database, object storage, auth, quotas, cancellation, retries, license enforcement, audit logs, and signed manifests reproduce local artifact hashes.",
            },
        ],
        "provenance": {
            "relicsArchive": "https://archive.stsci.edu/hlsp/relics",
            "relicsArchiveDoi": "10.17909/T9SP45",
            "redistribution": "Hosted registry contains metadata and hashes only; source archive terms apply to the underlying products.",
            "sourceFileSha256": {
                "p0633Preregistration": file_sha256(p0633_path),
                "p0640Acquisition": file_sha256(p0640_provenance_path),
                "p0641BaryonMaps": file_sha256(p0641_path),
                "p0710Acquisition": file_sha256(p0710_path),
                "p0710Report": file_sha256(p0710_report_path),
                "p0713Readiness": file_sha256(p0713_path),
                "p0735RawAdapter": file_sha256(p0735_path),
            },
            "p0710SampleSpent": p0710_report["P0633_sample_now_spent"],
        },
        "claimBoundary": [
            "This registry is an evidence inventory, not a successful cluster-lensing result.",
            "Inverse recovery against published lens maps generates candidate field laws; only forward prediction of unused raw observations can test those laws.",
            "All four P0633 clusters are spent and cannot be promoted back into holdouts.",
            "No per-cluster gravity parameter appears in the registered baryonic maps or readiness accounting.",
        ],
    }
    registry = {**core, "registrySha256": canonical_sha256(core)}
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    assert registry["schemaVersion"] == "sigma-resolved-cluster-evidence/1"
    assert len(registry["systems"]) == 4
    assert registry["sample"]["registeredBaryonMaps"] == 4
    assert registry["sample"]["inverseDiscoverySystems"] == 4
    assert registry["sample"]["rawCatalogReadinessGateSystems"] == 2
    assert registry["sample"]["rawForwardScoreReadySystems"] == 0
    assert registry["sample"]["prospectiveHoldoutSystems"] == 0
    assert all(item["sampleState"] == "spent_development" for item in registry["systems"])
    assert all(
        item["baryonicEvidence"]["scientificRole"] == "baryonic_input"
        for item in registry["systems"]
    )
    assert all(
        item["modelDerivedLensingEvidence"]["scientificRole"]
        == "model_derived_discovery_target"
        for item in registry["systems"]
    )
    assert all(
        item["rawLensingEvidence"]["scientificRole"] == "raw_observation"
        for item in registry["systems"]
    )
    assert all(
        item["baryonicEvidence"]["sharedUncertaintyPolicy"]["perClusterGravityParameters"]
        == 0
        for item in registry["systems"]
    )
    assert all(
        len(item["modelDerivedLensingEvidence"]["models"]) == 2
        for item in registry["systems"]
    )
    assert all(not item["readiness"]["rawForwardLensingScore"] for item in registry["systems"])
    core = {key: value for key, value in registry.items() if key != "registrySha256"}
    assert canonical_sha256(core) == registry["registrySha256"]


def write_registry(registry: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P0753 cluster evidence registry")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hosted-output", type=Path, default=DEFAULT_HOSTED_OUTPUT)
    args = parser.parse_args()

    registry = build_registry()
    write_registry(registry, args.output)
    write_registry(registry, args.hosted_output)
    print(
        json.dumps(
            {
                "systems": registry["sample"]["systemCount"],
                "inverseDiscoverySystems": registry["sample"]["inverseDiscoverySystems"],
                "rawForwardScoreReadySystems": registry["sample"]["rawForwardScoreReadySystems"],
                "registrySha256": registry["registrySha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
