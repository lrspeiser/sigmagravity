"""Build the combined tested-formula and published-prior-art registry.

The scored observational laws remain owned by ``build_formula_scorecard.py``.
This script does not retype or reinterpret their measurements.  It adds:

* final-theory eligibility flags, especially convenience-switch detection;
* links from every tested law to the closest published formula families;
* a curated primary-source prior-art registry; and
* an inventory of formula/action fragments in every Sigma v1--v17 protocol.

The result is deterministic and records hashes of both source registries.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "results" / "formula_scorecard" / "formula_scorecard.json"
PUBLISHED = ROOT / "configs" / "published_formula_registry.json"
ADDENDA = ROOT / "configs" / "project_formula_addenda.json"
OUT = ROOT / "results" / "formula_prior_art_registry"
MARKDOWN = ROOT / "docs" / "FORMULA_AND_PRIOR_ART_REGISTRY.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(*values: object) -> str:
    return " ".join(str(value).lower() for value in values if value is not None)


def closest_published_families(text: str) -> list[str]:
    """Return mechanistic overlaps, not claims of equation identity."""

    rules: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("newton", "g = g_bar", "poisson"), ("NEWTON-POISSON",)),
        (("nfw",), ("NFW-HALO",)),
        (("compact cluster halo",), ("NFW-HALO", "BURKERT-HALO", "EINASTO-HALO")),
        (("burkert",), ("BURKERT-HALO",)),
        (("isothermal tail",), ("PSEUDO-ISOTHERMAL-HALO",)),
        (("rar",), ("RAR-EMPIRICAL", "MOND-ALGEBRAIC")),
        (("mond", "low-acceleration", "a0"), ("MOND-ALGEBRAIC",)),
        (("aqual", "p-laplacian"), ("AQUAL",)),
        (("qumond", "source routing"), ("QUMOND",)),
        (("emond", "potential-dependent", "potential-moving"), ("EMOND",)),
        (
            (
                "refracted",
                "permittivity",
                "epsilon(rho",
                "density-only",
                "rg with",
                "rg +",
                "rg x",
                "sigma/rg",
                "partitioned rg",
                "g-rho",
            ),
            ("REFRACTED-GRAVITY", "COVARIANT-RG"),
        ),
        (
            ("void", "cf4", "environment law", "environmental", "density gate"),
            ("CHAMELEON", "SYMMETRON"),
        ),
        (("yukawa", "proca", "emog", "running g"), ("STVG-MOG",)),
        (("conformal",), ("CONFORMAL-GRAVITY", "BRANS-DICKE")),
        (("disformal",), ("DISFORMAL-METRIC",)),
        (
            (
                "curvature log",
                "curvature power",
                "curvature root",
                "curvature additive",
                "spherical spacetime",
            ),
            ("CONFORMAL-GRAVITY", "MULTIFRACTIONAL-SPACETIME"),
        ),
        (
            ("nonlocal", "memory", "path", "diffusion", "helmholtz", "catch-up"),
            ("NONLOCAL-GRAVITY",),
        ),
        (("fractional",), ("FRACTIONAL-LAPLACIAN-MOND",)),
        (("variable exponent", "distance exponent"), ("MULTIFRACTIONAL-SPACETIME",)),
        (("symmetron",), ("SYMMETRON",)),
        (("chameleon",), ("CHAMELEON",)),
        (("galileon", "vainshtein", "cubic hessian"), ("GALILEON-VAINSHTEIN",)),
        (("spin2", "spin-2", "drgt", "massive gravity"), ("DRGT-MASSIVE-GRAVITY",)),
        (("dhost", "degeneracy", "degenerate"), ("DHOST",)),
        (("nonmetricity", "f(q)", "stegr"), ("SYMMETRIC-TELEPARALLEL",)),
        (("aest",), ("AEST-MOND", "EINSTEIN-AETHER")),
        (
            (
                "aether",
                "c_13",
                "c_14",
                "c_123",
                "clock",
                "preferred frame",
                "spin-1 characteristic",
                "tensor alignment",
                "tensor dominance",
                "tensor competition",
                "tensor isotropic",
                "vector completion",
                "coherence completion",
            ),
            ("EINSTEIN-AETHER", "AEST-MOND"),
        ),
        (("scalar metric slip", "two-potential", "metric lens"), ("TEVES",)),
        (("coherence length",), ("AEST-MOND", "EMOND")),
        (("covariant weak-field metric",), ("BRANS-DICKE", "TEVES")),
        (("member vector", "member tidal", "tidal metric"), ("GR-EINSTEIN", "TEVES")),
        (("polarization",), ("DIPOLAR-POLARIZATION",)),
        (("elastic",), ("EMERGENT-GRAVITY",)),
        (
            (
                "spatial stress",
                "stress-energy",
                "theta_b",
                "q_total",
                "q_contrast",
                "kappa_gas",
                "thermal stress",
            ),
            ("GR-EINSTEIN", "FRT-GRAVITY", "EMSG"),
        ),
        (("basin phase",), ("SUPERFLUID-DM",)),
        (("canonical scalar", "massless canonical scalar"), ("BRANS-DICKE",)),
        (("potential screen", "potential threshold", "potential boundary"), ("EMOND", "CHAMELEON")),
    ]
    matches: list[str] = []
    for needles, ids in rules:
        if any(needle in text for needle in needles):
            matches.extend(ids)
    if not matches:
        matches.append("NEWTON-POISSON")
    return list(dict.fromkeys(matches))


def classify_tested_formula(row: dict[str, Any]) -> dict[str, Any]:
    text = normalized(row["family"], row["formula"], row["schematic_equation"], row["verdict"])

    convenience_switch = any(
        token in text
        for token in (
            "domain oracle",
            "cluster-retuned",
            "cluster-tuned diagnostic",
            "galaxy-only l",
            "galaxy only diagnostic",
            "cluster-only retuning",
        )
    )
    per_object_fit = any(
        token in text
        for token in (
            "per-galaxy nfw",
            "compact cluster halo",
            "clash nfw construction",
            "per-object",
            "object-specific",
        )
    )
    lensing_only_closure = (
        row["family"] in {"Refined scalar and spatial lens", "Metric lens closures"}
        or "lens-only" in text
        or "g_lens=" in text
    )
    empirical_gate = any(
        token in text
        for token in (
            "gate",
            "screen",
            "threshold",
            "coherence",
            "activation",
            "moving density",
            "domain oracle",
        )
    )
    empirical_composite = any(
        token in text
        for token in (
            " + ",
            " x ",
            "quadrature",
            "product",
            "hybrid",
            "plus ",
            "multiplied by",
        )
    )
    published_control = row["formula"] in {
        "Newtonian",
        "Fixed RAR",
        "Simple MOND",
        "Density-only refracted gravity",
        "Per-galaxy NFW halo",
        "CLASH NFW construction",
    }
    diagnostic = "diagnostic" in text or "oracle" in text or "inverse target" in text
    rejected = any(token in text for token in ("fail", "reject", "retire", "not universal"))

    if convenience_switch:
        eligibility = "prohibited_convenience_switch"
    elif per_object_fit:
        eligibility = "comparator_or_per_object_fit_not_final_theory"
    elif lensing_only_closure:
        eligibility = "lensing_only_closure_not_final_theory"
    elif published_control:
        eligibility = "published_control_not_novel"
    elif diagnostic:
        eligibility = "diagnostic_not_final_theory"
    elif empirical_gate or empirical_composite:
        eligibility = "requires_single_action_derivation"
    elif rejected:
        eligibility = "tested_and_rejected_or_failed"
    else:
        eligibility = "experimental_incomplete_candidate"

    return {
        "convenience_switch": convenience_switch,
        "per_object_gravity_fit": per_object_fit,
        "lensing_only_closure": lensing_only_closure,
        "empirical_gate_or_screen": empirical_gate,
        "empirical_composite": empirical_composite,
        "published_control": published_control,
        "diagnostic": diagnostic,
        "reported_failure_or_rejection": rejected,
        "final_theory_eligibility": eligibility,
    }


FORMULA_FIELD_NAMES = {
    "action",
    "action_addition",
    "action_class",
    "action_density",
    "action_envelope",
    "action_term",
    "action_under_test",
    "canonical_formula",
    "completion_lagrangian",
    "completion_operator",
    "covariant_diagnostic",
    "dimensionless_equation",
    "equations",
    "euler_lagrange_source",
    "formula",
    "fourier_solution",
    "helmholtz_operator",
    "localized_branch_action",
    "matter_action",
    "memory_equation",
    "mechanism",
    "mediator_equation",
    "polarization_lagrangian_inside_gravity_bracket",
    "projected_gas_proxy",
    "projected_member_proxy",
    "quadratic_action",
    "response_equation",
    "one_metric_response",
    "total_lagrangian",
}


def extract_formula_fragments(value: Any, path: str = "") -> list[dict[str, str]]:
    fragments: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key.lower() in FORMULA_FIELD_NAMES and isinstance(child, str):
                fragments.append({"json_path": child_path, "formula": child})
            fragments.extend(extract_formula_fragments(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fragments.extend(extract_formula_fragments(child, f"{path}[{index}]"))
    return fragments


def sigma_protocol_inventory(addenda: dict[str, Any]) -> list[dict[str, Any]]:
    addenda_by_config: dict[str, list[dict[str, Any]]] = {}
    for entry in addenda["entries"]:
        addenda_by_config.setdefault(entry["config"], []).append(entry)

    inventory: list[dict[str, Any]] = []
    for path in sorted((ROOT / "configs").glob("sigma_v*.json")):
        payload = load_json(path)
        fragments = extract_formula_fragments(payload)
        relative = path.relative_to(ROOT).as_posix()
        for entry in addenda_by_config.get(relative, []):
            fragments.append(
                {
                    "json_path": f"addendum:{entry['id']}",
                    "formula": entry["formula"],
                    "name": entry["name"],
                    "kind": entry["kind"],
                    "status": entry["status"],
                    "evidence_doc": entry["evidence_doc"],
                    "provenance": "project_formula_addenda.json",
                }
            )
        text = normalized(path.name, *(fragment["formula"] for fragment in fragments))
        for fragment in fragments:
            fragment["published_overlap_ids"] = closest_published_families(
                normalized(path.name, fragment["formula"])
            )
        inventory.append(
            {
                "config": relative,
                "protocol_version": payload.get("protocol_version", ""),
                "status": payload.get("status", ""),
                "role": "formula_or_action_protocol" if fragments else "gate_or_data_protocol_no_new_formula_fragment",
                "formula_fragments": fragments,
                "formula_fragment_count": len(fragments),
                "published_overlap_ids": closest_published_families(text),
                "sha256": sha256(path),
            }
        )
    return inventory


def compact_score(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    if row["galaxy_error"] is not None:
        pieces.append(f"gal {row['galaxy_error']:.4g} {row['galaxy_error_unit']}")
    if row["derived_lensing_error_dex"] is not None:
        pieces.append(f"derived lens {row['derived_lensing_error_dex']:.4g} dex")
    if row["raw_lensing_error_arcsec"] is not None:
        pieces.append(f"raw lens {row['raw_lensing_error_arcsec']:.4g} arcsec")
    return "; ".join(pieces) if pieces else "not scored"


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Formula and prior-art registry",
        "",
        f"Registry version: `{payload['registry_version']}`. Published-literature cutoff: `{payload['as_of_utc']}`.",
        "",
        "## What this file guarantees",
        "",
        f"This file imports all **{payload['counts']['tested_scored_formulas']} scientifically distinct, scored laws** in the project's authoritative formula scorecard, inventories **{payload['counts']['sigma_protocols']} Sigma v1-v17 protocol files** and their **{payload['counts']['sigma_formula_fragments']} explicit formula/action fragments** (including **{payload['counts']['project_formula_addenda']}** canonical equations recovered from result documents), and compares them with **{payload['counts']['published_families']} directly relevant published formula families**.",
        "",
        "It does **not** claim to contain every equation ever published in gravitation. It is a reproducible scientific prior-art screen of the families that overlap mechanisms we have actually explored. It is not a legal novelty or patent opinion.",
        "",
        "The scored rows are not silently collapsed: different equations, carrier choices, screens, and source constructions remain separate. Repeated numerical-resolution or data-acquisition protocols appear in the protocol inventory but are correctly marked as not introducing another formula.",
        "",
        "## Non-negotiable one-law rule",
        "",
        "A final candidate must follow from one action or one equation and use the same universal constants, source definition, and matter coupling in every system. It may have a smooth limiting regime only if that behavior is derived from a field invariant evaluated in exactly the same way everywhere.",
        "",
        "The following are not eligible final theories:",
        "",
        "- choosing a galaxy formula or cluster formula from an object label;",
        "- retuning a gravity constant for a galaxy or cluster;",
        "- adding a lens-only multiplier or a second post-hoc photon law;",
        "- fitting an object-specific halo/amplitude/scale/shear/orientation; or",
        "- inserting an empirical gate and calling the splice fundamental before deriving that gate from one healthy action.",
        "",
        "Combining published concepts is allowed, but the combination must interact inside one field theory. `A + B`, quadrature, or a sigmoid splice is only a phenomenological diagnostic until the combined form is derived and its constants are frozen before holdouts.",
        "",
        "## Convenience-switch audit",
        "",
        "| Tested formula | Reason it cannot be the final theory |",
        "|---|---|",
    ]
    disallowed = [
        row
        for row in payload["tested_formulas"]
        if row["classification"]["final_theory_eligibility"]
        in {
            "prohibited_convenience_switch",
            "comparator_or_per_object_fit_not_final_theory",
            "lensing_only_closure_not_final_theory",
        }
    ]
    for row in disallowed:
        lines.append(
            f"| {md_escape(row['formula'])} | {md_escape(row['classification']['final_theory_eligibility'])} |"
        )
    lines.extend(
        [
            "",
            "The current `RAR + squared coherence-gated RG` bridge is not in the explicit-switch list, because its gate is continuous. It is nevertheless classified `requires_single_action_derivation`: until one covariant invariant produces the gate, it remains a useful interpolation rather than a fundamental law.",
            "",
            "## Published formula families",
            "",
            "| ID | Published family | Canonical formula | Regime logic | Direct relevance | Primary source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in payload["published_families"]:
        lines.append(
            "| {id} | {name} ({year}) | `{formula}` | {logic} | {overlap} | [{title}]({url}) |".format(
                id=md_escape(item["id"]),
                name=md_escape(item["name"]),
                year=item["year"],
                formula=md_escape(item["canonical_formula"]),
                logic=md_escape(item["regime_logic"]),
                overlap=md_escape(item["overlap_note"]),
                title=md_escape(item["source_title"]),
                url=item["source_url"],
            )
        )
    lines.extend(
        [
            "",
            "## Every scored formula tested in this project",
            "",
            "Scores below retain the original heterogeneous test definitions. They are compact audit pointers, not a shared likelihood and not a probability that a law is correct.",
            "",
            "| ID | Family | Formula | Schematic equation | Available error | Eligibility | Closest published overlap | Verdict / evidence |",
            "|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["tested_formulas"]:
        overlap = ", ".join(row["published_overlap_ids"])
        lines.append(
            f"| {row['registry_id']} | {md_escape(row['family'])} | {md_escape(row['formula'])} | `{md_escape(row['schematic_equation'])}` | {md_escape(compact_score(row))} | {md_escape(row['classification']['final_theory_eligibility'])} | {md_escape(overlap)} | {md_escape(row['verdict'])}; `{md_escape(row['evidence'])}` |"
        )
    lines.extend(
        [
            "",
            "## Sigma action and protocol formula inventory",
            "",
            "This appendix closes a gap in the older scorecard: later action-health and source-selection cycles often failed before an observational score, so their action/equation fragments are inventoried from every `configs/sigma_v*.json` file. A protocol with zero fragments is a numerical gate, robustness test, or data protocol rather than another proposed law.",
            "",
            "| Protocol config | Role | Explicit formula/action fragments | Closest published overlap |",
            "|---|---|---|---|",
        ]
    )
    for protocol in payload["sigma_protocol_inventory"]:
        if protocol["formula_fragments"]:
            fragments = "<br>".join(
                f"`{md_escape(fragment['json_path'])}`: `{md_escape(fragment['formula'])}`"
                for fragment in protocol["formula_fragments"]
            )
        else:
            fragments = "none (no new formula in this protocol)"
        lines.append(
            f"| `{md_escape(protocol['config'])}` | {md_escape(protocol['role'])} | {fragments} | {md_escape(', '.join(protocol['published_overlap_ids']))} |"
        )
    lines.extend(
        [
            "",
            "## Novelty discipline for the next equation",
            "",
            "Before a new candidate is run, add its exact action/equation here and answer four questions: (1) which terms are copied from the published families above, (2) which term or coupling is actually new, (3) what invariant causes every limiting regime without an object label, and (4) what observation distinguishes the new combination from its closest parent. Freeze that record and the universal constants before opening a holdout.",
            "",
            "Source hashes:",
            "",
            f"- formula scorecard: `{payload['source_hashes']['formula_scorecard_sha256']}`",
            f"- published registry: `{payload['source_hashes']['published_registry_sha256']}`",
            f"- project formula addenda: `{payload['source_hashes']['project_formula_addenda_sha256']}`",
            "",
            "Machine-readable output: `results/formula_prior_art_registry/formula_prior_art_registry.json` and `formula_prior_art_registry.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    scorecard = load_json(SCORECARD)
    published = load_json(PUBLISHED)
    addenda = load_json(ADDENDA)
    published_ids = {item["id"] for item in published["families"]}

    tested: list[dict[str, Any]] = []
    for index, source_row in enumerate(scorecard["rows"], start=1):
        row = dict(source_row)
        row["registry_id"] = f"T{index:03d}"
        row["classification"] = classify_tested_formula(row)
        row["published_overlap_ids"] = closest_published_families(
            normalized(row["family"], row["formula"], row["schematic_equation"])
        )
        tested.append(row)

    protocols = sigma_protocol_inventory(addenda)
    referenced_ids = {
        overlap
        for row in tested
        for overlap in row["published_overlap_ids"]
    } | {
        overlap
        for protocol in protocols
        for overlap in protocol["published_overlap_ids"]
    }
    unknown_ids = referenced_ids - published_ids
    if unknown_ids:
        raise ValueError(f"Unknown published overlap IDs: {sorted(unknown_ids)}")

    eligibility_counts = Counter(
        row["classification"]["final_theory_eligibility"] for row in tested
    )
    payload: dict[str, Any] = {
        "registry_version": "FORMULA-PRIOR-ART-REGISTRY-1.0.0",
        "as_of_utc": published["as_of_utc"],
        "scope": published["scope"],
        "one_law_rule": published["eligibility_rule"],
        "counts": {
            "tested_scored_formulas": len(tested),
            "published_families": len(published["families"]),
            "sigma_protocols": len(protocols),
            "project_formula_addenda": len(addenda["entries"]),
            "sigma_formula_fragments": sum(
                item["formula_fragment_count"] for item in protocols
            ),
            "eligibility": dict(sorted(eligibility_counts.items())),
        },
        "source_hashes": {
            "formula_scorecard_sha256": sha256(SCORECARD),
            "published_registry_sha256": sha256(PUBLISHED),
            "project_formula_addenda_sha256": sha256(ADDENDA),
        },
        "published_families": published["families"],
        "tested_formulas": tested,
        "sigma_protocol_inventory": protocols,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "formula_prior_art_registry.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    csv_fields = [
        "registry_id",
        "family",
        "formula",
        "schematic_equation",
        "final_theory_eligibility",
        "convenience_switch",
        "per_object_gravity_fit",
        "lensing_only_closure",
        "empirical_gate_or_screen",
        "empirical_composite",
        "published_overlap_ids",
        "galaxy_error",
        "galaxy_error_unit",
        "derived_lensing_error_dex",
        "raw_lensing_error_arcsec",
        "verdict",
        "evidence",
    ]
    with (OUT / "formula_prior_art_registry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in tested:
            writer.writerow(
                {
                    "registry_id": row["registry_id"],
                    "family": row["family"],
                    "formula": row["formula"],
                    "schematic_equation": row["schematic_equation"],
                    "final_theory_eligibility": row["classification"][
                        "final_theory_eligibility"
                    ],
                    "convenience_switch": row["classification"]["convenience_switch"],
                    "per_object_gravity_fit": row["classification"][
                        "per_object_gravity_fit"
                    ],
                    "lensing_only_closure": row["classification"][
                        "lensing_only_closure"
                    ],
                    "empirical_gate_or_screen": row["classification"][
                        "empirical_gate_or_screen"
                    ],
                    "empirical_composite": row["classification"][
                        "empirical_composite"
                    ],
                    "published_overlap_ids": ";".join(row["published_overlap_ids"]),
                    "galaxy_error": row["galaxy_error"],
                    "galaxy_error_unit": row["galaxy_error_unit"],
                    "derived_lensing_error_dex": row["derived_lensing_error_dex"],
                    "raw_lensing_error_arcsec": row["raw_lensing_error_arcsec"],
                    "verdict": row["verdict"],
                    "evidence": row["evidence"],
                }
            )

    MARKDOWN.write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
