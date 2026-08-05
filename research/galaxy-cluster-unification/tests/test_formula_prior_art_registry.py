import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "formula_prior_art_registry" / "formula_prior_art_registry.json"
SCORECARD = ROOT / "results" / "formula_scorecard" / "formula_scorecard.json"
PUBLISHED = ROOT / "configs" / "published_formula_registry.json"
ADDENDA = ROOT / "configs" / "project_formula_addenda.json"
MARKDOWN = ROOT / "docs" / "FORMULA_AND_PRIOR_ART_REGISTRY.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tested_by_name(payload: dict) -> dict[str, dict]:
    return {row["formula"]: row for row in payload["tested_formulas"]}


def test_every_authoritative_scored_formula_is_present_once() -> None:
    report = _load(REPORT)
    scorecard = _load(SCORECARD)
    report_names = [row["formula"] for row in report["tested_formulas"]]
    source_names = [row["formula"] for row in scorecard["rows"]]

    assert report["counts"]["tested_scored_formulas"] == 128
    assert report_names == source_names
    assert len(report_names) == len(set(report_names))
    assert [row["registry_id"] for row in report["tested_formulas"]] == [
        f"T{index:03d}" for index in range(1, 129)
    ]


def test_published_registry_has_sources_formulas_and_unique_ids() -> None:
    report = _load(REPORT)
    families = report["published_families"]
    ids = [item["id"] for item in families]

    assert len(families) >= 30
    assert len(ids) == len(set(ids))
    for item in families:
        assert item["canonical_formula"]
        assert item["mechanism"]
        assert item["regime_logic"]
        assert item["overlap_note"]
        assert item["source_title"]
        assert item["source_url"].startswith("https://")


def test_switches_comparators_and_empirical_bridge_are_distinguished() -> None:
    rows = _tested_by_name(_load(REPORT))

    assert rows["domain oracle"]["classification"]["final_theory_eligibility"] == (
        "prohibited_convenience_switch"
    )
    assert rows["Cluster-retuned RAR"]["classification"]["convenience_switch"] is True
    assert rows["Per-galaxy NFW halo"]["classification"]["per_object_gravity_fit"] is True
    assert rows["Fixed RAR with scalar metric slip s=5"]["classification"][
        "lensing_only_closure"
    ] is True

    bridge = rows["RAR + squared coherence-gated RG (current empirical bridge)"]
    assert bridge["classification"]["convenience_switch"] is False
    assert bridge["classification"]["empirical_gate_or_screen"] is True
    assert bridge["classification"]["final_theory_eligibility"] == (
        "requires_single_action_derivation"
    )
    assert {"RAR-EMPIRICAL", "REFRACTED-GRAVITY"}.issubset(
        bridge["published_overlap_ids"]
    )


def test_every_sigma_protocol_is_inventoried_and_formula_paths_are_explicit() -> None:
    report = _load(REPORT)
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs").glob("sigma_v*.json")
    }
    actual = {item["config"] for item in report["sigma_protocol_inventory"]}

    assert actual == expected
    assert report["counts"]["sigma_protocols"] == len(expected)
    assert report["counts"]["project_formula_addenda"] >= 14
    assert report["counts"]["sigma_formula_fragments"] >= 50
    for protocol in report["sigma_protocol_inventory"]:
        assert protocol["formula_fragment_count"] == len(protocol["formula_fragments"])
        for fragment in protocol["formula_fragments"]:
            assert fragment["json_path"]
            assert fragment["formula"]
            assert fragment["published_overlap_ids"]


def test_registry_hashes_and_rendered_markdown_are_current() -> None:
    report = _load(REPORT)
    assert report["source_hashes"]["formula_scorecard_sha256"] == _sha256(SCORECARD)
    assert report["source_hashes"]["published_registry_sha256"] == _sha256(PUBLISHED)
    assert report["source_hashes"]["project_formula_addenda_sha256"] == _sha256(ADDENDA)

    markdown = MARKDOWN.read_text(encoding="utf-8")
    assert "# Formula and prior-art registry" in markdown
    assert "## Non-negotiable one-law rule" in markdown
    assert "## Every scored formula tested in this project" in markdown
    assert "## Sigma action and protocol formula inventory" in markdown
    for row in report["tested_formulas"]:
        assert row["formula"] in markdown


def test_v17_stress_prior_art_is_not_overclaimed_as_new() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17 = protocols["sigma_v17_dynamical_stress_data_gate.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17["formula_fragments"]
    )

    assert "Theta_b" in fragment_text
    assert {"GR-EINSTEIN", "FRT-GRAVITY", "EMSG"}.issubset(
        set(v17["published_overlap_ids"])
    )


def test_v17d_thermal_proxy_retains_its_stress_energy_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17d = protocols["sigma_v17d_thermal_stress_map.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17d["formula_fragments"]
    )

    assert "q_total" in fragment_text
    assert "q_contrast" in fragment_text
    assert {"GR-EINSTEIN", "FRT-GRAVITY", "EMSG"}.issubset(
        set(v17d["published_overlap_ids"])
    )


def test_v17g_pressure_metric_is_registered_as_prior_art_completion() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17g = protocols["sigma_v17g_pressure_metric_gate.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17g["formula_fragments"]
    )

    assert "J_X=alpha T+alpha E=3 alpha p" in fragment_text
    assert "g_tilde_mn=exp(2 alpha X)" in fragment_text
    assert {"TEVES", "DISFORMAL-METRIC", "GR-EINSTEIN"}.issubset(
        set(v17g["published_overlap_ids"])
    )


def test_v17h_susceptibility_screen_is_registered_as_aether_disformal_prior_art() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17h = protocols["sigma_v17h_susceptibility_screened_pressure.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17h["formula_fragments"]
    )

    assert "F_A(Z)=sqrt(1+Z)-1" in fragment_text
    assert "chi(Z)" in fragment_text
    assert {"TEVES", "DISFORMAL-METRIC", "EINSTEIN-AETHER"}.issubset(
        set(v17h["published_overlap_ids"])
    )


def test_v17i_localization_preserves_the_same_prior_art_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17i = protocols["sigma_v17i_localized_variation.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17i["formula_fragments"]
    )

    assert "B^m[A_m-U^n nabla_n U_m]" in fragment_text
    assert "J=T+E=3p" in fragment_text
    assert {"TEVES", "DISFORMAL-METRIC", "EINSTEIN-AETHER"}.issubset(
        set(v17i["published_overlap_ids"])
    )


def test_v17j_flat_kinetic_falsification_is_registered_as_aether_prior_art() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17j = protocols["sigma_v17j_flat_kinetic_gate.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17j["formula_fragments"]
    )

    assert "c_1=c_U, c_2=0, c_3=-c_U, c_4=-1" in fragment_text
    assert "s_1^2" in fragment_text
    assert "EINSTEIN-AETHER" in set(v17j["published_overlap_ids"])
    characteristic_fragments = [
        fragment
        for fragment in v17j["formula_fragments"]
        if fragment["json_path"]
        in {
            "frozen_action.einstein_aether_mapping.equations",
            "characteristic_gate.equations",
        }
    ]
    assert len(characteristic_fragments) == 2
    assert all(
        "EINSTEIN-AETHER" in fragment["published_overlap_ids"]
        for fragment in characteristic_fragments
    )


def test_v17k_luminal_carrier_is_registered_as_published_aether_completion() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17k = protocols["sigma_v17k_luminal_aether_pressure_carrier.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17k["formula_fragments"]
    )

    assert "c_1=epsilon, c_3=-epsilon, c_4=0" in fragment_text
    assert "s_2^2=1, s_1^2=1, s_0^2=1" in fragment_text
    assert {"EINSTEIN-AETHER", "TEVES", "DISFORMAL-METRIC"}.issubset(
        set(v17k["published_overlap_ids"])
    )


def test_v17l_localization_preserves_aether_and_disformal_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17l = protocols["sigma_v17l_localized_luminal_pressure.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17l["formula_fragments"]
    )

    assert "B^m[A_m-U^n nabla_n U_m]" in fragment_text
    assert "J=(1/2)H^mn D_mn" in fragment_text
    assert "T_U,mn" in fragment_text
    assert {"EINSTEIN-AETHER", "TEVES", "DISFORMAL-METRIC"}.issubset(
        set(v17l["published_overlap_ids"])
    )


def test_v17m_kinetic_gate_preserves_aether_and_disformal_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17m = protocols["sigma_v17m_active_pressure_kinetic_gate.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17m["formula_fragments"]
    )

    assert "c_14,eff=epsilon-q J_hat(q)/2" in fragment_text
    assert "rho_hat_crit" in fragment_text
    assert {"EINSTEIN-AETHER", "TEVES", "DISFORMAL-METRIC"}.issubset(
        set(v17m["published_overlap_ids"])
    )


def test_v17n_no_go_preserves_aether_and_disformal_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17n = protocols["sigma_v17n_decreasing_metric_screen_no_go.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17n["formula_fragments"]
    )

    assert "Delta K_T=2 q_base J_hat chi_prime(z_0)" in fragment_text
    assert "J_hat_crit" in fragment_text
    assert {"EINSTEIN-AETHER", "TEVES", "DISFORMAL-METRIC"}.issubset(
        set(v17n["published_overlap_ids"])
    )


def test_v17o_scale_audit_registers_published_halo_and_modified_gravity_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17o = protocols["sigma_v17o_halo_scale_driver_audit.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17o["formula_fragments"]
    )

    assert "MOND r_M=sqrt(GM_b/a_0)" in fragment_text
    assert "AeST r_C" in fragment_text
    assert "NFW r_s=r_200/c_200" in fragment_text
    assert {"NFW-HALO", "AEST-MOND", "COVARIANT-RG"}.issubset(
        set(v17o["published_overlap_ids"])
    )


def test_v17p_flux_no_go_registers_kinetic_and_metric_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17p = protocols["sigma_v17p_pressure_flux_screen_no_go.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17p["formula_fragments"]
    )

    assert "K-mouflage" in fragment_text
    assert "abs(gamma-1)" in fragment_text
    assert {"K-MOUFLAGE", "AQUAL", "TEVES", "DISFORMAL-METRIC"}.issubset(
        set(v17p["published_overlap_ids"])
    )


def test_v17q_pressure_symmetron_registers_screen_and_metric_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v17q = protocols["sigma_v17q_pressure_symmetron_no_go.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v17q["formula_fragments"]
    )

    assert "Symmetron" in fragment_text
    assert "Pi_sun/Pi_cluster" in fragment_text
    assert {"SYMMETRON", "TEVES", "DISFORMAL-METRIC"}.issubset(
        set(v17q["published_overlap_ids"])
    )


def test_v18_flux_selection_registers_newtonian_and_aqual_ancestry() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v18 = protocols["sigma_v18_post_pressure_flux_selection.json"]
    fragment_text = " ".join(
        fragment["formula"] for fragment in v18["formula_fragments"]
    )

    assert "rho_Sigma,eff" in fragment_text
    assert "Legendre-dual AQUAL" in fragment_text
    assert {"NEWTON-POISSON", "AQUAL"}.issubset(
        set(v18["published_overlap_ids"])
    )


def test_v18b_and_v18c_measurement_equations_are_not_lost() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }

    v18b = protocols["sigma_v18b_replacement_pair_readiness.json"]
    v18b_text = " ".join(
        fragment["formula"] for fragment in v18b["formula_fragments"]
    )
    assert "v_i=c" in v18b_text
    assert "M_star/M_sun" in v18b_text

    v18c = protocols["sigma_v18c_collisionless_stress_maps.json"]
    v18c_text = " ".join(
        fragment["formula"] for fragment in v18c["formula_fragments"]
    )
    assert "sigma_i=d_i,k/2" in v18c_text
    assert "q_member" in v18c_text
    assert {"GR-EINSTEIN", "FRT-GRAVITY", "EMSG"}.issubset(
        set(v18c["published_overlap_ids"])
    )


def test_v19_observability_gate_does_not_pretend_to_be_a_formula() -> None:
    report = _load(REPORT)
    protocols = {
        Path(item["config"]).name: item for item in report["sigma_protocol_inventory"]
    }
    v19 = protocols["sigma_v19_causal_assembly_observability_gate.json"]

    assert v19["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19["formula_fragment_count"] == 0

    v19a = protocols["sigma_v19a_assembly_history_readiness.json"]
    assert v19a["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19a["formula_fragment_count"] == 0

    v19b = protocols["sigma_v19b_replacement_cluster_screen.json"]
    assert v19b["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19b["formula_fragment_count"] == 0

    v19c = protocols["sigma_v19c_source_archive_acquisition.json"]
    assert v19c["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19c["formula_fragment_count"] == 0

    v19d = protocols["sigma_v19d_member_catalog_extraction.json"]
    assert v19d["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19d["formula_fragment_count"] == 0

    v19e = protocols["sigma_v19e_chandra_acquisition.json"]
    assert v19e["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19e["formula_fragment_count"] == 0

    v19f = protocols["sigma_v19f_chandra_source_reduction.json"]
    assert v19f["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19f["formula_fragment_count"] == 0

    v19g = protocols["sigma_v19g_gaia_hierarchical_astrometry.json"]
    assert v19g["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19g["formula_fragment_count"] == 0

    v19h = protocols["sigma_v19h_causal_observable_protocol.json"]
    assert v19h["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19h["formula_fragment_count"] == 0

    v19i = protocols["sigma_v19i_continuous_member_field.json"]
    assert v19i["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19i["formula_fragment_count"] == 0

    v19j = protocols["sigma_v19j_automated_front_implementation.json"]
    assert v19j["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19j["formula_fragment_count"] == 0

    v19k = protocols["sigma_v19k_smooth_null_front_likelihood.json"]
    assert v19k["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19k["formula_fragment_count"] == 0

    v19l = protocols["sigma_v19l_fixture_corrected_front_likelihood.json"]
    assert v19l["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19l["formula_fragment_count"] == 0

    v19m = protocols["sigma_v19m_adaptive_thermodynamic_regions.json"]
    assert v19m["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19m["formula_fragment_count"] == 0

    v19n = protocols["sigma_v19n_regional_response_workload.json"]
    assert v19n["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19n["formula_fragment_count"] == 0

    v19o = protocols["sigma_v19o_fov_filtered_response_workload.json"]
    assert v19o["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19o["formula_fragment_count"] == 0

    v19p = protocols["sigma_v19p_exact_flux_obs_support.json"]
    assert v19p["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19p["formula_fragment_count"] == 0

    v19q = protocols["sigma_v19q_positive_exposure_response_workload.json"]
    assert v19q["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19q["formula_fragment_count"] == 0

    v19r = protocols["sigma_v19r_response_commissioning.json"]
    assert v19r["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19r["formula_fragment_count"] == 0

    v19s = protocols["sigma_v19s_hi4pi_acquisition.json"]
    assert v19s["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19s["formula_fragment_count"] == 0

    v19t = protocols["sigma_v19t_temperature_fit_commissioning.json"]
    assert v19t["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19t["formula_fragment_count"] == 0

    v19u = protocols["sigma_v19u_response_production_plan.json"]
    assert v19u["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19u["formula_fragment_count"] == 0

    v19v = protocols["sigma_v19v_response_throughput_pilot.json"]
    assert v19v["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19v["formula_fragment_count"] == 0

    v19w2 = protocols["sigma_v19w2_exact_binmap_response_commissioning.json"]
    assert v19w2["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19w2["formula_fragment_count"] == 0

    v19w = protocols["sigma_v19w_full_response_production.json"]
    assert v19w["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19w["formula_fragment_count"] == 0

    v19x = protocols["sigma_v19x_spectral_combination_commissioning.json"]
    assert v19x["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19x["formula_fragment_count"] == 0

    v19y = protocols["sigma_v19y_hsc_member_photometry.json"]
    assert v19y["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19y["formula_fragment_count"] == 0

    v19z = protocols["sigma_v19z_nsc_member_photometry.json"]
    assert v19z["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19z["formula_fragment_count"] == 0

    v19be = protocols["sigma_v19be_long_wave_action_admission.json"]
    assert v19be["role"] == "gate_or_data_protocol_no_new_formula_fragment"
    assert v19be["formula_fragment_count"] == 0
