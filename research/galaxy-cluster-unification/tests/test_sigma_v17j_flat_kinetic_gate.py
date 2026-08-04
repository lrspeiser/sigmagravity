import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17j_flat_kinetic_gate.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17j_flat_kinetic_gate.py"
REPORT = ROOT / "results" / "sigma_v17j_flat_kinetic_gate" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17j_flat_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17j_is_theory_only_and_parent_hashes_are_current() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17J-FLAT-KINETIC-GATE-1.0.0"
    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    assert config["parent"]["sha256"] == _sha256(ROOT / config["parent"]["protocol"])
    assert config["parent"]["report_sha256"] == _sha256(ROOT / config["parent"]["report"])
    assert config["complexity"]["new_physical_constants_added"] == 0


def test_frozen_action_maps_to_the_claimed_einstein_aether_coefficients() -> None:
    runner = _load_runner()
    coefficients = runner.aether_coefficients(1.0, -1.0)
    quadratic = runner.quadratic_coefficients(1.0, -1.0)

    assert coefficients == {
        "c_1": 1.0,
        "c_2": 0.0,
        "c_3": -1.0,
        "c_4": -1.0,
        "c_13": 0.0,
        "c_14": 0.0,
        "c_123": 0.0,
    }
    assert quadratic["transverse_kinetic"] == 0.0
    assert quadratic["transverse_gradient"] == 1.0


def test_direct_square_root_expansion_confirms_the_zero_kinetic_coefficient() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expansion = report["quadratic_expansion"]

    assert expansion["verified"] is True
    assert expansion["analytic_time_hessian_c_U_plus_sigma_A"] == 0.0
    assert abs(expansion["numerical_time_hessian"]) <= 1e-8


@pytest.mark.parametrize(
    ("c_u", "expected_class"),
    [
        (-1.0, "wrong_sign"),
        (0.0, "degenerate"),
        (0.5, "negative_speed_squared"),
        (1.0, "singular"),
        (2.0, "outside_physical_cone"),
    ],
)
def test_exhaustive_vector_partition_has_no_allowed_interval(
    c_u: float, expected_class: str
) -> None:
    runner = _load_runner()
    row = runner.classify_point(
        c_u,
        -1.0,
        zero_tolerance=1e-12,
        cone_tolerance=1e-12,
    )
    speed = row["mode_speeds"]["spin_1_squared"]

    if expected_class in {"wrong_sign", "degenerate"}:
        assert row["gates"]["transverse_gradient_positive"] is False
    elif expected_class == "negative_speed_squared":
        assert speed is not None and speed < 0.0
    elif expected_class == "singular":
        assert speed is None
        assert row["gates"]["transverse_kinetic_positive"] is False
    else:
        assert speed is not None and speed > 1.0
        assert row["gates"]["spin_1_within_physical_cone"] is False


def test_frozen_action_is_retired_before_data_and_not_rescued_by_c_u() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "failed_flat_kinetic_gate"
    assert report["observational_data_opened"] is False
    assert report["empirical_fit_performed"] is False
    assert report["frozen_sign_scan"]["vector_gate_pass_count"] == 0
    assert report["frozen_sign_scan"]["full_mode_gate_pass_count"] == 0
    assert report["analytic_no_go"]["real_c_U_vector_pass_exists"] is False
    assert report["gates"]["flat_kinetic_gate_pass"] is False
    assert report["gates"]["holdout_authorized"] is False
    assert report["decision"]["outcome"] == ("retire_frozen_v17H_v17I_susceptibility_action")


def test_sign_flipped_control_repairs_only_vector_sector_not_full_theory() -> None:
    runner = _load_runner()
    control = runner.classify_point(
        0.5,
        1.0,
        zero_tolerance=1e-12,
        cone_tolerance=1e-12,
    )
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert control["quadratic"]["transverse_kinetic"] == pytest.approx(1.5)
    assert control["mode_speeds"]["spin_1_squared"] == pytest.approx(1.0 / 3.0)
    assert control["gates"]["spin_0_finite_positive"] is False
    assert report["sign_flipped_control_scan"]["vector_gate_pass_count"] > 0
    assert report["sign_flipped_control_scan"]["full_mode_gate_pass_count"] == 0
