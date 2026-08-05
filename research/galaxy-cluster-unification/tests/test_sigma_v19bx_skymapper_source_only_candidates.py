from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sigma_v19bx_skymapper_source_only_candidates.json"
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19bx_skymapper_source_only_candidates.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bx", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v19bx_projection_is_source_only_and_target_sealed() -> None:
    cfg = config()
    projection = " ".join(cfg["query_policy"]["projection"]).lower()
    assert not any(token.lower() in projection for token in cfg["forbidden_target_tokens"])
    boundary = cfg["access_boundary"]
    assert not boundary["wallaby_kinematic_table_rows_read"]
    assert not boundary["rotation_speed_or_velocity_field_read"]
    assert not boundary["optical_counterpart_selected"]
    assert not boundary["development_validation_holdout_split_selected"]
    assert not boundary["gravity_action_or_constant_changed"]
    assert not boundary["solar_system_optimization_performed"]


def test_v19bx_query_uses_exact_radius_projection_and_order() -> None:
    cfg = config()
    query = MODULE.build_query(cfg, 150.0, -30.0)
    assert f"SELECT TOP {cfg['query_policy']['maximum_rows_per_source']}" in query
    assert ",".join(cfg["query_policy"]["projection"]) in query
    assert "CIRCLE('ICRS',150.000000000000,-30.000000000000,0.016666666666667)" in query
    assert query.endswith("ORDER BY object_id")


def test_v19bx_parses_only_the_declared_projection() -> None:
    cfg = config()
    columns = cfg["query_policy"]["projection"]
    values = ["1", "150.0", "-30.0", *([""] * (len(columns) - 3))]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerow(values)
    rows = MODULE.parse_response(stream.getvalue().encode("utf-8"), columns)
    assert len(rows) == 1
    assert rows[0]["object_id"] == "1"


def test_v19bx_rejects_projection_drift() -> None:
    cfg = config()
    columns = cfg["query_policy"]["projection"]
    bad = (",".join(columns[:-1]) + "\n").encode("utf-8")
    try:
        MODULE.parse_response(bad, columns)
    except RuntimeError as error:
        assert "unexpected TAP projection" in str(error)
    else:  # pragma: no cover
        raise AssertionError("projection drift was not rejected")


def test_v19bx_diagnostic_flags_do_not_select_a_counterpart() -> None:
    row = {
        "r_ngood": "3",
        "r_flags": "0",
        "r_nimaflags": "0",
        "r_petro": "17.2",
        "e_r_petro": "0.1",
        "radius_petro": "8.0",
        "class_star": "0.2",
    }
    assert MODULE.usable_r(row)
    assert MODULE.extended(row)
    assert config()["query_policy"]["counterpart_selection"].startswith("forbidden")
