from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts" / "watch_r1_rxj2129_terminal.ps1"


def test_terminal_watcher_only_runs_frozen_audit_and_finalizer_after_outputs() -> None:
    text = WATCHER.read_text(encoding="utf-8")
    assert ".x4_response_products_complete" in text
    assert "results\\r1_rxj2129_hst_h2\\report.json" in text
    assert "audit_r1_rxj2129_xmm_x4_responses.py" in text
    assert "finalize_r1_rxj2129_terminal_observable_disposition.py" in text
    assert "audit_r0_r2_completion_evidence.py" in text
    assert "audit_r0_r2_goal_progress.py" in text
    assert "test_r0_r2_completion_evidence.py" in text
    assert "test_r1_ten_system_public_data_ceiling.py" in text
    assert "test_r1_rxj2129_terminal_disposition.py" in text
    assert "test_r1_rxj2129_hst_h2_execution.py" in text
    assert "Start-Sleep -Seconds $PollSeconds" in text
    assert "run_r1_rxj2129_hst_h2.py" not in text
    assert "run_r1_rxj2129_xmm_x4_responses.sh" not in text
    assert "fit_new_force" not in text
