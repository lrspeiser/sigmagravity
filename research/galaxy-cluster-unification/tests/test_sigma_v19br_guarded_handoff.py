from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import watch_sigma_v19w_then_run_v19br as watcher


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="watcher uses Linux procfs")
def test_pid_guard_rejects_absent_and_unrelated_processes() -> None:
    assert watcher.expected_process_is_live(999_999_999) == (False, "pid_absent")
    live, reason = watcher.expected_process_is_live(os.getpid())
    assert not live
    assert reason == "pid_reused_or_unexpected_command"


def test_handoff_uses_atomic_lock_and_runs_only_once(
    tmp_path: Path, monkeypatch
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "fake_terminal_runner.py"
    marker = tmp_path / "launches.txt"
    runner.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['SIGMA_WATCH_TEST_MARKER'])\n"
        "with path.open('a', encoding='utf-8') as handle:\n"
        "    handle.write('launch\\n')\n",
        encoding="utf-8",
    )
    lock = tmp_path / "handoff.lock"
    log_path = tmp_path / "watcher.log"
    monkeypatch.setenv("SIGMA_WATCH_TEST_MARKER", str(marker))
    monkeypatch.setattr(
        watcher,
        "expected_process_is_live",
        lambda _pid: (False, "pid_absent"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watcher",
            "--pid",
            "999999999",
            "--runner",
            str(runner),
            "--lock-dir",
            str(lock),
            "--log",
            str(log_path),
        ],
    )

    assert watcher.main() == 0
    assert watcher.main() == 0
    assert marker.read_text(encoding="utf-8").splitlines() == ["launch"]
    log_text = log_path.read_text(encoding="utf-8")
    assert "handoff_started" in log_text
    assert "handoff_finished returncode=0" in log_text
    assert "handoff_not_started lock_exists=" in log_text
