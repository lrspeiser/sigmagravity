#!/usr/bin/env python3
"""Run the frozen V19BR chain once the named V19W processes have exited.

This is an operational handoff for a long response-production run.  It does
not inspect science products or alter the frozen V19BR protocol.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
DEFAULT_RUNNER = HERE / "run_sigma_v19br_target_sealed_terminal_chain.py"
EXPECTED_PROCESS_MARKER = "run_sigma_v19w_full_response_production.py"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def expected_process_is_live(pid: int) -> tuple[bool, str]:
    """Return whether *pid* is the expected live V19W process and why."""

    proc = Path("/proc") / str(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, "pid_absent"
    except PermissionError:
        return True, "pid_exists_permission_denied"
    except OSError as exc:
        return True, f"pid_probe_uncertain:{type(exc).__name__}"
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError as exc:
        return True, f"pid_live_cmdline_unreadable:{type(exc).__name__}"
    if EXPECTED_PROCESS_MARKER not in command:
        return False, "pid_reused_or_unexpected_command"
    return True, "expected_v19w_process"


def log(handle, message: str) -> None:
    print(f"{utc_now()} {message}", file=handle, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pid",
        action="append",
        required=True,
        type=int,
        help="Expected V19W PID. Repeat for each protected parent process.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0:
        raise ValueError("poll and heartbeat intervals must be positive")
    runner = args.runner.resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"terminal runner not found: {runner}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8", buffering=1) as handle:
        log(handle, f"watcher_started pids={args.pid}")
        next_heartbeat = 0.0
        previous_live: tuple[int, ...] | None = None
        while True:
            states = {pid: expected_process_is_live(pid) for pid in args.pid}
            live = tuple(pid for pid, (is_live, _) in states.items() if is_live)
            monotonic_now = time.monotonic()
            if live != previous_live or monotonic_now >= next_heartbeat:
                reasons = {pid: reason for pid, (_, reason) in states.items()}
                log(handle, f"v19w_state live={list(live)} reasons={reasons}")
                previous_live = live
                next_heartbeat = monotonic_now + args.heartbeat_seconds
            if not live:
                break
            time.sleep(args.poll_seconds)

        try:
            args.lock_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            log(handle, f"handoff_not_started lock_exists={args.lock_dir}")
            return 0

        command = [sys.executable, str(runner), "--execute"]
        log(handle, f"handoff_started lock={args.lock_dir} command={command}")
        completed = subprocess.run(
            command,
            cwd=runner.parent.parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log(handle, f"handoff_finished returncode={completed.returncode}")
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
