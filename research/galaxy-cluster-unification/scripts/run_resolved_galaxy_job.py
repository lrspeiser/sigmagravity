#!/usr/bin/env python3
"""Execute one immutable resolved-galaxy extraction or generation job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.resolved_galaxy_job import execute_galaxy_request_file


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = execute_galaxy_request_file(arguments.request, arguments.output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "jobId": result["jobId"],
                "manifestSha256": result["manifestSha256"],
                "scientificResultSha256": result["scientificResultSha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schemaVersion": "sigma-galaxy-job-cli-error/1",
                    "state": "rejected_input",
                    "errorType": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
