from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from voidscreen.field_job import execute_request_file, package_array_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package or execute a content-addressed generic 2D/3D field job."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Create a verified array bundle from NPZ data")
    pack.add_argument("--arrays", type=Path, required=True)
    pack.add_argument("--metadata", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Execute an immutable field job request")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "pack":
        result = package_array_file(arguments.arrays, arguments.metadata, arguments.output)
        summary = {
            "state": "packaged",
            "bundleSha256": result["bundleSha256"],
            "output": str(arguments.output.resolve()),
        }
    else:
        result = execute_request_file(arguments.request, arguments.output)
        envelope = json.loads(arguments.request.read_text(encoding="utf-8"))
        output_value = (
            arguments.output
            if arguments.output is not None
            else Path(envelope["outputDirectory"])
        )
        effective_output = (
            output_value.resolve()
            if output_value.is_absolute()
            else (arguments.request.resolve().parent / output_value).resolve()
        )
        summary = {
            "state": result["state"],
            "jobId": result["jobId"],
            "manifestSha256": result["manifestSha256"],
            "output": str(effective_output),
        }
        if "scientificResultSha256" in result:
            summary["scientificResultSha256"] = result["scientificResultSha256"]
        if "failureSha256" in result:
            summary["failureSha256"] = result["failureSha256"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schemaVersion": "sigma-field-job-cli-error/1",
                    "state": "rejected_input",
                    "errorType": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
