#!/usr/bin/env python3
"""Run the frozen bounded tidal-tensor completion protocol."""

from __future__ import annotations

import sys
from pathlib import Path

from run_vector_completion_full_test import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        root = Path(__file__).resolve().parents[1]
        sys.argv.extend(
            ["--protocol", str(root / "configs/tensor_completion_full_test_protocol.json")]
        )
    main()
