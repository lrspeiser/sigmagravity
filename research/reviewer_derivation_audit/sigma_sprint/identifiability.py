"""Jacobian-rank audit for the canonical and submitted parameterizations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def matrix_rank_and_nullity(jacobian, parameter_names):
    matrix = np.atleast_2d(np.asarray(jacobian, dtype=float))
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "parameters": list(parameter_names),
        "shape": list(matrix.shape),
        "rank": rank,
        "nullity": int(matrix.shape[1] - rank),
        "jacobian": matrix.tolist(),
    }


def dependency_audit():
    """Return the minimal ranks implied by each dataset/observable."""
    log_lengths_one = np.log(np.array([600.0]) / 0.4)
    log_lengths_varied = np.log(np.array([0.2, 0.4, 1.0, 5.0, 600.0]) / 0.4)
    cluster_one_L = np.column_stack(
        [np.ones_like(log_lengths_one), log_lengths_one, -0.27 * np.ones_like(log_lengths_one)]
    )
    varied_L = np.column_stack(
        [np.ones_like(log_lengths_varied), log_lengths_varied, -0.27 * np.ones_like(log_lengths_varied)]
    )
    return {
        "deep_btfr": {
            **matrix_rank_and_nullity([[2.0, 1.0]], ["log_B", "log_g_dagger"]),
            "observable": "log(V^4/(G M_b)) = 2 log(B) + log(g_dagger)",
            "conclusion": "BTFR identifies B^2 g_dagger, not B and g_dagger separately.",
        },
        "fox_clusters_fixed_L": {
            **matrix_rank_and_nullity(
                cluster_one_L, ["log_A0", "n", "log_L0"]
            ),
            "conclusion": "With one L and C=1, n is a re-expression of one fitted B.",
        },
        "hypothetical_varied_L": {
            **matrix_rank_and_nullity(varied_L, ["log_A0", "n", "log_L0"]),
            "conclusion": "Even varied L leaves the A0/L0 normalization degenerate without an external anchor.",
        },
        "cluster_coherence_fixed_zero": {
            **matrix_rank_and_nullity([[1.0, 0.0]], ["density_coupling", "coherence_coupling"]),
            "conclusion": "Dispersion-supported clusters alone cannot identify a coherence-source coupling.",
        },
    }


def write_dependency_audit(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dependency_audit(), indent=2), encoding="utf-8")
