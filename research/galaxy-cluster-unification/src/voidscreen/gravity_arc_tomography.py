"""Utilities for inferring nonlocal baryonic gravity-routing paths."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


FIELD_DEFINITION = re.compile(r"#\s+(\d+)\s+(\S+)")


def read_relics_catalog(path: Path) -> pd.DataFrame:
    """Read a RELICS text catalog while preserving repeated filter columns."""
    definitions: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FIELD_DEFINITION.match(line)
        if match:
            definitions[int(match.group(1))] = match.group(2)
    if not definitions or sorted(definitions) != list(range(1, max(definitions) + 1)):
        raise ValueError(f"{path}: incomplete numbered column definitions")
    names = []
    seen: Counter[str] = Counter()
    for index in range(1, max(definitions) + 1):
        base = definitions[index]
        seen[base] += 1
        names.append(base if seen[base] == 1 else f"{base}__{seen[base]}")
    values = np.loadtxt(path, comments="#")
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(names):
        raise ValueError(
            f"{path}: {values.shape[1]} data columns but {len(names)} definitions"
        )
    return pd.DataFrame(values, columns=names)


def combine_f160_photometry(catalog: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Select the highest-significance valid F160W measurement per object."""
    flux_columns = [
        name
        for name in catalog
        if name == "f160w_fluxnJy" or name.startswith("f160w_fluxnJy__")
    ]
    sig_columns = [
        name for name in catalog if name == "f160w_sig" or name.startswith("f160w_sig__")
    ]
    if len(flux_columns) != len(sig_columns) or not flux_columns:
        raise ValueError("catalog lacks matched F160W flux/significance columns")
    flux = catalog[flux_columns].to_numpy(float)
    significance = catalog[sig_columns].to_numpy(float)
    valid = np.isfinite(flux) & np.isfinite(significance) & (flux > 0.0)
    ranking = np.where(valid, significance, -np.inf)
    choice = np.argmax(ranking, axis=1)
    rows = np.arange(len(catalog))
    selected_flux = flux[rows, choice]
    selected_significance = significance[rows, choice]
    none = ~np.any(valid, axis=1)
    selected_flux[none] = np.nan
    selected_significance[none] = np.nan
    return selected_flux, selected_significance


def photometric_membership_weights(
    catalog: pd.DataFrame,
    cluster_redshift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a hard interval selector and a smooth, non-probabilistic membership weight."""
    redshift = catalog["zb"].to_numpy(float)
    low = catalog["zbmin"].to_numpy(float)
    high = catalog["zbmax"].to_numpy(float)
    odds = np.clip(catalog["odds"].to_numpy(float), 0.0, 1.0)
    finite = np.isfinite(redshift) & np.isfinite(low) & np.isfinite(high)
    hard = (
        finite
        & (low <= float(cluster_redshift))
        & (high >= float(cluster_redshift))
        & (odds >= 0.5)
    )
    interval_sigma = 0.5 * np.maximum(high - low, 0.0)
    floor_sigma = 0.08 * (1.0 + float(cluster_redshift))
    sigma = np.maximum(interval_sigma, floor_sigma)
    soft = odds * np.exp(-0.5 * np.square((redshift - cluster_redshift) / sigma))
    soft[~finite] = 0.0
    return hard, np.clip(soft, 0.0, 1.0)


def sinkhorn_transport(
    source_weight: np.ndarray,
    target_weight: np.ndarray,
    cost: np.ndarray,
    *,
    entropy: float,
    iterations: int = 1000,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Balanced entropic transport plan for small source/target grids."""
    source = np.asarray(source_weight, dtype=float)
    target = np.asarray(target_weight, dtype=float)
    source = source / np.sum(source)
    target = target / np.sum(target)
    log_kernel = -np.asarray(cost, dtype=float) / float(entropy)
    log_source = np.log(np.maximum(source, np.finfo(float).tiny))
    log_target = np.log(np.maximum(target, np.finfo(float).tiny))
    log_left = np.zeros_like(source)
    log_right = np.zeros_like(target)
    for _ in range(int(iterations)):
        previous = log_left.copy()
        log_left = log_source - logsumexp(log_kernel + log_right[None, :], axis=1)
        log_right = log_target - logsumexp(log_kernel.T + log_left[None, :], axis=1)
        if np.max(np.abs(log_left - previous)) < tolerance:
            break
    plan = np.exp(log_left[:, None] + log_kernel + log_right[None, :])
    plan /= np.sum(plan)
    return plan
