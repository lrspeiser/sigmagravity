#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Stacked gravitational redshift (gRZ) pipeline (research sandbox)

Implements a minimal, robust stacking analysis as recommended:
- Per-cluster robust Δv profiles using biweight location and shifting-gapper
- Equal-weight stack over clusters with bootstrap errors
- Parallel prediction using Σ endpoint redshift via a provided geff(x)

Inputs (caller supplies):
- clusters: iterable of dicts with keys {id, RA, DEC, z_BCG, R200} (R200 in Mpc)
- members_by_cluster: dict id -> structured array with fields {Rproj_Mpc, z_spec}
- geff_callable: function geff(x[m]) -> a[m/s^2]

Outputs: stacked profile arrays suitable for CSV writing and plotting.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Iterable

try:
    from astropy.stats import biweight_location  # robust estimator
except Exception:
    # Fallback: Huber-like estimator if astropy is unavailable
    def biweight_location(a, c=6.0):
        a = np.asarray(a, float)
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med)) + 1e-12
        u = (a - med) / (c * mad)
        w = (1 - u**2)**2
        w[np.abs(u) >= 1] = 0.0
        num = np.nansum(w * a)
        den = np.nansum(w) + 1e-12
        return float(num / den)

from redshift.redshift import gravitational_redshift_endpoint, c


def shifting_gapper(R: np.ndarray, V: np.ndarray,
                    R_bins: np.ndarray = np.arange(0, 2.05, 0.1),
                    v_gap: float = 500.0) -> np.ndarray:
    """
    Simple shifting-gapper interloper rejection in (R, V) space.
    R: projected radius in R200 units; V: LOS velocity offset [km/s]
    Returns boolean mask of kept galaxies.
    """
    R = np.asarray(R, float)
    V = np.asarray(V, float)
    keep = np.ones_like(R, dtype=bool)
    for i in range(len(R_bins)-1):
        m = (R >= R_bins[i]) & (R < R_bins[i+1]) & keep
        if np.sum(m) < 8:
            continue
        vv = np.sort(V[m])
        gaps = np.diff(vv)
        # cut at first big gap
        idx = np.where(gaps > v_gap)[0]
        if idx.size:
            vmax = vv[idx.min()]
            keep[m] &= (V[m] <= vmax)
    return keep


def per_cluster_profile(cluster: Dict, members: np.ndarray,
                        r200_key: str = 'R200', max_r_mult: float = 2.0,
                        vmax: float = 3000.0,
                        rbins: Iterable[float] = (0.0, 0.3, 0.6, 1.0, 1.5, 2.0)) -> np.ndarray:
    """
    Compute per-cluster robust Δv profile vs r/R200 using biweight.
    members must include fields: Rproj_Mpc, z_spec.
    Returns array shape (Nbins, 2): [r_mid, dv_biweight_kms].
    """
    c_kms = c / 1000.0
    zc = float(cluster['z_BCG'])
    r200 = float(cluster[r200_key])

    R = np.asarray(members['Rproj_Mpc'], float) / max(r200, 1e-9)
    dv = c_kms * (np.asarray(members['z_spec'], float) - zc) / (1.0 + zc)

    m = (R < max_r_mult) & (np.abs(dv) < vmax)
    R, dv = R[m], dv[m]
    if R.size == 0:
        rb = np.asarray(list(rbins), float)
        return np.column_stack([0.5*(rb[:-1]+rb[1:]), np.full(len(rb)-1, np.nan)])

    keep = shifting_gapper(R, dv)
    R, dv = R[keep], dv[keep]

    rb = np.asarray(list(rbins), float)
    mids = 0.5*(rb[:-1] + rb[1:])
    out = np.full((len(mids), 2), np.nan, dtype=float)
    out[:, 0] = mids
    for i in range(len(rb)-1):
        b = (R >= rb[i]) & (R < rb[i+1])
        if np.sum(b) >= 10:
            out[i, 1] = float(biweight_location(dv[b]))
    return out


def stack_profiles(profiles: List[np.ndarray], n_boot: int = 500, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equal-weight stack over clusters per bin (ignore NaNs), with bootstrap errors
    over clusters.
    Returns: r_mid, mu (km/s), sig (km/s)
    """
    arr = np.stack(profiles)  # (Ncl, Nbins, 2)
    rmid = arr[0, :, 0]
    vals = arr[:, :, 1]
    mu = np.nanmean(vals, axis=0)

    rng = np.random.default_rng(int(seed))
    B = int(max(10, n_boot))
    boots = np.empty((B, vals.shape[1]), dtype=float)
    for b in range(B):
        idx = rng.integers(0, vals.shape[0], size=vals.shape[0])
        boots[b] = np.nanmean(vals[idx], axis=0)
    sig = np.nanstd(boots, axis=0, ddof=1)
    return rmid, mu, sig


def predict_cluster_profile(cluster: Dict, rbins: np.ndarray, geff_callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Σ endpoint Δv per bin from geff(x): Δv = [Ψ(r)-Ψ(0)]/c (km/s)
    BCG-relative: subtract the BCG term to match observational Δv definition.
    Geometry: BCG at origin, observer at +x 10 Mpc; emitters placed at y=r.
    """
    Mpc = 3.085677581e22
    x_obs = np.array([10.0*Mpc, 0.0, 0.0])
    x_bcg = np.array([0.0, 0.0, 0.0])

    # Compute BCG term once per cluster
    z_bcg = gravitational_redshift_endpoint(x_bcg, x_obs, geff_callable, r_max=5.0*Mpc, n_steps=2000)

    def dv_at_r(phys_r_Mpc: float) -> float:
        x_emit = np.array([0.0, phys_r_Mpc*Mpc, 0.0])
        z_emit = gravitational_redshift_endpoint(x_emit, x_obs, geff_callable, r_max=5.0*Mpc, n_steps=2000)
        return (z_emit - z_bcg) * (c / 1000.0)  # BCG-relative

    r200 = float(cluster['R200'])
    rb = np.asarray(rbins, float)
    rmid = 0.5*(rb[:-1] + rb[1:])
    dv_pred = np.array([dv_at_r(r * r200) for r in rmid], dtype=float)
    return rmid, dv_pred


def stack_predictions(clusters: List[Dict], rbins: np.ndarray, geff_callable) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    for cl in clusters:
        _, dvp = predict_cluster_profile(cl, rbins, geff_callable)
        preds.append(dvp)
    arr = np.stack(preds)
    dv_mean = np.nanmean(arr, axis=0)
    rmid = 0.5*(np.asarray(rbins)[:-1] + np.asarray(rbins)[1:])
    return rmid, dv_mean