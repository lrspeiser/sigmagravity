#!/usr/bin/env python3
"""
Generate rotation-curve gallery for SPARC galaxies using the paper's path‑spectrum kernel (Track‑2).

This script DOES NOT use the PDE/G³ surrogate. It uses:
- many_path_model/path_spectrum_kernel_track2.py (PathSpectrumKernel)
- g_total = g_bar * (1 + K), with K from many_path_boost_factor()

Outputs:
- many_path_model/paper_release/figures/rc_gallery.png

Usage (default SPARC path):
  python generate_rc_gallery.py \
    --sparc_dir data/Rotmod_LTG \
    --master   data/Rotmod_LTG/MasterSheet_SPARC.mrt \
    --galaxies UGC02953,UGC05253,NGC2403,UGC06787,UGC09133,UGC11914
"""
import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except Exception:
    HAS_SAVGOL = False

# Repo-relative imports
ROOT = Path(__file__).resolve().parents[3]  # repo root
KPC_TO_M = 3.0856776e19
KM_TO_M = 1000.0

import sys
sys.path.insert(0, str(ROOT))
from many_path_model.path_spectrum_kernel_track2 import (
    PathSpectrumKernel, PathSpectrumHyperparams
)


def load_rotmod(rotmod_path: Path):
    # Parse distance (optional)
    distance_mpc = None
    with open(rotmod_path, 'r') as f:
        for line in f:
            if 'Distance' in line:
                m = re.search(r'([\d.]+)\s*Mpc', line)
                if m:
                    distance_mpc = float(m.group(1))
                break
    # Load numeric columns
    rows = []
    with open(rotmod_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 8:
                try:
                    rows.append([float(x) for x in parts[:8]])
                except ValueError:
                    continue
    if not rows:
        raise ValueError(f'No numeric rows in {rotmod_path}')
    arr = np.array(rows)
    data = {
        'name': rotmod_path.stem.replace('_rotmod',''),
        'distance_mpc': distance_mpc,
        'R_kpc': arr[:,0],
        'Vobs': arr[:,1],
        'Verr': arr[:,2],
        'Vgas': arr[:,3],
        'Vdisk': arr[:,4],
        'Vbul': arr[:,5],
        'SBdisk': arr[:,6],
        'SBbul': arr[:,7],
    }
    return data


def _smooth(V: np.ndarray, win: int = 7) -> np.ndarray:
    win = max(3, win | 1)  # make odd
    if HAS_SAVGOL and V.size >= win:
        try:
            return savgol_filter(V, window_length=min(win, (V.size//2)*2+1), polyorder=2)
        except Exception:
            pass
    # Fallback: simple moving average
    k = min(5, max(3, V.size//6*2+1))
    out = np.copy(V)
    for i in range(V.size):
        i0 = max(0, i-k//2); i1 = min(V.size, i+k//2+1)
        out[i] = np.mean(V[i0:i1])
    return out


def predict_v_track2(gal, hp: PathSpectrumHyperparams, bar_strength: float = 0.0,
                      n_mc: int = 0) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    R = gal['R_kpc']
    Vgas = gal['Vgas']; Vdisk = gal['Vdisk']; Vbul = gal['Vbul']
    Vbar = np.sqrt(np.maximum(0.0, Vgas*Vgas + Vdisk*Vdisk + Vbul*Vbul))

    # B/T proxy per radius (velocity-squared weights)
    denom = np.maximum(1e-12, Vgas*Vgas + Vdisk*Vdisk + Vbul*Vbul)
    BT_rad = (Vbul*Vbul) / denom

    # Use observed V for shear (smoothed)
    Vobs = gal['Vobs']
    Vcirc = np.where(np.isfinite(Vobs) & (Vobs>0), Vobs, Vbar)
    Vcirc_s = _smooth(Vcirc)

    # g_bar from baryons in SI
    R_m = np.asarray(R)*KPC_TO_M
    g_bar = (np.asarray(Vbar)*KM_TO_M)**2 / np.maximum(R_m, 1e-9)

    # K per radius with local BT and smoothed V
    K = np.zeros_like(R, dtype=float)
    for i in range(len(R)):
        K[i] = float(kernel.many_path_boost_factor(
            r=float(R[i]), v_circ=float(Vcirc_s[i]), g_bar=float(g_bar[i]),
            BT=float(BT_rad[i]), bar_strength=float(bar_strength), r_bulge=1.0, r_bar=3.0, r_gate=0.5
        ))

    # g_total and V_pred
    g_tot = g_bar * (1.0 + np.asarray(K))
    Vpred = np.sqrt(np.clip(g_tot * R_m, 0.0, None)) / KM_TO_M

    # Optional Monte Carlo band from Vobs uncertainty
    V_lo = V_hi = None
    stats = {}
    if n_mc and 'Verr' in gal:
        Verr = np.asarray(gal['Verr'])
        draws = []
        rng = np.random.default_rng(42)
        for _ in range(n_mc):
            Vp = Vcirc + rng.normal(0.0, np.maximum(Verr, 2.0))
            Vp = _smooth(Vp)
            Kp = np.zeros_like(R, dtype=float)
            for i in range(len(R)):
                Kp[i] = float(kernel.many_path_boost_factor(
                    r=float(R[i]), v_circ=float(Vp[i]), g_bar=float(g_bar[i]),
                    BT=float(BT_rad[i]), bar_strength=float(bar_strength), r_bulge=1.0, r_bar=3.0, r_gate=0.5
                ))
            g_tot_p = g_bar * (1.0 + Kp)
            draws.append(np.sqrt(np.clip(g_tot_p * R_m, 0.0, None)) / KM_TO_M)
        D = np.vstack(draws)
        V_lo = np.percentile(D, 16, axis=0)
        V_hi = np.percentile(D, 84, axis=0)

    # Metrics
    Verr = np.asarray(gal['Verr'])
    mask = (Vobs>0)
    ape = float(np.mean(np.abs(Vpred[mask]-Vobs[mask]) / np.maximum(Vobs[mask], 1e-6))) * 100.0
    chi2 = float(np.sum(((Vpred[mask]-Vobs[mask]) / np.maximum(Verr[mask], 2.0))**2))
    chi2_red = chi2 / max(1, int(mask.sum())-1)
    stats = {'ape': ape, 'chi2_red': chi2_red}

    return Vpred, V_lo, V_hi, stats


def make_gallery(gal_list, out_png: Path, hp: PathSpectrumHyperparams,
                 bars_map: dict | None = None):
    n = len(gal_list)
    ncols = 3
    nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 3.8*nrows), squeeze=False)

    residuals_all = []
    for i, gal in enumerate(gal_list):
        r = i // ncols; c = i % ncols
        ax = axes[r][c]
        R = gal['R_kpc']
        Vobs = gal['Vobs']; Verr = gal['Verr']
        Vgas = gal['Vgas']; Vdisk = gal['Vdisk']; Vbul = gal['Vbul']
        Vbar = np.sqrt(np.maximum(0.0, Vgas*Vgas + Vdisk*Vdisk + Vbul*Vbul))
        bar_strength = 0.0
        if bars_map:
            key = gal['name']
            cls = bars_map.get(key) or bars_map.get(key.replace(' ',''))
            if cls:
                bar_strength = {'SB':0.7,'SAB':0.4,'SA':0.0,'S':0.0}.get(cls.upper(),0.0)
        Vpred, Vlo, Vhi, stats = predict_v_track2(gal, hp, bar_strength=bar_strength, n_mc=100)

        ax.errorbar(R, Vobs, yerr=Verr, fmt='k.', ms=3, lw=0.8, alpha=0.75, label='Observed')
        ax.plot(R, Vbar, 'c--', lw=1.2, label='GR (baryons)')
        if Vlo is not None and Vhi is not None:
            ax.fill_between(R, Vlo, Vhi, color='salmon', alpha=0.25, linewidth=0)
        ax.plot(R, Vpred, 'r-', lw=1.6, label='Σ‑Gravity (Track‑2)')
        ax.set_title(gal['name'])
        # annotate stats
        ax.text(0.02, 0.95, f"APE={stats['ape']:.1f}%\nχ²_ν={stats['chi2_red']:.2f}", transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        residuals_all.extend(list((Vpred - Vobs)))
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('V [km/s]')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Hide unused axes
    for j in range(i+1, nrows*ncols):
        r = j // ncols; c = j % ncols
        axes[r][c].axis('off')

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    # Residual histogram
    if residuals_all:
        fig2, ax2 = plt.subplots(1,1,figsize=(6,4))
        ax2.hist(residuals_all, bins=40, color='gray', alpha=0.8)
        ax2.set_xlabel('V_pred - V_obs [km/s]'); ax2.set_ylabel('Count')
        ax2.set_title('Rotation-curve residuals (Σ‑Gravity minus Observed)')
        hist_png = out_png.parent / 'rc_residual_hist.png'
        plt.tight_layout(); plt.savefig(hist_png, dpi=150); plt.close(fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc_dir', default=str(ROOT/'many_path_model'/'paper_release'/'data'/'Rotmod_LTG'))
    ap.add_argument('--master', default=str(ROOT/'many_path_model'/'paper_release'/'data'/'Rotmod_LTG'/'MasterSheet_SPARC.mrt'))
    ap.add_argument('--galaxies', default='UGC02953,UGC05253,NGC2403,UGC06787,UGC09133,UGC11914')
    ap.add_argument('--out', default=str(ROOT/'many_path_model'/'paper_release'/'figures'/'rc_gallery.png'))
    ap.add_argument('--hp', default=str(ROOT/'many_path_model'/'paper_release'/'config'/'hyperparams_track2.json'), help='Path to Track-2 hyperparameters JSON')
    args = ap.parse_args()

    names = [s.strip() for s in args.galaxies.split(',') if s.strip()]
    rotdir = Path(args.sparc_dir)

    gal_list = []
    for name in names:
        # rotmod files use no spaces in names
        rotmod = rotdir / f'{name}_rotmod.dat'
        if not rotmod.exists():
            # Try various common name normalizations
            alt = rotdir / f'{name.replace(" ", "")}_rotmod.dat'
            rotmod = alt if alt.exists() else rotmod
        if not rotmod.exists():
            raise SystemExit(f'Missing {rotmod}')
        gal_list.append(load_rotmod(rotmod))

    # Load hyperparameters
    import json
    hp_path = Path(args.hp)
    if hp_path.exists():
        with open(hp_path, 'r') as f:
            d = json.load(f)
        hp = PathSpectrumHyperparams.from_dict(d)
    else:
        hp = PathSpectrumHyperparams()

    # Load bars override mapping if present
    bars_path = ROOT/'many_path_model'/'paper_release'/'config'/'bars_override.json'
    bars_map = {}
    if bars_path.exists():
        try:
            bars_map = json.loads(bars_path.read_text())
        except Exception:
            bars_map = {}

    make_gallery(gal_list, Path(args.out), hp, bars_map=bars_map)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
