#!/usr/bin/env python3
"""
Generate RAR validation plot using Track‑2 kernel on SPARC subset.

Outputs: many_path_model/paper_release/figures/rar_sparc_validation.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except Exception:
    HAS_SAVGOL = False

ROOT = Path(__file__).resolve().parents[3]  # repo root
KPC_TO_M = 3.0856776e19
KM_TO_M = 1000.0

import sys
sys.path.insert(0, str(ROOT))
from many_path_model.path_spectrum_kernel_track2 import (
    PathSpectrumKernel, PathSpectrumHyperparams
)


def load_rotmod(rotmod_path: Path):
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
    a = np.array(rows)
    return {
        'R_kpc': a[:,0], 'Vobs': a[:,1], 'Verr': a[:,2],
        'Vgas': a[:,3], 'Vdisk': a[:,4], 'Vbul': a[:,5]
    }


def _smooth(V: np.ndarray, win: int = 7) -> np.ndarray:
    win = max(3, win | 1)
    if HAS_SAVGOL and V.size >= win:
        try:
            return savgol_filter(V, window_length=min(win, (V.size//2)*2+1), polyorder=2)
        except Exception:
            pass
    k = min(5, max(3, V.size//6*2+1))
    out = np.copy(V)
    for i in range(V.size):
        i0 = max(0, i-k//2); i1 = min(V.size, i+k//2+1)
        out[i] = np.mean(V[i0:i1])
    return out


def rar_points(gal, kernel, bar_strength: float = 0.0):
    R = gal['R_kpc']
    Vgas = gal['Vgas']; Vdisk = gal['Vdisk']; Vbul = gal['Vbul']
    Vobs = gal['Vobs']
    Vbar = np.sqrt(np.maximum(0.0, Vgas*Vgas + Vdisk*Vdisk + Vbul*Vbul))

    # B/T per radius
    denom = np.maximum(1e-12, Vgas*Vgas + Vdisk*Vdisk + Vbul*Vbul)
    BT_rad = (Vbul*Vbul)/denom

    # Smooth observed velocity used for shear
    Vcirc = np.where(np.isfinite(Vobs) & (Vobs>0), Vobs, Vbar)
    Vcirc_s = _smooth(Vcirc)

    R_m = np.asarray(R)*KPC_TO_M
    g_bar = (np.asarray(Vbar)*KM_TO_M)**2 / np.maximum(R_m, 1e-9)
    g_obs = (np.asarray(Vobs)*KM_TO_M)**2 / np.maximum(R_m, 1e-9)

    # K per radius
    K = np.zeros_like(R, dtype=float)
    for i in range(len(R)):
        K[i] = float(kernel.many_path_boost_factor(
            r=float(R[i]), v_circ=float(Vcirc_s[i]), g_bar=float(g_bar[i]),
            BT=float(BT_rad[i]), bar_strength=float(bar_strength), r_bulge=1.0, r_bar=3.0, r_gate=0.5
        ))
    g_model = g_bar * (1.0 + np.asarray(K))
    return g_bar, g_obs, g_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparc_dir', default=str(ROOT/'many_path_model'/'paper_release'/'data'/'Rotmod_LTG'))
    ap.add_argument('--out', default=str(ROOT/'many_path_model'/'paper_release'/'figures'/'rar_sparc_validation.png'))
    ap.add_argument('--hp', default=str(ROOT/'many_path_model'/'paper_release'/'config'/'hyperparams_track2.json'))
    args = ap.parse_args()

    rotdir = Path(args.sparc_dir)
    files = sorted(rotdir.glob('*_rotmod.dat'))

    # Load bars overrides
    import json
    bars_path = ROOT/'many_path_model'/'paper_release'/'config'/'bars_override.json'
    bars_map = {}
    if bars_path.exists():
        try:
            bars_map = json.loads(bars_path.read_text())
        except Exception:
            bars_map = {}
    if not files:
        raise SystemExit(f'No *_rotmod.dat files in {rotdir}')

    # Load hyperparameters
    import json
    hp_path = Path(args.hp)
    if hp_path.exists():
        d = json.loads(hp_path.read_text())
        hp = PathSpectrumHyperparams.from_dict(d)
    else:
        hp = PathSpectrumHyperparams()
    kernel = PathSpectrumKernel(hp, use_cupy=False)

    gb_all=[]; go_all=[]; gm_all=[]
    # Use up to 80 galaxies for speed
    for f in files[:80]:
        gal = load_rotmod(f)
        name = f.stem.replace('_rotmod','')
        cls = bars_map.get(name) or bars_map.get(name.replace(' ',''))
        bar_strength = {'SB':0.7,'SAB':0.4,'SA':0.0,'S':0.0}.get(str(cls).upper(),0.0) if cls else 0.0
        gb, go, gm = rar_points(gal, kernel, bar_strength=bar_strength)
        gb_all.append(gb); go_all.append(go); gm_all.append(gm)

    gb = np.concatenate(gb_all); go = np.concatenate(go_all); gm = np.concatenate(gm_all)

    fig, ax = plt.subplots(1,1,figsize=(7,6))
    ax.scatter(np.log10(gb), np.log10(go), s=3, c='k', alpha=0.35, label='Observed')
    ax.scatter(np.log10(gb), np.log10(gm), s=3, c='crimson', alpha=0.35, label='Σ‑Gravity (Track‑2)')
    x = np.linspace(np.log10(gb.min()), np.log10(gb.max()), 200)
    ax.plot(x, x, 'c--', lw=1.0, label='GR (baryons)')
    ax.set_xlabel('log10 g_bar [m/s²]'); ax.set_ylabel('log10 g [m/s²]')
    ax.set_title('RAR: SPARC subset with Track‑2 kernel')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(fig)
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
