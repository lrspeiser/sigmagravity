#!/usr/bin/env python3
"""
Tune amplitude A_0 only with fixed (p, n_coh) and gamma_bar=0 to realize a 4‑param model
(L_0, beta_bulge, alpha_disturb:=alpha_shear, A_0) for galaxy RAR.

Outputs tuned A_0 and test/train RAR; updates hyperparams_track2.json if --write is passed.
"""
import argparse, json
from pathlib import Path
import numpy as np

import sys
# Add repo root and many_path_model to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'many_path_model'))
from many_path_model.validation_suite import ValidationSuite
from many_path_model.path_spectrum_kernel_track2 import PathSpectrumHyperparams

PR = Path(__file__).resolve().parents[1]
CFG = PR / 'config' / 'hyperparams_track2.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=float, default=0.75)
    ap.add_argument('--ncoh', type=float, default=0.5)
    ap.add_argument('--gamma_bar', type=float, default=0.0)
    ap.add_argument('--grid', type=str, default='0.2:3.0:0.1', help='A0 grid start:stop:step')
    ap.add_argument('--write', action='store_true', help='Write tuned (A_0,p,n_coh,gamma_bar) into hyperparams_track2.json')
    args = ap.parse_args()

    # Load base HPs
    if CFG.exists():
        base = PathSpectrumHyperparams.from_dict(json.loads(CFG.read_text()))
    else:
        base = PathSpectrumHyperparams()

    vs = ValidationSuite(PR / 'results' / 'rar_tuning_4param', load_sparc=True)
    train_df, test_df = vs.perform_train_test_split()

    a_start, a_stop, a_step = map(float, args.grid.split(':'))
    grid = np.arange(a_start, a_stop + 1e-12, a_step)

    best = None
    best_rar = 1e9

    for A0 in grid:
        hp = PathSpectrumHyperparams(
            L_0=base.L_0,
            beta_bulge=base.beta_bulge,
            alpha_shear=base.alpha_shear,
            gamma_bar=args.gamma_bar,
            A_0=A0,
            p=args.p,
            n_coh=args.ncoh,
            g_dagger=base.g_dagger,
        )
        _, rar = vs.compute_btfr_rar(train_df, hp_override=hp)
        if rar < best_rar:
            best_rar = rar
            best = A0
        print(f"A0={A0:.3f} → train RAR={rar:.3f}")

    # Evaluate on test
    tuned_hp = PathSpectrumHyperparams(
        L_0=base.L_0,
        beta_bulge=base.beta_bulge,
        alpha_shear=base.alpha_shear,
        gamma_bar=args.gamma_bar,
        A_0=best,
        p=args.p,
        n_coh=args.ncoh,
        g_dagger=base.g_dagger,
    )
    btfr_t, rar_t = vs.compute_btfr_rar(test_df, hp_override=tuned_hp)

    print("\n=== Tuned 4‑param model (fixed p,n_coh; no bar gate) ===")
    print(f"A0* = {best:.3f}")
    print(f"RAR(train) ~ {best_rar:.3f} dex")
    print(f"RAR(test)  ~ {rar_t:.3f} dex")

    if args.write:
        d = base.to_dict()
        d.update({'A_0': float(best), 'p': float(args.p), 'n_coh': float(args.ncoh), 'gamma_bar': float(args.gamma_bar)})
        CFG.write_text(json.dumps(d, indent=2))
        print(f"Updated {CFG} with tuned (A_0,p,n_coh,gamma_bar)")

if __name__ == '__main__':
    main()
