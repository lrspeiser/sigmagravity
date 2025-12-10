#!/usr/bin/env python3
"""
Generate cluster figures via the SigmaGravity pipeline and copy into paper_release/figures.
This script relies on the validated cluster code path (projected Σ‑kernel + triaxial) and
therefore delegates to projects/SigmaGravity scripts.
"""
import subprocess as sp
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
SG = ROOT / 'projects' / 'SigmaGravity'
FIG_OUT = ROOT / 'many_path_model' / 'paper_release' / 'figures'


def run(cmd):
    print('>>>', ' '.join(str(x) for x in cmd))
    rc = sp.call(cmd)
    if rc != 0:
        print(f'WARN: command failed with {rc}:', ' '.join(str(x) for x in cmd))
    return rc


def main():
    # 1) Validate holdouts (paper selection)
    run([sys.executable, str(SG/'scripts'/'validate_holdout_paper.py')])
    # 2) Generate holdouts_pred_vs_obs.png
    run([sys.executable, str(SG/'scripts'/'generate_paper_figures.py')])

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    src = SG / 'figures' / 'holdouts_pred_vs_obs.png'
    if src.exists():
        shutil.copy2(src, FIG_OUT / 'holdouts_pred_vs_obs.png')
        print('Copied cluster figure into paper_release/figures')
    else:
        print('WARN: cluster figure not found, skipping copy')

if __name__ == '__main__':
    main()
