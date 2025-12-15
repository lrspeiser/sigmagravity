#!/usr/bin/env python3
"""Sweep the experimental high-g tail softening in h(g).

This script is designed to be run from the repo root, e.g.:

  python scripts/sweep_h_hi_power.py

It will call scripts/run_regression_experimental.py repeatedly with:
  --core
  --h=hi_power
  --h-hi-p <p_hi>
  --h-hi-x0 <x0>

and then parse the emitted JSON report to extract:
  - SPARC mean RMS
  - SPARC bulge/disk subset RMS (if present)
  - Cluster median ratio
  - Solar System |gamma-1|

Outputs:
  - scripts/regression_results/H_HI_POWER_SWEEP.json
  - scripts/regression_results/H_HI_POWER_SWEEP.md

Notes:
  - This sweep is CPU-expensive (SPARC has 171 galaxies). Keep grids small at first.
  - Use --coherence=jj etc as usual by adding them to EXTRA_ARGS below.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# USER-TUNABLE GRID
# ---------------------------
P_HI_GRID = [0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
X0_GRID = [1.0, 3.0, 10.0, 30.0]

# Add extra flags here if you want (examples):
#   EXTRA_ARGS = ['--coherence=jj', '--jj-xi-mult=0.4']
EXTRA_ARGS: List[str] = []

# ---------------------------
# Paths
# ---------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / 'scripts' / 'run_regression_experimental.py'
OUT_DIR = SCRIPT.parent / 'regression_results'
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class SweepRow:
    p_hi: float
    x0: float
    all_passed: bool
    sparc_rms: float
    sparc_bulge_rms: Optional[float]
    sparc_disk_rms: Optional[float]
    n_bulge: Optional[int]
    n_disk: Optional[int]
    clusters_ratio: float
    solar_gamma_minus_1: float
    report_path: str


def _find_result(report: Dict[str, Any], name: str) -> Dict[str, Any]:
    for r in report.get('results', []):
        if r.get('name') == name:
            return r
    raise KeyError(f"Result '{name}' not found in report")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def run_one(p_hi: float, x0: float) -> SweepRow:
    cmd = [
        'python', str(SCRIPT),
        '--core',
        '--h=hi_power',
        f'--h-hi-p={p_hi}',
        f'--h-hi-x0={x0}',
    ] + EXTRA_ARGS

    print(' '.join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))

    # Report location is determined by the regression script
    # (coherence model may change with EXTRA_ARGS)
    # We read the most recent experimental_report_*.json in OUT_DIR.
    reports = sorted(OUT_DIR.glob('experimental_report_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        raise FileNotFoundError(f"No experimental_report_*.json found in {OUT_DIR}")
    report_file = reports[0]

    report = json.loads(report_file.read_text())

    sparc = _find_result(report, 'SPARC Galaxies')
    clusters = _find_result(report, 'Clusters')
    solar = _find_result(report, 'Solar System')

    sparc_rms = float(sparc.get('metric', float('nan')))
    details = sparc.get('details', {}) or {}

    bulge_rms = _safe_float(details.get('mean_rms_bulge'))
    disk_rms = _safe_float(details.get('mean_rms_disk'))
    n_bulge = details.get('n_bulge')
    n_disk = details.get('n_disk')

    clusters_ratio = float(clusters.get('metric', float('nan')))
    solar_gamma = float(solar.get('metric', float('nan')))

    # Copy report to a uniquely named file for traceability
    tag = f"p{p_hi:g}_x0{x0:g}"
    copy_path = OUT_DIR / f"experimental_report_h_{tag}.json"
    copy_path.write_text(json.dumps(report, indent=2))

    return SweepRow(
        p_hi=float(p_hi),
        x0=float(x0),
        all_passed=bool(report.get('all_passed', False)),
        sparc_rms=sparc_rms,
        sparc_bulge_rms=bulge_rms,
        sparc_disk_rms=disk_rms,
        n_bulge=int(n_bulge) if isinstance(n_bulge, int) else None,
        n_disk=int(n_disk) if isinstance(n_disk, int) else None,
        clusters_ratio=clusters_ratio,
        solar_gamma_minus_1=solar_gamma,
        report_path=str(copy_path),
    )


def to_markdown(rows: List[SweepRow]) -> str:
    lines = []
    lines.append('# h(g) hi_power sweep')
    lines.append('')
    lines.append('This sweep varies the high-acceleration tail exponent `p_hi` and the knee `x0 = g/g†` where the modification turns on.')
    lines.append('')
    lines.append('| p_hi | x0 | SPARC RMS | Bulge RMS | Disk RMS | n_bulge | n_disk | Clusters ratio | Solar |γ-1| | all_passed | report |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--|')

    def fmt(x: Optional[float]) -> str:
        if x is None:
            return '—'
        if x != x:  # NaN
            return 'NaN'
        return f'{x:.3f}'

    for r in rows:
        lines.append(
            f"| {r.p_hi:g} | {r.x0:g} | {fmt(r.sparc_rms)} | {fmt(r.sparc_bulge_rms)} | {fmt(r.sparc_disk_rms)} | "
            f"{r.n_bulge if r.n_bulge is not None else '—'} | {r.n_disk if r.n_disk is not None else '—'} | {fmt(r.clusters_ratio)} | "
            f"{fmt(r.solar_gamma_minus_1)} | {'✓' if r.all_passed else '✗'} | {Path(r.report_path).name} |"
        )

    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    if not SCRIPT.exists():
        print(f"ERROR: {SCRIPT} not found. Edit SCRIPT path in this file.")
        return 2

    rows: List[SweepRow] = []
    for x0 in X0_GRID:
        for p_hi in P_HI_GRID:
            try:
                row = run_one(p_hi=p_hi, x0=x0)
                rows.append(row)
            except Exception as e:
                print(f"FAILED p_hi={p_hi}, x0={x0}: {e}")

    # Sort by SPARC RMS (then bulge RMS if present)
    def key(r: SweepRow) -> Tuple[float, float]:
        br = r.sparc_bulge_rms if r.sparc_bulge_rms is not None else 1e9
        return (r.sparc_rms, br)

    rows.sort(key=key)

    out_json = OUT_DIR / 'H_HI_POWER_SWEEP.json'
    out_md = OUT_DIR / 'H_HI_POWER_SWEEP.md'

    out_json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    out_md.write_text(to_markdown(rows))

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    if rows:
        best = rows[0]
        print(f"\nBest by SPARC RMS: p_hi={best.p_hi:g}, x0={best.x0:g}, RMS={best.sparc_rms:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


