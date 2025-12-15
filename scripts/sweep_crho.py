#!/usr/bin/env python3
"""Sweep CRHO (density/vorticity coherence) parameters.

This script sweeps CRHO_SCALE and rho_threshold to find optimal values
for improving bulge predictions without hurting disk or clusters.

Usage:
    python scripts/sweep_crho.py
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re


# ---------------------------
# USER-TUNABLE GRID
# ---------------------------
CRHO_SCALE_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
RHO_THRESHOLD_GRID = [1e-19, 3e-19, 5e-19, 1e-18, 3e-18]  # kg/m³

# ---------------------------
# Paths
# ---------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / 'scripts' / 'run_regression_experimental.py'
OUT_DIR = SCRIPT.parent / 'regression_results'
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class SweepRow:
    crho_scale: float
    rho_threshold: float
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


def run_one(scale: float, threshold: float) -> SweepRow:
    cmd = [
        'python', str(SCRIPT),
        '--core',
        '--coherence=crho',
        f'--crho-scale={scale}',
        f'--crho-threshold={threshold}',
    ]

    print(f'Running: CRHO_SCALE={scale}, rho_threshold={threshold:.2e}')
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    # Report location
    reports = sorted(OUT_DIR.glob('experimental_report_CRHO.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        raise FileNotFoundError(f"No experimental_report_CRHO.json found in {OUT_DIR}")
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

    # Copy report to a uniquely named file
    tag = f"s{scale:g}_t{threshold:.2e}".replace('.', 'p').replace('e', 'e').replace('+', '')
    copy_path = OUT_DIR / f"experimental_report_crho_{tag}.json"
    copy_path.write_text(json.dumps(report, indent=2))

    return SweepRow(
        crho_scale=float(scale),
        rho_threshold=float(threshold),
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
    lines.append('# CRHO parameter sweep')
    lines.append('')
    lines.append('This sweep varies CRHO_SCALE and rho_threshold to find optimal values for bulge improvement.')
    lines.append('')
    lines.append('| CRHO_SCALE | rho_threshold | SPARC RMS | Bulge RMS | Disk RMS | n_bulge | n_disk | Clusters ratio | Solar |γ-1| | all_passed | report |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--|')

    def fmt(x: Optional[float]) -> str:
        if x is None:
            return '—'
        if x != x:  # NaN
            return 'NaN'
        if abs(x) < 1e-6:
            return '0.000'
        return f'{x:.3f}'

    for r in rows:
        lines.append(
            f"| {r.crho_scale:g} | {r.rho_threshold:.2e} | {fmt(r.sparc_rms)} | {fmt(r.sparc_bulge_rms)} | {fmt(r.sparc_disk_rms)} | "
            f"{r.n_bulge if r.n_bulge is not None else '—'} | {r.n_disk if r.n_disk is not None else '—'} | {fmt(r.clusters_ratio)} | "
            f"{fmt(r.solar_gamma_minus_1)} | {'✓' if r.all_passed else '✗'} | {Path(r.report_path).name} |"
        )

    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    if not SCRIPT.exists():
        print(f"ERROR: {SCRIPT} not found.")
        return 2

    rows: List[SweepRow] = []
    for threshold in RHO_THRESHOLD_GRID:
        for scale in CRHO_SCALE_GRID:
            try:
                row = run_one(scale=scale, threshold=threshold)
                rows.append(row)
            except Exception as e:
                print(f"FAILED scale={scale}, threshold={threshold:.2e}: {e}")

    # Sort by bulge RMS (lower is better), then overall RMS
    def key(r: SweepRow) -> Tuple[float, float]:
        br = r.sparc_bulge_rms if r.sparc_bulge_rms is not None else 1e9
        return (br, r.sparc_rms)

    rows.sort(key=key)

    out_json = OUT_DIR / 'CRHO_SWEEP.json'
    out_md = OUT_DIR / 'CRHO_SWEEP.md'

    out_json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    out_md.write_text(to_markdown(rows))

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    if rows:
        best = rows[0]
        print(f"\nBest by bulge RMS: scale={best.crho_scale:g}, threshold={best.rho_threshold:.2e}")
        bulge_str = f"{best.sparc_bulge_rms:.3f}" if best.sparc_bulge_rms is not None else "N/A"
        print(f"  SPARC RMS: {best.sparc_rms:.3f}, Bulge RMS: {bulge_str}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

