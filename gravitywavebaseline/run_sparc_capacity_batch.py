"""
Batch-evaluate the Milky Way capacity law on every SPARC rotation curve.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Batch SPARC capacity test runner.")
    parser.add_argument("--sparc-list", default="data/sparc/sparc_combined.csv",
                        help="CSV with galaxy names (column 'galaxy_name').")
    parser.add_argument("--rotmod-dir", default="data/Rotmod_LTG",
                        help="Directory with *_rotmod.dat files.")
    parser.add_argument("--alpha", type=float, default=2.8271)
    parser.add_argument("--gamma", type=float, default=0.8579)
    parser.add_argument("--force-match", action="store_true")
    parser.add_argument("--results-file", default="gravitywavebaseline/sparc_results_batch.csv")
    parser.add_argument("--sparc-summary", default="data/sparc/sparc_combined.csv")
    parser.add_argument("--shell-mode", choices=["data", "adaptive", "mass_adaptive", "gravity_adaptive"], default="data")
    parser.add_argument("--shell-r-factor", type=float, default=4.0)
    parser.add_argument("--shell-n-min", type=int, default=8)
    parser.add_argument("--shell-n-max", type=int, default=40)
    parser.add_argument("--alpha-scaling", choices=["constant", "mass_power", "sigma_power"],
                        default="constant")
    parser.add_argument("--gamma-scaling", choices=["constant", "mass_log", "sigma_log"],
                        default="constant")
    parser.add_argument("--mass-ref", type=float, default=6.0e10)
    parser.add_argument("--sigma-ref", type=float, default=30.0)
    parser.add_argument("--alpha-beta", type=float, default=0.0)
    parser.add_argument("--gamma-beta", type=float, default=0.0)
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=100.0)
    parser.add_argument("--gamma-min", type=float, default=-5.0)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument("--budget-factor", type=float, default=1.0)
    parser.add_argument("--g-floor", type=float, default=1e-5)
    parser.add_argument("--coh-radius-factor", type=float, default=3.0)
    parser.add_argument("--spillover-mode", choices=["capped", "mass_formula", "radius_formula", "hybrid"],
                        default="capped")
    parser.add_argument("--spill-alpha", type=float, default=0.5)
    parser.add_argument("--spill-beta", type=float, default=1.0)
    parser.add_argument("--spill-gamma", type=float, default=1.0)
    parser.add_argument("--spill-delta", type=float, default=2.0)
    parser.add_argument("--spill-gcut", type=float, default=1e-5)
    parser.add_argument("--max-galaxies", type=int, default=None,
                        help="Optional limit for quick tests.")
    args = parser.parse_args()

    df = pd.read_csv(args.sparc_list)
    galaxies = df["galaxy_name"].str.strip().tolist()
    if args.max_galaxies:
        galaxies = galaxies[: args.max_galaxies]

    results = []
    script = Path(__file__).resolve().parent / "sparc_capacity_test.py"

    for name in galaxies:
        cmd = [
            sys.executable,
            str(script),
            "--galaxy",
            name,
            "--rotmod-dir",
            args.rotmod_dir,
            "--alpha",
            str(args.alpha),
            "--gamma",
            str(args.gamma),
            "--sparc-summary",
            args.sparc_summary,
            "--shell-mode",
            args.shell_mode,
            "--shell-r-factor",
            str(args.shell_r_factor),
            "--shell-n-min",
            str(args.shell_n_min),
            "--shell-n-max",
            str(args.shell_n_max),
            "--g-floor",
            str(args.g_floor),
            "--coh-radius-factor",
            str(args.coh_radius_factor),
            "--alpha-scaling",
            args.alpha_scaling,
            "--gamma-scaling",
            args.gamma_scaling,
            "--mass-ref",
            str(args.mass_ref),
            "--sigma-ref",
            str(args.sigma_ref),
            "--alpha-beta",
            str(args.alpha_beta),
            "--gamma-beta",
            str(args.gamma_beta),
            "--alpha-min",
            str(args.alpha_min),
            "--alpha-max",
            str(args.alpha_max),
            "--gamma-min",
            str(args.gamma_min),
            "--gamma-max",
            str(args.gamma_max),
            "--budget-factor",
            str(args.budget_factor),
            "--spillover-mode",
            args.spillover_mode,
            "--spill-alpha",
            str(args.spill_alpha),
            "--spill-beta",
            str(args.spill_beta),
            "--spill-gamma",
            str(args.spill_gamma),
            "--spill-delta",
            str(args.spill_delta),
            "--spill-gcut",
            str(args.spill_gcut),
        ]
        if args.force_match:
            cmd.append("--force-match")
        print(f"\n[RUN] {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            results.append({"galaxy": name, "status": "error", "message": proc.stderr.strip()})
            continue

        report_path = (
            Path("gravitywavebaseline/sparc_results") / f"{name}_capacity_test.json"
        )
        if report_path.exists():
            import json

            data = json.loads(report_path.read_text())
            results.append(
                {
                    "galaxy": name,
                    "status": "ok",
                    "n_points": data["n_points"],
                    "alpha": data["alpha"],
                    "gamma": data["gamma"],
                    "alpha_eff": data.get("alpha_eff", data["alpha"]),
                    "gamma_eff": data.get("gamma_eff", data["gamma"]),
                    "alpha_scaling": data.get("alpha_scaling", args.alpha_scaling),
                    "gamma_scaling": data.get("gamma_scaling", args.gamma_scaling),
                    "shell_mode": data.get("shell_mode", args.shell_mode),
                    "shell_params": data.get("shell_parameters", {}),
                    "budget_factor": data.get("budget_factor", args.budget_factor),
                    "force_match": data["force_match"],
                    "rms_velocity": data["rms_velocity"],
                }
            )
        else:
            results.append({"galaxy": name, "status": "no_output"})

    pd.DataFrame(results).to_csv(args.results_file, index=False)
    print(f"\n[OK] Batch results saved to {args.results_file}")


if __name__ == "__main__":
    main()

