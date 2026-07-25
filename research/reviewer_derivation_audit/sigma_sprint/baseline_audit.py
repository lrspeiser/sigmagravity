"""Static/data audit of the submitted production baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_rotmod_radii(path):
    radii = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    radius, vobs, vgas, vdisk = (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                    vbulge = float(parts[5])
                except ValueError:
                    continue
                vbar2 = np.sign(vgas) * vgas**2 + 0.5 * vdisk**2 + 0.7 * vbulge**2
                if radius > 0 and vobs > 0 and vbar2 > 0:
                    radii.append(radius)
    return np.asarray(radii)


def sparc_scale_length_audit(true_rdisk_csv, rotmod_directory):
    truth = pd.read_csv(true_rdisk_csv).set_index("Name")
    rows = []
    for path in sorted(Path(rotmod_directory).glob("*_rotmod.dat")):
        radii = _load_rotmod_radii(path)
        galaxy = path.stem.replace("_rotmod", "")
        if len(radii) < 5 or galaxy not in truth.index:
            continue
        index = len(radii) // 3
        heuristic = radii[index] if index > 0 else radii[-1] / 2.0
        actual = float(truth.loc[galaxy, "Rdisk"])
        rows.append(
            {
                "galaxy": galaxy,
                "Rdisk_catalog_kpc": actual,
                "Rdisk_production_heuristic_kpc": heuristic,
                "heuristic_to_catalog_ratio": heuristic / actual,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "n_galaxies": int(len(frame)),
        "pearson_correlation": float(
            frame[["Rdisk_catalog_kpc", "Rdisk_production_heuristic_kpc"]].corr().iloc[0, 1]
        ),
        "median_heuristic_to_catalog_ratio": float(frame["heuristic_to_catalog_ratio"].median()),
        "catalog_scale_length_used_by_baseline_prediction": False,
    }
    return summary, frame


def static_baseline_audit(production_script):
    text = Path(production_script).read_text(encoding="utf-8")
    checks = {
        "coherence_uses_iterated_predicted_velocity": (
            "C = C_coherence(V, sigma_kms)" in text
            and "V_new = V_bar * np.sqrt(Sigma)" in text
        ),
        "cluster_C_is_one": "Sigma_baseline = 1 + A_cluster_baseline * h" in text,
        "cluster_length_fixed_600_kpc": "L_cluster = 600" in text,
        "cluster_baryon_shortcut_present": "M_bar_200 = 0.4 * f_baryon * M500" in text,
        "baseline_prediction_uses_Rdisk_argument": False,
        "unused_window_function_present": "def W_coherence" in text,
    }
    # R_d appears in the signature but not the executable body before the next section.
    start = text.index("def predict_velocity_baseline")
    end = text.index("# =============================================================================\n# NEW MODEL", start)
    body = text[start:end]
    checks["baseline_prediction_uses_Rdisk_argument"] = body.count("R_d") > 1
    return checks
