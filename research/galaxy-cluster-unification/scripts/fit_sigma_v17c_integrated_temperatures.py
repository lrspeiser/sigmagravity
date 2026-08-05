#!/usr/bin/env python3
"""Fit the frozen Sigma v17C integrated Chandra temperature spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"
DEFAULT_SPECTRA = ROOT / "results" / "sigma_v17c_integrated_spectra" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17c_integrated_temperatures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return str(value)
    return rendered if math.isfinite(rendered) else str(rendered)


def result_attributes(result: object, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: json_value(getattr(result, name))
        for name in names
        if hasattr(result, name)
    }


def primary_optimization_method(protocol: object) -> str:
    """Return the Sherpa method token from the frozen prose protocol."""

    if not isinstance(protocol, str):
        raise TypeError("optimization protocol must be a string")
    method = protocol.partition(";")[0].strip().lower()
    if method != "levmar":
        raise ValueError(
            "the frozen v17C optimization protocol must start with levmar"
        )
    return method


def configure_apec_data(ui, headas: Path | None = None) -> dict[str, Any]:
    """Bind XSPEC APEC to the AtomDB version declared by this installation."""

    if headas is None:
        raw_headas = os.environ.get("HEADAS")
        if not raw_headas:
            raise RuntimeError("HEADAS is not set; XSPEC model data are unavailable")
        headas = Path(raw_headas)
    headas = headas.resolve()
    init_candidates = (
        headas / "manager" / "Xspec.init",
        headas.parent / "spectral" / "manager" / "Xspec.init",
    )
    init_path = next((path for path in init_candidates if path.is_file()), None)
    if init_path is None:
        raise RuntimeError("the active XSPEC installation has no Xspec.init")
    match = re.search(
        r"^\s*ATOMDB_VERSION\s*:\s*([^\s#]+)",
        init_path.read_text(encoding="utf-8", errors="replace"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("Xspec.init does not declare ATOMDB_VERSION")
    version = match.group(1)
    data_candidates = (
        headas / "modelData",
        headas.parent / "spectral" / "modelData",
    )
    data_dir = next((path for path in data_candidates if path.is_dir()), None)
    if data_dir is None:
        raise RuntimeError("the active XSPEC installation has no modelData directory")
    root = data_dir / f"apec_v{version}"
    continuum = Path(f"{root}_coco.fits")
    lines = Path(f"{root}_line.fits")
    if not continuum.is_file() or not lines.is_file():
        raise RuntimeError(
            f"AtomDB {version} is declared but its APEC continuum/line files are absent"
        )
    ui.set_xsxset("APECROOT", str(root))
    return {
        "atomdb_version": version,
        "xspec_init": str(init_path),
        "xspec_init_sha256": sha256(init_path),
        "apec_root": str(root),
        "continuum_sha256": sha256(continuum),
        "line_sha256": sha256(lines),
    }


def evaluate_apec_probe(thermal, fit_lo: float, fit_hi: float) -> dict[str, Any]:
    """Fail closed if XSPEC cannot evaluate the configured plasma tables."""

    probe_hi = min(fit_hi, fit_lo + 0.5)
    probe_mid = 0.5 * (fit_lo + probe_hi)
    values = [
        float(value)
        for value in thermal(
            [fit_lo, probe_mid],
            [probe_mid, probe_hi],
        )
    ]
    total = sum(values)
    if not values or not all(finite_number(value) for value in values) or total <= 0:
        raise RuntimeError("APEC model-data probe returned no finite positive flux")
    return {
        "energy_bins_keV": [[fit_lo, probe_mid], [probe_mid, probe_hi]],
        "integrated_flux": total,
    }


def product_path(cluster: dict, role: str) -> Path:
    matches = [
        ROOT / item["relative_path"]
        for item in cluster["frozen_snapshot"]["products"]
        if item["role"] == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{cluster['cluster']} expected one {role}, found {matches}")
    path = matches[0]
    expected = next(
        item["sha256"]
        for item in cluster["frozen_snapshot"]["products"]
        if item["role"] == role
    )
    if sha256(path) != expected:
        raise RuntimeError(f"frozen spectrum product changed: {path}")
    return path


def run_fit(
    ui,
    thermal,
    method: str,
    sherpa_error: type[Exception],
) -> tuple[object, list[str]]:
    attempts = []
    ui.set_method(method)
    attempts.append(method)
    result = None
    try:
        ui.fit(1)
        result = ui.get_fit_results()
    except (sherpa_error, RuntimeError, TypeError, ValueError):
        pass
    if (
        result is not None
        and bool(getattr(result, "succeeded", False))
        and finite_number(getattr(result, "statval", None))
    ):
        return result, attempts

    ui.set_method("neldermead")
    attempts.append("neldermead")
    ui.fit(1)
    ui.set_method("levmar")
    attempts.append("levmar_polish")
    ui.fit(1)
    result = ui.get_fit_results()
    if not bool(getattr(result, "succeeded", False)) or not finite_number(
        getattr(result, "statval", None)
    ):
        raise RuntimeError(f"all optimization attempts failed for {thermal.name}")
    return result, attempts


def fit_cluster(cluster: dict, config: dict) -> dict[str, Any]:
    import numpy as np
    from sherpa.astro import ui
    from sherpa.utils.err import SherpaErr

    cluster_name = cluster["cluster"]
    cluster_config = config["clusters"][cluster_name]
    model_config = config["model"]
    source_pha = product_path(cluster, "grouped_source_spectrum")
    background_pha = product_path(cluster, "background_spectrum")
    arf = product_path(cluster, "source_arf")
    rmf = product_path(cluster, "source_rmf")

    ui.clean()
    apec_data = configure_apec_data(ui)
    abundance_table_token = model_config["abundance_table"].split(maxsplit=1)[0]
    ui.set_xsabund(abundance_table_token)
    ui.load_pha(1, str(source_pha))
    ui.set_analysis(1, "energy", "counts")
    fit_lo, fit_hi = map(float, model_config["fit_energy_keV"])
    ui.notice_id(1, fit_lo, fit_hi)
    data = ui.get_data(1)
    if not getattr(data, "background_ids", []):
        raise RuntimeError(f"{source_pha} does not reference a background spectrum")
    filtered_counts = float(np.sum(np.asarray(data.get_dep(filter=True), dtype=float)))
    exposure = float(data.exposure)
    if not finite_number(filtered_counts) or filtered_counts <= 0 or exposure <= 0:
        raise RuntimeError(f"invalid filtered counts or exposure in {source_pha}")
    count_rate = filtered_counts / exposure
    norm_initial = max(
        1e-8,
        0.01 * count_rate,
    )

    ui.subtract(1)
    absorption = ui.create_model_component("xstbabs", f"tbabs_{cluster_name.lower()}")
    thermal = ui.create_model_component("xsapec", f"apec_{cluster_name.lower()}")
    ui.set_source(1, absorption * thermal)

    absorption.nH = float(cluster_config["weighted_HI4PI_nH_cm2"]) / 1e22
    ui.freeze(absorption.nH)
    thermal.kT = float(model_config["temperature_keV"]["initial"])
    thermal.kT.min = float(model_config["temperature_keV"]["minimum"])
    thermal.kT.max = float(model_config["temperature_keV"]["maximum"])
    ui.thaw(thermal.kT)
    thermal.Abundanc = float(model_config["abundance_solar"]["integrated_initial"])
    thermal.Abundanc.min = float(model_config["abundance_solar"]["minimum"])
    thermal.Abundanc.max = float(model_config["abundance_solar"]["maximum"])
    ui.thaw(thermal.Abundanc)
    thermal.Redshift = float(cluster_config["redshift"])
    ui.freeze(thermal.Redshift)
    thermal.norm = norm_initial
    thermal.norm.min = float(model_config["normalization"]["minimum"])
    thermal.norm.max = float(model_config["normalization"]["maximum"])
    ui.thaw(thermal.norm)
    apec_probe = evaluate_apec_probe(thermal, fit_lo, fit_hi)

    ui.set_stat(model_config["statistic"])
    fit_result, attempts = run_fit(
        ui,
        thermal,
        primary_optimization_method(model_config["optimization"]),
        SherpaErr,
    )
    temperature = float(thermal.kT.val)
    abundance = float(thermal.Abundanc.val)
    normalization = float(thermal.norm.val)
    statval = float(fit_result.statval)
    dof = int(fit_result.dof)
    reduced_statistic = statval / dof if dof > 0 else math.nan

    conf_error = ""
    conf_result = None
    lower_delta = math.nan
    upper_delta = math.nan
    try:
        ui.set_conf_opt("sigma", 1.0)
        ui.conf(thermal.kT)
        conf_result = ui.get_conf_results()
        temperature_index = list(conf_result.parnames).index(thermal.kT.fullname)
        lower_delta = float(conf_result.parmins[temperature_index])
        upper_delta = float(conf_result.parmaxes[temperature_index])
    except (SherpaErr, RuntimeError, TypeError, ValueError) as exc:
        conf_error = f"{type(exc).__name__}: {exc}"
    lower = temperature + lower_delta
    upper = temperature + upper_delta
    published = float(cluster_config["published_global_temperature_keV_validation_only"])
    fractional_published_difference = abs(temperature / published - 1.0)

    finite_parameters = all(
        finite_number(value)
        for value in (temperature, abundance, normalization, lower, upper)
    )
    interval_ordered = finite_parameters and lower < temperature < upper
    gates = {
        "finite_temperature_abundance_and_interval": finite_parameters and interval_ordered,
        "reduced_statistic_at_most_1_5": finite_number(reduced_statistic)
        and reduced_statistic <= float(config["gates"]["integrated"]["maximum_reduced_statistic"]),
        "published_temperature_difference_at_most_20_percent": (
            fractional_published_difference
            <= float(
                config["gates"]["integrated"][
                    "maximum_fractional_difference_from_published_temperature"
                ]
            )
        ),
    }
    gates["all_passed"] = all(gates.values())

    return {
        "cluster": cluster_name,
        "fit_completed": True,
        "fit_exception": "",
        "source_spectrum": str(source_pha),
        "source_spectrum_sha256": sha256(source_pha),
        "background_spectrum": str(background_pha),
        "background_spectrum_sha256": sha256(background_pha),
        "arf_sha256": sha256(arf),
        "rmf_sha256": sha256(rmf),
        "fit_band_keV": [fit_lo, fit_hi],
        "background_subtracted": True,
        "grouped_source_counts_in_fit_band": filtered_counts,
        "source_exposure_s": exposure,
        "background_unsubtracted_count_rate_s": count_rate,
        "normalization_initial": norm_initial,
        "model": model_config["expression"],
        "xspec_atomic_data": apec_data,
        "apec_model_probe": apec_probe,
        "abundance_table": model_config["abundance_table"],
        "xspec_abundance_table_token": abundance_table_token,
        "statistic": model_config["statistic"],
        "optimization_attempts": attempts,
        "parameters": {
            "nH_1e22_cm2_fixed": float(absorption.nH.val),
            "redshift_fixed": float(thermal.Redshift.val),
            "temperature_keV": temperature,
            "abundance_solar": abundance,
            "normalization": normalization,
        },
        "temperature_confidence_68_percent": {
            "lower_delta_keV": json_value(lower_delta),
            "upper_delta_keV": json_value(upper_delta),
            "lower_keV": json_value(lower),
            "upper_keV": json_value(upper),
            "error": conf_error,
            "raw": result_attributes(
                conf_result,
                ("datasets", "methodname", "iterfitname", "fitname", "statname", "sigma", "percent", "parnames", "parvals", "parmins", "parmaxes", "nfits"),
            )
            if conf_result is not None
            else None,
        },
        "fit": {
            "statval": statval,
            "dof": dof,
            "reduced_statistic": json_value(reduced_statistic),
            "raw": result_attributes(
                fit_result,
                ("datasets", "methodname", "statname", "succeeded", "message", "nfev", "istatval", "statval", "dstatval", "numpoints", "dof", "qval", "rstat", "parnames", "parvals"),
            ),
        },
        "published_validation": {
            "temperature_keV": published,
            "fractional_difference": fractional_published_difference,
        },
        "gates": gates,
    }


def failed_cluster_result(cluster: dict, exc: Exception) -> dict[str, Any]:
    """Retain an attempted integrated fit as an explicit failed gate row."""

    return {
        "cluster": cluster["cluster"],
        "fit_completed": False,
        "fit_exception": f"{type(exc).__name__}: {exc}",
        "parameters": {
            "nH_1e22_cm2_fixed": None,
            "redshift_fixed": None,
            "temperature_keV": None,
            "abundance_solar": None,
            "normalization": None,
        },
        "temperature_confidence_68_percent": {
            "lower_delta_keV": None,
            "upper_delta_keV": None,
            "lower_keV": None,
            "upper_keV": None,
            "error": f"fit execution failed: {type(exc).__name__}: {exc}",
            "raw": None,
        },
        "fit": {
            "statval": None,
            "dof": None,
            "reduced_statistic": None,
            "raw": None,
        },
        "gates": {
            "finite_temperature_abundance_and_interval": False,
            "reduced_statistic_at_most_1_5": False,
            "published_temperature_difference_at_most_20_percent": False,
            "all_passed": False,
        },
    }


def main() -> None:
    from sherpa.utils.err import SherpaErr

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--spectra", type=Path, default=DEFAULT_SPECTRA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    spectra_path = args.spectra.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spectra = json.loads(spectra_path.read_text(encoding="utf-8"))
    if spectra["protocol_version"] != config["protocol_version"]:
        raise RuntimeError("spectral extraction and fit protocols differ")
    if spectra["status"] != "both_frozen_integrated_spectra_extracted_combined_and_grouped":
        raise RuntimeError("frozen integrated spectral extraction is incomplete")
    if spectra["config_sha256"] != sha256(config_path):
        raise RuntimeError("frozen spectral config changed after extraction")

    fits = []
    for cluster in spectra["clusters"]:
        try:
            fits.append(fit_cluster(cluster, config))
        except (SherpaErr, RuntimeError, TypeError, ValueError, OSError) as exc:
            fits.append(failed_cluster_result(cluster, exc))
    all_passed = all(
        row["fit_completed"] and row["gates"]["all_passed"] for row in fits
    )
    report = {
        "status": "both_integrated_temperature_gates_passed"
        if all_passed
        else "integrated_temperature_gate_failed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "spectral_extraction_report_sha256": sha256(spectra_path),
        "clusters": fits,
        "all_integrated_gates_passed": all_passed,
        "regional_fit_authorized": all_passed,
        "thermal_stress_constructed": False,
        "lensing_target_opened": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    for row in fits:
        if not row["fit_completed"]:
            print(
                f"{row['cluster']}: fit execution failed; pass=False; "
                f"{row['fit_exception']}",
                flush=True,
            )
            continue
        parameters = row["parameters"]
        print(
            f"{row['cluster']}: kT={parameters['temperature_keV']:.4f} keV, "
            f"Z={parameters['abundance_solar']:.4f}, "
            f"reduced statistic={row['fit']['reduced_statistic']}, "
            f"pass={row['gates']['all_passed']}",
            flush=True,
        )
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
