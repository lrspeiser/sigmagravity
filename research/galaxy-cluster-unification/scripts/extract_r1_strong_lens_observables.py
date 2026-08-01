from __future__ import annotations

import argparse
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_tar_text(path: Path, member: str) -> str:
    with tarfile.open(path) as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"{path}: missing {member}")
        return handle.read().decode("utf-8", errors="replace")


def _source_family(image_id: str, *, a2537_clump_scheme: bool = False) -> int:
    prefix = image_id.split(".", 1)[0]
    number = int(re.match(r"\d+", prefix).group(0))
    if a2537_clump_scheme and number >= 10:
        return int(str(number)[0])
    return number


def _clean_latex_cell(value: str) -> str:
    value = value.strip().replace("$", "")
    value = value.replace(r"\ldots", "...").replace(r"\dots", "...")
    value = value.replace(r"\\", "").strip()
    return value


def _newman_family(system: str, image_id: str) -> str:
    plain = re.sub(r"[^A-Za-z0-9.]", "", image_id)
    if system == "A2390":
        if plain.startswith("41"):
            return "41"
        if plain.startswith("51"):
            return "51"
        if plain.startswith("B"):
            return "B"
        if plain.startswith("H32"):
            return "H3"
        if plain.startswith("H51"):
            return "H5"
    match = re.match(r"(\d+)", plain)
    if match is None:
        raise ValueError(f"cannot derive source family for {system} image {image_id}")
    return match.group(1)


def _parse_newman_table(path: Path) -> pd.DataFrame:
    """Parse the BCG-relative image-plane inputs in Newman et al. (2013)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    label = text.index(r"\tablecaption{Positions of multiple images")
    start = text.index(r"\startdata", label)
    end = text.index(r"\enddata", start)
    selected = {"A2390", "A2537", "A2667", "A383", "A611", "MS2137"}
    current_system = [None, None]
    parsed = []
    for raw_line in text[start + len(r"\startdata") : end].splitlines():
        fields = raw_line.split("&")
        if len(fields) < 6:
            continue
        if len(fields) < 12:
            fields += [""] * (12 - len(fields))
        for side, offset in enumerate((0, 6)):
            cells = [_clean_latex_cell(value) for value in fields[offset : offset + 6]]
            cluster_cell, image_cell, dx_cell, dy_cell, redshift_cell, source_cell = cells
            if cluster_cell and cluster_cell not in {"...", "---"}:
                current_system[side] = re.sub(r"[^A-Za-z0-9]", "", cluster_cell)
            system = current_system[side]
            if system not in selected or not image_cell:
                continue
            image_plain = re.sub(r"\^\{.*?\}|\^.|[${}]", "", image_cell).strip()
            if not re.search(r"[A-Za-z0-9]", image_plain):
                continue
            try:
                delta_x = float(_clean_latex_cell(dx_cell))
                delta_y = float(_clean_latex_cell(dy_cell))
            except ValueError:
                continue
            redshift_text = _clean_latex_cell(redshift_cell)
            try:
                redshift = float(redshift_text)
            except ValueError:
                redshift = np.nan
            used = not (system == "A383" and "dagger" in image_cell.lower())
            parsed.append(
                {
                    "source_sample": "Newman2013",
                    "system": system,
                    "image_id": image_plain,
                    "source_family": _newman_family(system, image_plain),
                    "ra_deg": np.nan,
                    "dec_deg": np.nan,
                    "delta_x_west_arcsec": delta_x,
                    "delta_y_north_arcsec": delta_y,
                    "coordinate_status": "published_offsets_relative_to_bcg",
                    "image_position_sigma_arcsec": 1.0 if system == "A2390" else 0.5,
                    "source_redshift": redshift,
                    "redshift_kind": "spectroscopic" if np.isfinite(redshift) else "not_available",
                    "published_model_redshift": np.nan,
                    "used_in_published_fiducial": used,
                    "observable_level_image_position": used,
                    "source_archive": _display_path(path),
                    "source_member": "paper1.tex",
                    "source_table": "tab:slimages",
                    "position_covariance_status": (
                        "published independent diagonal image-plane likelihood; "
                        + ("1.0" if system == "A2390" else "0.5")
                        + " arcsec per coordinate; no systematic covariance"
                    ),
                    "likelihood_source_archive": _display_path(path),
                    "likelihood_source_member": "paper1.tex",
                    "likelihood_source_equation": "published image-plane chi-square, eq. chi2_SL",
                    "classification_note": (
                        "published image-plane constraint"
                        if used
                        else "published but explicitly not used as a lens constraint"
                    ),
                }
            )
    frame = pd.DataFrame(parsed)
    expected = {"A2390": 13, "A2537": 16, "A2667": 13, "A383": 15, "A611": 12, "MS2137": 11}
    observed = frame.groupby("system").size().to_dict()
    if observed != expected:
        raise ValueError(f"unexpected Newman multiple-image counts: {observed}; expected {expected}")
    explicit_redshifts = (
        frame.loc[frame["source_redshift"].notna(), ["system", "source_family", "source_redshift"]]
        .drop_duplicates(["system", "source_family"])
        .set_index(["system", "source_family"])["source_redshift"]
    )
    for index, row in frame.iterrows():
        key = (row["system"], row["source_family"])
        if key in explicit_redshifts.index:
            frame.at[index, "source_redshift"] = float(explicit_redshifts.loc[key])
            frame.at[index, "redshift_kind"] = "spectroscopic"
    return frame


def _parse_kaleidoscope_table(path: Path, *, label: str, system: str) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    label_index = text.index(r"\label{" + label + "}")
    start = text.index(r"\begin{tabular}", label_index)
    end = text.index(r"\end{tabular}", start)
    block = text[start:end]
    pattern = re.compile(
        r"(?m)^\s*([0-9]+\.[0-9]+\*?)\s*&\s*"
        r"([0-9.]+)\s*&\s*(-?[0-9.]+)\s*&\s*([0-9.]+)\s*&"
    )
    rows = []
    for match in pattern.finditer(block):
        image_id_raw, ra_text, dec_text, redshift_text = match.groups()
        image_id = image_id_raw.rstrip("*")
        family = image_id.split(".", 1)[0]
        ra_deg = float(ra_text)
        dec_deg = float(dec_text)
        if system == "MACS J0326":
            # The text explicitly names 1.1 and 2.4 as predictions and says four
            # of twelve are predictions. 3.4 and 3.5 are the two remaining table
            # entries without a same-redshift MUSE source match; keep the
            # inference visible and exclude all four from measured observables.
            if image_id in {"1.1", "2.4"}:
                coordinate_status = "model_predicted_image_explicit_in_text"
                observable = False
                note = "paper explicitly identifies this as a model-predicted image"
            elif image_id in {"3.4", "3.5"}:
                coordinate_status = "model_predicted_image_inferred_from_muse_nonmatch"
                observable = False
                note = "inferred remaining prediction: no same-redshift MUSE-coordinate match"
            else:
                coordinate_status = "published_observed_image_coordinate"
                observable = True
                note = "published image coordinate; not among four excluded predictions"
        else:
            corrupt = image_id in {"1.1", "1.2"} and dec_deg == ra_deg
            predicted = image_id_raw.endswith("*")
            if corrupt:
                coordinate_status = "source_table_coordinate_corrupt_dec_equals_ra"
                observable = False
                note = "source table repeats R.A. in the Decl. column"
            elif predicted:
                coordinate_status = "model_predicted_image_marked_by_source"
                observable = False
                note = "source table marks this image with an asterisk as predicted"
            else:
                coordinate_status = "published_observed_image_coordinate"
                observable = True
                note = "published observed image coordinate"
        rows.append(
            {
                "source_sample": "Kaleidoscope2025",
                "system": system,
                "image_id": image_id,
                "source_family": family,
                "ra_deg": ra_deg if coordinate_status != "source_table_coordinate_corrupt_dec_equals_ra" else np.nan,
                "dec_deg": dec_deg if coordinate_status != "source_table_coordinate_corrupt_dec_equals_ra" else np.nan,
                "delta_x_west_arcsec": np.nan,
                "delta_y_north_arcsec": np.nan,
                "coordinate_status": coordinate_status,
                "image_position_sigma_arcsec": 0.5,
                "source_redshift": float(redshift_text),
                "redshift_kind": "spectroscopic",
                "published_model_redshift": np.nan,
                "used_in_published_fiducial": True,
                "observable_level_image_position": observable,
                "source_archive": _display_path(path),
                "source_member": path.name,
                "source_table": label,
                "position_covariance_status": (
                    "published independent diagonal image-plane likelihood; "
                    "0.5 arcsec per image; no systematic covariance"
                ),
                "likelihood_source_archive": _display_path(path),
                "likelihood_source_member": path.name,
                "likelihood_source_equation": "published image-plane chi-square, 0.5 arcsec sigma_ij",
                "classification_note": note,
            }
        )
    frame = pd.DataFrame(rows)
    expected = 12 if system == "MACS J0326" else 6
    if len(frame) != expected:
        raise ValueError(f"expected {expected} {system} table entries, parsed {len(frame)}")
    return frame


def _parse_a2537(path: Path) -> pd.DataFrame:
    text = _read_tar_text(path, "main.tex")
    start = text.index(r"\tablecaption{Abell~2537 -- Properties of lensed galaxies")
    end = text.index(r"\enddata", start)
    block = text[start:end]
    pattern = re.compile(
        r"(?m)^\s*([0-9]+\.[0-9]+)\s*&\s*"
        r"([0-9]{2}:[0-9]{2}:[0-9.]+)\s*&\s*"
        r"([+-][0-9]{2}:[0-9]{2}:[0-9.]+)\s*&"
    )
    redshifts = {
        1: (np.nan, "model_optimized_not_observable", 2.11),
        2: (3.611, "spectroscopic", 3.611),
        3: (3.2, "fixed_photometric_point_estimate", 3.2),
        4: (np.nan, "model_optimized_not_observable", 4.24),
    }
    rows = []
    for match in pattern.finditer(block):
        image_id, ra_text, dec_text = match.groups()
        coordinate = SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg), frame="icrs")
        family = _source_family(image_id, a2537_clump_scheme=True)
        redshift, redshift_kind, published_model_redshift = redshifts[family]
        rows.append(
            {
                "source_sample": "Cerny2018",
                "system": "A2537",
                "image_id": image_id,
                "source_family": family,
                "ra_deg": coordinate.ra.deg,
                "dec_deg": coordinate.dec.deg,
                "image_position_sigma_arcsec": 0.3,
                "source_redshift": redshift,
                "redshift_kind": redshift_kind,
                "published_model_redshift": published_model_redshift,
                "used_in_published_fiducial": True,
                "source_archive": _display_path(path),
                "source_member": "main.tex",
                "source_table": "tab.A2537arcs",
                "position_covariance_status": "published 0.3 arcsec per-image uncertainty; no systematic covariance",
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 27:
        raise ValueError(f"expected 27 A2537 image constraints, parsed {len(frame)}")
    return frame


def _parse_macs0417(path: Path) -> pd.DataFrame:
    text = _read_tar_text(path, "arcs_bronze.tex")
    start = text.index(r"\startdata")
    end = text.index(r"\enddata", start)
    block = text[start:end]
    pattern = re.compile(
        r"(?m)^\s*([0-9]+[a-z]?\.[0-9]+)\s*&\s*"
        r"([0-9.]+)\s*&\s*(-?[0-9.]+)\s*&"
    )
    spectroscopic = {1: 0.8710, 2: 1.0460, 3: 1.0460}
    model_redshifts = {
        4: 2.26,
        5: 2.27,
        6: 2.34,
        7: 2.09,
        8: 2.39,
        9: 5.97,
        10: 2.02,
        11: 3.47,
        12: 2.84,
        13: 2.89,
        14: 4.40,
        15: 2.11,
        16: 4.50,
        17: 2.30,
    }
    bronze_families = {7, 14}
    rows = []
    for match in pattern.finditer(block):
        image_id, ra_text, dec_text = match.groups()
        family = _source_family(image_id)
        mixed_bronze = image_id == "10.3"
        used = family not in bronze_families and not mixed_bronze
        if family in spectroscopic:
            redshift = spectroscopic[family]
            redshift_kind = "spectroscopic"
            published_model_redshift = redshift
        else:
            redshift = np.nan
            redshift_kind = "model_optimized_not_observable"
            published_model_redshift = model_redshifts[family]
        rows.append(
            {
                "source_sample": "Mahler2019",
                "system": "MACS J0417",
                "image_id": image_id,
                "source_family": family,
                "ra_deg": float(ra_text),
                "dec_deg": float(dec_text),
                "image_position_sigma_arcsec": 0.5,
                "source_redshift": redshift,
                "redshift_kind": redshift_kind,
                "published_model_redshift": published_model_redshift,
                "used_in_published_fiducial": used,
                "source_archive": _display_path(path),
                "source_member": "arcs_bronze.tex",
                "source_table": "tab:arcs",
                "position_covariance_status": (
                    "Kaleidoscope2025 publishes the reused-model image-plane likelihood "
                    "with 0.5 arcsec independent errors; no systematic covariance"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 57:
        raise ValueError(f"expected 57 MACS J0417 image constraints, parsed {len(frame)}")
    return frame


def _parse_macs0949(path: Path) -> pd.DataFrame:
    text = _read_tar_text(path, "version22.tex")
    label = text.index(r"\label{tab:spectro_mul_m0949}")
    start = text.index(r"\begin{tabular}", label)
    end = text.index(r"\end{tabular}", start)
    block = text[start:end]
    pattern = re.compile(
        r"(?m)^\s*([0-9]+\.[0-9]+)\s*&\s*"
        r"([0-9.]+)\s*&\s*([0-9.]+)\s*&"
    )
    redshifts = {
        1: (4.8902, "spectroscopic", 4.8902),
        2: (4.8844, "spectroscopic", 4.8844),
        3: (np.nan, "model_optimized_not_observable", 4.85),
        4: (np.nan, "model_optimized_not_observable", 3.76),
        5: (np.nan, "model_optimized_not_observable", 3.63),
        6: (np.nan, "model_optimized_not_observable", 3.57),
    }
    rows = []
    for match in pattern.finditer(block):
        image_id, ra_text, dec_text = match.groups()
        family = _source_family(image_id)
        redshift, redshift_kind, published_model_redshift = redshifts[family]
        rows.append(
            {
                "source_sample": "Allingham2023",
                "system": "MACS J0949",
                "image_id": image_id,
                "source_family": family,
                "ra_deg": float(ra_text),
                "dec_deg": float(dec_text),
                "image_position_sigma_arcsec": 0.5,
                "source_redshift": redshift,
                "redshift_kind": redshift_kind,
                "published_model_redshift": published_model_redshift,
                "used_in_published_fiducial": True,
                "source_archive": _display_path(path),
                "source_member": "version22.tex",
                "source_table": "tab:spectro_mul_m0949",
                "position_covariance_status": (
                    "Kaleidoscope2025 publishes the reused-model image-plane likelihood "
                    "with 0.5 arcsec independent errors; no systematic covariance"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 20:
        raise ValueError(f"expected 20 MACS J0949 image constraints, parsed {len(frame)}")
    return frame


def extract(
    *,
    source_dir: Path,
    replacement_source_dir: Path,
    sample_config_path: Path,
    center_config_path: Path,
    dynamics_path: Path,
    image_output: Path,
    support_output: Path,
    report_output: Path,
) -> dict:
    images = pd.concat(
        [
            _parse_newman_table(replacement_source_dir / "paper1.tex"),
            _parse_kaleidoscope_table(
                replacement_source_dir / "kaleidoscope2025_main.tex",
                label="tab.ms0326arcs",
                system="MACS J0326",
            ),
            _parse_kaleidoscope_table(
                replacement_source_dir / "kaleidoscope2025_main.tex",
                label="tab.ms1427arcs",
                system="MACS J1427",
            ),
            _parse_macs0417(source_dir / "mahler2019_macs0417_relics_source.tar"),
            _parse_macs0949(source_dir / "allingham2023_macs0949_source.tar"),
        ],
        ignore_index=True,
    )
    kaleidoscope_likelihood_path = replacement_source_dir / "kaleidoscope2025_main.tex"
    reused_model_rows = images["system"].isin(["MACS J0417", "MACS J0949"])
    images.loc[reused_model_rows, "likelihood_source_archive"] = _display_path(
        kaleidoscope_likelihood_path
    )
    images.loc[reused_model_rows, "likelihood_source_member"] = kaleidoscope_likelihood_path.name
    images.loc[reused_model_rows, "likelihood_source_equation"] = (
        "published image-plane chi-square, 0.5 arcsec sigma_ij"
    )
    for column, default in {
        "delta_x_west_arcsec": np.nan,
        "delta_y_north_arcsec": np.nan,
        "coordinate_status": "published_observed_image_coordinate",
        "classification_note": "published observed image coordinate",
        "observable_level_image_position": True,
    }.items():
        images[column] = (
            images[column].where(images[column].notna(), default) if column in images else default
        )
    images["source_family"] = images["source_family"].astype(str)
    images["observable_level_image_position"] = images["observable_level_image_position"].astype(bool)
    images["alternative_metric_likelihood_ready"] = (
        images["observable_level_image_position"]
        & images["used_in_published_fiducial"].astype(bool)
        & images["image_position_sigma_arcsec"].notna()
        & images["source_redshift"].notna()
        & images["redshift_kind"].isin(["spectroscopic"])
    )

    centers = json.loads(center_config_path.read_text(encoding="utf-8"))["systems"]
    sample_records = pd.DataFrame(
        json.loads(sample_config_path.read_text(encoding="utf-8"))["records"]
    )
    selected_sources = (
        sample_records.sort_values("source_priority", ascending=False)
        .drop_duplicates("system", keep="first")
        .set_index("system")["source_sample"]
        .to_dict()
    )
    dynamics = pd.read_csv(dynamics_path)
    support_rows = []
    summaries = {}
    for system, group in images.groupby("system", sort=True):
        center = centers.get(system, {"centering_verified": False})
        selected_source = selected_sources[system]
        selected_dynamics = dynamics.loc[
            (dynamics["system"] == system) & (dynamics["source_sample"] == selected_source)
        ]
        dynamics_max = float(selected_dynamics["bin_max_arcsec"].max())
        direct_offsets = group["delta_x_west_arcsec"].notna() & group["delta_y_north_arcsec"].notna()
        if direct_offsets.all():
            radii = np.hypot(
                group["delta_x_west_arcsec"].to_numpy(dtype=float),
                group["delta_y_north_arcsec"].to_numpy(dtype=float),
            )
            centering_verified = True
            radial_support_auditable = True
            radial_reference = "published offsets relative to the BCG"
        elif center["centering_verified"]:
            bcg = SkyCoord(center["bcg_ra_deg"] * u.deg, center["bcg_dec_deg"] * u.deg)
            image_coordinates = SkyCoord(group["ra_deg"].to_numpy() * u.deg, group["dec_deg"].to_numpy() * u.deg)
            radii = bcg.separation(image_coordinates).arcsec
            centering_verified = True
            radial_support_auditable = bool(np.isfinite(radii).any())
            radial_reference = "verified absolute BCG coordinate"
        else:
            radii = np.full(len(group), np.nan)
            centering_verified = False
            radial_support_auditable = False
            radial_reference = "not auditable from published coordinates"
        used = group["used_in_published_fiducial"].to_numpy(dtype=bool)
        observable = group["observable_level_image_position"].to_numpy(dtype=bool)
        likelihood_ready = group["alternative_metric_likelihood_ready"].to_numpy(dtype=bool)
        within = np.isfinite(radii) & (radii <= dynamics_max) & used & observable
        strict_within = within & likelihood_ready
        for row, radius, is_within in zip(group.itertuples(index=False), radii, within, strict=True):
            support_rows.append(
                {
                    "system": system,
                    "image_id": row.image_id,
                    "source_family": row.source_family,
                    "bcg_centering_verified": centering_verified,
                    "radial_reference": radial_reference,
                    "bcg_centric_radius_arcsec": radius,
                    "dynamics_r_max_arcsec": dynamics_max,
                    "used_in_published_fiducial": row.used_in_published_fiducial,
                    "observable_level_image_position": row.observable_level_image_position,
                    "alternative_metric_likelihood_ready": row.alternative_metric_likelihood_ready,
                    "inside_dynamics_support": bool(is_within),
                }
            )
        used_group = group.loc[
            group["used_in_published_fiducial"] & group["observable_level_image_position"]
        ]
        within_families = set(group.loc[within, "source_family"])
        strict_within_families = set(group.loc[strict_within, "source_family"])
        finite_used_indices = np.flatnonzero(used & observable & np.isfinite(radii))
        if len(finite_used_indices):
            nearest_index = int(finite_used_indices[np.argmin(radii[finite_used_indices])])
            nearest_radius = float(radii[nearest_index])
            nearest_image_id = str(group.iloc[nearest_index]["image_id"])
        else:
            nearest_radius = None
            nearest_image_id = None
        summaries[system] = {
            "published_image_positions": len(group),
            "observable_level_image_positions": int(group["observable_level_image_position"].sum()),
            "fiducial_image_positions": len(used_group),
            "fiducial_source_families": int(used_group["source_family"].nunique()),
            "fiducial_images_with_spectroscopic_source_redshift": int(
                ((used_group["redshift_kind"] == "spectroscopic")).sum()
            ),
            "position_uncertainty_published": bool(
                used_group["image_position_sigma_arcsec"].notna().all()
            ),
            "strict_position_and_spectroscopic_redshift_likelihood_inputs": int(
                group["alternative_metric_likelihood_ready"].sum()
            ),
            "bcg_centering_verified": centering_verified,
            "radial_support_auditable": radial_support_auditable,
            "radial_reference": radial_reference,
            "nearest_fiducial_image_id": nearest_image_id,
            "nearest_fiducial_image_radius_arcsec": nearest_radius,
            "dynamics_support_max_arcsec": dynamics_max,
            "selected_dynamics_source": selected_source,
            "image_positions_inside_dynamics_support": int(within.sum()),
            "source_families_inside_dynamics_support": len(within_families),
            "strict_likelihood_source_families_inside_dynamics_support": len(strict_within_families),
            "passes_three_independent_lensing_families_inside_dynamics_support": len(strict_within_families) >= 3,
        }

    support = pd.DataFrame(support_rows)
    for output in (image_output, support_output, report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    images.to_csv(image_output, index=False)
    support.to_csv(support_output, index=False)
    report = {
        "audit_version": "R1-strong-lens-observables-0.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "systems": summaries,
        "summary": {
            "systems": len(summaries),
            "systems_with_published_image_tables": len(summaries),
            "systems_with_at_least_one_observable_level_image_position": sum(
                value["observable_level_image_positions"] > 0 for value in summaries.values()
            ),
            "published_image_positions": len(images),
            "observable_level_image_positions": int(images["observable_level_image_position"].sum()),
            "images_with_strict_position_and_spectroscopic_redshift_likelihood_inputs": int(
                images["alternative_metric_likelihood_ready"].sum()
            ),
            "systems_with_published_position_uncertainty": sum(
                value["position_uncertainty_published"] for value in summaries.values()
            ),
            "systems_with_three_verified_lensing_families_inside_dynamics_support": sum(
                value["passes_three_independent_lensing_families_inside_dynamics_support"]
                for value in summaries.values()
            ),
        },
        "classification": {
            "raw_observable": "multiple-image sky coordinates and spectroscopic source redshifts",
            "alternative_metric_use": "positions can be forward-modeled with a declared lens equation",
            "not_observable": "published model-optimized source redshifts are retained only as provenance and must not be scored as data",
            "covariance_limit": (
                "all ten systems have a published independent Gaussian image-position error model, "
                "but none publishes a full systematic covariance"
            ),
            "table_quality_limit": (
                "MACS J0326 mixes four model-predicted positions into its 12-row table; "
                "MACS J1427 has two corrupted coordinates and two source-marked predictions"
            ),
            "radial_support_rule": "count distinct lensed source families, not map pixels or multiple knots, inside verified BCG-centered dynamics support",
        },
        "outputs": {
            "image_observables": _display_path(image_output),
            "radial_support": _display_path(support_output),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "r1_lens_paper_sources",
    )
    parser.add_argument(
        "--replacement-source-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "replacement_sample_audit",
    )
    parser.add_argument(
        "--center-config", type=Path, default=ROOT / "configs" / "r1_lens_centers.json"
    )
    parser.add_argument(
        "--sample-config",
        type=Path,
        default=ROOT / "configs" / "r1_replacement_sample_gate.json",
    )
    parser.add_argument(
        "--dynamics",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_published_bcg_velocity_profiles.csv",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_strong_lens_image_observables.csv",
    )
    parser.add_argument(
        "--support-output",
        type=Path,
        default=ROOT / "data" / "derived" / "r1_strong_lens_radial_support.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "results" / "r1_strong_lens_observables" / "report.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            extract(
                source_dir=args.source_dir,
                replacement_source_dir=args.replacement_source_dir,
                sample_config_path=args.sample_config,
                center_config_path=args.center_config,
                dynamics_path=args.dynamics,
                image_output=args.image_output,
                support_output=args.support_output,
                report_output=args.report_output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
