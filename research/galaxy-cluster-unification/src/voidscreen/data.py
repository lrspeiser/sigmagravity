from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KPC_M = 3.085677581491367e19


@dataclass(frozen=True)
class GalaxyMetadata:
    name: str
    distance_mpc: float
    distance_error_mpc: float
    inclination_deg: float
    inclination_error_deg: float
    disk_scale_kpc: float
    quality: int


@dataclass(frozen=True)
class RotationCurve:
    metadata: GalaxyMetadata
    radius_kpc: np.ndarray
    velocity_observed_kms: np.ndarray
    velocity_error_kms: np.ndarray
    velocity_gas_kms: np.ndarray
    velocity_disk_unit_ml_kms: np.ndarray
    velocity_bulge_unit_ml_kms: np.ndarray


@dataclass(frozen=True)
class PackedDataset:
    galaxy_names: tuple[str, ...]
    galaxy_index: np.ndarray
    radius_kpc: np.ndarray
    velocity_observed_kms: np.ndarray
    velocity_error_kms: np.ndarray
    velocity_gas_kms: np.ndarray
    velocity_disk_unit_ml_kms: np.ndarray
    velocity_bulge_unit_ml_kms: np.ndarray
    train_mask: np.ndarray
    distance_mpc: np.ndarray
    distance_fractional_error: np.ndarray
    inclination_deg: np.ndarray
    inclination_error_deg: np.ndarray
    quality: np.ndarray
    disk_scale_kpc: np.ndarray
    environment_raw: np.ndarray
    environment_standardized: np.ndarray
    environment_score_column: str | None
    environment_fingerprint: str | None
    data_fingerprint: str

    @property
    def n_galaxies(self) -> int:
        return len(self.galaxy_names)

    @property
    def n_points(self) -> int:
        return int(self.radius_kpc.size)

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_holdout(self) -> int:
        return int((~self.train_mask).sum())


def parse_table1(path: Path) -> dict[str, GalaxyMetadata]:
    """Parse the CDS fixed-width SPARC Table 1 file."""
    rows: dict[str, GalaxyMetadata] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                name = line[0:11].strip()
                metadata = GalaxyMetadata(
                    name=name,
                    distance_mpc=float(line[15:21]),
                    distance_error_mpc=float(line[22:27]),
                    inclination_deg=float(line[30:34]),
                    inclination_error_deg=float(line[35:39]),
                    disk_scale_kpc=float(line[71:76]),
                    quality=int(line[112:115]),
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed SPARC metadata at {path}:{line_number}") from exc
            if not name:
                raise ValueError(f"Blank galaxy name at {path}:{line_number}")
            rows[name] = metadata
    if len(rows) != 175:
        raise ValueError(f"Expected 175 SPARC metadata rows, found {len(rows)} in {path}")
    return rows


def parse_rotation_curve(path: Path, metadata: GalaxyMetadata) -> RotationCurve:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Expected six mass-model columns at {path}:{line_number}")
            try:
                rows.append([float(value) for value in parts[:6]])
            except ValueError as exc:
                raise ValueError(f"Non-numeric mass-model row at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No mass-model rows found in {path}")
    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    return RotationCurve(
        metadata=metadata,
        radius_kpc=values[:, 0],
        velocity_observed_kms=values[:, 1],
        velocity_error_kms=values[:, 2],
        velocity_gas_kms=values[:, 3],
        velocity_disk_unit_ml_kms=values[:, 4],
        velocity_bulge_unit_ml_kms=values[:, 5],
    )


def load_curves(data_dir: Path) -> list[RotationCurve]:
    data_dir = Path(data_dir)
    table_path = data_dir / "table1.dat"
    rotmod_dir = data_dir / "rotmod"
    if not table_path.exists() or not rotmod_dir.exists():
        raise FileNotFoundError(
            f"SPARC snapshot not found under {data_dir}. Run scripts/import_sigmagravity_data.ps1."
        )
    metadata = parse_table1(table_path)
    paths = sorted(rotmod_dir.glob("*_rotmod.dat"))
    if len(paths) != 175:
        raise ValueError(f"Expected 175 rotation-curve files, found {len(paths)} in {rotmod_dir}")
    curves: list[RotationCurve] = []
    for path in paths:
        name = path.name.removesuffix("_rotmod.dat")
        if name not in metadata:
            raise KeyError(f"No metadata row for {name}")
        curves.append(parse_rotation_curve(path, metadata[name]))
    return curves


def _load_environment(
    path: Path | None, score_column: str
) -> dict[str, float] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"galaxy", score_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Environment CSV must contain {sorted(required)}")
    if frame["galaxy"].duplicated().any():
        duplicates = frame.loc[frame["galaxy"].duplicated(), "galaxy"].tolist()
        raise ValueError(f"Duplicate environment rows: {duplicates[:5]}")
    if not np.isfinite(frame[score_column].to_numpy(dtype=float)).all():
        raise ValueError("Environment scores must all be finite")
    return dict(
        zip(frame["galaxy"].astype(str), frame[score_column].astype(float), strict=True)
    )


def data_fingerprint(data_dir: Path) -> str:
    """Hash input content while ignoring timestamp and checkout-location metadata."""
    data_dir = Path(data_dir)
    manifest = data_dir / "provenance.json"
    digest = hashlib.sha256()
    if manifest.exists():
        provenance = json.loads(manifest.read_text(encoding="utf-8-sig"))
        for record in sorted(provenance["files"], key=lambda item: item["path"]):
            digest.update(record["path"].encode("utf-8"))
            digest.update(record["sha256"].encode("ascii"))
            digest.update(str(record["bytes"]).encode("ascii"))
    else:
        paths = [data_dir / "table1.dat", *sorted((data_dir / "rotmod").glob("*.dat"))]
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def load_provenance(data_dir: Path) -> dict[str, object] | None:
    path = Path(data_dir) / "provenance.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pack_dataset(
    data_dir: Path,
    *,
    quality_max: int = 2,
    minimum_inclination_deg: float = 30.0,
    minimum_points: int = 8,
    train_fraction: float = 0.7,
    minimum_train_points: int = 5,
    minimum_holdout_points: int = 2,
    environment_csv: Path | None = None,
    environment_score_column: str = "void_score",
) -> PackedDataset:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    environment = _load_environment(environment_csv, environment_score_column)
    retained: list[tuple[RotationCurve, np.ndarray]] = []
    for curve in load_curves(data_dir):
        meta = curve.metadata
        valid = (
            np.isfinite(curve.radius_kpc)
            & np.isfinite(curve.velocity_observed_kms)
            & np.isfinite(curve.velocity_error_kms)
            & np.isfinite(curve.velocity_gas_kms)
            & np.isfinite(curve.velocity_disk_unit_ml_kms)
            & np.isfinite(curve.velocity_bulge_unit_ml_kms)
            & (curve.radius_kpc > 0.0)
            & (curve.velocity_observed_kms > 0.0)
            & (curve.velocity_error_kms > 0.0)
        )
        count = int(valid.sum())
        if meta.quality > quality_max or meta.inclination_deg < minimum_inclination_deg:
            continue
        if count < max(minimum_points, minimum_train_points + minimum_holdout_points):
            continue
        if environment is not None and meta.name not in environment:
            raise ValueError(f"Missing independent environment score for retained galaxy {meta.name}")
        retained.append((curve, valid))
    if not retained:
        raise ValueError("No galaxies survived the declared cuts")

    names = tuple(curve.metadata.name for curve, _ in retained)
    point_fields: dict[str, list[np.ndarray]] = {
        "radius": [],
        "observed": [],
        "error": [],
        "gas": [],
        "disk": [],
        "bulge": [],
        "galaxy_index": [],
        "train_mask": [],
    }
    for galaxy_index, (curve, valid) in enumerate(retained):
        n_points = int(valid.sum())
        n_train = int(np.floor(train_fraction * n_points))
        n_train = max(minimum_train_points, n_train)
        n_train = min(n_points - minimum_holdout_points, n_train)
        split = np.zeros(n_points, dtype=bool)
        split[:n_train] = True
        point_fields["radius"].append(curve.radius_kpc[valid])
        point_fields["observed"].append(curve.velocity_observed_kms[valid])
        point_fields["error"].append(curve.velocity_error_kms[valid])
        point_fields["gas"].append(curve.velocity_gas_kms[valid])
        point_fields["disk"].append(curve.velocity_disk_unit_ml_kms[valid])
        point_fields["bulge"].append(curve.velocity_bulge_unit_ml_kms[valid])
        point_fields["galaxy_index"].append(np.full(n_points, galaxy_index, dtype=np.int64))
        point_fields["train_mask"].append(split)

    metadata = [curve.metadata for curve, _ in retained]
    raw_environment = np.asarray(
        [0.0 if environment is None else environment[name] for name in names], dtype=np.float64
    )
    if environment is None:
        standardized_environment = raw_environment.copy()
    else:
        standard_deviation = float(raw_environment.std(ddof=0))
        if standard_deviation <= 0.0:
            raise ValueError("Environment scores have zero variance after cuts")
        standardized_environment = (raw_environment - raw_environment.mean()) / standard_deviation

    return PackedDataset(
        galaxy_names=names,
        galaxy_index=np.concatenate(point_fields["galaxy_index"]),
        radius_kpc=np.concatenate(point_fields["radius"]),
        velocity_observed_kms=np.concatenate(point_fields["observed"]),
        velocity_error_kms=np.concatenate(point_fields["error"]),
        velocity_gas_kms=np.concatenate(point_fields["gas"]),
        velocity_disk_unit_ml_kms=np.concatenate(point_fields["disk"]),
        velocity_bulge_unit_ml_kms=np.concatenate(point_fields["bulge"]),
        train_mask=np.concatenate(point_fields["train_mask"]),
        distance_mpc=np.asarray([m.distance_mpc for m in metadata], dtype=np.float64),
        distance_fractional_error=np.asarray(
            [m.distance_error_mpc / max(m.distance_mpc, 1e-12) for m in metadata], dtype=np.float64
        ),
        inclination_deg=np.asarray([m.inclination_deg for m in metadata], dtype=np.float64),
        inclination_error_deg=np.asarray([m.inclination_error_deg for m in metadata], dtype=np.float64),
        quality=np.asarray([m.quality for m in metadata], dtype=np.int64),
        disk_scale_kpc=np.asarray([m.disk_scale_kpc for m in metadata], dtype=np.float64),
        environment_raw=raw_environment,
        environment_standardized=standardized_environment,
        environment_score_column=(environment_score_column if environment is not None else None),
        environment_fingerprint=(
            hashlib.sha256(Path(environment_csv).read_bytes()).hexdigest()
            if environment_csv is not None
            else None
        ),
        data_fingerprint=data_fingerprint(data_dir),
    )
