import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_sigma_v19cy_a2319_arfs as generator
import generate_sigma_v19cy_a2319_response_components as components


def test_arf_command_uses_one_frozen_image_source_and_exact_response_inputs(tmp_path: Path):
    config, _, _ = components.validate_inputs()
    command = generator.arf_command(
        config,
        workdir=tmp_path,
        raytrace=tmp_path / "branch_raytrace.fits",
        expmap=tmp_path / "branch_expmap.fits",
        rmf=tmp_path / "region.rmf",
        image=tmp_path / "chandra.img",
        region_file=tmp_path / "detector.reg",
        output=tmp_path / "region.arf",
    )
    assert "sourcetype=IMAGE" in command
    assert "erange='2.0 10.5 0.5 7.0'" in command
    assert "numphoton=600000" in command
    assert "minphoton=100" in command
    assert "seed=7" in command
    assert "source_ra=290.299" in command
    assert "source_dec=43.9345" in command
    assert "regmode=DET" in command
    assert "rslgapreg=no" in command
    assert "qefile=CALDB" in command
    assert "mirrorfile=CALDB" in command
    assert "imgfile=" in command and "chandra.img" in command
    assert "xrtevtfile=" in command and "branch_raytrace.fits" in command
    assert "emapfile=" in command and "branch_expmap.fits" in command
    assert "rmffile=" in command and "region.rmf" in command
    assert "regionfile=" in command and "detector.reg" in command


def test_arf_stage_keeps_validation_and_holdout_sealed_by_construction():
    config, _, _ = components.validate_inputs()
    assert config["authorization"]["access_A3667_validation"] is False
    assert config["authorization"]["access_A754_holdout"] is False
    assert config["authorization"]["open_lensing_halo_or_gravity_targets"] is False
    assert generator.REPORT.name == "development_response_arfs.json"


def test_arf_stage_has_exactly_three_branch_scoped_raytraces_and_ten_arfs():
    config, _, _ = components.validate_inputs()
    assert len(config["branches"]) == 3
    assert sum(len(row["regions"]) for row in config["branches"]) == 10
    names = [row["name"] for row in config["branches"]]
    assert len(names) == len(set(names))
    assert generator.ARF_TIMEOUT_SECONDS == 43_200
