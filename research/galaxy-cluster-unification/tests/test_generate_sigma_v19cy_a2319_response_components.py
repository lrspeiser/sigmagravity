import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_sigma_v19cy_a2319_response_components as generator


def test_frozen_inputs_pass_and_targets_remain_sealed():
    config, prep, chandra = generator.validate_inputs()
    assert prep["terminal_gate_passed"] is True
    assert chandra["terminal_gate_passed"] is True
    assert config["authorization"]["access_A3667_validation"] is False
    assert config["authorization"]["access_A754_holdout"] is False


def test_pixel_serializations_are_tool_specific_and_exact():
    pixels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 28, 29, 30, 31, 32, 33, 34, 35]
    clause = generator.pixel_clause(pixels)
    assert clause.startswith("(PIXEL==0||PIXEL==1")
    assert clause.endswith("PIXEL==35)")
    assert generator.nxb_pixel_list(pixels) == "0,1,2,3,4,5,6,7,8,28,29,30,31,32,33,34,35"


def test_wsl_native_output_paths_are_shortened_without_changing_windows_inputs():
    config, _, _ = generator.validate_inputs()
    native = Path("//wsl.localhost/Ubuntu-24.04/tmp/frozen/source.pha")
    assert generator.tool_path(config, native) == "/tmp/frozen/source.pha"
    windows = ROOT / "data/processed/example.fits"
    assert generator.tool_path(config, windows).startswith("/mnt/c/")


def test_ftcopy_materializes_hp_filter_and_extractor_uses_plain_filename(tmp_path: Path):
    config, _, _ = generator.validate_inputs()
    event = ROOT / "data/processed/sigma_v19cy_a2319_response_aware_spectral/000101_open_0_cross_obsid/corrected_branch.evt"
    source_event = tmp_path / "source_hp.evt"
    selection = config["event_selections"]["source_hp"] + "&&" + generator.pixel_clause([9, 10])
    materialize = generator.ftcopy_event_command(config, event, source_event, selection)
    extract = generator.extractor_command(config, source_event, tmp_path / "source.pha")
    assert "ITYPE==0" in materialize
    assert "PIXEL==9||PIXEL==10" in materialize
    assert "copyall=yes" in materialize
    assert "[EVENTS][" not in extract
    assert "ecol=PI" in extract
    assert "ccol=NONE" in extract
    assert "phamax=" not in extract
    assert "wtmapb=no" in extract


def test_rmf_command_explicitly_uses_observation_date_and_hyphen_runs(tmp_path: Path):
    config, _, _ = generator.validate_inputs()
    event = ROOT / "data/processed/sigma_v19cy_a2319_response_aware_spectral/000101_open_0_cross_obsid/corrected_branch.evt"
    command = generator.rmf_command(config, event, tmp_path / "response", [0, 1, 2, 7, 8])
    assert "time=2023-10-14T00:38:42" in command
    assert "pixlist=0-2,7-8" in command
    assert "whichrmf=L" in command
    assert "resolist=0" in command
    assert "[EVENTS][" not in command


def test_nxb_command_uses_public_local_database_and_frozen_sorting(tmp_path: Path):
    config, _, _ = generator.validate_inputs()
    event = ROOT / "data/processed/sigma_v19cy_a2319_response_aware_spectral/000102_open_0_cross_obsid/corrected_branch.evt"
    ehk = ROOT / config["observation_support"]["000102000"]["ehk"]["path"]
    region_file = ROOT / "data/processed/sigma_v19cy_a2319_response_aware_spectral/detector_e_prime.reg"
    command = generator.nxb_command(config, event, ehk, region_file, tmp_path / "nxb.pha", [18, 19])
    assert "database=LOCAL" in command
    assert "sortcol=CORTIME" in command
    assert "sortbin=6,8,10,12,99" in command
    assert "timefirst=-300" in command
    assert "timelast=+300" in command
    assert "pixels=18-19" in command
    assert "detector_e_prime.reg" in command


def test_expmap_command_uses_exact_branch_gti_and_fine_attitude_bins(tmp_path: Path):
    config, _, _ = generator.validate_inputs()
    support = config["observation_support"]["000101000"]
    event = ROOT / "data/processed/sigma_v19cy_a2319_response_aware_spectral/000101_open_0_cross_obsid/corrected_branch.evt"
    command = generator.expmap_command(
        config,
        ROOT / support["ehk"]["path"],
        event,
        ROOT / support["pixel_gti"]["path"],
        tmp_path / "expmap.fits",
    )
    assert "corrected_branch.evt[GTI]" in command
    assert "instrume=RESOLVE" in command
    assert "delta=0.25" in command
    assert "numphi=4" in command
    assert "maskcalsrc=yes" in command
