import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ax_delve_dr3_coadd_acquisition.json"
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19ax_delve_dr3_coadds.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ax", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v19ax_is_acquisition_only_and_frozen():
    config = json.loads(CONFIG.read_text())
    assert config["status"] == "frozen_before_full_coadd_pixel_acquisition"
    assert config["source"]["endpoint"].endswith("/sia/delve_dr3")
    assert config["source"]["cutout_size_deg"] == 0.17
    assert config["gates"]["exact_products"] == 12
    assert config["gates"]["exact_candidates"] == 568
    assert not config["authorization"]["measure_anchor_or_candidate_flux"]
    assert not config["authorization"]["choose_photometry_or_deblend_model"]
    assert not config["authorization"]["select_or_rank_counterparts"]
    assert not config["authorization"]["infer_mass_or_current"]


def test_standard_product_classifier_excludes_nobkg_and_wrong_extension():
    config = json.loads(CONFIG.read_text())
    base = "https://datalab.noirlab.edu/svc/cutout?col=delve_dr3&"
    standard = base + "siaRef=DES0659-5540_r6435p01_g.fits.fz&extn=1"
    nobkg = base + "siaRef=DES0659-5540_r6435p01_g_nobkg.fits.fz&extn=1"
    wrong_extension = base + "siaRef=DES0659-5540_r6435p01_g.fits.fz&extn=2"
    assert MODULE.classify_product(standard, "g", "image", config) == (
        "DES0659-5540_r6435p01_g.fits.fz",
        1,
    )
    assert MODULE.classify_product(nobkg, "g", "image", config) is None
    assert MODULE.classify_product(wrong_extension, "g", "image", config) is None


def test_sia_query_contains_only_frozen_field_parameters():
    config = json.loads(CONFIG.read_text())
    url = MODULE.sia_query_url(config)
    assert "POS=104.6247543743987%2C-55.94659781854907" in url
    assert "SIZE=0.17" in url
    assert "candidate" not in url.lower()
