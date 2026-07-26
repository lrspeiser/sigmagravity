from pathlib import Path

import pandas as pd
import pytest

from voidscreen.data import pack_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "sparc"


def test_environment_input_requires_every_retained_galaxy(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"galaxy": ["CamB"], "void_score": [0.1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing independent environment score"):
        pack_dataset(DATA, environment_csv=path)

