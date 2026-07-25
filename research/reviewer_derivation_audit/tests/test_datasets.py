from pathlib import Path

from sigma_sprint.datasets import load_tian2020, normalize_cluster_name


def test_tian_loader_preserves_groups(tmp_path: Path):
    path = tmp_path / "fig2.dat"
    path.write_text(
        "A209 14.3 -10.015 -9.579 0.043 0.089\n"
        "A209 100 -10.100 -9.600 0.040 0.080\n"
        "MACS0416 14.3 -10.670 -9.541 0.043 0.096\n",
        encoding="utf-8",
    )
    frame = load_tian2020(path)
    assert len(frame) == 3
    assert frame["group_id"].nunique() == 2
    assert normalize_cluster_name("MACS J0416.1-2403") == "macs0416"
