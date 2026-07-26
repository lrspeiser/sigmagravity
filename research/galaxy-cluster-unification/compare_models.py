import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.cli import main_compare

if __name__ == "__main__":
    raise SystemExit(main_compare())

