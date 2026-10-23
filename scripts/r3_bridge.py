"""CLI entry for the R3 CSV/TOML bridge."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_bridge import main

if __name__ == '__main__':
    main()
