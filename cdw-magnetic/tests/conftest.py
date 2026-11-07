import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # cdw-magnetic/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))              # reference implementations
