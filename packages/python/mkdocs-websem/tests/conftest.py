from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1]
WORKSPACE_PYTHON = PACKAGE_ROOT.parent

sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_PYTHON / "websem-types" / "src"))
