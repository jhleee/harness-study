"""pytest 설정 — editable 설치 없이도 `src/`를 import 할 수 있도록 sys.path 에 주입."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
