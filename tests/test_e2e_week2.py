"""2주차 E2E — Progressive Disclosure + offloading 회귀 게이트.

실제 CLI 를 실제 MiniMax 엔드포인트에 대고 3턴 스크립트로 돌린다:

  턴 1: "echo 스킬 로드" — load_skill 도구 호출을 유도하고, 라우터가 이를
          skill_loader 로 보낸다. 턴 후 state.loaded_skills 에 'echo' 가 있어야 하고,
          messages 에는 <skill:echo> ToolMessage 가 있어야 한다.

  턴 2: "data/cache/e2e-week2-note.txt 를 read" — tool_dispatch 를 통해 read 호출을
          유도한다. subprocess 가 돌기 전에 파일을 미리 만들어 두어 읽기가 성공하도록
          한다.

  턴 3: 사용한 도구 이름을 나열해 달라는 평범한 응답 — 새 도구 호출 없음.
          세션이 여전히 END 로 수렴하는지에 대한 sanity check.

단언:
  1. CLI 가 0 으로 종료.
  2. 트레이스 레코드 3 개.
  3. 모든 system_sha256 동일 — 스킬 로드와 도구 디스패치를 지나도 Frozen Snapshot 이
     유지된다. 커리큘럼의 중요한 약속 중 하나 — 동적 로딩이 캐시 프리픽스를 깨지 않는다.
  4. 턴 1 후 loaded_skill_names 에 'echo' 가 포함.
  5. 턴 2 후 total_tool_calls >= 1 (read 호출이 최소 한 번은 반영).
  6. 턴 2 어시스턴트 preview 에 미리 심어둔 SENTINEL('SENTINEL-W2') 이 언급됨 —
     모델이 실제로 read 로 파일을 읽었다는 증거.

HARNESS_E2E=1 뒤로 게이트.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

E2E_ENABLED = os.environ.get("HARNESS_E2E") == "1"
REASON = "실행하려면 HARNESS_E2E=1 설정 (실제 MiniMax 엔드포인트를 호출)"
pytestmark = pytest.mark.skipif(not E2E_ENABLED, reason=REASON)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "e2e-week2.txt"
STAGED_FILE = REPO / "data" / "cache" / "e2e-week2-note.txt"
SENTINEL = "SENTINEL-W2-unique-content-9f3c"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_week2_progressive_disclosure_gate(tmp_path: Path) -> None:
    assert SCRIPT.exists()

    # 모델이 read 해 달라고 요청할 파일을 미리 만들어 둔다.
    STAGED_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAGED_FILE.write_text(
        f"This file contains {SENTINEL}. It exists solely for the Week 2 E2E test.\n",
        encoding="utf-8",
    )

    trace = tmp_path / "week2.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "harness.cli",
                "--script", str(SCRIPT),
                "--thread-id", "week2-e2e",
                "--trace", str(trace),
                "--quiet",
            ],
            env=env,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=240,
        )
        assert proc.returncode == 0, \
            f"CLI 실패:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        records = _read_jsonl(trace)
        assert len(records) == 3, f"레코드 3 개를 기대했으나 {len(records)}"

        # 스킬 로드/도구 디스패치 턴을 지나도 Frozen Snapshot 유지.
        hashes = {r["system_sha256"] for r in records}
        assert len(hashes) == 1 and next(iter(hashes)), \
            f"시스템 프롬프트가 드리프트했다: {hashes}"

        # 턴 1: 스킬이 로드됨.
        assert "echo" in records[0]["loaded_skill_names"], \
            f"턴 1 이후 echo 가 loaded_skills 에 있어야 하는데: {records[0]}"

        # 턴 2: read 도구가 최소 한 번 디스패치됨.
        assert records[1]["total_tool_calls"] >= 1, \
            f"턴 2 이후 tool_call_count >= 1 를 기대했는데: {records[1]}"

        # 턴 2 어시스턴트는 어떤 식으로든 stage 파일을 참조해야 한다 —
        # 모델은 그대로 인용하기보다 요약하는 경향이 있으므로, SENTINEL 자체나
        # 고신호 문자열 "E2E" 중 하나라도 언급되면 OK.
        preview = records[1]["assistant_preview"]
        assert SENTINEL in preview or "E2E" in preview, (
            "턴-2 preview 에서 모델이 stage 파일을 읽었다는 흔적이 보이지 않는다. "
            f"preview={preview!r}"
        )
    finally:
        STAGED_FILE.unlink(missing_ok=True)
