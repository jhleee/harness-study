"""3주차 E2E — subagent RPC + compactor pinning.

실제 MiniMax 로 돌리는 시나리오:

  턴 1: load_skill('echo')  → skill_loader 가 본문을 주입, loaded_skills state 갱신.
  턴 2: spawn_subagent(task="2+2")  → 부모가 위임, 자식이 응답, 부모가 요약을 보고
          "자식이 답한 숫자: 4" 를 출력.
  턴 3: 사용한 도구 나열 — 새 도구 호출 없이 평범한 응답.

compact_threshold=600 을 강제해 세션 동안 compactor 가 실제로 돌도록 만든다 (기본 80K 는
세 턴짜리 세션에서 도달 불가). 가이드 §3-3 (a) 회귀 확인용 — compaction 이 터진 뒤에도
턴 1 의 <skill:echo> ToolMessage 가 살아남아야 한다. 그렇지 않으면 턴 3 의
loaded_skill_names 에 "echo" 가 빠진다.

단언:
  1. CLI 가 0 으로 종료.
  2. 트레이스 레코드 3 개.
  3. Frozen Snapshot: 턴 전반에 system_sha256 동일.
  4. 턴 1: 'echo' 가 loaded_skill_names 에.
  5. 턴 2 preview 에 '4' 포함 — subagent 의 답이 RPC 요약을 통해 부모로 전달된 증거.
  6. 턴 3 loaded_skill_names 에 여전히 'echo' — 낮은 임계치에서 compactor 가 돌았지만
     스킬을 pin 했다.

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
SCRIPT = REPO / "scripts" / "e2e-week3.txt"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_week3_subagent_and_compactor_gate(tmp_path: Path) -> None:
    assert SCRIPT.exists()
    trace = tmp_path / "week3.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [
            sys.executable, "-m", "harness.cli",
            "--script", str(SCRIPT),
            "--thread-id", "week3-e2e",
            "--trace", str(trace),
            "--quiet",
            "--compact-threshold", "600",
        ],
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, \
        f"CLI 실패:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    records = _read_jsonl(trace)
    assert len(records) == 3, f"레코드 3 개를 기대했으나 {len(records)}"

    hashes = {r["system_sha256"] for r in records}
    assert len(hashes) == 1, f"시스템 프롬프트가 드리프트했다: {hashes}"

    assert "echo" in records[0]["loaded_skill_names"], \
        f"턴 1 에서 echo 가 로드돼야 하는데: {records[0]}"

    # 턴 2 — subagent 가 2+2=4 로 답했고, 부모가 이를 언급해야 한다.
    assert "4" in records[1]["assistant_preview"], (
        "턴 2 preview 에 subagent 의 답 '4' 가 보이지 않는다 — "
        f"preview={records[1]['assistant_preview']!r}"
    )

    # 턴 3 — compaction 하에서 스킬 pinning. compact 임계치가 낮아 턴 3 이전에
    # compactor 가 최소 한 번은 돌았을 것이다; 그래도 'echo' 가 loaded_skill_names 에
    # 남아 있어야 한다 (= <skill:echo> ToolMessage 가 살아남았다).
    assert "echo" in records[2]["loaded_skill_names"], (
        "턴 3 에서 echo 가 loaded_skill_names 에서 사라졌다 — compactor 가 "
        "스킬 ToolMessage 를 축출했을 수 있다 (가이드 §3-3 (a) 회귀). "
        f"{records[2]}"
    )
