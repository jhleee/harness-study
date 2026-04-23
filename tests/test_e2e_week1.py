"""1주차 E2E — Frozen Snapshot 회귀 게이트.

실제 CLI 를 subprocess 로 돌려 실제 MiniMax 엔드포인트에 요청한 뒤, 커리큘럼의
1주차 인수 기준을 단언한다:

  1. 3턴 스크립트를 소비하고 CLI 가 종료코드 0 으로 끝난다.
  2. 트레이스 JSONL 은 스크립트 한 줄당 한 레코드, 순서대로 (turn=1..3) 기록된다.
  3. 모든 턴의 system_sha256 이 동일하다 (Frozen Snapshot 불변식).
  4. 턴마다 input_tokens 가 증가한다 (checkpointer 가 실제로 대화 히스토리를 배관
     하고 있음을 증명. 매 턴 state 가 새로 시작되는 게 아님).
  5. 모델의 턴-3 응답이 턴-1 사용자 텍스트를 그대로 포함한다 (checkpointer 가
     SystemMessage 뿐 아니라 USER 메시지도 배관한다는 증거).

HARNESS_E2E=1 뒤로 게이트되어 있어 CI / 일반 `pytest` 실행이 우연히 네트워크를
치지 않도록 한다. 인수 게이트가 중요할 때만 명시적으로 실행한다.
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
SCRIPT = REPO / "scripts" / "e2e-week1.txt"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_week1_frozen_snapshot_gate(tmp_path: Path) -> None:
    assert SCRIPT.exists(), f"E2E 스크립트가 없다: {SCRIPT}"

    trace = tmp_path / "week1.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [
            sys.executable, "-m", "harness.cli",
            "--script", str(SCRIPT),
            "--thread-id", "week1-e2e",
            "--trace", str(trace),
            "--quiet",
        ],
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert proc.returncode == 0, f"CLI 실패:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    records = _read_jsonl(trace)
    assert len(records) == 3, f"트레이스 레코드 3 개를 기대했으나 {len(records)} 개"

    # 턴 번호
    assert [r["turn"] for r in records] == [1, 2, 3]

    # Frozen Snapshot 불변식 — 1주차의 핵심 속성.
    hashes = {r["system_sha256"] for r in records}
    assert len(hashes) == 1 and next(iter(hashes)), \
        f"시스템 프롬프트가 턴을 넘기며 드리프트했다: {hashes}"

    # 토큰 증가 = checkpointer 가 히스토리를 누적하고 있다는 증거.
    inputs = [r["input_tokens"] for r in records]
    assert inputs[0] < inputs[1] < inputs[2], \
        f"input_tokens 가 턴마다 증가해야 하는데 {inputs}"

    # 턴-3 어시스턴트 preview 가 턴-1 사용자 문자열을 포함해야 한다.
    # (턴-3 질문이 "처음에 뭐라고 했어?" 이므로 어떤 합리적 응답이라도
    # 첫 문구를 그대로 인용할 것이다.)
    first_phrase = "누구야"   # 턴-1 입력에서 안정적으로 반향될 고신호 부분 문자열
    assert first_phrase in records[2]["assistant_preview"], (
        "턴-3 어시스턴트 preview 가 턴-1 내용을 반향하지 않는다 — "
        "checkpointer 가 사용자 히스토리를 배관하지 못하고 있을 수 있다. "
        f"preview={records[2]['assistant_preview']!r}"
    )
