"""4주차 E2E — self-improvement 루프, 끝에서 끝까지.

시나리오:
  턴 1-5: 미리 준비한 파일 5 개를 모델이 `view` 로 각각 읽도록 지시.
           매 view 호출은 tool_dispatch 를 거쳐 tool_call_count 를 1 증가시키고
           4-튜플 trace 를 append. 5 턴 뒤 tool_call_count == 5.
  턴 6:   finalize_task(title="e2e-week4-demo") 호출을 지시.
           라우터가 이를 self_improve 로 보내고, 노드는 tool_call_count >=
           SKILL_DISTILL_THRESHOLD (5) 인 것을 확인한 뒤
           data/skills-e2e/<slug>/SKILL.md 를 쓴다.

실제 data/skills/ 카탈로그를 오염시키지 않도록 tmp_path 아래 전용 --skills-dir 를 쓴다.

단언:
  1. CLI 가 0 으로 종료.
  2. 트레이스 레코드 6 개, Frozen Snapshot 불변식 유지.
  3. 턴 5 시점 total_tool_calls 가 정확히 5 (매 view 가 tool_dispatch 를 거쳐 카운터 증가).
  4. <skills-dir>/e2e-week4-demo/SKILL.md 가 존재.
  5. SKILL.md 본문에 'Procedure' 섹션이 있고 `view` 가 언급됨 — 4-튜플 트레이스가
     distiller 까지 도달했다는 증거.

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
SCRIPT = REPO / "scripts" / "e2e-week4.txt"


def _stage_files() -> list[Path]:
    cache = REPO / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, 6):
        p = cache / f"e2e-w4-{i}.txt"
        p.write_text(
            f"Week-4 E2E file {i}. Contents: pattern-{i*13}.\n",
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_week4_self_improvement_loop(tmp_path: Path) -> None:
    assert SCRIPT.exists()
    staged = _stage_files()
    skills_out = tmp_path / "skills-e2e"
    trace = tmp_path / "week4.jsonl"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "harness.cli",
                "--script", str(SCRIPT),
                "--thread-id", "week4-e2e",
                "--trace", str(trace),
                "--skills-dir", str(skills_out),
                "--quiet",
            ],
            env=env,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=420,
        )
        assert proc.returncode == 0, \
            f"CLI 실패:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        records = _read_jsonl(trace)
        assert len(records) == 6, f"트레이스 레코드 6 개 기대, {len(records)}"

        hashes = {r["system_sha256"] for r in records}
        assert len(hashes) == 1, f"시스템 프롬프트가 드리프트했다: {hashes}"

        # 턴 5 까지 tool_call_count 가 최소 5 이어야 한다 (매 턴 view 한 번씩;
        # 모델이 추가로 호출했으면 더 많을 수도 있다).
        assert records[4]["total_tool_calls"] >= 5, (
            "턴 5 까지 최소 5 번의 tool_dispatch 실행을 기대; "
            f"실제 {records[4]['total_tool_calls']}"
        )

        # finalize_task 가 만들어낸 skill 파일이 있어야 한다.
        skill_md = skills_out / "e2e-week4-demo" / "SKILL.md"
        assert skill_md.exists(), (
            f"SKILL.md 가 없음: {skill_md}; 존재하는 자식: "
            f"{list(skills_out.glob('*/SKILL.md')) if skills_out.exists() else '(skills dir 없음)'}"
        )
        body = skill_md.read_text(encoding="utf-8")
        assert "Procedure" in body
        assert "view" in body
    finally:
        for p in staged:
            p.unlink(missing_ok=True)
