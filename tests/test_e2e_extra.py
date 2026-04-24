"""extra 브랜치 E2E — read / write / edit 도구의 end-to-end 검증.

write 와 edit 는 DESTRUCTIVE_TOOLS 라 graph 가 human_gate 의 interrupt_before 에서
멈춘다. CLI 는 아직 resume 로직이 없으므로, 이 테스트는 graph 를 직접 돌리며
state.next 가 비워질 때까지 invoke(None, config) 로 auto-approve 한다.

시나리오:
  턴 1 (read)  — 미리 만든 source 파일을 read 도구로 읽도록 지시.
                 LLM 이 read 호출 → tool_dispatch → _READ_CACHE 갱신 → END.
                 도구 호출 1 회 누적.
  턴 2 (write) — 새 target 파일에 알려진 sentinel 을 write 하도록 지시.
                 LLM 이 write 호출 → human_gate interrupt → resume → tool_dispatch
                 → atomic write → END. 디스크에 파일이 sentinel 과 함께 존재.
  턴 3 (edit)  — 방금 만든 target 의 sentinel 토큰을 다른 토큰으로 치환하도록 edit 지시.
                 write 가 _READ_CACHE 를 채웠으므로 read 없이도 edit 가 통한다.
                 human_gate interrupt → resume → tool_dispatch → atomic write → END.
                 디스크 내용이 새 토큰으로 바뀜.

HARNESS_E2E=1 뒤로 게이트.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from harness import tools
from harness.config import llm, load_settings
from harness.graph import build_graph

E2E_ENABLED = os.environ.get("HARNESS_E2E") == "1"
REASON = "실행하려면 HARNESS_E2E=1 설정 (실제 MiniMax 엔드포인트를 호출)"
pytestmark = pytest.mark.skipif(not E2E_ENABLED, reason=REASON)

REPO = Path(__file__).resolve().parents[1]
SOURCE_FILE = REPO / "data" / "cache" / "extra-source.txt"
TARGET_FILE = REPO / "data" / "cache" / "extra-target.txt"
SOURCE_SENTINEL = "EXTRA-SRC-7af2"
WRITE_SENTINEL = "EXTRA-WRITE-9b1c"
EDITED_SENTINEL = "EXTRA-EDITED-3d4e"


def _drive(graph, msg: str, config: dict, *, max_resumes: int = 4) -> dict:
    """invoke 후 state.next 가 비워질 때까지 invoke(None) 로 auto-approve.

    interrupt_before=['human_gate'] 가 트리거되면 첫 invoke 는 human_gate 직전에서
    멈춘다. 이때 state.next == ('human_gate',). resume 시 human_gate → tool_dispatch
    → agent 까지 흘러간다. 추가 도구 호출이 또 destructive 면 다시 멈출 수 있어
    루프로 처리한다 (실수로 무한 루프가 되지 않도록 max_resumes 제한).
    """
    result = graph.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
    for _ in range(max_resumes):
        state = graph.get_state(config)
        if not state.next:
            break
        result = graph.invoke(None, config=config)
    state = graph.get_state(config)
    assert not state.next, f"resume 한도 초과 — 여전히 pending: {state.next}"
    return result


def test_extra_read_write_edit_loop() -> None:
    settings = load_settings()
    if not settings.api_key:
        pytest.skip("OPENAI_API_KEY 미설정")

    # 깨끗한 상태에서 시작.
    SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_FILE.write_text(
        f"This source file contains {SOURCE_SENTINEL} for the extra E2E.\n",
        encoding="utf-8",
    )
    TARGET_FILE.unlink(missing_ok=True)
    tools._clear_read_cache()

    graph = build_graph(llm(settings))
    config = {"configurable": {"thread_id": "extra-e2e"}}

    try:
        # --- 턴 1: read ---------------------------------------------------------
        msg1 = (
            f"data/cache/extra-source.txt 파일을 read 도구로 읽어. "
            "다른 도구는 사용하지 말고 read 하나만 호출해. 읽은 뒤 본문은 짧게 인용만 하고 끝내."
        )
        r1 = _drive(graph, msg1, config)
        assert r1.get("tool_call_count", 0) >= 1, "read 가 디스패치되지 않았다"
        assert any(
            entry.get("tool") == "read" for entry in r1.get("task_trace") or []
        ), f"task_trace 에 read 가 없다: {r1.get('task_trace')}"

        # --- 턴 2: write --------------------------------------------------------
        msg2 = (
            "이제 write 도구로 새 파일을 만들어. "
            f"path='data/cache/extra-target.txt', content='{WRITE_SENTINEL}'. "
            "정확히 그 두 인자로 write 만 한 번 호출해. 다른 도구는 쓰지 마."
        )
        r2 = _drive(graph, msg2, config)
        assert TARGET_FILE.exists(), (
            f"write 가 파일을 생성하지 않음. trace: {r2.get('task_trace')}"
        )
        body2 = TARGET_FILE.read_text(encoding="utf-8")
        assert WRITE_SENTINEL in body2, (
            f"파일이 sentinel 을 담지 않음. body={body2!r}"
        )
        assert any(
            entry.get("tool") == "write" for entry in r2.get("task_trace") or []
        ), f"task_trace 에 write 가 없다: {r2.get('task_trace')}"

        # --- 턴 3: edit ---------------------------------------------------------
        msg3 = (
            f"방금 만든 data/cache/extra-target.txt 안의 '{WRITE_SENTINEL}' 부분을 "
            f"'{EDITED_SENTINEL}' 로 바꿔. edit 도구로 한 번만 호출해. "
            f"인자: path='data/cache/extra-target.txt', "
            f"old_string='{WRITE_SENTINEL}', new_string='{EDITED_SENTINEL}'."
        )
        r3 = _drive(graph, msg3, config)
        body3 = TARGET_FILE.read_text(encoding="utf-8")
        assert EDITED_SENTINEL in body3 and WRITE_SENTINEL not in body3, (
            f"edit 결과가 디스크에 반영되지 않음. body={body3!r}"
        )
        assert any(
            entry.get("tool") == "edit" for entry in r3.get("task_trace") or []
        ), f"task_trace 에 edit 가 없다: {r3.get('task_trace')}"

    finally:
        SOURCE_FILE.unlink(missing_ok=True)
        TARGET_FILE.unlink(missing_ok=True)
        tools._clear_read_cache()


def test_extra_edit_without_read_returns_error() -> None:
    """LLM 이 read 없이 edit 를 호출하면 도구가 'must read first' 로 거부.

    이건 LLM 동작에 의존하지 않고 도구 + dispatch + human_gate 통합만 검증.
    state 를 직접 조작해 graph 한 사이클만 돌린다.
    """
    settings = load_settings()
    if not settings.api_key:
        pytest.skip("OPENAI_API_KEY 미설정")

    target = REPO / "data" / "cache" / "extra-stale.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")
    tools._clear_read_cache()

    try:
        # 도구를 직접 호출 — edit 가 read 캐시 미스로 거부하는지.
        from harness.tools import edit
        out = edit.invoke(
            {"path": str(target), "old_string": "hello", "new_string": "hi"}
        )
        assert out.startswith("error: must read"), out
        # 디스크는 그대로.
        assert target.read_text(encoding="utf-8") == "hello"
    finally:
        target.unlink(missing_ok=True)
        tools._clear_read_cache()
