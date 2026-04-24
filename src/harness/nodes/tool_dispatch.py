"""tool_dispatch — offload 기능을 갖춘 범용 도구 실행기.

가이드 §4-2: "긴 tool 결과는 파일로 내보내고 messages에는 포인터만 남긴다."
여기서 그 레시피를 그대로 구현한다:

1. 주입된 레지스트리에서 이름으로 도구를 찾는다.
2. 실행한다 (try/except — 도구가 터져도 턴 전체가 죽어서는 안 된다).
3. 결과를 문자열로 바꾼 길이가 OFFLOAD_THRESHOLD 자를 넘으면
   data/cache/out_<uuid>.txt 로 쓰고, 컨텍스트용 preview 500 자를 담은
   포인터 ToolMessage 를 돌려준다.
4. 4-튜플 트레이스 레코드(reasoning, tool, args, observation) 를 append 한다 —
   4주차 self_improve 가 나중에 증류할 재료.
5. tool_call_count 를 증가.

경로 선택(가이드 §4-2 경고): /tmp (컨테이너 재시작이나 서버리스 워커 이동에서 휘발) 이
아니라 data/cache/ (프로젝트 기준 상대, 영속).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from harness.config import CACHE_DIR
from harness.state import HarnessState

OFFLOAD_THRESHOLD = 2000
OFFLOAD_PREVIEW_CHARS = 500


def _first_tool_call(state: HarnessState) -> dict | None:
    if not state.get("messages"):
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    calls = getattr(last, "tool_calls", None) or []
    return calls[0] if calls else None


def _run_tool(tool: BaseTool, args: dict) -> str:
    try:
        result = tool.invoke(args)
    except Exception as exc:  # 도구 에러는 턴을 죽이지 말고 ToolMessage 로 만든다
        return f"error: tool {tool.name} raised {type(exc).__name__}: {exc}"
    if not isinstance(result, str):
        result = str(result)
    return result


def _maybe_offload(content: str, cache_dir: Path) -> str:
    if len(content) <= OFFLOAD_THRESHOLD:
        return content
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"out_{uuid.uuid4().hex[:12]}.txt"
    path.write_text(content, encoding="utf-8")
    preview = content[:OFFLOAD_PREVIEW_CHARS]
    return (
        f"<result path='{path}' bytes={len(content)}>\n"
        f"{preview}"
        + ("…(잘림, 전체 내용은 read 도구로 읽으세요)" if len(content) > OFFLOAD_PREVIEW_CHARS else "")
        + "\n</result>"
    )


def make_tool_dispatch(
    tools_by_name: dict[str, BaseTool],
    *,
    cache_dir: Path | None = None,
) -> Callable[[HarnessState], dict]:
    def tool_dispatch(state: HarnessState) -> dict:
        call = _first_tool_call(state)
        if call is None:
            return {}

        tool_name = call.get("name", "")
        args = call.get("args") or {}
        call_id = call.get("id", "")
        last_ai: AIMessage = state["messages"][-1]  # type: ignore[assignment]
        reasoning = last_ai.content if isinstance(last_ai.content, str) else ""

        tool = tools_by_name.get(tool_name)
        if tool is None:
            content = f"error: unknown tool '{tool_name}'"
        else:
            content = _run_tool(tool, args)

        resolved_cache = cache_dir if cache_dir is not None else CACHE_DIR
        final_content = _maybe_offload(content, resolved_cache)
        tool_msg = ToolMessage(content=final_content, tool_call_id=call_id)

        trace_entry: dict[str, Any] = {
            "reasoning": reasoning,
            "tool": tool_name,
            "args": args,
            "observation": final_content[:500],
        }
        return {
            "messages": [tool_msg],
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "task_trace": [trace_entry],
        }

    return tool_dispatch
