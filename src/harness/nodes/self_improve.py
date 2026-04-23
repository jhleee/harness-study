"""self_improve — 트레이스를 SKILL.md 로 증류하는 노드.

가이드 §2 + §3-3 (c): 에이전트가 finalize_task 를 명시적으로 호출했을 때만 돈다.
'no tool_calls' 를 증류 트리거로 쓰는 ReAct/Plan-Execute 라우터(평범한 모든 응답을
증류 실행으로 바꿔 버리는 함정)는 쓰지 않는다.

증류 규칙:
  tool_call_count < SKILL_DISTILL_THRESHOLD → 아무 것도 하지 않고 안내만.
  이 아래면 재사용 가능한 절차를 만들기엔 트레이스가 너무 얄팍하다.
  finalize 가 반영되지 않았음을 LLM 이 알 수 있도록 에러 ToolMessage 로 알려,
  작업을 더 한 뒤 재시도할 수 있게 한다.

Trace -> SKILL.md 템플릿:
  주입된 distiller 가 4-튜플 트레이스(reasoning, tool, args, observation) 를
  Procedure / Pitfalls / Validation 섹션이 있는 마크다운 문서로 바꾼다. 테스트에서는
  결정적 fallback 을 써서 단위 레이어가 오프라인으로 유지된다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, ToolMessage

from harness.config import SKILLS_DIR
from harness.state import HarnessState

SKILL_DISTILL_THRESHOLD = 5


def _slugify(title: str) -> str:
    # 보수적인 slug — 소문자, 비alnum → 하이픈, 연속 하이픈 압축, 양끝 하이픈 제거.
    base = title.strip().lower()
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE)
    base = re.sub(r"[\s_-]+", "-", base)
    return base.strip("-") or "untitled-skill"


def _render_trace(trace: list[dict[str, Any]]) -> str:
    lines = []
    for i, entry in enumerate(trace, 1):
        lines.append(f"## Step {i}")
        lines.append(f"**Reasoning:** {entry.get('reasoning', '')}")
        tool = entry.get("tool", "")
        args = entry.get("args", {})
        lines.append(f"**Tool:** `{tool}({args})`")
        lines.append(f"**Observation:** {entry.get('observation', '')}")
        lines.append("")
    return "\n".join(lines)


def _fallback_distill(title: str, trace: list[dict[str, Any]]) -> str:
    """결정적 템플릿 — LLM distiller 가 주입되지 않았을 때 사용."""
    header = f"# {title} — 성공한 트레이스에서 자동 증류\n"
    procedure = "## Procedure\n" + "\n".join(
        f"{i}. `{e.get('tool', '')}` 를 args `{e.get('args', {})}` 로 호출."
        for i, e in enumerate(trace, 1)
    )
    detail = "## Trace (참고)\n" + _render_trace(trace)
    return f"{header}\n{procedure}\n\n{detail}\n"


def _extract_finalize_call(state: HarnessState) -> dict | None:
    if not state.get("messages"):
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    for call in getattr(last, "tool_calls", None) or []:
        if call.get("name") == "finalize_task":
            return call
    return None


def make_self_improve_node(
    *,
    skills_dir: Path | None = None,
    distiller: Callable[[str, list[dict[str, Any]]], str] | None = None,
    threshold: int = SKILL_DISTILL_THRESHOLD,
) -> Callable[[HarnessState], dict]:
    def self_improve(state: HarnessState) -> dict:
        call = _extract_finalize_call(state)
        if call is None:
            return {}

        call_id = call.get("id", "")
        title = (call.get("args") or {}).get("title") or ""
        trace = list(state.get("task_trace") or [])
        tool_calls = int(state.get("tool_call_count") or 0)

        if tool_calls < threshold:
            msg = ToolMessage(
                content=(
                    f"error: finalize_task 거부 — 지금까지 도구 호출 {tool_calls} 회, "
                    f"임계치는 {threshold} 회입니다. finalize 전에 더 작업하세요."
                ),
                tool_call_id=call_id,
            )
            return {"messages": [msg]}

        target_dir = (skills_dir or SKILLS_DIR) / _slugify(title)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "SKILL.md"

        body = (distiller or _fallback_distill)(title, trace)
        target.write_text(body, encoding="utf-8")

        return {
            "messages": [
                ToolMessage(
                    content=f"saved: {target.as_posix()}",
                    tool_call_id=call_id,
                )
            ]
        }

    return self_improve
