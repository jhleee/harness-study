"""skill_loader — Level 1 Progressive Disclosure.

에이전트의 마지막 AIMessage.tool_calls[0] 가 load_skill 일 때 라우터가 이 노드로
분기한다. 스킬 본문을 새 SystemMessage 가 아니라 반드시 ToolMessage 로 내보낸다 —
가이드 §5 의 'Progressive Disclosure 는 시스템 프롬프트가 아니라 messages 에 관한
이야기' 라는 원칙 그 자체.

콘텐츠 포맷: ToolMessage content 를 <skill:NAME> ... </skill> 태그로 감싼다.
3주차 compactor 가 이 태그를 보고 해당 메시지를 pin 해 둔다 (가이드 §2 compactor
본문의 `_is_skill_msg` 필터).

loaded_skills / skill_last_used 는 순전히 부기용으로 state 에 남긴다: 본문은 이
노드가 돌고 나면 이미 messages 에 들어가 있으므로, 이후 주차(축출, 재사용 통계) 가
메시지 히스토리를 훑지 않고도 이 사실을 추론할 수 있게 해 준다.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from harness.state import HarnessState
from harness.tools import skill_body_path


def _extract_first_load_skill(state: HarnessState) -> tuple[str, str] | None:
    """가장 최신 AIMessage 에서 첫 번째 load_skill 호출의 (name, tool_call_id) 를
    돌려준다. 없으면 None."""
    if not state.get("messages"):
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    for call in getattr(last, "tool_calls", None) or []:
        if call.get("name") == "load_skill":
            name = (call.get("args") or {}).get("name")
            if isinstance(name, str) and name:
                return name, call.get("id", "")
    return None


def skill_loader(state: HarnessState) -> dict:
    parsed = _extract_first_load_skill(state)
    if parsed is None:
        return {}
    name, call_id = parsed
    loaded = dict(state.get("loaded_skills") or {})
    last_used = dict(state.get("skill_last_used") or {})
    turn = state.get("turn", 0)

    body = loaded.get(name)
    if body is None:
        path: Path = skill_body_path(name)
        if not path.exists():
            err = ToolMessage(
                content=f"error: unknown skill '{name}'",
                tool_call_id=call_id,
            )
            return {"messages": [err]}
        body = path.read_text(encoding="utf-8")

    tool_msg = ToolMessage(
        content=f"<skill:{name}>\n{body}\n</skill>",
        tool_call_id=call_id,
    )
    loaded[name] = body
    last_used[name] = turn
    return {
        "messages": [tool_msg],
        "loaded_skills": loaded,
        "skill_last_used": last_used,
    }


def is_skill_message(msg) -> bool:
    return (
        isinstance(msg, ToolMessage)
        and isinstance(msg.content, str)
        and msg.content.startswith("<skill:")
    )
