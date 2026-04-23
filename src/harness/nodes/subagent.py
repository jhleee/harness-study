"""subagent — fork 가 아니라 RPC 스타일 위임.

가이드 §3-1 에 명시: subagent 는 부모의 메시지 히스토리를 절대 상속하지 않는다.
deepcopy(parent_state) 를 해 버리면 자식이 부모보다 더 많은 컨텍스트로 시작하게 되고,
'isolation' 주장이 무너진다.

계약:
- 부모는 spawn_subagent 도구 스키마를 통해 `task` (필수) 와 선택적으로
  `context`, `constraints` 만 제공한다.
- 자식 그래프는 브리핑 필드로 조립한 HumanMessage 하나만 담긴 새 메시지 리스트로 호출된다.
- 자식의 skills_catalog 와 memory_snapshot 은 read-only 참조로 넘어간다 — 자식은
  bootstrap 을 다시 돌릴 필요 없이 부모와 같은 시스템 프롬프트를 본다.
- 자식 결과는 한 줄 요약으로 압축되어, 부모에게는 tool_call 응답인 ToolMessage 한 개로 보인다.

모듈 import 시점에 컴파일된 그래프를 import 할 수 없다 (그래프도 이 노드를 import 하므로
순환). 대신 {graph: compiled_graph} 라는 mutable 컨테이너 참조를 받고, build_graph 가
compile 뒤에 여기에 값을 채운다. LangGraph 자체 서브그래프 예제와 같은 패턴이다.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from harness.state import HarnessState

SUBAGENT_RECURSION_LIMIT = 30


def _extract_spawn_call(state: HarnessState) -> dict | None:
    if not state.get("messages"):
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    for call in getattr(last, "tool_calls", None) or []:
        if call.get("name") == "spawn_subagent":
            return call
    return None


def _default_summarize(messages: list[BaseMessage]) -> str:
    ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]
    if ai_msgs:
        final = ai_msgs[-1].content
        if isinstance(final, str):
            return final[:2000]
    return "(subagent 가 최종 응답을 내지 못했습니다)"


def make_subagent_node(
    graph_ref: dict[str, Any],
    *,
    summarizer: Callable[[list[BaseMessage]], str] | None = None,
    recursion_limit: int = SUBAGENT_RECURSION_LIMIT,
) -> Callable[[HarnessState], dict]:
    """`graph_ref["graph"]` 는 build_graph 가 compile 뒤에 채우므로, 그래프 자체가
    아니라 컨테이너를 받는다."""
    summarize = summarizer or _default_summarize

    def subagent(state: HarnessState) -> dict:
        call = _extract_spawn_call(state)
        if call is None:
            return {}

        args = call.get("args") or {}
        task = args.get("task") or ""
        ctx = args.get("context") or ""
        constraints = args.get("constraints") or ""
        call_id = call.get("id", "")

        briefing = (
            f"<task>\n{task}\n</task>\n"
            f"<relevant_context>\n{ctx}\n</relevant_context>\n"
            f"<constraints>\n{constraints}\n</constraints>"
        )

        compiled = graph_ref.get("graph")
        if compiled is None:
            return {
                "messages": [
                    ToolMessage(
                        content="error: subagent runtime not wired",
                        tool_call_id=call_id,
                    )
                ]
            }

        # 새로운 state — parent.messages 는 상속하지 않고, snapshot + catalog 만
        # read-only 참조로 넘겨서 자식이 bootstrap 을 다시 돌리지 않고도 같은 시스템
        # 프롬프트를 보게 한다.
        child_state: HarnessState = {
            "messages": [HumanMessage(content=briefing)],
            "memory_snapshot": state.get("memory_snapshot", ""),
            "skills_catalog": state.get("skills_catalog") or {},
            "loaded_skills": {},
            "skill_last_used": {},
            "tool_call_count": 0,
            "task_trace": [],
            "channel": "subagent",
            "turn": 0,
        }
        child_thread = f"sub-{uuid.uuid4().hex[:10]}"
        try:
            result = compiled.invoke(
                child_state,
                config={
                    "configurable": {"thread_id": child_thread},
                    "recursion_limit": recursion_limit,
                },
            )
            summary = summarize(result.get("messages") or [])
        except Exception as exc:
            summary = f"error: subagent raised {type(exc).__name__}: {exc}"

        return {
            "messages": [ToolMessage(content=summary, tool_call_id=call_id)],
        }

    return subagent
