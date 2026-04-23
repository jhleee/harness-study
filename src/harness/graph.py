"""그래프 조립.

1주차:
    START → gateway → session_bootstrap → agent → END

2주차(이 리비전) 에 Progressive Disclosure 와 tool dispatch 가 더해진다:

    START → gateway → session_bootstrap → agent
              agent --load_skill-->   skill_loader   --> agent
              agent --그 외 도구 -->  tool_dispatch  --> agent
              agent --tool_calls 없음-> END (다음 사용자 턴)

llm.bind_tools(ALL_TOOLS) 로 스키마를 바인딩해 모델이 도구 호출 능력을 광고하도록 한다.
agent 노드에는 tool-바인딩 된 LLM 을, 디스패처 노드에는 실행용 tools_by_name 레지스트리를
주입한다.

이후 주차에서는 더 많은 분기(subagent, human_gate, self_improve, compactor) 가 추가된다 —
전체 mermaid 는 가이드 §2 참고.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from harness.nodes.agent import make_agent_node
from harness.nodes.bootstrap import session_bootstrap
from harness.nodes.gateway import gateway
from harness.nodes.skill_loader import skill_loader
from harness.nodes.tool_dispatch import make_tool_dispatch
from harness.state import HarnessState
from harness.tools import ALL_TOOLS


def route_after_agent(state: HarnessState) -> str:
    """agent 노드에서 나가는 조건부 엣지.

    순서가 중요하다 — load_skill 은 전용 노드(스킬 본문 pinning) 를 받고, 그 외는
    범용 디스패치로 흘러간다. `finalize_task`, `spawn_subagent`, 파괴적 도구는
    이후 주차에서 도입된다.
    """
    if not state.get("messages"):
        return END
    last: Any = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return END
    name = calls[0].get("name", "")
    if name == "load_skill":
        return "skill_loader"
    return "tool_dispatch"


def build_graph(
    llm: BaseChatModel,
    checkpointer=None,
    *,
    tools=None,
):
    tools = tools if tools is not None else ALL_TOOLS
    tools_by_name = {t.name: t for t in tools}

    # 스키마를 bind 하여 LLM 이 tool-calling 능력을 광고하게 한다. 일부 테스트 페이크
    # (예: FakeListChatModel) 는 NotImplementedError 를 던지는 bind_tools 스텁을
    # 가지고 있다 — 그럴 때는 '여기서는 도구가 필요 없다' 로 간주.
    try:
        bound_llm = llm.bind_tools(tools) if hasattr(llm, "bind_tools") else llm
    except NotImplementedError:
        bound_llm = llm

    g = StateGraph(HarnessState)
    g.add_node("gateway", gateway)
    g.add_node("session_bootstrap", session_bootstrap)
    g.add_node("agent", make_agent_node(bound_llm))
    g.add_node("skill_loader", skill_loader)
    g.add_node("tool_dispatch", make_tool_dispatch(tools_by_name))

    g.add_edge(START, "gateway")
    g.add_edge("gateway", "session_bootstrap")
    g.add_edge("session_bootstrap", "agent")

    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "skill_loader": "skill_loader",
            "tool_dispatch": "tool_dispatch",
            END: END,
        },
    )
    g.add_edge("skill_loader", "agent")
    g.add_edge("tool_dispatch", "agent")

    return g.compile(checkpointer=checkpointer or MemorySaver())
