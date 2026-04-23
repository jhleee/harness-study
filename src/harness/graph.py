"""그래프 조립.

1주차:
    START → gateway → session_bootstrap → agent → END

2주차: Progressive Disclosure + tool dispatch 추가.

3주차 (이 리비전): compactor + subagent.

    agent --spawn_subagent--> subagent       (새 자식 그래프 호출)
    agent --load_skill-->     skill_loader   → route_after_tool
    agent --그 외 도구 -->    tool_dispatch  → route_after_tool
    agent --tool_calls 없음--> END

    route_after_tool: tokens >= threshold ? compactor : agent

llm.bind_tools(ALL_TOOLS) 로 LLM 에 스키마를 바인딩한다.
subagent 는 compile 이후 graph_ref 컨테이너에 담긴 *같은* 컴파일 그래프로 재귀 호출한다.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from harness.nodes.agent import make_agent_node
from harness.nodes.bootstrap import session_bootstrap
from harness.nodes.compactor import (
    DEFAULT_COMPACT_THRESHOLD,
    estimate_tokens,
    make_compactor,
)
from harness.nodes.gateway import gateway
from harness.nodes.skill_loader import skill_loader
from harness.nodes.subagent import make_subagent_node
from harness.nodes.tool_dispatch import make_tool_dispatch
from harness.state import HarnessState
from harness.tools import ALL_TOOLS


def route_after_agent(state: HarnessState) -> str:
    """agent 노드에서 나가는 조건부 엣지.

    순서가 중요하다 — load_skill 은 전용 노드(스킬 본문 pinning) 를, spawn_subagent 는
    자식 그래프 노드를 받고, 그 외는 범용 디스패치로 흘러간다.
    `finalize_task` 같은 파괴적/종결 도구는 이후 주차에서 도입된다.
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
    if name == "spawn_subagent":
        return "subagent"
    return "tool_dispatch"


def make_route_after_tool(threshold: int) -> Callable[[HarnessState], str]:
    def route(state: HarnessState) -> str:
        if estimate_tokens(state.get("messages") or []) >= threshold:
            return "compactor"
        return "agent"
    return route


def _make_llm_summarizer(llm: BaseChatModel) -> Callable[[list[BaseMessage]], str]:
    """compactor 용 단발 요약기로 LLM 을 감싼다. 지시는 HumanMessage 로 넣어
    프로바이더 호환(MiniMax) 을 유지한다."""
    instructions = (
        "다음 대화 턴들을 이 작업을 이어갈 에이전트를 위해 요약하라. "
        "결정, 미해결 질문, 핵심 사실은 유지하라. 인사말과 잡음은 버려라. "
        "5개 이하의 bullet 로 답하라."
    )

    def _summarize(messages: list[BaseMessage]) -> str:
        rendered = "\n\n".join(
            f"[{m.__class__.__name__}] "
            + (m.content if isinstance(m.content, str) else str(m.content))
            for m in messages
        )
        response = llm.invoke(
            [HumanMessage(content=f"<system>\n{instructions}\n</system>"),
             HumanMessage(content=rendered)]
        )
        content = response.content
        return content if isinstance(content, str) else str(content)

    return _summarize


def build_graph(
    llm: BaseChatModel,
    checkpointer=None,
    *,
    tools=None,
    compact_threshold: int = DEFAULT_COMPACT_THRESHOLD,
    use_llm_summarizer: bool = True,
):
    tools = tools if tools is not None else ALL_TOOLS
    tools_by_name = {t.name: t for t in tools}

    # 스키마를 bind 해 LLM 이 tool-calling 능력을 광고하게 한다. 일부 테스트 페이크
    # (예: FakeListChatModel) 는 NotImplementedError 를 던지는 bind_tools 스텁을 가졌다 —
    # 그럴 때는 '여기서는 도구가 필요 없다' 로 간주.
    try:
        bound_llm = llm.bind_tools(tools) if hasattr(llm, "bind_tools") else llm
    except NotImplementedError:
        bound_llm = llm

    # subagent 는 자기가 재귀 호출할 컴파일된 그래프를 참조해야 한다. compile() 이후에
    # 이 컨테이너를 채워 준다.
    graph_ref: dict[str, Any] = {"graph": None}

    summarizer = _make_llm_summarizer(llm) if use_llm_summarizer else None

    g = StateGraph(HarnessState)
    g.add_node("gateway", gateway)
    g.add_node("session_bootstrap", session_bootstrap)
    g.add_node("agent", make_agent_node(bound_llm))
    g.add_node("skill_loader", skill_loader)
    g.add_node("tool_dispatch", make_tool_dispatch(tools_by_name))
    g.add_node("subagent", make_subagent_node(graph_ref))
    g.add_node("compactor", make_compactor(summarizer, threshold=compact_threshold))

    g.add_edge(START, "gateway")
    g.add_edge("gateway", "session_bootstrap")
    g.add_edge("session_bootstrap", "agent")

    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "skill_loader": "skill_loader",
            "tool_dispatch": "tool_dispatch",
            "subagent": "subagent",
            END: END,
        },
    )

    route_after_tool = make_route_after_tool(compact_threshold)
    for tool_node in ("skill_loader", "tool_dispatch", "subagent"):
        g.add_conditional_edges(
            tool_node,
            route_after_tool,
            {"compactor": "compactor", "agent": "agent"},
        )
    g.add_edge("compactor", "agent")

    compiled = g.compile(checkpointer=checkpointer or MemorySaver())
    graph_ref["graph"] = compiled
    return compiled
