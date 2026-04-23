"""그래프 조립.

진화 요약:
  1주차 — frozen snapshot (gateway → bootstrap → agent → END)
  2주차 — Progressive Disclosure (+ skill_loader, tool_dispatch)
  3주차 — externalization (+ compactor, subagent)
  4주차 (이 리비전) — 자기개선 + 승인 게이트
           + self_improve (finalize_task 로 SKILL.md 증류)
           + human_gate    (파괴적 도구에 대한 interrupt_before)

전체 라우팅 mermaid — 가이드 §2 참고.
"""
from __future__ import annotations

from pathlib import Path
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
from harness.nodes.human_gate import human_gate
from harness.nodes.self_improve import make_self_improve_node
from harness.nodes.skill_loader import skill_loader
from harness.nodes.subagent import make_subagent_node
from harness.nodes.tool_dispatch import make_tool_dispatch
from harness.state import HarnessState
from harness.tools import ALL_TOOLS, DESTRUCTIVE_TOOLS


def route_after_agent(state: HarnessState) -> str:
    """agent 노드에서 나가는 조건부 엣지.

    순서가 중요하다 — finalize_task 는 self_improve 로, load_skill 은 전용 노드
    (스킬 본문 pinning), spawn_subagent 는 자식 그래프 노드, DESTRUCTIVE_TOOLS 는
    human_gate 로, 그 외는 범용 디스패치로.
    """
    if not state.get("messages"):
        return END
    last: Any = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return END
    name = calls[0].get("name", "")
    if name == "finalize_task":
        return "self_improve"
    if name == "load_skill":
        return "skill_loader"
    if name == "spawn_subagent":
        return "subagent"
    if name in DESTRUCTIVE_TOOLS:
        return "human_gate"
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
    skills_dir: Path | None = None,
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
    g.add_node("self_improve", make_self_improve_node(skills_dir=skills_dir))
    g.add_node("human_gate", human_gate)

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
            "self_improve": "self_improve",
            "human_gate": "human_gate",
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

    # self_improve 는 이번 턴의 종착점 — SKILL.md 가 쓰였고 LLM 은 확인 ToolMessage 를
    # 받았으니, 다시 agent 로 돌려 보내 추가 숙고를 시킬 필요가 없다.
    g.add_edge("self_improve", END)

    # 사람 승인이 끝나 그래프가 resume 되면, 파괴적 도구를 일반 도구처럼 tool_dispatch
    # 로 흘려 보낸다 — human_gate 는 승인 기록만 하고, 실제 실행은 디스패처의 몫.
    g.add_edge("human_gate", "tool_dispatch")

    compiled = g.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["human_gate"],
    )
    graph_ref["graph"] = compiled
    return compiled


def make_sqlite_checkpointer(db_path: str | Path):
    """호출자가 langgraph 내부를 import 하지 않도록 하는 얇은 래퍼."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
