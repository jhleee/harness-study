"""그래프 조립 — 1주차 최소 배선.

    START → gateway → session_bootstrap → agent → END

여기서 END 는 '이 invoke 는 끝났고 다음 사용자 턴을 기다린다'는 의미
(가이드 §3-3 (c) 참고) 이지 세션 종료가 아니다. 이후 주차에서는 이 자리에
노드(skill_loader, tool_dispatch, human_gate, subagent, self_improve, compactor) 와
`agent` 에서 나가는 조건부 엣지가 더해진다.

Checkpointer 는 기본값이 MemorySaver 라 같은 thread_id 로 invoke 를 반복하면
메시지가 누적된다. 영속성이 필요하면 SqliteSaver 를 넘겨 주면 된다.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from harness.nodes.agent import make_agent_node
from harness.nodes.bootstrap import session_bootstrap
from harness.nodes.gateway import gateway
from harness.state import HarnessState


def build_graph(llm: BaseChatModel, checkpointer=None):
    g = StateGraph(HarnessState)
    g.add_node("gateway", gateway)
    g.add_node("session_bootstrap", session_bootstrap)
    g.add_node("agent", make_agent_node(llm))

    g.add_edge(START, "gateway")
    g.add_edge("gateway", "session_bootstrap")
    g.add_edge("session_bootstrap", "agent")
    g.add_edge("agent", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
