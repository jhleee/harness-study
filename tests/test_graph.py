from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from harness.graph import build_graph
from harness.nodes.bootstrap import SYSTEM_MESSAGE_ID


def _invoke(graph, text: str, thread_id: str = "t-test"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
    )


def test_graph_runs_end_to_end_with_fake_llm() -> None:
    llm = FakeListChatModel(responses=["hello-from-fake"])
    graph = build_graph(llm)
    result = _invoke(graph, "hi")

    msgs = result["messages"]
    assert any(isinstance(m, SystemMessage) for m in msgs)
    assert msgs[-1].content == "hello-from-fake"
    assert result["turn"] == 1
    assert "memory_snapshot" in result and result["memory_snapshot"]
    assert "skills_catalog" in result and result["skills_catalog"]


def test_system_prompt_is_byte_identical_across_turns() -> None:
    """Frozen Snapshot 불변식 — 1주차에서 가장 중요한 단일 속성이다.
    bootstrap 의 short-circuit 과 checkpointer 의 state 재사용이 맞물려
    SystemMessage 의 content 가 절대 바뀌지 않아야 한다."""
    llm = FakeListChatModel(responses=["r1", "r2", "r3"])
    graph = build_graph(llm)

    snapshots: list[str] = []
    for text in ["turn-1", "turn-2", "turn-3"]:
        result = _invoke(graph, text, thread_id="t-frozen")
        sys_msg = next(m for m in result["messages"] if m.id == SYSTEM_MESSAGE_ID)
        snapshots.append(sys_msg.content)

    assert len(set(snapshots)) == 1, "시스템 프롬프트가 턴을 넘기며 드리프트 했다"


def test_bootstrap_only_runs_once_per_thread() -> None:
    """같은 thread_id 로 세 번 턴을 돌려도 누적 메시지 히스토리에는
    SystemMessage 가 단 하나만 존재해야 한다."""
    llm = FakeListChatModel(responses=["a", "b", "c"])
    graph = build_graph(llm)

    for text in ["t1", "t2", "t3"]:
        result = _invoke(graph, text, thread_id="t-single-sys")

    system_msgs = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
