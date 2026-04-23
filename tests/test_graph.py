from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from harness.graph import build_graph
from harness.nodes.bootstrap import SYSTEM_MESSAGE_ID


class _ScriptedToolCallingLLM(BaseChatModel):
    """최소한의 tool-calling 페이크. 매 .invoke() 마다 스크립트에서 AIMessage 를
    하나씩 소비한다. bind_tools() 는 self 를 반환하는 no-op — 실제로 스키마를
    내보낼 필요가 없다. 응답은 테스트가 공급한다."""

    responses: list[AIMessage] = []
    call_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self.call_count
        object.__setattr__(self, "call_count", idx + 1)
        msg = self.responses[idx]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "_scripted_tool_calling_fake"


def _invoke(graph, text: str, thread_id: str = "t-test"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
    )


# ------ 1주차 회귀 테스트 (2주차 변경 후에도 계속 통과해야 함) -----

def test_graph_runs_end_to_end_with_fake_llm() -> None:
    llm = FakeListChatModel(responses=["hello-from-fake"])
    graph = build_graph(llm)
    result = _invoke(graph, "hi")

    msgs = result["messages"]
    assert any(isinstance(m, SystemMessage) for m in msgs)
    assert msgs[-1].content == "hello-from-fake"
    assert result["turn"] == 1


def test_system_prompt_is_byte_identical_across_turns() -> None:
    llm = FakeListChatModel(responses=["r1", "r2", "r3"])
    graph = build_graph(llm)
    snapshots: list[str] = []
    for text in ["t-1", "t-2", "t-3"]:
        result = _invoke(graph, text, thread_id="t-frozen")
        sys_msg = next(m for m in result["messages"] if m.id == SYSTEM_MESSAGE_ID)
        snapshots.append(sys_msg.content)
    assert len(set(snapshots)) == 1


def test_bootstrap_only_runs_once_per_thread() -> None:
    llm = FakeListChatModel(responses=["a", "b", "c"])
    graph = build_graph(llm)
    for text in ["t1", "t2", "t3"]:
        result = _invoke(graph, text, thread_id="t-single-sys")
    system_msgs = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1


# ------------------------- 2주차 라우팅 테스트 ----------------------------

def test_load_skill_tool_call_routes_through_skill_loader() -> None:
    """에이전트가 턴 1 에 load_skill(echo) 를 부르고, 턴 2 에 응답."""
    llm = _ScriptedToolCallingLLM(
        responses=[
            AIMessage(
                content="I need the echo skill.",
                tool_calls=[
                    {"name": "load_skill", "args": {"name": "echo"}, "id": "c1"}
                ],
            ),
            AIMessage(content="OK I have echo now."),
        ]
    )
    graph = build_graph(llm)
    result = _invoke(graph, "hi", thread_id="t-skill")

    msgs: list[BaseMessage] = result["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content.startswith("<skill:echo>")
    assert "echo" in result["loaded_skills"]
    assert result["skill_last_used"]["echo"] == result["turn"] - 1  # 두 번째 agent 증가 전에 기록


def test_view_tool_call_routes_through_tool_dispatch(tmp_path) -> None:
    """에이전트가 턴 1 에 view(path) 호출 → tool_dispatch 가 파일 읽기 → agent."""
    from harness.config import DATA_DIR
    target = DATA_DIR / "cache" / "test_view_graph.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("file-body", encoding="utf-8")

    llm = _ScriptedToolCallingLLM(
        responses=[
            AIMessage(
                content="Reading the file.",
                tool_calls=[
                    {"name": "view", "args": {"path": str(target)}, "id": "v1"}
                ],
            ),
            AIMessage(content="Saw it."),
        ]
    )
    graph = build_graph(llm)
    try:
        result = _invoke(graph, "go", thread_id="t-view")
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "file-body" in tool_msgs[0].content
        assert result["tool_call_count"] == 1
        assert result["task_trace"][0]["tool"] == "view"
    finally:
        target.unlink(missing_ok=True)


def test_no_tool_calls_routes_to_end() -> None:
    """tool_calls 가 없는 평범한 AIMessage → END. turn 카운터는 한 번만 증가."""
    llm = _ScriptedToolCallingLLM(responses=[AIMessage(content="done.")])
    graph = build_graph(llm)
    result = _invoke(graph, "hi", thread_id="t-end")
    assert result["turn"] == 1
    assert result["messages"][-1].content == "done."
