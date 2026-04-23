from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from harness.nodes.agent import _wrap_system_as_human, make_agent_node


def test_agent_appends_response_and_bumps_turn() -> None:
    llm = FakeListChatModel(responses=["reply-1", "reply-2"])
    agent = make_agent_node(llm)

    state = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="hi")],
        "turn": 0,
    }
    out = agent(state)  # type: ignore[arg-type]
    assert out["turn"] == 1
    assert len(out["messages"]) == 1
    assert out["messages"][0].content == "reply-1"


def test_agent_handles_missing_turn_field() -> None:
    """bootstrap 이 돌지 않았을 때(예: 단위 테스트 직접 호출), turn 은 0 으로 기본값."""
    llm = FakeListChatModel(responses=["reply-1"])
    agent = make_agent_node(llm)
    out = agent({"messages": [HumanMessage(content="hi")]})  # type: ignore[arg-type]
    assert out["turn"] == 1


def test_wrap_system_as_human_converts_only_system_messages() -> None:
    msgs: list[BaseMessage] = [
        SystemMessage(content="rules", id="sys-1"),
        HumanMessage(content="hi"),
    ]
    out = _wrap_system_as_human(msgs)
    assert isinstance(out[0], HumanMessage)
    assert out[0].id == "sys-1"
    assert "<system>" in out[0].content and "rules" in out[0].content
    # SystemMessage 가 아닌 메시지는 원본 그대로.
    assert out[1] is msgs[1]


class _RecordingLLM:
    """에이전트가 실제로 보낸 메시지를 그대로 기록하는 작은 스텁."""

    def __init__(self) -> None:
        self.seen: list[list[BaseMessage]] = []

    def invoke(self, messages, config=None):
        self.seen.append(list(messages))
        return HumanMessage(content="noop")


def test_agent_without_adapter_sends_raw_system_message() -> None:
    llm = _RecordingLLM()
    agent = make_agent_node(llm, system_as_human=False)  # type: ignore[arg-type]
    state = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="u")],
        "turn": 0,
    }
    agent(state)  # type: ignore[arg-type]
    assert isinstance(llm.seen[0][0], SystemMessage)


def test_agent_with_adapter_sends_wrapped_human_message() -> None:
    llm = _RecordingLLM()
    agent = make_agent_node(llm, system_as_human=True)  # type: ignore[arg-type]
    state = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="u")],
        "turn": 0,
    }
    agent(state)  # type: ignore[arg-type]
    first = llm.seen[0][0]
    assert isinstance(first, HumanMessage)
    assert first.content.startswith("<system>")
