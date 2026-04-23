from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from harness.nodes.agent import make_agent_node


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
