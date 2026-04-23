from langchain_core.messages import HumanMessage

from harness.state import HarnessState


def test_state_accepts_required_fields() -> None:
    state: HarnessState = {
        "messages": [HumanMessage(content="hi")],
        "channel": "cli",
        "memory_snapshot": "snapshot text",
        "skills_catalog": {"echo": "Echo user input"},
        "turn": 0,
    }
    assert state["channel"] == "cli"
    assert state["turn"] == 0
    assert state["skills_catalog"]["echo"].startswith("Echo")
