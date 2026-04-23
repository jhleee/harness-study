from langchain_core.messages import AIMessage, HumanMessage

from harness.nodes.human_gate import human_gate


def test_human_gate_records_pending_call() -> None:
    state = {
        "messages": [
            AIMessage(
                content="about to run destructive tool",
                tool_calls=[{"name": "bash_rm", "args": {"path": "/tmp"}, "id": "d1"}],
            )
        ]
    }
    out = human_gate(state)  # type: ignore[arg-type]
    assert out["pending_approval"]["name"] == "bash_rm"
    assert out["pending_approval"]["id"] == "d1"


def test_human_gate_noop_when_no_tool_calls() -> None:
    out = human_gate({  # type: ignore[arg-type]
        "messages": [AIMessage(content="plain reply")]
    })
    assert out == {}


def test_human_gate_noop_when_last_is_human() -> None:
    out = human_gate({  # type: ignore[arg-type]
        "messages": [HumanMessage(content="hi")]
    })
    assert out == {}
