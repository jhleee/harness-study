from langchain_core.messages import HumanMessage

from harness.state import HarnessState


def test_state_accepts_week1_fields() -> None:
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


def test_state_accepts_week2_fields() -> None:
    state: HarnessState = {
        "messages": [HumanMessage(content="hi")],
        "loaded_skills": {"echo": "# echo body"},
        "skill_last_used": {"echo": 3},
        "tool_call_count": 2,
        "task_trace": [
            {"reasoning": "r", "tool": "load_skill",
             "args": {"name": "echo"}, "observation": "ok"}
        ],
    }
    assert state["loaded_skills"]["echo"].startswith("# echo")
    assert state["tool_call_count"] == 2
    assert state["task_trace"][0]["tool"] == "load_skill"
