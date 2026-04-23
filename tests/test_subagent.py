from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness.nodes.subagent import make_subagent_node


class _FakeCompiledGraph:
    """호출된 sub_state 를 기록하고 미리 준비된 결과를 돌려준다."""

    def __init__(self, canned_response: str = "canned-summary"):
        self.canned = canned_response
        self.invocations: list[tuple[dict, dict]] = []

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        self.invocations.append((state, config or {}))
        return {
            "messages": [
                HumanMessage(content="briefing echo"),
                AIMessage(content=self.canned),
            ]
        }


def _parent_state_with_spawn(task: str, ctx: str = "", cons: str = "") -> dict:
    return {
        "messages": [
            HumanMessage(content="parent prompt"),
            AIMessage(
                content="delegating",
                tool_calls=[
                    {
                        "name": "spawn_subagent",
                        "args": {"task": task, "context": ctx, "constraints": cons},
                        "id": "sub-1",
                    }
                ],
            ),
        ],
        "memory_snapshot": "mem-snap",
        "skills_catalog": {"echo": "Echo skill"},
        "loaded_skills": {"echo": "body"},
        "tool_call_count": 5,
    }


def test_subagent_returns_summary_tool_message() -> None:
    fake = _FakeCompiledGraph(canned_response="child-final-answer")
    node = make_subagent_node({"graph": fake})
    out = node(_parent_state_with_spawn("find X"))
    msgs = out["messages"]
    assert len(msgs) == 1 and isinstance(msgs[0], ToolMessage)
    assert msgs[0].content == "child-final-answer"
    assert msgs[0].tool_call_id == "sub-1"


def test_subagent_child_gets_briefing_only_not_parent_messages() -> None:
    """가이드 §3-1: 자식은 부모 메시지를 상속하지 않는다.
    spawn_subagent 도구 args 로부터 만든 briefing 만 받는다."""
    fake = _FakeCompiledGraph()
    node = make_subagent_node({"graph": fake})
    _ = node(_parent_state_with_spawn(
        task="find something",
        ctx="look in /tmp/foo",
        cons="read-only",
    ))
    child_state, child_config = fake.invocations[0]
    assert len(child_state["messages"]) == 1
    first = child_state["messages"][0]
    assert isinstance(first, HumanMessage)
    assert "<task>" in first.content and "find something" in first.content
    assert "<relevant_context>" in first.content and "look in /tmp/foo" in first.content
    assert "<constraints>" in first.content and "read-only" in first.content
    # 그리고 가장 중요한 것 — 부모의 "parent prompt" 는 자식에 들어가면 안 된다.
    assert "parent prompt" not in first.content


def test_subagent_child_inherits_readonly_references() -> None:
    fake = _FakeCompiledGraph()
    node = make_subagent_node({"graph": fake})
    node(_parent_state_with_spawn("t"))
    child_state, _ = fake.invocations[0]

    # catalog 와 snapshot 을 넘겨줘서, 자식의 bootstrap 이 디스크 접근 없이
    # 부모와 같은 시스템 프롬프트를 읽게 한다.
    assert child_state["memory_snapshot"] == "mem-snap"
    assert child_state["skills_catalog"] == {"echo": "Echo skill"}
    # loaded_skills, task_trace, tool_call_count 는 반드시 새로 시작되어야 한다.
    assert child_state["loaded_skills"] == {}
    assert child_state["task_trace"] == []
    assert child_state["tool_call_count"] == 0
    assert child_state["channel"] == "subagent"


def test_subagent_child_gets_its_own_thread_id() -> None:
    """같은 부모에서 파생된 두 형제 subagent 는 checkpointer thread state 를
    공유해서는 안 된다. 호출마다 thread id 가 달라야 한다."""
    fake = _FakeCompiledGraph()
    node = make_subagent_node({"graph": fake})
    node(_parent_state_with_spawn("t1"))
    node(_parent_state_with_spawn("t2"))

    id1 = fake.invocations[0][1]["configurable"]["thread_id"]
    id2 = fake.invocations[1][1]["configurable"]["thread_id"]
    assert id1 != id2
    assert id1.startswith("sub-") and id2.startswith("sub-")


def test_subagent_errors_surface_as_tool_message() -> None:
    class _Boom:
        def invoke(self, *_a, **_k):
            raise RuntimeError("child exploded")

    node = make_subagent_node({"graph": _Boom()})
    out = node(_parent_state_with_spawn("anything"))
    content = out["messages"][0].content
    assert content.startswith("error:")
    assert "child exploded" in content


def test_subagent_no_op_if_last_is_not_spawn_call() -> None:
    node = make_subagent_node({"graph": _FakeCompiledGraph()})
    assert node({
        "messages": [HumanMessage(content="hi")],
    }) == {}
