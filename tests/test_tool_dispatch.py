from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from harness.nodes.tool_dispatch import (
    OFFLOAD_THRESHOLD,
    _maybe_offload,
    make_tool_dispatch,
)


@tool
def _echo_tool(text: str) -> str:
    """입력 텍스트를 그대로 돌려준다."""
    return text


@tool
def _crash_tool() -> str:
    """항상 예외를 던진다 — 에러 경로 커버리지용."""
    raise RuntimeError("boom")


@tool
def _huge_tool(n: int) -> str:
    """길이 n 의 문자열을 돌려준다."""
    return "x" * n


def _ai_with_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="deciding to use the tool",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def test_dispatch_runs_tool_and_appends_tool_message(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    state = {"messages": [_ai_with_call("_echo_tool", {"text": "hello"})]}
    out = dispatch(state)  # type: ignore[arg-type]

    msgs = out["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert msgs[0].content == "hello"
    assert msgs[0].tool_call_id == "c1"


def test_dispatch_records_trace_4_tuple(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    out = dispatch({  # type: ignore[arg-type]
        "messages": [_ai_with_call("_echo_tool", {"text": "hi"})],
    })
    assert len(out["task_trace"]) == 1
    entry = out["task_trace"][0]
    assert set(entry.keys()) == {"reasoning", "tool", "args", "observation"}
    assert entry["tool"] == "_echo_tool"
    assert entry["args"] == {"text": "hi"}
    assert entry["reasoning"].startswith("deciding to use")


def test_dispatch_increments_tool_call_count(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    out = dispatch({  # type: ignore[arg-type]
        "messages": [_ai_with_call("_echo_tool", {"text": "x"})],
        "tool_call_count": 4,
    })
    assert out["tool_call_count"] == 5


def test_dispatch_returns_error_tool_message_on_unknown(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    out = dispatch({  # type: ignore[arg-type]
        "messages": [_ai_with_call("no_such_tool", {})],
    })
    assert "unknown tool" in out["messages"][0].content


def test_dispatch_catches_tool_exception(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_crash_tool": _crash_tool}, cache_dir=tmp_path)
    out = dispatch({  # type: ignore[arg-type]
        "messages": [_ai_with_call("_crash_tool", {})],
    })
    content = out["messages"][0].content
    assert content.startswith("error:")
    assert "RuntimeError" in content


def test_dispatch_noops_when_last_message_is_not_ai(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    assert dispatch({"messages": [HumanMessage(content="hi")]}) == {}  # type: ignore[arg-type]


def test_offload_replaces_content_with_pointer(tmp_path: Path) -> None:
    big = "A" * (OFFLOAD_THRESHOLD + 50)
    out = _maybe_offload(big, tmp_path)
    assert out.startswith("<result path=")
    assert "truncated" in out
    # Exactly one file written
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == big


def test_offload_passes_through_small_results(tmp_path: Path) -> None:
    small = "A" * 500
    out = _maybe_offload(small, tmp_path)
    assert out == small
    assert list(tmp_path.iterdir()) == []


def test_dispatch_offloads_large_tool_output(tmp_path: Path) -> None:
    dispatch = make_tool_dispatch({"_huge_tool": _huge_tool}, cache_dir=tmp_path)
    out = dispatch({  # type: ignore[arg-type]
        "messages": [_ai_with_call("_huge_tool", {"n": OFFLOAD_THRESHOLD + 200})],
    })
    content = out["messages"][0].content
    assert content.startswith("<result path=")
    files = list(tmp_path.iterdir())
    assert len(files) == 1


def test_dispatch_handles_parallel_tool_calls(tmp_path: Path) -> None:
    """모델이 한 AIMessage 에 tool_calls 를 여러 개 담아 보내면, dispatch 는
    호출당 1개씩 ToolMessage 를 만들어야 한다 — 이걸 빼먹으면 다음 LLM invoke 가
    'tool call and result not match' 로 거부된다.
    """
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    parallel = AIMessage(
        content="calling echo three times",
        tool_calls=[
            {"name": "_echo_tool", "args": {"text": "a"}, "id": "c1"},
            {"name": "_echo_tool", "args": {"text": "b"}, "id": "c2"},
            {"name": "_echo_tool", "args": {"text": "c"}, "id": "c3"},
        ],
    )
    out = dispatch({"messages": [parallel]})  # type: ignore[arg-type]

    # tool_call 개수만큼 ToolMessage 가 있고 id 가 1:1 로 매칭되어야 한다.
    msgs = out["messages"]
    assert len(msgs) == 3
    assert [m.tool_call_id for m in msgs] == ["c1", "c2", "c3"]
    assert [m.content for m in msgs] == ["a", "b", "c"]
    # 카운터/트레이스도 호출 수만큼 누적.
    assert out["tool_call_count"] == 3
    assert len(out["task_trace"]) == 3
    assert [t["args"]["text"] for t in out["task_trace"]] == ["a", "b", "c"]


def test_dispatch_unknown_tool_in_parallel_still_emits_message(tmp_path: Path) -> None:
    """알 수 없는 도구가 끼어 있어도 다른 호출 응답은 정상으로 나오고 해당
    호출에는 error ToolMessage 가 매겨진다 — 응답 누락으로 시퀀스를 깨면 안 된다."""
    dispatch = make_tool_dispatch({"_echo_tool": _echo_tool}, cache_dir=tmp_path)
    parallel = AIMessage(
        content="mixed",
        tool_calls=[
            {"name": "_echo_tool", "args": {"text": "ok"}, "id": "c1"},
            {"name": "_nope", "args": {}, "id": "c2"},
        ],
    )
    out = dispatch({"messages": [parallel]})  # type: ignore[arg-type]
    assert [m.tool_call_id for m in out["messages"]] == ["c1", "c2"]
    assert out["messages"][0].content == "ok"
    assert "unknown tool" in out["messages"][1].content
