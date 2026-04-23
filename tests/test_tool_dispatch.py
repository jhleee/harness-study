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
