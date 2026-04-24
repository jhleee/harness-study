"""CLI 테스트 — argparse 와 I/O 배관 위주. 실제 MiniMax 연동은
E2E 테스트(test_e2e_week1.py) 에서 검증한다. 단위 테스트는 빠르고 오프라인으로 유지."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from harness.cli import (
    _drain_interrupts,
    _iter_inputs,
    _last_tool_call,
    _parse_args,
)


def test_parse_args_defaults() -> None:
    args = _parse_args([])
    assert args.thread_id is None
    assert args.trace is None
    assert args.script is None
    assert args.quiet is False
    assert args.auto_approve is False


def test_parse_args_full() -> None:
    args = _parse_args([
        "--thread-id", "t9",
        "--trace", "out.jsonl",
        "--script", "inputs.txt",
        "--quiet",
        "--auto-approve",
    ])
    assert args.thread_id == "t9"
    assert args.trace == "out.jsonl"
    assert args.script == "inputs.txt"
    assert args.quiet is True
    assert args.auto_approve is True


def test_iter_inputs_reads_script_skips_comments_and_blanks(tmp_path: Path) -> None:
    f = tmp_path / "in.txt"
    f.write_text(
        "hello\n"
        "\n"
        "# this is a comment\n"
        "world\n"
        "   \n"
        "three\n",
        encoding="utf-8",
    )
    assert list(_iter_inputs(f)) == ["hello", "world", "three"]


# --- interrupt handling -----------------------------------------------------

def test_last_tool_call_normal_case() -> None:
    msgs = [
        AIMessage(content="ok"),
        AIMessage(
            content="thinking, then call",
            tool_calls=[{"name": "write", "args": {"path": "x"}, "id": "c1"}],
        ),
    ]
    call = _last_tool_call(msgs)
    assert call is not None
    assert call["name"] == "write"
    assert call["id"] == "c1"


class _FakeGraph:
    """get_state / invoke / update_state 만 구현한 미니멀 페이크."""

    def __init__(self, scenario: list[tuple[str | None, list]]) -> None:
        # scenario: 각 step 의 (next, messages). next=None 이면 종료.
        self._steps = list(scenario)
        self.invocations: list[None | dict] = []
        self.updates: list[dict] = []

    def get_state(self, _config) -> SimpleNamespace:
        nxt, msgs = self._steps[0]
        return SimpleNamespace(
            next=(nxt,) if nxt else (),
            values={"messages": msgs},
        )

    def invoke(self, payload, config=None):
        self.invocations.append(payload)
        # invoke 가 끝나면 다음 step 으로 진행한다.
        self._steps.pop(0)
        return {"messages": self._steps[0][1] if self._steps else []}

    def update_state(self, _config, values, as_node=None) -> None:
        self.updates.append({"values": values, "as_node": as_node})
        # denial 후 graph 는 종료된 상태로 본다.
        self._steps[0] = (None, self._steps[0][1] + values["messages"])


def test_drain_interrupts_no_pending_returns_fallback() -> None:
    g = _FakeGraph([(None, [AIMessage(content="hi")])])
    out = _drain_interrupts(g, {}, auto_approve=True, fallback={"sentinel": True})
    assert out == {"sentinel": True}
    assert g.invocations == []


def test_drain_interrupts_auto_approve_resumes() -> None:
    pending_msg = AIMessage(
        content="want to write",
        tool_calls=[{"name": "write", "args": {"path": "x"}, "id": "c1"}],
    )
    g = _FakeGraph([
        ("human_gate", [pending_msg]),
        (None, [pending_msg, ToolMessage(content="created", tool_call_id="c1")]),
    ])
    out = _drain_interrupts(g, {}, auto_approve=True, fallback={})
    # invoke(None) 한 번으로 종료해야 한다.
    assert g.invocations == [None]
    assert any(isinstance(m, ToolMessage) for m in out["messages"])


def test_drain_interrupts_rejection_injects_denial(monkeypatch) -> None:
    pending_msg = AIMessage(
        content="want to write",
        tool_calls=[{"name": "write", "args": {"path": "x"}, "id": "c1"}],
    )
    g = _FakeGraph([
        ("human_gate", [pending_msg]),
        (None, [pending_msg]),  # 거부 후 invoke(None) 가 진행할 종료 step.
    ])
    # 사용자에게 묻지 않고 거부로 가도록 _prompt_approval 패치.
    monkeypatch.setattr("harness.cli._prompt_approval", lambda _call: False)

    _drain_interrupts(g, {}, auto_approve=False, fallback={})

    assert len(g.updates) == 1
    upd = g.updates[0]
    assert upd["as_node"] == "tool_dispatch"
    denial = upd["values"]["messages"][0]
    assert isinstance(denial, ToolMessage)
    assert denial.tool_call_id == "c1"
    assert "denied" in denial.content
