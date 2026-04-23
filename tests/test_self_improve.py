from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness.nodes.self_improve import (
    SKILL_DISTILL_THRESHOLD,
    _slugify,
    make_self_improve_node,
)


def _ai_finalize(title: str, call_id: str = "fin-1") -> AIMessage:
    return AIMessage(
        content="task done",
        tool_calls=[{
            "name": "finalize_task",
            "args": {"title": title},
            "id": call_id,
        }],
    )


def _trace_entries(n: int) -> list[dict]:
    return [
        {
            "reasoning": f"step {i}",
            "tool": "view",
            "args": {"path": f"/tmp/{i}"},
            "observation": f"result-{i}",
        }
        for i in range(n)
    ]


def test_slugify_basics() -> None:
    assert _slugify("My Cool Skill!") == "my-cool-skill"
    assert _slugify("  Lots   of  spaces ") == "lots-of-spaces"
    assert _slugify("") == "untitled-skill"


def test_self_improve_below_threshold_returns_error(tmp_path: Path) -> None:
    node = make_self_improve_node(skills_dir=tmp_path)
    out = node({
        "messages": [_ai_finalize("too-early")],
        "task_trace": _trace_entries(2),
        "tool_call_count": 2,
    })
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.content.startswith("error:")
    assert "threshold" in msg.content
    # Crucially, nothing was written.
    assert list(tmp_path.iterdir()) == []


def test_self_improve_at_threshold_writes_skill_md(tmp_path: Path) -> None:
    node = make_self_improve_node(skills_dir=tmp_path)
    out = node({
        "messages": [_ai_finalize("git-log-search")],
        "task_trace": _trace_entries(SKILL_DISTILL_THRESHOLD),
        "tool_call_count": SKILL_DISTILL_THRESHOLD,
    })
    written = tmp_path / "git-log-search" / "SKILL.md"
    assert written.exists()
    body = written.read_text(encoding="utf-8")
    assert "git-log-search" in body
    assert "Procedure" in body
    assert "view" in body
    # Ack message references the path.
    assert "saved:" in out["messages"][0].content


def test_self_improve_uses_injected_distiller(tmp_path: Path) -> None:
    def _fake(title, trace):
        return f"# custom body for {title} — {len(trace)} steps\n"

    node = make_self_improve_node(skills_dir=tmp_path, distiller=_fake)
    out = node({
        "messages": [_ai_finalize("inject-test")],
        "task_trace": _trace_entries(7),
        "tool_call_count": 7,
    })
    body = (tmp_path / "inject-test" / "SKILL.md").read_text(encoding="utf-8")
    assert body.startswith("# custom body for inject-test")


def test_self_improve_noop_without_finalize_call(tmp_path: Path) -> None:
    node = make_self_improve_node(skills_dir=tmp_path)
    out = node({
        "messages": [HumanMessage(content="hi")],
        "task_trace": _trace_entries(10),
        "tool_call_count": 10,
    })
    assert out == {}
