from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness import tools
from harness.nodes import skill_loader as skill_loader_mod
from harness.nodes.skill_loader import is_skill_message, skill_loader


def _make_ai_with_load_skill(name: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "load_skill", "args": {"name": name}, "id": call_id}],
    )


@pytest.fixture
def fake_skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    (d / "echo").mkdir()
    (d / "echo" / "SKILL.md").write_text(
        "# echo — echo the input\n\nbody-of-echo\n", encoding="utf-8"
    )
    monkeypatch.setattr(tools, "SKILLS_DIR", d)
    return d


def test_skill_loader_first_time_reads_from_disk(fake_skills_dir: Path) -> None:
    state = {
        "messages": [_make_ai_with_load_skill("echo")],
        "turn": 3,
        "loaded_skills": {},
        "skill_last_used": {},
    }
    out = skill_loader(state)  # type: ignore[arg-type]
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert msg.content.startswith("<skill:echo>")
    assert "body-of-echo" in msg.content
    assert msg.tool_call_id == "call-1"
    assert "echo" in out["loaded_skills"]
    assert out["skill_last_used"]["echo"] == 3


def test_skill_loader_uses_cache_on_second_load(
    fake_skills_dir: Path,
) -> None:
    """본문이 이미 loaded_skills 에 있으면 loader 는 디스크를 다시 치면 안 된다
    (첫 로드 뒤에 파일을 지워 버려서 검증)."""
    cached_body = "# echo — cached body\n"
    (fake_skills_dir / "echo" / "SKILL.md").unlink()

    state = {
        "messages": [_make_ai_with_load_skill("echo", call_id="c-2")],
        "turn": 5,
        "loaded_skills": {"echo": cached_body},
        "skill_last_used": {"echo": 1},
    }
    out = skill_loader(state)  # type: ignore[arg-type]
    assert "cached body" in out["messages"][0].content
    assert out["skill_last_used"]["echo"] == 5


def test_skill_loader_returns_error_for_unknown_skill(
    fake_skills_dir: Path,
) -> None:
    state = {
        "messages": [_make_ai_with_load_skill("nope")],
        "turn": 1,
        "loaded_skills": {},
        "skill_last_used": {},
    }
    out = skill_loader(state)  # type: ignore[arg-type]
    msg = out["messages"][0]
    assert isinstance(msg, ToolMessage)
    assert "unknown skill" in msg.content


def test_skill_loader_noop_when_last_message_is_not_ai(
    fake_skills_dir: Path,
) -> None:
    out = skill_loader({  # type: ignore[arg-type]
        "messages": [HumanMessage(content="hi")],
        "turn": 0,
    })
    assert out == {}


def test_is_skill_message_detects_tagged_tool_messages() -> None:
    skill = ToolMessage(content="<skill:git>\nbody\n</skill>", tool_call_id="x")
    other = ToolMessage(content="regular tool result", tool_call_id="y")
    assert is_skill_message(skill)
    assert not is_skill_message(other)
