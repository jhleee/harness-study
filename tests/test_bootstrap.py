from __future__ import annotations

from pathlib import Path

import pytest

from harness.nodes import bootstrap
from harness.nodes.bootstrap import (
    SYSTEM_MESSAGE_ID,
    build_catalog,
    build_system_content,
    session_bootstrap,
)


@pytest.fixture
def fake_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """내용이 정해진 가짜 data/ 트리를 만들어, bootstrap 의 출력이
    테스트 입력으로 완전히 결정되도록 한다."""
    mem = tmp_path / "memories"
    skills = tmp_path / "skills"
    mem.mkdir()
    skills.mkdir()
    (mem / "MEMORY.md").write_text("facts: alpha\n", encoding="utf-8")
    (mem / "USER.md").write_text("user: beta\n", encoding="utf-8")
    for name, desc in [
        ("zeta", "Zeta skill description"),
        ("alpha", "Alpha skill description"),
        ("mu", "Mu skill description"),
    ]:
        d = skills / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"# {name} — {desc}\n\nBody.\n", encoding="utf-8"
        )
    monkeypatch.setattr(bootstrap, "MEMORIES_DIR", mem)
    monkeypatch.setattr(bootstrap, "SKILLS_DIR", skills)
    return tmp_path


def test_build_catalog_sorts_by_name(fake_layout: Path) -> None:
    catalog = build_catalog(fake_layout / "skills")
    assert list(catalog.keys()) == ["alpha", "mu", "zeta"]
    assert catalog["alpha"] == "alpha — Alpha skill description"


def test_build_system_content_is_byte_deterministic(fake_layout: Path) -> None:
    """N 번 실행해서 모든 출력이 바이트 단위로 동일해야 한다 — 이 속성이
    프롬프트 캐시 프리픽스를 턴을 넘나들며 뜨겁게 유지해 준다."""
    outputs = {build_system_content()[0] for _ in range(5)}
    assert len(outputs) == 1, "시스템 콘텐츠가 바이트 단위로 결정적이지 않다"


def test_system_content_contains_all_sections(fake_layout: Path) -> None:
    content, snapshot, catalog = build_system_content()
    assert "<memory>" in content and "facts: alpha" in content
    assert "<user>" in content and "user: beta" in content
    assert "<skills_catalog>" in content
    assert "alpha" in content and "mu" in content and "zeta" in content
    assert snapshot.startswith("facts: alpha")
    assert "alpha" in catalog


def test_session_bootstrap_sets_state_and_emits_one_system(fake_layout: Path) -> None:
    out = session_bootstrap({})  # type: ignore[arg-type]
    assert out["turn"] == 0
    assert len(out["messages"]) == 1
    assert out["messages"][0].id == SYSTEM_MESSAGE_ID
    assert out["messages"][0].type == "system"
    assert "alpha" in out["skills_catalog"]


def test_session_bootstrap_short_circuits_when_already_bootstrapped(
    fake_layout: Path,
) -> None:
    out = session_bootstrap({"memory_snapshot": "already there"})  # type: ignore[arg-type]
    assert out == {}, "2턴째부터 bootstrap 이 시스템 프롬프트를 다시 방출하면 안 된다"
