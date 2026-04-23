from __future__ import annotations

from pathlib import Path

import pytest

from harness import tools
from harness.tools import ALL_TOOLS, _read_safe, load_skill, view


def test_tool_schemas_are_registered() -> None:
    names = {t.name for t in ALL_TOOLS}
    assert names == {"load_skill", "view"}


def test_load_skill_schema_has_name_param() -> None:
    schema = load_skill.args_schema.model_json_schema()
    assert "name" in schema["properties"]


def test_load_skill_body_is_sentinel() -> None:
    """@tool 본문은 의도적으로 NotImplementedError — 실제 실행은
    라우팅을 통해 skill_loader 노드가 담당한다."""
    with pytest.raises(NotImplementedError):
        load_skill.invoke({"name": "echo"})


def test_view_reads_file_inside_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    (data_root / "cache").mkdir(parents=True)
    f = data_root / "cache" / "x.txt"
    f.write_text("hello from cache", encoding="utf-8")

    monkeypatch.setattr(tools, "DATA_DIR", data_root)

    assert _read_safe(str(f)) == "hello from cache"


def test_view_refuses_path_outside_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    monkeypatch.setattr(tools, "DATA_DIR", data_root)

    assert _read_safe(str(outside)).startswith("error: path outside data/")


def test_view_tool_returns_not_found_for_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(tools, "DATA_DIR", data_root)

    result = view.invoke({"path": str(data_root / "nope.txt")})
    assert result.startswith("error: not found")
