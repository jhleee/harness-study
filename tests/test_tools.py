from __future__ import annotations

from pathlib import Path

import pytest

from harness import tools
from harness.tools import (
    ALL_TOOLS,
    DESTRUCTIVE_TOOLS,
    _clear_read_cache,
    edit,
    load_skill,
    read,
    write,
)


@pytest.fixture(autouse=True)
def _reset_read_cache() -> None:
    _clear_read_cache()


def test_tool_schemas_are_registered() -> None:
    names = {t.name for t in ALL_TOOLS}
    assert names == {
        "load_skill",
        "read",
        "write",
        "edit",
        "spawn_subagent",
        "finalize_task",
    }


def test_destructive_tools_are_gated() -> None:
    assert {"write", "edit"} <= DESTRUCTIVE_TOOLS


def test_load_skill_schema_has_name_param() -> None:
    schema = load_skill.args_schema.model_json_schema()
    assert "name" in schema["properties"]


def test_load_skill_body_is_sentinel() -> None:
    """@tool 본문은 의도적으로 NotImplementedError — 실제 실행은
    라우팅을 통해 skill_loader 노드가 담당한다."""
    with pytest.raises(NotImplementedError):
        load_skill.invoke({"name": "echo"})


# --- read --------------------------------------------------------------------

def _set_data_dir(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tools, "DATA_DIR", root)
    return root


def test_read_returns_full_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_bytes(b"line1\nline2\nline3\n")
    assert read.invoke({"path": str(f)}) == "line1\nline2\nline3\n"


def test_read_view_range_slices_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_bytes(b"a\nb\nc\nd\n")
    assert read.invoke({"path": str(f), "view_range": [2, 3]}) == "b\nc\n"
    assert read.invoke({"path": str(f), "view_range": [3, -1]}) == "c\nd\n"


def test_read_rejects_path_outside_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    assert read.invoke({"path": str(outside)}).startswith("error: path outside")


def test_read_rejects_long_path_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_data_dir(monkeypatch, tmp_path / "data")
    assert read.invoke({"path": "\\\\?\\C:\\evil.txt"}).startswith(
        "error: long-path prefix"
    )


def test_read_caches_sha_for_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("hello", encoding="utf-8")
    read.invoke({"path": str(f)})
    assert str(f.resolve()) in tools._READ_CACHE


def test_read_max_bytes_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "big.txt"
    f.write_text("x" * 100, encoding="utf-8")
    result = read.invoke({"path": str(f), "max_bytes": 10})
    assert result.startswith("error: file exceeds max_bytes")


# --- write -------------------------------------------------------------------

def test_write_creates_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    target = data_root / "notes" / "n.md"
    out = write.invoke({"path": str(target), "content": "hi"})
    assert out.startswith("created:")
    assert target.read_text(encoding="utf-8") == "hi"


def test_write_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("orig", encoding="utf-8")
    out = write.invoke({"path": str(f), "content": "new"})
    assert out.startswith("error: already exists")
    assert f.read_text(encoding="utf-8") == "orig"


def test_write_rejects_outside_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_data_dir(monkeypatch, tmp_path / "data")
    out = write.invoke({"path": str(tmp_path / "x.txt"), "content": "x"})
    assert out.startswith("error: path outside")


def test_write_rejects_windows_reserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    out = write.invoke({"path": str(data_root / "nul.txt"), "content": "x"})
    assert out.startswith("error: reserved name")


def test_write_atomic_no_partial_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_atomic_write 가 실패하면 target 은 건드려지지 않아야 한다."""
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    target = data_root / "atomic.txt"

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(tools, "_atomic_write", _boom)
    out = write.invoke({"path": str(target), "content": "x"})
    assert out.startswith("error: write failed")
    assert not target.exists()


# --- edit --------------------------------------------------------------------

def test_edit_requires_prior_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    out = edit.invoke({"path": str(f), "old_string": "hello", "new_string": "hi"})
    assert out.startswith("error: must read")


def test_edit_unique_match_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    read.invoke({"path": str(f)})
    out = edit.invoke({"path": str(f), "old_string": "hello", "new_string": "hi"})
    assert out.startswith("edited:")
    assert f.read_text(encoding="utf-8") == "hi world"


def test_edit_zero_match_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("hello", encoding="utf-8")
    read.invoke({"path": str(f)})
    out = edit.invoke({"path": str(f), "old_string": "nope", "new_string": "x"})
    assert "did not appear verbatim" in out


def test_edit_multi_match_fails_without_replace_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_bytes(b"foo\nfoo\nfoo\n")
    read.invoke({"path": str(f)})
    out = edit.invoke({"path": str(f), "old_string": "foo", "new_string": "bar"})
    assert "occurrences" in out and "lines [1, 2, 3]" in out
    assert f.read_bytes() == b"foo\nfoo\nfoo\n"


def test_edit_multi_match_with_replace_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("foo foo foo", encoding="utf-8")
    read.invoke({"path": str(f)})
    out = edit.invoke(
        {"path": str(f), "old_string": "foo", "new_string": "bar", "replace_all": True}
    )
    assert out.startswith("edited:")
    assert f.read_text(encoding="utf-8") == "bar bar bar"


def test_edit_stale_after_external_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("hello", encoding="utf-8")
    read.invoke({"path": str(f)})
    f.write_text("hello (changed)", encoding="utf-8")
    out = edit.invoke({"path": str(f), "old_string": "hello", "new_string": "hi"})
    assert "changed since last read" in out


def test_edit_consecutive_edits_without_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """edit 성공 후 캐시가 새 sha 로 갱신돼 추가 read 없이도 다음 edit 가 가능."""
    data_root = _set_data_dir(monkeypatch, tmp_path / "data")
    f = data_root / "a.txt"
    f.write_text("a b c", encoding="utf-8")
    read.invoke({"path": str(f)})
    edit.invoke({"path": str(f), "old_string": "a", "new_string": "A"})
    out = edit.invoke({"path": str(f), "old_string": "b", "new_string": "B"})
    assert out.startswith("edited:")
    assert f.read_text(encoding="utf-8") == "A B c"


