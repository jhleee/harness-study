"""CLI 테스트 — argparse 와 I/O 배관 위주. 실제 MiniMax 연동은
E2E 테스트(test_e2e_week1.py) 에서 검증한다. 단위 테스트는 빠르고 오프라인으로 유지."""
from __future__ import annotations

from pathlib import Path

from harness.cli import _iter_inputs, _parse_args


def test_parse_args_defaults() -> None:
    args = _parse_args([])
    assert args.thread_id is None
    assert args.trace is None
    assert args.script is None
    assert args.quiet is False


def test_parse_args_full() -> None:
    args = _parse_args([
        "--thread-id", "t9",
        "--trace", "out.jsonl",
        "--script", "inputs.txt",
        "--quiet",
    ])
    assert args.thread_id == "t9"
    assert args.trace == "out.jsonl"
    assert args.script == "inputs.txt"
    assert args.quiet is True


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
