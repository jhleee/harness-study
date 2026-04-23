from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from harness.metrics import (
    TraceRecord,
    TraceWriter,
    extract_usage,
    hash_system_prompt,
)


def test_extract_usage_reads_standard_fields() -> None:
    msg = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 80},
        },
    )
    u = extract_usage(msg)
    assert u == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cached_tokens": 80,
    }


def test_extract_usage_handles_missing_metadata() -> None:
    msg = AIMessage(content="hi")
    u = extract_usage(msg)
    assert u == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }


def test_hash_system_prompt_stable() -> None:
    msgs = [SystemMessage(content="abc"), HumanMessage(content="user")]
    h1 = hash_system_prompt(msgs)
    h2 = hash_system_prompt(msgs)
    assert h1 == h2 and len(h1) == 64


def test_hash_system_prompt_detects_drift() -> None:
    a = [SystemMessage(content="abc")]
    b = [SystemMessage(content="abd")]
    assert hash_system_prompt(a) != hash_system_prompt(b)


def test_trace_writer_roundtrip(tmp_path: Path) -> None:
    w = TraceWriter(tmp_path / "trace.jsonl")
    w.write(TraceRecord(turn=1, thread_id="t1", input_tokens=10, output_tokens=5))
    w.write(TraceRecord(turn=2, thread_id="t1", input_tokens=12, output_tokens=7))
    records = w.read_all()
    assert [r["turn"] for r in records] == [1, 2]
    assert records[0]["input_tokens"] == 10
