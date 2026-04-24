"""CLI 게이트웨이 — 1주차의 주 채널.

사용법:
    python -m harness.cli                       # 대화형 REPL
    python -m harness.cli --script inputs.txt   # 비대화형 (E2E 테스트)
    python -m harness.cli --thread-id t1 --trace data/traces/t1.jsonl

커리큘럼 권고에 따라 CLI 는 테스트와 회귀 검증의 주된 채널이다. 매 invoke 마다
TraceRecord 한 줄을 JSONL 에 기록하므로, E2E 테스트가 시스템 프롬프트의 바이트 안정성과
턴별 토큰 사용량 형태를 단언할 수 있다.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Iterable, Iterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness.config import TRACE_DIR, llm, load_settings
from harness.graph import build_graph, make_sqlite_checkpointer
from harness.metrics import (
    TraceRecord,
    TraceWriter,
    extract_usage,
    hash_system_prompt,
)

PROMPT = "you> "
SLASH_EXIT = {"/exit", "/quit", "/q"}


def _iter_inputs(script: Path | None) -> Iterator[str]:
    if script is not None:
        for line in script.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                yield line
        return
    while True:
        try:
            line = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.strip() in SLASH_EXIT:
            return
        if line.strip():
            yield line


def run(
    inputs: Iterable[str],
    thread_id: str,
    trace_path: Path,
    *,
    verbose_trace: bool = True,
    compact_threshold: int | None = None,
    db_path: Path | None = None,
    skills_dir: Path | None = None,
    auto_approve: bool = False,
) -> int:
    settings = load_settings()
    if not settings.api_key:
        print("error: OPENAI_API_KEY 가 설정되지 않았습니다 (.env 를 확인하세요)", file=sys.stderr)
        return 2
    graph_kwargs: dict = {}
    if compact_threshold is not None:
        graph_kwargs["compact_threshold"] = compact_threshold
    if db_path is not None:
        graph_kwargs["checkpointer"] = make_sqlite_checkpointer(db_path)
    if skills_dir is not None:
        graph_kwargs["skills_dir"] = skills_dir
    graph = build_graph(llm(settings), **graph_kwargs)
    writer = TraceWriter(trace_path)
    config = {"configurable": {"thread_id": thread_id}}

    turn = 0
    for user_text in inputs:
        turn += 1
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )
        # human_gate 가 interrupt_before 로 그래프를 멈춰뒀으면 사용자 승인을 받아 resume.
        # 이 단계를 빠뜨리면 다음 턴 invoke 가 ToolMessage 없이 AIMessage(tool_call) 만
        # 들고 LLM 에 가서 'tool call result does not follow tool call' 로 거부당한다.
        result = _drain_interrupts(
            graph, config, auto_approve=auto_approve, fallback=result
        )
        assistant = result["messages"][-1]
        print(f"assistant> {assistant.content}")

        usage = extract_usage(assistant)
        record = TraceRecord(
            turn=result.get("turn", turn),
            thread_id=thread_id,
            user_preview=_preview(user_text),
            assistant_preview=_preview(getattr(assistant, "content", "")),
            system_sha256=hash_system_prompt(result["messages"]),
            total_tool_calls=int(result.get("tool_call_count", 0)),
            loaded_skill_names=sorted((result.get("loaded_skills") or {}).keys()),
            **usage,
        )
        writer.write(record)
        if verbose_trace:
            print(
                f"  [trace turn={record.turn} "
                f"sys_sha={record.system_sha256[:12]} "
                f"in={record.input_tokens} out={record.output_tokens} "
                f"cached={record.cached_tokens}]",
                file=sys.stderr,
            )
    return 0


def _last_tool_call(messages: list) -> dict | None:
    """state.messages 끝에서부터 가장 최근 AIMessage 의 첫 tool_call 을 돌려준다."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            calls = getattr(m, "tool_calls", None) or []
            return calls[0] if calls else None
    return None


def _prompt_approval(call: dict | None) -> bool:
    if call is None:
        return True
    name = call.get("name", "?")
    args = call.get("args", {})
    print(f"\n[승인 요청] 파괴적 도구 호출: {name}({args})", file=sys.stderr)
    try:
        ans = input("승인하시겠습니까? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def _drain_interrupts(
    graph,
    config: dict,
    *,
    auto_approve: bool,
    fallback: dict,
    max_iters: int = 8,
) -> dict:
    """state.next 가 비워질 때까지 human_gate interrupt 를 처리한다.

    interrupt_before=['human_gate'] 가 트리거되면 첫 invoke 는 human_gate 직전에서
    멈춘다. 사용자(또는 auto_approve) 가 승인하면 invoke(None) 으로 이어 달리고,
    거부하면 update_state 로 denial ToolMessage 를 tool_dispatch 가 만든 것처럼
    삽입해 그래프가 정상 종료되도록 한다.
    """
    state = graph.get_state(config)
    if not state.next:
        return fallback
    for _ in range(max_iters):
        if not state.next:
            break
        pending = _last_tool_call(state.values.get("messages") or [])
        approved = True if auto_approve else _prompt_approval(pending)
        if approved:
            graph.invoke(None, config=config)
        else:
            tool_id = (pending or {}).get("id", "")
            graph.update_state(
                config,
                {"messages": [ToolMessage(
                    content="error: tool call denied by user",
                    tool_call_id=tool_id,
                )]},
                as_node="tool_dispatch",
            )
            graph.invoke(None, config=config)
        state = graph.get_state(config)
    if state.next:
        print(
            f"warning: {max_iters} 회 resume 후에도 pending 노드가 남음: {state.next}",
            file=sys.stderr,
        )
    return state.values


def _preview(text: str, n: int = 400) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="harness", description="에이전트 하니스 CLI")
    p.add_argument("--thread-id", default=None,
                   help="대화 thread id (기본: 새 uuid4)")
    p.add_argument("--trace", default=None,
                   help="JSONL 트레이스 파일 경로 (기본: data/traces/<thread>.jsonl)")
    p.add_argument("--script", default=None,
                   help="비대화형: 이 파일에서 한 줄에 하나씩 입력을 읽는다")
    p.add_argument("--quiet", action="store_true",
                   help="stderr 로 찍히는 턴별 트레이스 라인을 끈다")
    p.add_argument("--compact-threshold", type=int, default=None,
                   help="compactor 의 문자 수 임계치 (디버그/E2E 용).")
    p.add_argument("--db", default=None,
                   help="영속 checkpoint 용 SqliteSaver 경로.")
    p.add_argument("--skills-dir", default=None,
                   help="self_improve 출력 디렉터리 오버라이드 (E2E/디버그).")
    p.add_argument("--auto-approve", action="store_true",
                   help="human_gate 에서 파괴적 도구 호출을 자동 승인 "
                        "(--script 모드에서는 기본 활성화).")
    return p.parse_args(argv)


def _force_utf8_stdio() -> None:
    """한국어 로케일의 Windows 콘솔은 기본값이 cp949 라, 모델이 CJK / em-dash /
    스마트 따옴표 문자를 내보내면 터진다. CLI 하니스에는 UTF-8 + errors='replace' 가
    가장 덜 놀라운 기본값이다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = _parse_args(argv)
    thread_id = args.thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    trace_path = Path(args.trace) if args.trace else (TRACE_DIR / f"{thread_id}.jsonl")
    script_path = Path(args.script) if args.script else None
    # 스크립트 모드는 사람이 옆에 없으니 기본 자동 승인. 명시적 --auto-approve 도 존중.
    auto_approve = args.auto_approve or script_path is not None
    return run(
        _iter_inputs(script_path),
        thread_id=thread_id,
        trace_path=trace_path,
        verbose_trace=not args.quiet,
        compact_threshold=args.compact_threshold,
        db_path=Path(args.db) if args.db else None,
        skills_dir=Path(args.skills_dir) if args.skills_dir else None,
        auto_approve=auto_approve,
    )


if __name__ == "__main__":
    sys.exit(main())
