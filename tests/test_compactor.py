from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from harness.nodes.compactor import estimate_tokens, make_compactor


def _msg(cls, content, mid):
    # 모든 메시지 타입이 생성자에서 id= 를 받는다.
    return cls(content=content, id=mid)


def _skill(name: str, mid: str) -> ToolMessage:
    return ToolMessage(
        content=f"<skill:{name}>\nbody of {name}\n</skill>",
        tool_call_id=f"call-{name}",
        id=mid,
    )


def test_compactor_noops_below_threshold() -> None:
    compactor = make_compactor(threshold=10_000_000)
    out = compactor({
        "messages": [HumanMessage(content="hi", id="h1")],
    })
    assert out == {}


def test_compactor_removes_old_messages_and_appends_summary() -> None:
    # 임계치를 아주 작게 강제.
    msgs = [
        SystemMessage(content="sys content", id="sys-1"),
        HumanMessage(content="q1 " * 2000, id="h1"),
        AIMessage(content="a1 " * 2000, id="a1"),
        HumanMessage(content="q2 " * 2000, id="h2"),
        AIMessage(content="a2 " * 2000, id="a2"),
        HumanMessage(content="q3", id="h3"),
    ]
    compactor = make_compactor(threshold=100, tail_keep=2)
    out = compactor({"messages": msgs})
    ops = out["messages"]
    remove_ids = {op.id for op in ops if isinstance(op, RemoveMessage)}
    # sys-1 과 마지막 tail (tail_keep=2 → 마지막 2개) 은 보호된다.
    assert "sys-1" not in remove_ids
    assert "h3" not in remove_ids
    # 최소한 옛 메시지 하나는 삭제돼야 한다.
    assert {"h1", "a1", "h2"} & remove_ids

    # 요약은 HumanMessage 로 감싼다 — 가이드 §3-3 (b) 교정.
    summary = [op for op in ops if isinstance(op, HumanMessage)]
    assert len(summary) == 1
    assert "<prior_conversation_summary>" in summary[0].content


def test_compactor_pins_skill_messages_even_when_old() -> None:
    """가이드 §3-3 (a) 재현: pinning 이 없으면 옛 스킬 ToolMessage 가 축출되어
    loaded_skills 가 현실과 어긋난다."""
    skill_msg = _skill("git", "skill-git-1")
    msgs = [
        SystemMessage(content="sys", id="sys-1"),
        skill_msg,
        HumanMessage(content="q " * 3000, id="h1"),
        AIMessage(content="a " * 3000, id="a1"),
        HumanMessage(content="q2 " * 3000, id="h2"),
        AIMessage(content="a2 " * 3000, id="a2"),
        HumanMessage(content="tail", id="tail"),
    ]
    compactor = make_compactor(threshold=100, tail_keep=1)
    out = compactor({"messages": msgs})
    remove_ids = {op.id for op in out["messages"] if isinstance(op, RemoveMessage)}
    assert "skill-git-1" not in remove_ids, \
        "compactor 가 <skill:...> ToolMessage 를 축출했다 — 가이드 §3-3 (a) 위반"


def test_compactor_uses_injected_summarizer() -> None:
    def _fake(removed):
        return f"SUMMARY of {len(removed)} msgs"

    msgs = [
        SystemMessage(content="sys", id="sys-1"),
        HumanMessage(content="x" * 5000, id="h1"),
        AIMessage(content="y" * 5000, id="a1"),
        HumanMessage(content="z" * 5000, id="h2"),
        HumanMessage(content="tail", id="tail"),
    ]
    compactor = make_compactor(summarizer=_fake, threshold=100, tail_keep=1)
    out = compactor({"messages": msgs})
    summary = [op for op in out["messages"] if isinstance(op, HumanMessage)][0]
    assert "SUMMARY of" in summary.content


def test_estimate_tokens_monotonic() -> None:
    short = [HumanMessage(content="x" * 40, id="s1")]
    long = [HumanMessage(content="x" * 4000, id="l1")]
    assert estimate_tokens(short) < estimate_tokens(long)