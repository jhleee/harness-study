"""compactor — 스킬 pinning 과 HumanMessage 래핑 요약을 갖춘 슬라이딩 윈도우 축약기.

가이드 §3-3 (a) + (b) 가 이 노드가 피해야 하는 함정이다:

(a) 순진한 `messages[:-8]` 삭제는 로드된 <skill:NAME> ToolMessage 까지 내보낸다.
    그런데 loaded_skills state 는 '이미 로드됐음' 이라고 말하고 있으므로, LLM 은
    절차 없이 돌아가서 환각을 일으킨다. 해법: 스킬 메시지를 pin 한다.

(b) 요약을 배열 중간의 SystemMessage 로 감싸면 Anthropic Messages API 가 깨진다
    (거기서는 role 이 user/assistant 여야 함). MiniMax 는 더 엄격해서 1주차부터
    SystemMessage 자체를 거부한다. 해법: <prior_conversation_summary> 태그를
    두른 HumanMessage 로 감싸 LLM 이 주입 컨텍스트로 읽도록 한다.

이 노드는 GATED — 임계치 아래에서는 {} 를 돌려준다. 비싼 LLM 요약 호출은 실제로
토큰 예산 근처일 때만 돌아가므로, 평상시 턴 비용은 0 이다.

summarizer 는 주입 가능해서 실 LLM 없이도 테스트가 동작을 단언할 수 있다.
"""
from __future__ import annotations

from typing import Callable, Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from harness.nodes.skill_loader import is_skill_message
from harness.state import HarnessState

TAIL_KEEP = 8
DEFAULT_COMPACT_THRESHOLD = 80_000


def estimate_tokens(messages: Iterable[BaseMessage]) -> int:
    """char/4 대략 휴리스틱 — 여기서 토크나이저 수준의 정확도는 필요 없고,
    임계치 게이트가 대략 적절한 대화 길이에서 트립되도록 단조성만 있으면 된다.
    우리가 진짜 신경 쓰는 건 캐시 프리픽스 비용이고, 그게 문자 수에 대략 선형으로
    비례한다."""
    total = 0
    for m in messages:
        content = m.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                total += len(str(part))
    return total // 4


def _summarize_fallback(removed: list[BaseMessage]) -> str:
    """LLM summarizer 가 주입되지 않았을 때 사용. 의미적으로 풍부하지는 않지만
    절대 실패하지 않으므로 노드의 오프라인 테스트 가능성을 지켜 준다."""
    lines = []
    for m in removed:
        role = m.__class__.__name__.replace("Message", "")
        text = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {text[:200]}")
    return "Prior turns (dropped for brevity):\n" + "\n".join(lines)


def make_compactor(
    summarizer: Callable[[list[BaseMessage]], str] | None = None,
    *,
    threshold: int = DEFAULT_COMPACT_THRESHOLD,
    tail_keep: int = TAIL_KEEP,
) -> Callable[[HarnessState], dict]:
    def compactor(state: HarnessState) -> dict:
        messages: list[BaseMessage] = list(state.get("messages") or [])
        if estimate_tokens(messages) < threshold:
            return {}

        tail_ids = {m.id for m in messages[-tail_keep:] if getattr(m, "id", None)}
        skill_ids = {m.id for m in messages if is_skill_message(m) and getattr(m, "id", None)}
        # SystemMessage 도 pin — 우리가 새로 추가하지는 않지만, 여기서 빠지면
        # Frozen Snapshot 바이트 안정성 테스트가 깨진다.
        sys_ids = {m.id for m in messages if isinstance(m, SystemMessage) and getattr(m, "id", None)}
        protected = tail_ids | skill_ids | sys_ids

        to_remove = [
            m for m in messages
            if getattr(m, "id", None) and m.id not in protected
        ]
        if not to_remove:
            return {}

        if summarizer is None:
            summary_text = _summarize_fallback(to_remove)
        else:
            summary_text = summarizer(to_remove)

        wrapper = HumanMessage(
            content=(
                "<prior_conversation_summary>\n"
                f"{summary_text}\n"
                "</prior_conversation_summary>\n"
                "(위는 자동 축약된 이전 대화다. 계속 진행하라.)"
            ),
        )

        return {
            "messages": [RemoveMessage(id=m.id) for m in to_remove] + [wrapper],
        }

    return compactor
