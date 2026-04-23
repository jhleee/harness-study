"""Agent 노드 — LLM 호출 자리.

모듈 레벨 함수가 아니라 make_agent_node(llm) 팩토리로 구현한 이유:
- 테스트에서 FakeListChatModel 을 바인딩할 때 monkeypatch 가 필요 없다.
- 그래프 조립(M6) 에서는 MiniMax 를 향한 실제 ChatOpenAI 를 바인딩한다.
- Subagent 스폰(3주차) 은 부모 노드를 건드리지 않고 자식 그래프에 다르게 설정된
  LLM 을 바인딩할 수 있다.

LangGraph 관례상 노드 자체는 state 를 받는 단순 callable 이므로, 이 팩토리가
그 callable 을 돌려준다.

1주차 범위는 의도적으로 좁다: invoke → 응답 append → turn 카운터 증가.
tool 바인딩, 스킬 로딩, finalize_task 라우팅은 이후 주차 브랜치의 M3 이후에서 추가된다.

프로바이더 호환 shim (system_as_human):
  MiniMax 의 OpenAI 호환 /v1/chat/completions 는 messages 배열의 role="system" 을
  거부한다 (에러 2013: "invalid message role: system"). 가이드 §3-3 (b) 에서 Anthropic
  API 가 배열 중간의 SystemMessage 를 거부하는 것과 구조적으로 같은 문제이며, 해법도
  같다: 시스템 콘텐츠를 HumanMessage 로 감싸 전송에는 통과시키되 모델은 여전히 이를
  지시로 읽도록 한다.

  이 변환은 호출 시점에만 적용한다 — `state["messages"]` 는 실제 SystemMessage 를
  그대로 보존하므로, 커리큘럼의 Frozen Snapshot 불변식이 측정하는 대상을 그대로
  metrics.hash_system_prompt 가 확인할 수 있다.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from harness.state import HarnessState


def _wrap_system_as_human(messages: list[BaseMessage]) -> list[BaseMessage]:
    adapted: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            adapted.append(
                HumanMessage(content=f"<system>\n{m.content}\n</system>", id=m.id)
            )
        else:
            adapted.append(m)
    return adapted


def make_agent_node(
    llm: BaseChatModel,
    *,
    system_as_human: bool = True,
) -> Callable[[HarnessState], dict]:
    def agent(state: HarnessState) -> dict:
        msgs = state["messages"]
        if system_as_human:
            msgs = _wrap_system_as_human(msgs)
        response = llm.invoke(msgs)
        return {
            "messages": [response],
            "turn": state.get("turn", 0) + 1,
        }

    return agent
