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
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel

from harness.state import HarnessState


def make_agent_node(llm: BaseChatModel) -> Callable[[HarnessState], dict]:
    def agent(state: HarnessState) -> dict:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response],
            "turn": state.get("turn", 0) + 1,
        }

    return agent
