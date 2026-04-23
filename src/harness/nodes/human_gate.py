"""human_gate — 파괴적 도구 호출에 대한 인터럽트 지점.

그래프는 interrupt_before=['human_gate'] 로 컴파일되므로, 라우터가 agent 의
AIMessage 를 이 노드로 보내는 순간 LangGraph 가 실행을 멈추고 제어권을 호출자에게
돌려준다. 외부(out-of-band) 검토자가 `state.pending_approval` 을 보고:

  - 승인  → 그래프를 resume. 이 노드가 돌면서 승인되었음을 기록하고,
            일반 tool_dispatch 파이프라인이 도구를 실제로 실행하게 한다.
  - 거부  → 호출자가 실행을 취소하고, 보통 거부 이유를 설명하는 ToolMessage 를 붙인다.

노드 자체는 의도적으로 최소로 유지 — 검토자가 볼 수 있도록 state 에 pending call 을
기록할 뿐이다. 실제 suspend/resume 은 LangGraph 의 interrupt 시맨틱이 담당한다.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from harness.state import HarnessState


def human_gate(state: HarnessState) -> dict:
    if not state.get("messages"):
        return {}
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return {}
    return {"pending_approval": calls[0]}
