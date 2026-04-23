"""HarnessState — LangGraph 의 모든 노드를 관통하는 공유 TypedDict.

1주차에서는 가이드에 기술된 필드 중 일부만 사용한다. 이후 주차에서
(skill_loader, tool_dispatch, compactor, subagent, self_improve, human_gate)
TypedDict 에 필드가 추가될 것이다. 일단은 정의를 좁게 유지하는 편이
Frozen-Snapshot 불변식을 추론하기 쉽다:
`messages` + `memory_snapshot` + `skills_catalog` 만으로 충분하다.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

Channel = Literal["cli", "telegram", "slack", "cron", "subagent"]


class HarnessState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    channel: Channel

    memory_snapshot: str
    skills_catalog: dict[str, str]

    turn: int
