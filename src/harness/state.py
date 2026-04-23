"""HarnessState — LangGraph 의 모든 노드를 관통하는 공유 TypedDict.

새 기능이 올라올 때마다 주차별로 한 번씩 확장된다:

1주차 (Frozen Snapshot):
  messages, channel, memory_snapshot, skills_catalog, turn

2주차 (Progressive Disclosure):
  + loaded_skills     — {name: body}  이미 messages 에 주입된 스킬 본문.
                        덕분에 skill_loader 가 reload 요청마다 디스크를
                        다시 읽을 필요가 없다.
  + skill_last_used   — {name: turn}  축출(eviction) 휴리스틱.
  + tool_call_count   — 성공한 도구 실행 횟수를 세는 카운터.
                        4주차 self_improve 의 임계치에서 참조된다.
  + task_trace        — append-only list of {reasoning, tool, args,
                        observation} 4-튜플. 가이드 §3-3 (d) —
                        2-튜플 형태는 '에이전트가 왜 이 도구를 이 인자로
                        선택했는가' 를 잃어버리므로, self_improve 가
                        재사용 가능한 절차를 증류해 낼 수가 없다.

`total=False` 라 각 노드는 자기가 신경쓰는 필드만 건드리면 된다. 이후 주차의
노드들은 필드가 없어도 실패하지 않고 합리적인 기본값으로 short-circuit 한다.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Channel = Literal["cli", "telegram", "slack", "cron", "subagent"]


class HarnessState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    channel: Channel

    memory_snapshot: str
    skills_catalog: dict[str, str]

    loaded_skills: dict[str, str]
    skill_last_used: dict[str, int]

    tool_call_count: int
    task_trace: Annotated[list[dict[str, Any]], operator.add]

    turn: int
