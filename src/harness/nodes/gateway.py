"""Gateway 노드 — 하니스 루프를 위한 채널 멀티플렉서.

Gateway 는 여러 가능한 채널(CLI, Telegram, Slack, cron, subagent RPC, ...) 을
하나의 에이전트 루프로 합쳐 주는 단일 진입점이다. 1주차에는 CLI 채널만 필요하므로
이 노드는 아주 단순하다: `channel` 필드가 비어 있으면 채워 주고, 그 외에는 그대로 통과시킨다.

멘션 필터링이나 권한 검사는 이후 주차에서 이 자리에 들어가게 된다.
"""
from __future__ import annotations

from harness.state import HarnessState


def gateway(state: HarnessState) -> dict:
    if not state.get("channel"):
        return {"channel": "cli"}
    return {}
