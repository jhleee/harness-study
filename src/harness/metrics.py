"""메트릭 및 트레이스 로깅.

1주차 커리큘럼이 명시적으로 요구하는 것:
  - 턴별 cache_creation / cache_read 입력 토큰
  - 시스템 프롬프트 바이트가 턴을 넘나들며 안정적이라는 증거

CLI(M7) 와 E2E 테스트(M9) 가 조합해 쓰는 원시 함수들을 노출한다:

- extract_usage(msg)        — LangChain 의 표준 usage_metadata 와 프로바이더가
                              노출하는 캐시 필드
- hash_system_prompt(msgs)  — SystemMessage content 의 sha256, 턴 간 바이트
                              안정성을 단언하는 데 사용
- TraceWriter               — append-only JSONL 라이터

MiniMax 의 OpenAI 호환 API 는 (이 구현 시점 기준) 일부 OpenAI 모델처럼
prompt_tokens_details 에 cached-token 수를 보고하지 않는다. 그래서 extractor 는
실패하는 대신 0 을 보고한다 — 동작은 커리큘럼의 트레이스 JSONL 스키마를 그대로
행사하고, 1주차의 중요한 단언(시스템 바이트가 턴을 넘기며 안정적이다) 은
프로바이더 보고 캐시 메트릭이 아니라 hash_system_prompt 로 검증한다.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import BaseMessage, SystemMessage


@dataclass
class TraceRecord:
    turn: int
    thread_id: str
    node: str = "agent"
    ts: float = field(default_factory=lambda: time.time())
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    system_sha256: str = ""
    user_preview: str = ""
    assistant_preview: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def extract_usage(msg: BaseMessage) -> dict[str, int]:
    usage = getattr(msg, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    details = usage.get("input_token_details") or {}
    cached = int(details.get("cache_read") or details.get("cached") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached,
    }


def hash_system_prompt(messages: Iterable[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, SystemMessage):
            return hashlib.sha256(m.content.encode("utf-8")).hexdigest()
    return ""


def _preview(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


class TraceWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TraceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
