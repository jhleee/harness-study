"""설정 로더와 LLM 팩토리.

python-dotenv 로 `.env` 를 한 번만 읽어 들이고, 프로젝트의 `.env` 파일에 지정된
MiniMax 엔드포인트를 향하도록 구성된 `ChatOpenAI` 를 돌려주는 `llm()` 팩토리를 노출한다:

    OPENAI_API_KEY=sk-cp-...
    OPENAI_BASE_URL=https://api.minimax.io/v1
    OPENAI_MODEL=MiniMax-M2.7-highspeed

이렇게 한 곳으로 모아두면 테스트에서는 사소하게 교체할 수 있고, 각 노드는 백엔드
모델이 Anthropic 인지 OpenAI 인지 MiniMax 인지 알 필요가 없다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MEMORIES_DIR = DATA_DIR / "memories"
SKILLS_DIR = DATA_DIR / "skills"
CACHE_DIR = DATA_DIR / "cache"
TRACE_DIR = DATA_DIR / "traces"

# frozen memory 파일에 대한 Hermes 스타일 문자 수 상한.
# 파일 자체의 크기 제한이 아니라, 시스템 프롬프트에 주입되는 스냅샷 바이트의 상한이다.
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7


def load_settings(dotenv_path: Path | None = None) -> Settings:
    """`.env` 가 있으면 `os.environ` 에 반영한 뒤 Settings 를 돌려준다.

    `load_dotenv` 는 파일이 없으면 아무 일도 하지 않는다 — 테스트에서는 환경변수를
    직접 설정하고 디스크에 .env 없이 이 함수를 호출하면 되므로 편리하다.
    """
    env_file = dotenv_path or (PROJECT_ROOT / ".env")
    load_dotenv(env_file, override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return Settings(api_key=api_key, base_url=base_url, model=model)


def llm(settings: Settings | None = None, **overrides) -> ChatOpenAI:
    s = settings or load_settings()
    params: dict = {
        "model": s.model,
        "api_key": s.api_key,
        "base_url": s.base_url,
        "temperature": s.temperature,
    }
    params.update(overrides)
    return ChatOpenAI(**params)
