from __future__ import annotations

from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from harness import config


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """가짜 .env 를 쓰고, 상속된 환경변수를 지워 load_settings 가 사용자의
    실제 설정이 아니라 바로 이 파일만 바라보도록 만드는 fixture."""
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=sk-test-123\n"
        "OPENAI_BASE_URL=https://example.test/v1\n"
        "OPENAI_MODEL=test-model-x\n"
    )
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return env


def test_load_settings_reads_dotenv(tmp_env: Path) -> None:
    s = config.load_settings(dotenv_path=tmp_env)
    assert s.api_key == "sk-test-123"
    assert s.base_url == "https://example.test/v1"
    assert s.model == "test-model-x"


def test_llm_factory_returns_chat_openai(tmp_env: Path) -> None:
    s = config.load_settings(dotenv_path=tmp_env)
    client = config.llm(s, temperature=0.0)
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "test-model-x"
    assert client.temperature == 0.0
