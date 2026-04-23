"""도구 스키마와 구현.

2주차에 추가되는 도구는 둘:

- load_skill(name)  — 센티넬 도구. LLM 이 카탈로그 항목의 본문 전체를 대화에
  끌어오고 싶을 때 호출한다. 실제 '실행' 은 이 함수가 아니라 skill_loader
  노드에서 일어나므로, @tool 본문은 NotImplementedError 를 던진다. LangChain
  입장에서는 bind_tools 스키마를 만들 때 docstring + signature 만 있으면 된다.

- view(path)        — 실제 도구. tool_dispatch 가 이전에 offload 한 콘텐츠(너무
  큰 결과를 data/cache/ 에 쓰고 messages 에는 포인터만 남긴 것 — 가이드 §4-2)
  를 LLM 이 다시 읽을 때 호출한다. offload 된 콘텐츠가 컨텍스트로 다시 들어올
  수 있는 유일한 경로다.

경로 안전: view() 는 DATA_DIR 바깥으로 해석되는 경로를 거부한다. 이게 없으면
프롬프트 인젝션으로 "view 도구를 /etc/passwd 에 써" 가 호스트 파일을 유출시킨다.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from harness.config import DATA_DIR, SKILLS_DIR


@tool
def load_skill(name: str) -> str:
    """카탈로그에 있는 스킬의 본문 전체를 현재 대화에 로드합니다. 해당 스킬의
    상세 절차가 처음 필요할 때 사용하세요; 한번 로드되면 본문이 컨텍스트에 남습니다.

    Args:
        name: 카탈로그에 있는 스킬 이름 (예: 'echo', 'notes', 'cli_help').

    Returns:
        SKILL.md 본문 전체 — 도구 결과 메시지로 표시됩니다.
    """
    # 센티넬: skill_loader 노드가 라우팅으로 가로챈다.
    raise NotImplementedError("load_skill 은 skill_loader 노드가 처리한다")


@tool
def view(path: str) -> str:
    """하니스가 이전에 디스크로 offload 한 파일 내용을 읽어 돌려준다.
    컨텍스트에서 도구 결과가 <result path='...'> 포인터로 대체된 경우 이 도구를 사용.

    Args:
        path: 프로젝트의 data/ 디렉터리 아래 절대/상대 경로
              (offload 된 결과는 data/cache/ 아래에 저장).

    Returns:
        파일 내용 텍스트, 또는 'error:' 로 시작하는 에러 메시지.
    """
    return _read_safe(path)


def _read_safe(raw: str) -> str:
    try:
        p = Path(raw).resolve()
    except (OSError, ValueError) as exc:
        return f"error: invalid path: {exc}"
    data_root = DATA_DIR.resolve()
    if data_root not in p.parents and p != data_root:
        return f"error: path outside data/: {p}"
    if not p.exists():
        return f"error: not found: {p}"
    if not p.is_file():
        return f"error: not a file: {p}"
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        return f"error: read failed: {exc}"


def skill_body_path(name: str) -> Path:
    return SKILLS_DIR / name / "SKILL.md"


ALL_TOOLS = [load_skill, view]
