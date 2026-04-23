"""session_bootstrap — Frozen Snapshot 주입 노드.

가이드 §5 핵심 규칙: *시스템 프롬프트는 턴을 넘나들며 바이트 단위로 동일해야 한다.*
이 노드는 그래프에서 SystemMessage 를 쓰는 유일한 지점이며, 2턴째부터는
`memory_snapshot` 센티넬을 보고 short-circuit 한다. 덕분에 동일한 프롬프트 프리픽스가
불필요하게 다시 방출되지 않는다 (add_messages 의 ID 기반 dedupe 가 어차피 같은 복사본으로
교체는 해 주겠지만, 그 자체가 낭비이므로).

콘텐츠 레이아웃 (가이드 §5 그림 참고):
    <memory>   ← MEMORY.md 를 MEMORY_CHAR_LIMIT 로 잘라서
    <user>     ← USER.md   를 USER_CHAR_LIMIT  로 잘라서
    <skills_catalog>   ← {name: 한 줄 설명}  (Level 0)

바이트 동일성을 지키기 위한 결정론 요구사항:
- sorted() 로 스킬을 순회한다. glob() 만으로는 순서가 결정적이지 않다.
- json.dumps(..., sort_keys=True, ensure_ascii=False) 로 카탈로그 직렬화 순서를 고정.
- 파일 읽기 시점에 UTF-8 줄바꿈을 "\n" 으로 정규화 (Windows 체크아웃의 CRLF 혼입 방지).
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import SystemMessage

from harness.config import (
    MEMORIES_DIR,
    MEMORY_CHAR_LIMIT,
    SKILLS_DIR,
    USER_CHAR_LIMIT,
)
from harness.state import HarnessState

SYSTEM_MESSAGE_ID = "system-bootstrap"


def _read_capped(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return text[:limit]


def _extract_description(skill_md: Path) -> str:
    """첫 번째 `# heading` 라인이 카탈로그 설명이 된다. 헤딩이 없으면 디렉터리 이름으로 폴백."""
    try:
        for line in skill_md.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
    except OSError:
        pass
    return skill_md.parent.name


def build_catalog(skills_dir: Path | None = None) -> dict[str, str]:
    # 호출 시점에 해석해야 테스트가 모듈 레벨 SKILLS_DIR 를 monkeypatch 할 때
    # 모든 호출 지점에 fixture 를 넘겨주지 않아도 된다.
    skills_dir = skills_dir or SKILLS_DIR
    return {
        p.parent.name: _extract_description(p)
        for p in sorted(skills_dir.glob("*/SKILL.md"))
    }


def build_system_content(
    memories_dir: Path | None = None,
    skills_dir: Path | None = None,
) -> tuple[str, str, dict[str, str]]:
    """(system_content, memory_snapshot, catalog) 튜플을 돌려준다. 순수 함수라
    그래프 전체를 띄우지 않고도 바이트 동일성 속성을 테스트할 수 있다."""
    memories_dir = memories_dir or MEMORIES_DIR
    skills_dir = skills_dir or SKILLS_DIR
    memory = _read_capped(memories_dir / "MEMORY.md", MEMORY_CHAR_LIMIT)
    user = _read_capped(memories_dir / "USER.md", USER_CHAR_LIMIT)
    catalog = build_catalog(skills_dir)
    content = (
        f"<memory>\n{memory}\n</memory>\n"
        f"<user>\n{user}\n</user>\n"
        f"<skills_catalog>\n"
        f"{json.dumps(catalog, sort_keys=True, ensure_ascii=False, indent=2)}\n"
        f"</skills_catalog>"
    )
    return content, memory + "\n\n" + user, catalog


def session_bootstrap(state: HarnessState) -> dict:
    if state.get("memory_snapshot"):
        return {}
    content, snapshot, catalog = build_system_content()
    system = SystemMessage(content=content, id=SYSTEM_MESSAGE_ID)
    return {
        "messages": [system],
        "memory_snapshot": snapshot,
        "skills_catalog": catalog,
        "turn": 0,
    }
