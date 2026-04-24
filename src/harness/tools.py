r"""도구 스키마와 구현.

도구 목록:

- load_skill(name) — 센티넬. LLM 이 카탈로그 항목의 본문 전체를 대화에 끌어오고
  싶을 때 호출. 실제 '실행' 은 이 함수가 아니라 skill_loader 노드가 라우팅으로
  가로채 처리한다. @tool 본문은 NotImplementedError — LangChain 의 bind_tools
  는 docstring + signature 만 보면 된다.

- spawn_subagent(...) / finalize_task(...) — 마찬가지로 센티넬. 각각 subagent
  노드와 self_improve 노드에서 처리.

- read / write / edit — 실제 파일 도구. Anthropic str_replace_based_edit_tool
  규약을 미러링한다.
    - read  : data/ 아래 파일을 읽어 본문 반환. view_range / max_bytes 지원.
              tool_dispatch 가 이전에 offload 한 콘텐츠(가이드 §4-2 — 큰 결과를
              data/cache/ 에 쓰고 messages 에는 <result path='...'> 포인터만
              남긴 것) 를 LLM 이 다시 끌어오는 유일한 경로.
              read 는 _READ_CACHE 에 sha 를 적어, 후속 edit 가 stale 인지
              검증할 수 있게 한다.
    - write : create-only. 기존 파일이면 거부 — 덮어쓰려면 edit.
    - edit  : str_replace, old_string 정확히 1회 일치. 직전에 read 로 본 적이
              없거나 디스크가 바뀌었으면 거부 (stale 방지). 모두 atomic write.

경로 안전: 모든 파일 도구는 _resolve_safe 로 입력을 정규화하고 DATA_DIR 바깥은
거부한다. 이게 없으면 프롬프트 인젝션으로 "/etc/passwd 를 read 해" 가 호스트
파일을 유출시킨다. write/edit 는 추가로 Windows 예약 이름(CON/PRN/AUX/NUL/
COM1-9/LPT1-9 — 확장자 무관) 과 \\?\ 롱패스 prefix 를 거부한다.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
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


MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000

_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

# 단일 프로세스 하니스 가정. read 가 본 파일의 sha256 을 기억해 두고, edit 가
# 호출되기 전에 디스크 상태가 그대로인지 확인한다. 멀티 프로세스/스레드에서는
# 깨지므로 해당 환경으로 이전할 때 InjectedState 로 옮겨야 한다.
_READ_CACHE: dict[str, str] = {}


def _clear_read_cache() -> None:
    """테스트용 — 모듈 레벨 캐시를 비운다."""
    _READ_CACHE.clear()


def _resolve_safe(raw: str):
    """입력 경로를 DATA_DIR 안의 절대 Path 로 해석. 실패 시 'error: ...' 문자열.

    write/edit 가 새 파일을 만들 수 있어야 하므로 strict=False (존재 확인은 호출자가).
    """
    if not raw:
        return "error: empty path"
    if "\x00" in raw:
        return "error: NUL byte in path"
    if raw.startswith("\\\\?\\") or raw.startswith("//?/"):
        return "error: long-path prefix not allowed"
    try:
        p = Path(raw).resolve()
    except (OSError, ValueError) as exc:
        return f"error: invalid path: {exc}"
    data_root = DATA_DIR.resolve()
    if p != data_root and data_root not in p.parents:
        return "error: path outside data/"
    base = p.name.split(".")[0].lower()
    if base in _WINDOWS_RESERVED:
        return f"error: reserved name: {p.name}"
    return p


def _atomic_write(target: Path, data: bytes) -> None:
    """temp 파일을 같은 디렉터리에 만들어 fsync 후 os.replace — 부분 쓰기 방지."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=".tmp.", suffix=".swp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(target))
    except Exception:
        try:
            Path(tmp_name).unlink()
        except OSError:
            pass
        raise


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def skill_body_path(name: str) -> Path:
    return SKILLS_DIR / name / "SKILL.md"


@tool
def spawn_subagent(task: str, context: str = "", constraints: str = "") -> str:
    """탐색적이거나 범위가 제한된 작업을 자식 에이전트에게 위임한다. 자식은 자신의
    컨텍스트에서 돌며, 당신의 메시지는 전달되지 않는다. `context` 와 `constraints` 에
    담은 내용만이 자식에게 전해진다. 중간 턴이 많이 발생하는 작업(파일 검색, 로그 grep,
    읽기 중심의 긴 탐색) 에서 그 부산물이 당신의 컨텍스트에 남는 것을 피하고 싶을 때 사용한다.

    Args:
        task:        자식이 해결할 질문 또는 작업. 보통 한 문장. context 가 없으면
                     자식은 그 외 아무 정보도 보지 못한다.
        context:     선택 사항 — 당신의 컨텍스트에서 자식에게 필요한 정보만 추려서
                     (파일 경로, 최근 발견, 사용자로부터의 제약). 짧게 유지하라;
                     길어지면 isolation 의 목적이 사라진다.
        constraints: 선택 사항 — 자식이 따라야 할 규칙 (도구 화이트리스트, 출력 포맷, 예산 등).

    Returns:
        자식이 찾거나 수행한 내용의 짧은 요약.
    """
    # 센티넬 — 라우팅을 통해 subagent 노드가 처리한다.
    raise NotImplementedError("spawn_subagent 는 subagent 노드가 처리한다")


@tool
def finalize_task(title: str) -> str:
    """현재 작업이 완료됐음을 선언해, 하니스가 성공한 트레이스를 재사용 가능한
    SKILL.md 로 증류하도록 만든다. 다음 조건일 때만 호출:

      - 실제로 작업을 해결했을 때 (포기했으면 부르지 말 것).
      - 최소 ~5 회의 의미있는 도구 호출이 해결로 이어졌을 때 — 더 적으면
        LLM 요약 왕복에 값하는 증류물이 나오지 않는다.

    Args:
        title: 스킬의 짧은 이름 (kebab-case 권장). data/skills/ 아래 디렉터리
               이름으로 그대로 사용된다.

    Returns:
        저장된 것(또는 저장하지 않은 이유) 에 대한 한 줄 확인 메시지.
    """
    raise NotImplementedError("finalize_task 는 self_improve 노드가 처리한다")


@tool
def read(
    path: str,
    view_range: list[int] | None = None,
    max_bytes: int = MAX_READ_BYTES,
) -> str:
    """파일을 읽어 돌려준다 (data/ 아래 한정). edit 도구를 쓰려면 먼저 read 로
    파일을 본 적이 있어야 하며, read 이후 디스크가 바뀌면 edit 가 거부한다.

    Args:
        path: data/ 아래 절대/상대 경로.
        view_range: 선택. [start, end] 1-indexed 라인 범위 (end=-1 이면 EOF).
        max_bytes: 한 번에 읽을 최대 바이트. 초과 시 view_range 사용 권장.

    Returns:
        파일 내용 텍스트, 또는 'error:' 로 시작하는 에러 메시지.
    """
    resolved = _resolve_safe(path)
    if isinstance(resolved, str):
        return resolved
    if not resolved.exists():
        return f"error: not found: {resolved}"
    if not resolved.is_file():
        return f"error: not a file: {resolved}"
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        return f"error: read failed: {exc}"
    if len(data) > max_bytes and view_range is None:
        return f"error: file exceeds max_bytes={max_bytes}; use view_range"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return f"error: not utf-8: {exc}"
    if view_range is not None:
        if len(view_range) != 2:
            return "error: view_range must be [start, end]"
        start, end = view_range
        lines = text.splitlines(keepends=True)
        n = len(lines)
        if start < 1 or start > n:
            return f"error: start out of range (file has {n} lines)"
        end = n if end == -1 else end
        if end < start:
            return "error: end < start"
        text = "".join(lines[start - 1 : end])
    _READ_CACHE[str(resolved)] = _sha(data)
    return text


@tool
def write(path: str, content: str) -> str:
    """새 파일을 생성한다 (data/ 아래). 기존 파일이면 거부 — 덮어쓰려면 edit.

    Args:
        path: data/ 아래 절대/상대 경로.
        content: UTF-8 텍스트.

    Returns:
        'created: ...' 한 줄 확인, 또는 'error:' 메시지.
    """
    resolved = _resolve_safe(path)
    if isinstance(resolved, str):
        return resolved
    if resolved.exists():
        return f"error: already exists: {resolved} (use edit to modify)"
    if "\x00" in content:
        return "error: NUL byte in content"
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return f"error: content exceeds max_bytes={MAX_WRITE_BYTES}"
    try:
        _atomic_write(resolved, encoded)
    except OSError as exc:
        return f"error: write failed: {exc}"
    _READ_CACHE[str(resolved)] = _sha(encoded)
    return f"created: {resolved} ({len(encoded)} bytes)"


@tool
def edit(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """파일에서 old_string 을 new_string 으로 치환한다. old_string 은 파일 내에
    정확히 1회 일치해야 한다 (verbatim, 공백/줄바꿈 포함). 0 회/2 회 이상이면
    실패 — 더 많은 주변 컨텍스트로 unique 하게 만들거나 replace_all=True 사용.

    edit 호출 전에 같은 파일을 read 로 본 적이 있어야 한다 (마지막 read 이후
    디스크가 바뀌었으면 거부; stale 편집 방지). 모두 atomic write 로 처리.

    Args:
        path: data/ 아래 경로.
        old_string: 파일 내에 정확히 일치해야 하는 원본 부분 문자열.
        new_string: 치환 후 문자열.
        replace_all: True 면 모든 일치를 치환.

    Returns:
        'edited: ...' 한 줄 확인, 또는 'error:' 메시지.
    """
    resolved = _resolve_safe(path)
    if isinstance(resolved, str):
        return resolved
    if not resolved.exists():
        return f"error: not found: {resolved}"
    if not resolved.is_file():
        return f"error: not a file: {resolved}"
    if old_string == new_string:
        return "error: old_string equals new_string (no-op)"
    if "\x00" in new_string:
        return "error: NUL byte in new_string"
    key = str(resolved)
    cached = _READ_CACHE.get(key)
    if cached is None:
        return (
            f"error: must read {resolved} before editing "
            "(call read first to register the current contents)"
        )
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        return f"error: read failed: {exc}"
    if _sha(data) != cached:
        return (
            f"error: {resolved} changed since last read; "
            "re-run read before editing"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return f"error: not utf-8: {exc}"
    count = text.count(old_string)
    if count == 0:
        return (
            f"error: old_string did not appear verbatim in {resolved} "
            "— re-read the file and copy the exact bytes"
        )
    if count > 1 and not replace_all:
        line_hits = [
            ln for ln, line in enumerate(text.splitlines(keepends=True), start=1)
            if old_string in line
        ]
        hint = f" at lines {line_hits}" if line_hits else ""
        return (
            f"error: {count} occurrences of old_string found{hint}. "
            "Provide more surrounding context to make it unique, or set replace_all=true."
        )
    new_text = (
        text.replace(old_string, new_string)
        if replace_all
        else text.replace(old_string, new_string, 1)
    )
    encoded = new_text.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return f"error: result exceeds max_bytes={MAX_WRITE_BYTES}"
    try:
        _atomic_write(resolved, encoded)
    except OSError as exc:
        return f"error: write failed: {exc}"
    _READ_CACHE[key] = _sha(encoded)
    n_replaced = count if replace_all else 1
    return f"edited: {resolved} ({n_replaced} replacement{'s' if n_replaced != 1 else ''})"


ALL_TOOLS = [load_skill, read, write, edit, spawn_subagent, finalize_task]

# 파괴적인 도구 이름 — 라우터가 이 도구들을 human_gate 의 interrupt_before 승인으로 보낸다.
# 추가/제거가 쉽도록 set 으로 유지.
DESTRUCTIVE_TOOLS: set[str] = {
    "write",
    "edit",
}
