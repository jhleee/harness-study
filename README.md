# Agent Harness Study — LangGraph로 구현하는 하네스 패턴

OpenClaw / Hermes Agent 계열 하네스의 작동 원리를 **LangGraph**로 재구성하며 학습하는 4주 집중 스터디 프로젝트.

> "루프 모양이 아니라 **system prompt에 무엇을 언제 넣느냐**가 본질이다."

---

## 프로젝트 목적

Hermes Agent / OpenClaw가 가진 **로컬-퍼스트 메신저 게이트웨이 + Skills 루프** 구조를 LangGraph의 노드·엣지 모델로 옮기면서, 실제로 하네스가 무너지는 지점(오구현)을 직접 밟고 교정하는 것을 목표로 한다.

ReAct / Plan-Execute와 구분되는 이 계열의 핵심은 **런타임 학습**:
- Skills progressive disclosure (카탈로그만 상시 로드, 본문은 on-demand)
- Self-improvement loop (성공 궤적을 `SKILL.md`로 증류)

## 다룰 핵심 패턴

| # | 패턴 | 역할 |
|---|---|---|
| 1 | Gateway 멀티플렉싱 | 여러 채널 → 단일 agent loop |
| 2 | Frozen snapshot memory | `MEMORY.md` / `USER.md`를 system에 주입 후 고정 |
| 3 | Skills progressive disclosure | 카탈로그(이름+설명)만 상시, 본문은 on-demand |
| 4 | Self-improvement loop | 복잡 태스크 종료 후 성공 경로를 `SKILL.md`로 증류 |
| 5 | Subagent 위임 | 병렬 워크스트림을 별도 컨텍스트로 격리 (RPC 모델) |
| 6 | Cron/스케줄러 | 사용자 입력 없이 에이전트 loop 트리거 |

## 4주 커리큘럼 요약

빌드업 축: **맥락을 어디에 쌓느냐**의 4단계.

### Week 1 — Frozen Snapshot: System을 얼린다
- **명제**: *System prompt는 세션 내내 바이트 단위로 동일해야 한다.*
- 최소 그래프: `gateway` → `session_bootstrap` → `agent` → END
- `MEMORY.md`/`USER.md`/`skills_catalog`를 system에 **한 번만** 주입
- cache 측정(`cache_creation_input_tokens`, `cache_read_input_tokens`) 로깅
- 함정 재현: skill 본문을 system에 끼워넣어 cache miss 관찰

### Week 2 — Messages Side: 본문은 messages로 끌어온다
- **명제**: *동적 로딩의 "동적" 부분은 system이 아니라 messages에서 일어난다.*
- `skill_loader` + `tool_dispatch` 추가
- `load_skill(name)` 도구 — 본문을 **ToolMessage로** 반환
- Tool result offloading (2000자 초과 시 `~/.hermes/cache/`로 내보내고 포인터만 남김)
- 함정 재현: `SystemMessage` 삽입 오류, `/tmp` 휘발성 문제

### Week 3 — 맥락 외재화: Compaction · Subagent
- **명제**: *Subagent는 예방책, compaction은 치료제. 서로 다른 타이밍의 도구.*
- `compactor` — skill ToolMessage **pinning** + `HumanMessage` 래핑
- `subagent` — RPC 모델, briefing 3-필드 스키마 (`task`, `context`, `constraints`)
- 함정 재현: `deepcopy(parent_state)` fork / `messages[:-8]` 슬라이싱 / `SystemMessage` 요약 삽입

### Week 4 — Self-Improvement Loop + 통합
- **명제**: *하네스는 자기가 해낸 일을 파일로 되돌려 다음 세션의 자산으로 만든다.*
- `finalize_task` **명시 호출 시에만** `self_improve` 진입
- trace 4-튜플(`reasoning`, `tool`, `args`, `observation`) 기록
- `human_gate` + `interrupt_before` + SQLite checkpointer
- 공통 평가 시나리오 3종(탐색+수정 / 장문 처리 / 학습 루프)으로 상호 리뷰

## 목표 아키텍처 (v1.0)

```
gateway → session_bootstrap → agent
                               ├─ load_skill       → skill_loader
                               ├─ normal tool      → tool_dispatch (+ offloading + trace)
                               ├─ destructive tool → human_gate (interrupt)
                               ├─ spawn_subagent   → subagent (RPC briefing)
                               ├─ finalize_task    → self_improve (SKILL.md 증류)
                               └─ no tool_calls    → END (다음 유저 턴 대기)

모든 tool 출구 → compactor 체크 (토큰 예산 초과 시) → agent
```

## 스택

- **Language**: Python
- **Agent runtime**: LangGraph
- **Model**: Claude Sonnet 4.x (Anthropic SDK)
- **Persistence**: SQLite checkpointer, `~/.hermes/{skills,memories,cache}/`

## 문서

- [`docs/agent-harness-langgraph-guide.md`](docs/agent-harness-langgraph-guide.md) — LangGraph 구현 가이드 (State 정의 / 노드 / 라우팅 / 오구현 교정 / prompt caching)
- [`docs/harness_study_curriculum_v2.md`](docs/harness_study_curriculum_v2.md) — 4주 커리큘럼 (주차별 과제 / 함정 / 평가 시나리오)

## 학습 원칙

매 주 **"오구현 → 교정"** 사이클을 포함한다. 가이드 §3의 흔한 오구현 섹션이 이 스터디 가치의 절반이다.

- messages 배열은 "다음 턴에 LLM이 실제로 봐야 할 것"만 담는다
- system prompt는 세션 동안 얼어 있다
- subagent는 fork가 아니라 RPC다
- progressive disclosure의 기본값은 "로드는 적극적으로, 언로드는 보수적으로" (cache prefix 보호)

## 공통 트랙

- **Trace JSONL 스키마**는 Week 1에서 합의해 전 주차 누적 (turn / node / cache 지표 / tool call / 4-튜플)
- **의사결정 로그**: 각자 매주 "이 결정을 왜 했는가"를 README에 추가
- **자기검증**: system 재조립 코드 경로 존재 여부 / subagent의 부모 messages 참조 여부 / `tool_calls is None`일 때 self_improve 라우팅 여부
