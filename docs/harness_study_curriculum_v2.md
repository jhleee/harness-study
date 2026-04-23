# 에이전트 하네스 스터디 커리큘럼 (4주 집중)

**레퍼런스**: `agent-harness-langgraph-guide-1.md` (OpenClaw / Hermes Agent 계열을 LangGraph로 구현)

---

## 설계 축

가이드의 핵심 주장 — *"루프 모양이 아니라 system prompt에 무엇을 언제 넣느냐가 본질"* — 을 스터디 축으로 삼는다.

빌드업은 **맥락을 어디에 쌓느냐**의 4단계로 간다:

1. **Week 1** — system에 뭘 얼려둘지 (Frozen snapshot, skills catalog)
2. **Week 2** — messages로 뭘 끌어올지 (Progressive disclosure, tool offloading)
3. **Week 3** — messages에서 뭘 빼낼지 (Compaction, subagent, structured state)
4. **Week 4** — 성공 궤적을 어떻게 파일로 되돌릴지 (Self-improvement loop) + 통합

매 주 **"오구현 → 교정"** 사이클을 넣는다. 가이드 섹션 3이 이 스터디의 절반 가치다.

**공통 스택**: Python + LangGraph + Anthropic SDK. 모델은 Claude Sonnet 4.x.

---

## Week 1 — Frozen Snapshot: System을 얼린다

**핵심 명제**: *System prompt는 세션 내내 바이트 단위로 동일해야 한다.*

### 다룰 개념 (가이드 매핑)
- §5 Prompt Caching과 System Prompt 불변성 (**전부**)
- §1의 Hermes `MEMORY.md`(2200자) / `USER.md`(1375자) — 캐릭터 리밋을 둔 이유
- §4-3 Structured state — 왜 TODO를 messages에 안 쌓는가
- Progressive disclosure의 **Level 0**만 구현 (skills_catalog, 이름+설명)

### 사전 구현 과제
LangGraph로 **`gateway` → `session_bootstrap` → `agent` → END**만 있는 최소 그래프.

필수:
1. `HarnessState` TypedDict 정의 (가이드 State 정의 섹션 그대로 출발해도 됨)
2. `session_bootstrap`: `MEMORY.md` + `USER.md` + `skills_catalog`를 **system에 한 번만** 주입
3. `skills/` 디렉터리 하나 만들고 dummy skill 3개 (`SKILL.md`의 첫 줄 description만 catalog에 실림)
4. **cache 측정**: 매 턴 API 응답의 `cache_creation_input_tokens` / `cache_read_input_tokens`를 로그로 출력
5. 같은 세션에서 5턴 이상 돌려서 2턴부터 cache hit가 뜨는지 확인

### 일부러 밟을 함정
세션 중간에 "관련 있어 보이는 skill 본문을 system 상단에 끼워 넣는" 코드를 한 번 작성해본다 (가이드 §5 체크리스트의 ❌ 항목). → cache miss가 연속으로 뜨는 걸 눈으로 본다.

### 세션 리뷰 체크리스트
- system prompt를 바이트 비교했을 때 1턴↔5턴이 정말 동일한가 (공백/시간 포맷 문제 자주 발생)
- `cache_read_input_tokens`가 실제로 증가하는 그래프를 보였는가
- MEMORY.md의 2200자 제한을 두는 이유를 본인 언어로 설명 가능한가 (힌트: 토큰 비용을 **상수**로 만드는 것)
- TODO/scratchpad를 messages에 쌓은 사람 vs state에 둔 사람 — 토큰 차이

---

## Week 2 — Messages Side: 본문은 messages로 끌어온다

**핵심 명제**: *동적 로딩의 "동적" 부분은 system이 아니라 messages에서 일어난다.*

### 다룰 개념
- §1 Hermes의 Progressive disclosure **Level 1** (skill 본문 on-demand 로드)
- §2의 `skill_loader` 노드 — **왜 ToolMessage로 반환하는가**
- §4-2 Tool result offloading — 큰 결과는 파일 포인터로
- §4-4 Skill body eviction과 **"기본값은 그냥 두기"**의 이유 (cache prefix 보호)
- §3-2의 증상↔기법 매트릭스 일부 (offloading 타이밍)

### 사전 구현 과제
Week 1 그래프에 `skill_loader` + `tool_dispatch` 추가.

필수:
1. `load_skill(name)` 도구 구현: 본문을 **ToolMessage로** 반환 (system 건드리기 금지)
2. `tool_dispatch`에 **offloading 로직**: 결과가 2000자 초과면 `~/.hermes/cache/`에 쓰고 messages에는 포인터만 남김
3. `view(path)` 도구 추가 — 오프로드된 결과 재소환용
4. skill을 1회 로드한 뒤 10턴 더 돌아갈 때 cache hit가 유지되는지 확인 (ToolMessage도 prefix에 포함됨)
5. 일부러 50KB짜리 파일을 읽는 태스크 돌려서 offloading이 트리거되는지 검증

### 일부러 밟을 함정
처음엔 skill 본문을 `SystemMessage`로 삽입하도록 구현 → 어댑터가 받아주는지/거부하는지 관찰 → 가이드 §3-2의 교정으로 전환.

동시에 offloading 경로를 `/tmp`로 짜본다. 컨테이너 재시작 시뮬레이션(프로세스 kill 후 재실행)으로 파일 증발 재현 → `~/.hermes/cache/`로 교정 (가이드 §3-3 (e)).

### 세션 리뷰 체크리스트
- skill 로드 전후 `cache_read_input_tokens` 변화 — 본문이 messages에 들어간 뒤에도 캐시가 깨지지 않는가
- offloading 포인터 포맷이 서로 어떻게 다른가 (`<result path=... bytes=...>` 이런 태그 스키마 합의해두면 W3에서 편함)
- eviction을 구현한 사람이 있다면 왜 했는가 — 가이드 §4-4의 "기본은 유지"를 납득시킬 수 있는 시나리오인가

---

## Week 3 — 맥락 외재화: Compaction, Subagent, 그리고 사전/사후 구분

**핵심 명제**: *Subagent는 예방책, compaction은 치료제다. 둘은 다른 타이밍의 도구다.*

이번 주가 4주 중 가장 무겁다. 두 노드를 동시에 얹는다.

### 다룰 개념
- §2의 `compactor` 노드 — **skill pinning**과 **HumanMessage 래핑**
- §2의 `subagent` 노드 — **RPC 모델** (fork 아님)
- §3-1 Subagent는 fork가 아니다 (교정 전체)
- §3-2 증상↔기법 매트릭스 (이번 주 핵심 표)
- §3-3 (a) compactor ↔ skill 증발 충돌
- §3-3 (b) Anthropic API의 중간 SystemMessage 거부
- §3-3 (f) subagent briefing 스키마

### 사전 구현 과제
Week 2 그래프에 `compactor`와 `subagent` 추가. 가이드의 `route_after_tool`도 붙여서 **모든 tool 출구가 compactor 체크를 거치게**.

필수:
1. `compactor`: 토큰 임계치 초과 시 `messages[:-8]`을 요약해서 `HumanMessage(<prior_conversation_summary>)`로 치환. **skill ToolMessage는 pinning**
2. `subagent` 도구 스키마를 **3필드로 명시**: `task`, `context`, `constraints`. 부모 messages 이력 승계 금지
3. 자식은 `compiled_graph.invoke(sub_state, config={"recursion_limit": 30})`로 **같은 그래프를 재귀 호출**
4. 자식의 수만 토큰 메시지 → `summarize()` → 부모에는 ToolMessage 하나만 반환
5. 탐색형 시나리오로 검증: "이 대규모 레포에서 X 기능이 어디에 구현돼 있는지 찾아라" 같은 grep-heavy 작업을 subagent에 위임

### 일부러 밟을 함정 (가이드 §3-1, §3-3 (a)(b) 재현)
- 먼저 `deepcopy(parent_state)`로 subagent 구현 → 자식 컨텍스트가 부모보다 긴 기현상 측정
- 먼저 compactor에서 `messages[:-8]` 단순 슬라이싱 → skill 본문이 증발한 뒤 LLM이 "git 명령어를 모르겠다"고 하는 할루시네이션 관찰
- 먼저 요약본을 `SystemMessage`로 삽입 → Anthropic API 에러 메시지를 직접 받아본다
- 위 3개를 모두 교정한 뒤 재측정

### 세션 리뷰 체크리스트
- 가이드 §3-2의 증상↔기법 매트릭스를 **본인 코드의 어느 라인이 담당하는지** 짚을 수 있는가
- subagent briefing의 `context` 필드에 무엇을 넣었는가 — 너무 많이 넣지는 않았는가 (자식이 비대해짐)
- compactor 발동 시점에 `cache_creation_input_tokens`가 튀는 걸 확인했는가 (한 번의 재캐싱 비용, 가이드 §4-1)
- 10턴 넘는 긴 세션에서 토큰 사용량 그래프 비교 — compaction 있는 버전 vs 없는 버전

---

## Week 4 — Self-Improvement Loop + 통합 + 상호 평가

**핵심 명제**: *하네스는 자기가 해낸 일을 파일로 되돌려 다음 세션의 자산으로 만든다.*

### 다룰 개념
- §1 Hermes의 5+ tool calls 후 SKILL.md 자동 증류
- §2의 `self_improve` 노드 — **`finalize_task` 명시 호출 시에만** 진입
- §3-3 (c) 일반 대화가 매 턴 self_improve로 가지 않게 하는 라우팅
- §3-3 (d) trace 4-튜플 `{reasoning, tool, args, observation}`의 이유
- §6 본질적 설계 포인트 4가지 (통합 회고 소재)
- Human gate (§2 `human_gate`, interrupt_before) — 안전 장치

### 사전 구현 과제 (v1.0 프로토타입)
Week 1~3을 통합하고 `self_improve` + `human_gate` 추가.

필수:
1. `finalize_task` 도구 추가 — 에이전트가 **명시 호출할 때만** `self_improve`로 라우팅
2. `tool_dispatch`의 trace 기록을 **4-튜플**로 수정 (가이드 §3-3 (d))
3. `self_improve`에서 `task_trace`가 5개 이상일 때만 SKILL.md 생성 → `~/.hermes/skills/<slug>/SKILL.md`에 저장
4. `DESTRUCTIVE` 세트 정의 + `human_gate` 노드 + `interrupt_before=["human_gate"]`
5. **SQLite checkpointer** 연결 — 세션 재개 가능하게
6. README에 **설계 결정 기록**: Week 1~3에서 밟은 함정과 교정 과정 요약

### 공통 평가 시나리오 (Week 2 끝에 미리 정해두기)
모든 참가자의 하네스로 동일 태스크 실행 → trace 비교.

제안:
1. **탐색+수정**: "이 레포의 failing test 1개를 찾아 수정하라" — subagent 격리가 작동하는지
2. **장문 처리**: "이 50페이지 PDF에서 X 조항만 추출해 요약본 작성" — offloading과 compaction이 작동하는지
3. **학습 루프**: 같은 유형의 태스크를 **2번 연속** 실행. 1번째에서 `finalize_task` → SKILL.md 생성 → 2번째에서 그 skill을 `load_skill`로 사용하는지

### 마지막 세션 진행
1. 각자 하네스로 공통 태스크 3개 실행 → trace JSONL 제출 (스키마: turn / messages 스냅샷 / cache 지표 / tool call / 4-튜플)
2. 상호 리뷰:
   - 2번째 실행에서 SKILL.md 재사용이 실제로 일어났는가 (가이드 §1 self-improvement의 검증)
   - cache hit 비율 비교
   - subagent 토큰 사용량과 부모 토큰 사용량의 비율
3. 회고:
   - 가이드 §6의 4가지 설계 포인트 각각에 대해, 내 구현이 지켜낸 것 / 무너진 것
   - "system prompt를 얼리는 것"이 왜 어려웠는지 (실무 감각)

---

## 공통 트랙 (매주 누적)

### Trace JSONL 스키마 (Week 1에서 합의)
```json
{
  "turn": 3,
  "node": "tool_dispatch",
  "cache_creation": 0,
  "cache_read": 8421,
  "input_tokens": 124,
  "output_tokens": 87,
  "tool": "load_skill",
  "args": {"name": "git"},
  "reasoning": "...",
  "observation_preview": "..."
}
```
비교 리뷰의 언어를 통일하기 위해 첫 주에 고정한다.

### 의사결정 로그
각자 README에 **"이 결정을 왜 했는가"**를 매주 추가. 가이드 §6 설계 포인트 4가지와 매핑되게.

### 반드시 피할 것 (매주 자기검증용)
- system prompt를 턴마다 재조립하는 코드 경로가 **한 곳이라도** 있는가
- subagent가 부모 state를 참조만 해도 안 되는가? (읽기 전용 `memory_snapshot`/`skills_catalog`는 OK, messages는 NO)
- `tool_calls is None`일 때 self_improve로 가는 라우팅이 있는가 (§3-3 (c))

---

## 이전 커리큘럼과의 차이 (요약)

| 축 | 이전 (일반론) | 지금 (가이드 기반) |
|---|---|---|
| 축 | 맥락의 수명주기 (생성/흐름/오염/압축) | **맥락을 어디에 두느냐** (system / messages / state / 파일) |
| 스택 | 스택 자유 | LangGraph 고정 |
| 레퍼런스 | Claude Code, ReAct, smolagents | **OpenClaw, Hermes** |
| W1 중심 | 단일 턴 payload 조립 | Frozen snapshot + cache 측정 |
| W2 중심 | 멀티턴 루프 + 토큰 추적 | Progressive disclosure + offloading |
| W3 중심 | Compaction 전략 선택 | **Subagent(사전) vs Compactor(사후) 구분** |
| W4 중심 | 평가 태스크 통합 | **Self-improvement loop** + 통합 |
| 오구현 섹션 | 없음 | **매주 1개 이상 의도적 재현** |

가이드의 §3(흔한 오구현)과 §5(Prompt Caching)가 커리큘럼의 절반 무게를 차지하도록 조정했다 — 이 두 섹션이 가이드의 실전 가치 대부분이기 때문.
