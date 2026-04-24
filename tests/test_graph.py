from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from harness.graph import build_graph
from harness.nodes.bootstrap import SYSTEM_MESSAGE_ID


class _ScriptedToolCallingLLM(BaseChatModel):
    """최소한의 tool-calling 페이크. 매 .invoke() 마다 스크립트에서 AIMessage 를
    하나씩 소비한다. bind_tools() 는 self 를 반환하는 no-op — 실제로 스키마를
    내보낼 필요가 없다. 응답은 테스트가 공급한다."""

    responses: list[AIMessage] = []
    call_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self.call_count
        object.__setattr__(self, "call_count", idx + 1)
        msg = self.responses[idx]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "_scripted_tool_calling_fake"


def _invoke(graph, text: str, thread_id: str = "t-test"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
    )


# ------ 1주차 회귀 테스트 (2주차 변경 후에도 계속 통과해야 함) -----

def test_graph_runs_end_to_end_with_fake_llm() -> None:
    llm = FakeListChatModel(responses=["hello-from-fake"])
    graph = build_graph(llm)
    result = _invoke(graph, "hi")

    msgs = result["messages"]
    assert any(isinstance(m, SystemMessage) for m in msgs)
    assert msgs[-1].content == "hello-from-fake"
    assert result["turn"] == 1


def test_system_prompt_is_byte_identical_across_turns() -> None:
    llm = FakeListChatModel(responses=["r1", "r2", "r3"])
    graph = build_graph(llm)
    snapshots: list[str] = []
    for text in ["t-1", "t-2", "t-3"]:
        result = _invoke(graph, text, thread_id="t-frozen")
        sys_msg = next(m for m in result["messages"] if m.id == SYSTEM_MESSAGE_ID)
        snapshots.append(sys_msg.content)
    assert len(set(snapshots)) == 1


def test_bootstrap_only_runs_once_per_thread() -> None:
    llm = FakeListChatModel(responses=["a", "b", "c"])
    graph = build_graph(llm)
    for text in ["t1", "t2", "t3"]:
        result = _invoke(graph, text, thread_id="t-single-sys")
    system_msgs = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1


# ------------------------- 2주차 라우팅 테스트 ----------------------------

def test_load_skill_tool_call_routes_through_skill_loader() -> None:
    """에이전트가 턴 1 에 load_skill(echo) 를 부르고, 턴 2 에 응답."""
    llm = _ScriptedToolCallingLLM(
        responses=[
            AIMessage(
                content="I need the echo skill.",
                tool_calls=[
                    {"name": "load_skill", "args": {"name": "echo"}, "id": "c1"}
                ],
            ),
            AIMessage(content="OK I have echo now."),
        ]
    )
    graph = build_graph(llm)
    result = _invoke(graph, "hi", thread_id="t-skill")

    msgs: list[BaseMessage] = result["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content.startswith("<skill:echo>")
    assert "echo" in result["loaded_skills"]
    assert result["skill_last_used"]["echo"] == result["turn"] - 1  # 두 번째 agent 증가 전에 기록


def test_read_tool_call_routes_through_tool_dispatch(tmp_path) -> None:
    """에이전트가 턴 1 에 read(path) 호출 → tool_dispatch 가 파일 읽기 → agent."""
    from harness.config import DATA_DIR
    target = DATA_DIR / "cache" / "test_read_graph.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("file-body", encoding="utf-8")

    llm = _ScriptedToolCallingLLM(
        responses=[
            AIMessage(
                content="Reading the file.",
                tool_calls=[
                    {"name": "read", "args": {"path": str(target)}, "id": "v1"}
                ],
            ),
            AIMessage(content="Saw it."),
        ]
    )
    graph = build_graph(llm)
    try:
        result = _invoke(graph, "go", thread_id="t-read")
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "file-body" in tool_msgs[0].content
        assert result["tool_call_count"] == 1
        assert result["task_trace"][0]["tool"] == "read"
    finally:
        target.unlink(missing_ok=True)


def test_no_tool_calls_routes_to_end() -> None:
    """tool_calls 가 없는 평범한 AIMessage → END. turn 카운터는 한 번만 증가."""
    llm = _ScriptedToolCallingLLM(responses=[AIMessage(content="done.")])
    graph = build_graph(llm)
    result = _invoke(graph, "hi", thread_id="t-end")
    assert result["turn"] == 1
    assert result["messages"][-1].content == "done."


# ---------------------- route_after_agent 단위 테스트 -------------------

def test_route_destructive_call_in_parallel_goes_to_human_gate() -> None:
    """[read, write] 같이 destructive 가 한 호출이라도 끼면 묶음 전체 human_gate.
    이를 빼먹으면 write 가 게이트를 우회해 실행된다."""
    from harness.graph import route_after_agent
    state = {
        "messages": [AIMessage(
            content="mixed",
            tool_calls=[
                {"name": "read", "args": {"path": "x"}, "id": "c1"},
                {"name": "write", "args": {"path": "y", "content": "z"}, "id": "c2"},
            ],
        )]
    }
    assert route_after_agent(state) == "human_gate"  # type: ignore[arg-type]


def test_route_sentinel_in_parallel_falls_back_to_dispatch() -> None:
    """sentinel(load_skill 등) 은 단독 호출만 전용 노드로. 묶음에 끼면
    tool_dispatch 로 보내 1:1 ToolMessage 가 매겨지게 한다 (응답 누락 방지)."""
    from harness.graph import route_after_agent
    state = {
        "messages": [AIMessage(
            content="mixed sentinels",
            tool_calls=[
                {"name": "load_skill", "args": {"name": "echo"}, "id": "c1"},
                {"name": "read", "args": {"path": "x"}, "id": "c2"},
            ],
        )]
    }
    assert route_after_agent(state) == "tool_dispatch"  # type: ignore[arg-type]


def test_route_solo_sentinel_still_goes_to_dedicated_node() -> None:
    from harness.graph import route_after_agent
    for name, expected in [
        ("finalize_task", "self_improve"),
        ("load_skill", "skill_loader"),
        ("spawn_subagent", "subagent"),
    ]:
        state = {
            "messages": [AIMessage(
                content="solo",
                tool_calls=[{"name": name, "args": {}, "id": "c1"}],
            )]
        }
        assert route_after_agent(state) == expected, name  # type: ignore[arg-type]


# --------------------------- 3주차 배선 테스트 --------------------------

def test_spawn_subagent_tool_routes_through_subagent_node() -> None:
    """부모가 spawn_subagent 호출 → subagent 노드 → ToolMessage 반환 →
    부모 agent 가 최종 응답을 낸다."""
    llm = _ScriptedToolCallingLLM(
        responses=[
            # 부모 턴 1: 자식 spawn.
            AIMessage(
                content="Delegating to child.",
                tool_calls=[{
                    "name": "spawn_subagent",
                    "args": {
                        "task": "count to 3",
                        "context": "ignore",
                        "constraints": "no tools",
                    },
                    "id": "s1",
                }],
            ),
            # 자식 그래프 첫 스텝: 자식 응답 (도구 없음).
            AIMessage(content="one-two-three"),
            # 부모 턴 2: 자식 요약을 보고 최종 응답.
            AIMessage(content="Child said: one-two-three"),
        ]
    )
    graph = build_graph(llm, use_llm_summarizer=False)
    result = _invoke(graph, "do it", thread_id="t-sub")
    # subagent 가 돌아온 뒤에 부모의 최종 응답이 나온다.
    assert result["messages"][-1].content.startswith("Child said")
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    # ToolMessage 하나 = subagent 의 요약이 부모로 돌아간 것.
    assert len(tool_msgs) == 1
    assert "one-two-three" in tool_msgs[0].content


def test_finalize_task_routes_to_self_improve_and_writes_skill(tmp_path) -> None:
    """도구 호출을 충분히 한 뒤 agent 가 finalize_task 를 호출하면 → self_improve 가
    돌아 data/skills/<slug>/SKILL.md 가 쓰인다."""
    # seed: 첫 턴이 도구 호출을 만드므로 tool_call_count >= 1.
    # 5 개의 더미 trace 로 self_improve 임계치를 넘긴다.
    llm = _ScriptedToolCallingLLM(
        responses=[
            AIMessage(
                content="done finalizing",
                tool_calls=[{
                    "name": "finalize_task",
                    "args": {"title": "week4-test-skill"},
                    "id": "f1",
                }],
            ),
        ]
    )
    # 실제 도구를 돌리지 않고 finalize 가 임계치를 넘도록 state 를 seed.
    graph = build_graph(llm, use_llm_summarizer=False, skills_dir=tmp_path)
    seeded_trace = [
        {"reasoning": f"r{i}", "tool": "read",
         "args": {"path": f"/x/{i}"}, "observation": f"o{i}"}
        for i in range(5)
    ]
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="finalize please")],
            "task_trace": seeded_trace,
            "tool_call_count": 5,
        },
        config={"configurable": {"thread_id": "t-fin"}},
    )
    skill_md = tmp_path / "week4-test-skill" / "SKILL.md"
    assert skill_md.exists(), "self_improve 가 SKILL.md 를 쓰지 않았다"
    body = skill_md.read_text(encoding="utf-8")
    assert "Procedure" in body


def test_sqlite_checkpointer_persists_across_builds(tmp_path) -> None:
    """같은 sqlite db 와 같은 thread_id 로 그래프를 두 번 만들면 — 두 번째 빌드는
    첫 실행의 메시지를 봐야 한다."""
    from harness.graph import make_sqlite_checkpointer

    db = tmp_path / "ckp.db"

    # 첫 세션.
    llm1 = _ScriptedToolCallingLLM(responses=[AIMessage(content="first")])
    graph1 = build_graph(llm1, checkpointer=make_sqlite_checkpointer(db))
    r1 = _invoke(graph1, "turn-one", thread_id="t-persist")
    assert r1["messages"][-1].content == "first"

    # 두 번째 세션, 새 그래프, 새 LLM — 하지만 같은 db + thread.
    llm2 = _ScriptedToolCallingLLM(responses=[AIMessage(content="second")])
    graph2 = build_graph(llm2, checkpointer=make_sqlite_checkpointer(db))
    r2 = _invoke(graph2, "turn-two", thread_id="t-persist")

    # state 에 첫 세션의 'first' + 새 'second' 가 누적된다.
    ai_contents = [
        m.content for m in r2["messages"]
        if isinstance(m, AIMessage) and isinstance(m.content, str)
    ]
    assert "first" in ai_contents and "second" in ai_contents


def test_compactor_triggers_above_threshold() -> None:
    """임계치를 아주 작게 강제해 도구 턴 한 번 뒤에 compactor 가 돌게 한다."""
    llm = _ScriptedToolCallingLLM(
        responses=[
            AIMessage(
                content="X" * 10_000,
                tool_calls=[{
                    "name": "load_skill",
                    "args": {"name": "echo"},
                    "id": "c1",
                }],
            ),
            AIMessage(content="done."),
        ]
    )
    graph = build_graph(
        llm,
        compact_threshold=100,        # 터무니없이 낮춰 compaction 강제
        use_llm_summarizer=False,     # 오프라인 fallback summarizer 사용
    )
    result = _invoke(graph, "go", thread_id="t-compact")
    # compaction 이후에도 messages 는 수렴하고 응답이 나와야 한다.
    assert result["messages"][-1].content == "done."
    # 스킬 메시지는 축출되면 안 된다 (pinning).
    remaining = [
        m for m in result["messages"]
        if isinstance(m, ToolMessage) and m.content.startswith("<skill:echo>")
    ]
    assert len(remaining) == 1, "compactor 가 skill ToolMessage 를 축출했다"
