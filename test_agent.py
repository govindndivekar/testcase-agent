"""Deterministic tests: fake LLM, real graph, real javac, in-memory checkpointer."""
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent

PLAN = agent.TestPlan(
    class_name="LoginTest", base_url="https://example.com/login",
    steps=[agent.PlanStep(action="Click Sign in", locator='By.id("signin")',
                          expected="Dashboard shown", assertion="assertTrue(banner.isDisplayed())")],
)
BROKEN = "public class LoginTest { void t() { int x = } }"
FIXED = "public class LoginTest { void t() { int x = 1; } }"


class FakeLLM:
    """Stands in for ChatAnthropic: returns scripted plans and code."""
    def __init__(self, code_replies):
        self.code_replies = list(code_replies)

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if "test plan" in messages[0].content:        # the planner prompt
            return PLAN
        return AIMessage(content=self.code_replies.pop(0))


@pytest.fixture
def graph(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "OUTPUT_DIR", tmp_path)
    return agent.build_graph(InMemorySaver())


CFG = {"configurable": {"thread_id": "t1"}}
START = {"test_case": {"key": "IR-T101", "steps": []}, "attempts": 0}


def test_pauses_for_review_before_any_code(graph, monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM([FIXED]))
    result = graph.invoke(START, CFG)
    assert result["__interrupt__"][0].value["plan"]["class_name"] == "LoginTest"
    assert "java_code" not in result


def test_approve_then_fix_loop_recovers(graph, monkeypatch):
    fake = FakeLLM([BROKEN, FIXED])
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    graph.invoke(START, CFG)
    result = graph.invoke(Command(resume={"action": "approve"}), CFG)
    assert result["status"] == "generated"
    assert result["attempts"] == 2                    # one failed compile, one fix
    assert result["output_path"].endswith("LoginTest.java")


def test_gives_up_after_max_attempts(graph, monkeypatch):
    fake = FakeLLM([BROKEN] * agent.MAX_ATTEMPTS)
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    graph.invoke(START, CFG)
    result = graph.invoke(Command(resume={"action": "approve"}), CFG)
    assert result["status"] == "failed_compilation"
    assert result["attempts"] == agent.MAX_ATTEMPTS


def test_revise_loops_back_to_planner(graph, monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM([FIXED]))
    graph.invoke(START, CFG)
    result = graph.invoke(Command(resume={"action": "revise", "feedback": "use CSS"}), CFG)
    assert "__interrupt__" in result                  # planner ran again, paused again
    assert result["review_feedback"] == ""            # feedback consumed by the planner


def test_reject_ends_run(graph, monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM([]))
    graph.invoke(START, CFG)
    result = graph.invoke(Command(resume={"action": "reject"}), CFG)
    assert result["status"] == "rejected"


def test_state_history_supports_time_travel(graph, monkeypatch):
    fake = FakeLLM([BROKEN, FIXED])
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    graph.invoke(START, CFG)
    graph.invoke(Command(resume={"action": "approve"}), CFG)
    history = list(graph.get_state_history(CFG))
    nodes_run = [h.next for h in history]
    assert ("compile_check",) in nodes_run            # every step was checkpointed
