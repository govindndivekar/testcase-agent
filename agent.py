"""Test Case -> TestNG Generator: a LangGraph agent built from scratch.

Flow: plan_test -> human_review (interrupt) -> generate_code -> compile_check -> write_output
      with a revise loop (review -> plan) and a fix loop (compile errors -> generate).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

MAX_ATTEMPTS = 3
PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")   # "openrouter" or "anthropic"
MODEL = os.getenv("AGENT_MODEL", "")                # e.g. a slug copied from openrouter.ai/models
# Selenium + TestNG jars; e.g. copy them into ./lib with `mvn dependency:copy-dependencies`
CLASSPATH = os.getenv("TEST_CLASSPATH", "lib/*")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "generated"))

class PlanStep(BaseModel):
    action: str = Field(description="What the test does in this step")
    locator: str = Field(description="Selenium locator, e.g. By.id(\"username\")")
    expected: str = Field(description="Expected result from the test case")
    assertion: str = Field(description="TestNG assertion that verifies the expected result")


class TestPlan(BaseModel):
    class_name: str = Field(description="PascalCase Java class name ending in 'Test'")
    package: str = Field(default="com.example.tests")
    base_url: str
    steps: list[PlanStep]


class AgentState(TypedDict, total=False):
    test_case: dict            # input: the manual test case
    plan: dict                 # TestPlan as a dict (JSON-serialisable for the checkpointer)
    review_feedback: str       # reviewer notes when they ask for a revision
    java_code: str             # latest generated source
    compile_errors: str        # "" when the last compile passed
    attempts: int              # code-generation attempts so far
    output_path: str
    status: str                # generated | rejected | failed_compilation

def get_llm():
    """Single place to choose the model; tests monkeypatch this."""
    if PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MODEL,                                   # OpenRouter slug, e.g. "vendor/model-name"
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0,
            max_tokens=4096,
            default_headers={"X-OpenRouter-Title": "tc-agent"},   # optional app attribution
        )
    from langchain_anthropic import ChatAnthropic          # direct Anthropic key
    return ChatAnthropic(model=MODEL or "claude-sonnet-5", temperature=0, max_tokens=4096)


@tool
def compile_java(java_code: str) -> str:
    """Compile one Java source file with javac. Returns 'OK' or the compiler errors."""
    match = re.search(r"public\s+class\s+(\w+)", java_code)
    if not match:
        return "error: no 'public class' declaration found"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{match.group(1)}.java"
        src.write_text(java_code)
        proc = subprocess.run(
            ["javac", "-cp", CLASSPATH, "-d", tmp, str(src)],
            capture_output=True, text=True, timeout=120,
        )
        return "OK" if proc.returncode == 0 else proc.stderr[-4000:]


def strip_fences(text: str) -> str:
    """Remove markdown code fences if the model added them anyway."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    return re.sub(r"\n```\s*$", "", text).strip()

PLAN_PROMPT = """You are a senior QA automation engineer.
Convert the manual test case into a Selenium 4 + TestNG test plan.
Prefer By.id, then By.cssSelector; use By.xpath only as a last resort.
Write one assertion per expected result."""

CODE_PROMPT = """Write ONE complete Java 17 class that implements the plan with Selenium 4 and TestNG 7.
Rules:
- Use the package and class name from the plan.
- Create ChromeDriver in @BeforeMethod and quit it in @AfterMethod.
- Use WebDriverWait with Duration.ofSeconds(10); never Thread.sleep.
- Use org.testng.Assert for every assertion in the plan.
Return only Java source, no markdown fences, no explanation."""


def plan_test(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(TestPlan)
    messages = [SystemMessage(PLAN_PROMPT),
                HumanMessage(json.dumps(state["test_case"], indent=2))]
    if state.get("review_feedback"):
        messages.append(HumanMessage(
            "Revise your previous plan using this reviewer feedback:\n"
            f"{state['review_feedback']}\n\nPrevious plan:\n{json.dumps(state['plan'], indent=2)}"))
    plan: TestPlan = llm.invoke(messages)
    return {"plan": plan.model_dump(), "review_feedback": ""}


def human_review(state: AgentState) -> Command[Literal["generate_code", "plan_test", "__end__"]]:
    # Everything before interrupt() re-runs on resume, so keep this node side-effect free.
    decision = interrupt({
        "question": "Approve this test plan? (approve / revise / reject)",
        "plan": state["plan"],
    })
    action = decision.get("action", "reject")
    if action == "approve":
        return Command(goto="generate_code")
    if action == "revise":
        return Command(update={"review_feedback": decision.get("feedback", "")}, goto="plan_test")
    return Command(update={"status": "rejected"}, goto=END)


def generate_code(state: AgentState) -> dict:
    messages = [SystemMessage(CODE_PROMPT),
                HumanMessage("Plan:\n" + json.dumps(state["plan"], indent=2))]
    if state.get("compile_errors"):
        messages.append(HumanMessage(
            f"Your previous code failed to compile:\n{state['compile_errors']}\n\n"
            f"Previous code:\n{state['java_code']}\n\nReturn the full corrected class."))
    response = get_llm().invoke(messages)
    return {"java_code": strip_fences(response.text),
            "attempts": state.get("attempts", 0) + 1}


def compile_check(state: AgentState) -> dict:
    result = compile_java.invoke({"java_code": state["java_code"]})
    errors = "" if result == "OK" else result
    update: dict = {"compile_errors": errors}
    if errors and state["attempts"] >= MAX_ATTEMPTS:
        update["status"] = "failed_compilation"
    return update


def write_output(state: AgentState) -> dict:
    name = re.search(r"public\s+class\s+(\w+)", state["java_code"]).group(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.java"
    path.write_text(state["java_code"])
    return {"output_path": str(path), "status": "generated"}

def route_after_compile(state: AgentState) -> str:
    if not state["compile_errors"]:
        return "write_output"
    if state["attempts"] < MAX_ATTEMPTS:
        return "generate_code"      # fix loop
    return END                      # give up after MAX_ATTEMPTS

def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("plan_test", plan_test)
    builder.add_node("human_review", human_review,
                     destinations=("generate_code", "plan_test", END))
    builder.add_node("generate_code", generate_code)
    builder.add_node("compile_check", compile_check)
    builder.add_node("write_output", write_output)

    builder.add_edge(START, "plan_test")
    builder.add_edge("plan_test", "human_review")
    builder.add_edge("generate_code", "compile_check")
    builder.add_conditional_edges("compile_check", route_after_compile,
                                  ["write_output", "generate_code", END])
    builder.add_edge("write_output", END)
    return builder.compile(checkpointer=checkpointer)
