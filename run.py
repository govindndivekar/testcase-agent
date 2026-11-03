"""CLI runner: start a run, answer the review interrupt, resume after a crash."""
import json
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import build_graph


def ask_reviewer(request: dict) -> dict:
    print("\n=== PLAN FOR REVIEW ===")
    print(json.dumps(request["plan"], indent=2))
    action = input(f"\n{request['question']} > ").strip().lower()
    feedback = input("Feedback for the planner > ") if action == "revise" else ""
    return {"action": action, "feedback": feedback}


def main(test_case_file: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = build_graph(checkpointer)

        snapshot = graph.get_state(config)
        if snapshot.interrupts:                 # paused earlier: resume, don't restart
            decision = ask_reviewer(snapshot.interrupts[0].value)
            result = graph.invoke(Command(resume=decision), config)
        else:
            with open(test_case_file) as f:
                result = graph.invoke({"test_case": json.load(f), "attempts": 0}, config)

        while "__interrupt__" in result:        # every pause surfaces here
            decision = ask_reviewer(result["__interrupt__"][0].value)
            result = graph.invoke(Command(resume=decision), config)

    print(f"\nstatus={result.get('status')} attempts={result.get('attempts')} "
          f"output={result.get('output_path')}")
    if result.get("compile_errors"):
        print("Last compiler errors:\n" + result["compile_errors"])


if __name__ == "__main__":
    # usage: python run.py testcases/IR-T101.json IR-T101
    main(sys.argv[1], sys.argv[2])
