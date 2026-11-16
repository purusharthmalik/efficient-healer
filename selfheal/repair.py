"""Choose a repair hint for every never-solved task from its failure labels.

The hint is added to the system prompt for the retry only. It is removed from
the conversation before a successful retry becomes a training example.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Format failures are read off the trajectory by rule; the LLM taxonomy has no label for them.
HINTS_BY_RULE = {
    "loop_collapse": "Do not repeat a tool call you already made. Use the result you already have.",
    "no_answer_tag": "When you have the number, reply with only <answer>NUMBER</answer>.",
    "step_cap": "Use at most four tool calls, then reply with only <answer>NUMBER</answer>.",
    "tool_errors": "Use only the listed tools and their exact argument names. Read the table schema before you query it.",
    "no_tool_evidence": "Answer only from tool results. Look the values up before you answer.",
}

# For runs that used the tools and still gave a wrong number, the LLM label picks the hint.
HINTS_BY_ROOT_CAUSE = {
    "wrong_routing": "List the company's tables first and choose the table whose name matches the question's topic.",
    "wrong_tool": "Use only the listed tools and their exact argument names. Read the table schema before you query it.",
    "incomplete_reasoning": "Query the exact rows and years the question asks about, then compute the final number with the calculator.",
    "hallucinated_output": "Answer only from tool results. Never guess a table name or a value.",
    "loop_collapse": "Do not repeat a tool call you already made. Use the result you already have.",
}


def choose_hint(rule_label: str, llm_label: str | None) -> tuple[str, str]:
    """Return (label used, hint text)."""
    if rule_label in HINTS_BY_RULE:
        return rule_label, HINTS_BY_RULE[rule_label]
    root = llm_label if llm_label in HINTS_BY_ROOT_CAUSE else "incomplete_reasoning"
    return root, HINTS_BY_ROOT_CAUSE[root]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True, help="output of selfheal.failures")
    ap.add_argument("--out", required=True, help="directory for hints.json, hint_labels.json, never_ids.json")
    a = ap.parse_args()
    tasks = json.loads(Path(a.failures).read_text())["tasks"]
    hints, labels = {}, {}
    for task_id, lab in tasks.items():
        labels[task_id], hints[task_id] = choose_hint(lab["rule_label"], lab.get("llm_label"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hints.json").write_text(json.dumps(hints, indent=1))
    (out / "hint_labels.json").write_text(json.dumps(labels, indent=1))
    (out / "never_ids.json").write_text(json.dumps(sorted(tasks)))
    print(json.dumps({"never_solved_tasks": len(tasks), "hint_labels": dict(Counter(labels.values()))}, indent=2))
