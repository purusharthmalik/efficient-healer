"""Label why a rollout failed.

Two labels per failed rollout:
  rule_label  deterministic, read off the trajectory (loop, tool errors, no tagged answer, ...)
  llm_label   AgentTune's FailureClassifier, one of its five root causes

AgentTune's classifier turns any error into `incomplete_reasoning` with
confidence 0.0, so a rollout keeps both labels and we report how often they agree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from .summary import by_task, load, sh

# Rule label -> the AgentTune root cause it corresponds to (None: the rules cannot tell).
RULE_TO_ROOT = {
    "rollout_error": None,
    "no_tool_evidence": "hallucinated_output",
    "loop_collapse": "loop_collapse",
    "step_cap": "incomplete_reasoning",
    "tool_errors": "wrong_tool",
    "no_answer_tag": "incomplete_reasoning",
    "wrong_number": None,
}


def tool_calls(rec: dict) -> list[tuple[str, str]]:
    """(name, canonical arguments) for every tool call in order."""
    calls = []
    for step in rec.get("steps", []):
        action = step.get("action") or {}
        for c in action.get("tool_calls") or ([action] if action.get("name") else []):
            calls.append((c.get("name", ""), json.dumps(c.get("arguments", {}), sort_keys=True)))
    return calls


def rule_label(rec: dict) -> str:
    s = sh(rec)
    if s.get("error"):
        return "rollout_error"
    calls = tool_calls(rec)
    counts = Counter(calls)
    repeated_in_a_row = any(a == b for a, b in zip(calls, calls[1:]))
    if repeated_in_a_row or (counts and max(counts.values()) >= 3):
        return "loop_collapse"
    if s["n_tool_calls"] - s["n_tool_failures"] < 1:
        return "no_tool_evidence"
    if s["verify_rule"] == "no_answer_tag":
        return "step_cap" if s["n_steps"] >= s["max_steps"] else "no_answer_tag"
    if s["n_tool_failures"] >= max(1, s["n_tool_calls"] // 2):
        return "tool_errors"
    return "wrong_number"


def _context(rec: dict) -> dict:
    """A compact view of the failure. AgentTune truncates the context to 2,000 characters."""
    s = sh(rec)
    steps = []
    for step in rec.get("steps", [])[:8]:
        action = step.get("action") or {}
        for c in action.get("tool_calls") or ([action] if action.get("name") else []):
            steps.append(
                {"call": f"{c.get('name')}({json.dumps(c.get('arguments', {}))[:110]})",
                 "result": str(step.get("observation", ""))[:90]}
            )
    return {
        "question": rec.get("task", "")[:260] if isinstance(rec.get("task"), str) else str(s["task_id"]),
        "expected_answer": s["gold"],
        "agent_answer": s["pred"] if s["pred"] is not None else (rec.get("final_response") or "")[-160:],
        "steps": steps,
    }


def classify_with_agenttune(failed: list[dict], model: str, api_base: str, batch: int = 32,
                            thinking: bool = True) -> list[dict]:
    from agenttune.decide.closed_loop.contracts import Failure
    from agenttune.decide.closed_loop.failure_classifier import FailureClassifier

    if not thinking:
        # The classifier calls litellm.acompletion with fixed arguments; add the server-side switch
        # that turns the teacher's thinking off, which makes labelling several times faster.
        import functools

        import litellm

        if not getattr(litellm.acompletion, "_selfheal_patched", False):
            patched = functools.partial(litellm.acompletion,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            patched._selfheal_patched = True
            litellm.acompletion = patched

    os.environ.setdefault("OPENAI_API_KEY", "unused")
    clf = FailureClassifier(model_name=model, api_base=api_base)
    failures = [
        Failure(
            trajectory_id=r["trajectory_id"],
            failure_type=rule_label(r),
            failed_stage_name="finqa_agent",
            context=_context(r),
            error_message=sh(r).get("error") or f"final answer check: {sh(r)['verify_rule']}",
        )
        for r in failed
    ]
    out = []
    for i in range(0, len(failures), batch):
        chunk = asyncio.run(clf.classify_batch(failures[i : i + batch]))
        out.extend(
            {"trajectory_id": c.failure.trajectory_id, "llm_label": c.root_cause,
             "llm_confidence": c.confidence, "llm_analysis": c.analysis[:400]}
            for c in chunk
        )
        print(f"[classify] {len(out)}/{len(failures)}", flush=True)
    return out


def label_failures(recs: list[dict], model: str | None, api_base: str | None, per_task: int = 2,
                   never_only: bool = False, thinking: bool = True) -> dict:
    failed = [r for r in recs if not sh(r)["success"]]
    rows = {r["trajectory_id"]: {"trajectory_id": r["trajectory_id"], "task_id": sh(r)["task_id"],
                                 "rule_label": rule_label(r)} for r in failed}
    if model:
        # The LLM label is used per task, so classify at most `per_task` failed rollouts of each task.
        seen: Counter = Counter()
        subset = []
        never_ids = {t for t, g in by_task(recs).items() if not any(sh(r)["success"] for r in g)}
        for r in failed:
            if never_only and sh(r)["task_id"] not in never_ids:  # the LLM label only picks hints for these
                continue
            if seen[sh(r)["task_id"]] < per_task:
                seen[sh(r)["task_id"]] += 1
                subset.append(r)
        for c in classify_with_agenttune(subset, model, api_base, thinking=thinking):
            rows[c["trajectory_id"]].update(c)

    labelled = [r for r in rows.values() if r.get("llm_confidence", 0) > 0]
    comparable = [r for r in labelled if RULE_TO_ROOT[r["rule_label"]] is not None]
    agree = sum(1 for r in comparable if RULE_TO_ROOT[r["rule_label"]] == r["llm_label"])

    # One label per never-solved task: the most common label among its failed rollouts.
    never = {t for t, g in by_task(recs).items() if not any(sh(r)["success"] for r in g)}
    per_task = {}
    for t in never:
        mine = [r for r in rows.values() if r["task_id"] == t]
        per_task[t] = {
            "rule_label": Counter(r["rule_label"] for r in mine).most_common(1)[0][0],
            "llm_label": Counter(r["llm_label"] for r in mine if r.get("llm_confidence", 0) > 0).most_common(1)[0][0]
            if any(r.get("llm_confidence", 0) > 0 for r in mine) else None,
        }

    return {
        "rollouts": list(rows.values()),
        "tasks": per_task,
        "stats": {
            "failed_rollouts": len(failed),
            "rule_labels": dict(Counter(r["rule_label"] for r in rows.values())),
            "llm_labels": dict(Counter(r["llm_label"] for r in labelled)),
            "llm_classified": len([r for r in rows.values() if "llm_label" in r]) if model else None,
            "llm_classifier_errors": len([r for r in rows.values() if "llm_label" in r and r.get("llm_confidence", 0) <= 0]) if model else None,
            "rule_vs_llm_agreement": f"{agree}/{len(comparable)}" if comparable else None,
            "never_solved_tasks": len(never),
            "task_rule_labels": dict(Counter(v["rule_label"] for v in per_task.values())),
            "task_llm_labels": dict(Counter(str(v["llm_label"]) for v in per_task.values())),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help='LiteLLM model name, e.g. "openai/teacher"')
    ap.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--never-only", action="store_true", help="LLM-label only tasks with no success")
    ap.add_argument("--no-thinking", action="store_true", help="turn the classifier model's thinking off")
    a = ap.parse_args()
    result = label_failures(load(a.paths), a.model, a.api_base, a.per_task, a.never_only, not a.no_thinking)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["stats"], indent=2))
