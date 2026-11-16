"""Split tasks by how often the student solved them, and pick training targets.

  always     solved in all k rollouts  -> shortest success, the "efficiency" examples
  sometimes  solved in 1..k-1 rollouts -> shortest success, the "reliability" examples
  never      solved in 0 rollouts      -> sent to failure labelling and repair

"Shortest" means fewest tool calls, then fewest generated tokens. A trajectory is
only eligible as a target if it passes the evidence guard (see evidence_ok).
Without the guard, shortest-success would prefer a lucky answer that used no evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .summary import by_task, load, sh


DATA_TOOLS = {"query_finqa_tables", "get_table_schema"}


def evidence_ok(rec: dict) -> bool:
    """The answer must rest on table data that was really fetched: at least one
    query_finqa_tables or get_table_schema call that returned without error (the
    schema result includes real sample rows), and no assistant turn that writes
    a tool result itself (a lucky answer on invented data)."""
    if not sh(rec)["success"]:
        return False
    fetched = False
    for step in rec.get("steps", []):
        action = step.get("action") or {}
        names = [c.get("name") for c in action.get("tool_calls") or ([action] if action.get("name") else [])]
        result = str(step.get("observation", "")).lstrip('"\' ')
        if DATA_TOOLS & set(names) and result and not result.startswith("Error"):
            fetched = True
    invented = any(
        m.get("role") == "assistant" and "Tool result" in (m.get("content") or "")
        for m in rec["metadata"].get("conversation", [])
    )
    return fetched and not invented


def length_key(rec: dict) -> tuple:
    s = sh(rec)
    return (s["n_tool_calls"], s.get("completion_tokens", 0))


def pick_target(successes: list[dict], prefer_clean: bool) -> dict | None:
    eligible = [r for r in successes if evidence_ok(r)]
    if not eligible:
        return None
    if prefer_clean:  # a run with no failed tool call beats a shorter run that has one
        return min(eligible, key=lambda r: (sh(r)["n_tool_failures"] > 0, *length_key(r)))
    return min(eligible, key=length_key)


def make_buckets(recs: list[dict], prefer_clean: bool = True) -> dict:
    tasks = []
    for task_id, group in by_task(recs).items():
        successes = [r for r in group if sh(r)["success"]]
        n, k = len(successes), len(group)
        bucket = "always" if n == k else "never" if n == 0 else "sometimes"
        target = pick_target(successes, prefer_clean)
        tasks.append(
            {
                "task_id": task_id,
                "bucket": bucket,
                "solved": n,
                "k": k,
                "target_id": target["trajectory_id"] if target else None,
                "target_calls": sh(target)["n_tool_calls"] if target else None,
                "success_calls": [sh(r)["n_tool_calls"] for r in successes],
                "question_type": sh(group[0])["question_type"],
            }
        )

    def stats(name: str) -> dict:
        rows = [t for t in tasks if t["bucket"] == name]
        with_target = [t for t in rows if t["target_id"]]
        return {
            "tasks": len(rows),
            "with_target": len(with_target),
            "dropped_by_evidence_guard": sum(1 for t in rows if t["solved"] and not t["target_id"]),
            "mean_calls_all_successes": round(mean(c for t in rows for c in t["success_calls"]), 2) if any(t["success_calls"] for t in rows) else None,
            "mean_calls_target": round(mean(t["target_calls"] for t in with_target), 2) if with_target else None,
        }

    return {"tasks": tasks, "stats": {b: stats(b) for b in ("always", "sometimes", "never")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-prefer-clean", action="store_true")
    a = ap.parse_args()
    result = make_buckets(load(a.paths), prefer_clean=not a.no_prefer_clean)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["stats"], indent=2))
