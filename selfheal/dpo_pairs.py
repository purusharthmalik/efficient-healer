"""Step-level preference pairs for DPO.

For a task with a verified target run and a failed student run, walk both runs
turn by turn while they are identical (same calls, same tool results). At the
first agent turn where they differ:
  prompt   = the conversation up to that point (identical in both runs)
  chosen   = the verified run's next agent turn
  rejected = the failed run's next agent turn
The pair isolates one decision, most often which table to open. It is a heuristic:
the first difference from the verified run is not always the failed run's mistake.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOL_RESULT_FORMAT

from .sft_data import student_messages
from .summary import load, sh


def first_divergence(target: list[dict], failed: list[dict]) -> int | None:
    """Index of the first assistant turn that differs, with identical messages before it."""
    for i in range(min(len(target), len(failed))):
        a, b = target[i], failed[i]
        if a["role"] != b["role"]:
            return None
        if a["content"] != b["content"]:
            return i if a["role"] == "assistant" and i >= 2 else None
    return None


def build_pairs(rows: list[dict], student_recs: list[dict]) -> list[dict]:
    target_by_task = {r["task_id"]: r for r in rows}
    pairs, seen = [], set()
    for rec in student_recs:
        s = sh(rec)
        if s["success"] or s.get("error") or s["task_id"] not in target_by_task:
            continue
        target = target_by_task[s["task_id"]]["messages"]
        failed = student_messages(rec, clean=True, tool_result_format=DEFAULT_TOOL_RESULT_FORMAT)
        i = first_divergence(target, failed)
        if i is None or not failed[i]["content"].strip():
            continue
        key = (s["task_id"], i, failed[i]["content"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "task_id": s["task_id"],
            "turn": (i - 1) // 2,  # 0 = first agent turn
            "chosen_kind": "call" if target[i]["content"].startswith("[{") else "answer",
            "rejected_kind": "call" if failed[i]["content"].startswith("[{") else "answer",
            "prompt": target[:i],
            "chosen": [target[i]],
            "rejected": [failed[i]],
        })
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--student", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in Path(a.rows).open()]
    pairs = build_pairs(rows, load(a.student))
    stats = {"pairs_found": len(pairs), "tasks_with_a_pair": len({p["task_id"] for p in pairs}),
             "by_turn": dict(sorted(Counter(p["turn"] for p in pairs).items())),
             "chosen_vs_rejected_kind": dict(Counter(f"{p['chosen_kind']}/{p['rejected_kind']}" for p in pairs))}
    random.Random(a.seed).shuffle(pairs)
    pairs = pairs[: a.max_pairs]
    stats["pairs_used"] = len(pairs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(json.dumps(stats, indent=1))
