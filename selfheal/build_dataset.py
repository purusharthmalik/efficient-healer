"""Assemble the SFT set for one round from every source of verified trajectories.

Sources, one target per task, taken in this order:
  efficiency       task solved in all k student rollouts  -> its shortest success
  reliability      task solved in some student rollouts   -> its shortest success
  repair_hint      never solved; a hinted student retry succeeded -> shortest success
  repair_teacher   never solved; a teacher rollout succeeded      -> shortest success
A task with no verified success from any source is dropped and counted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOL_RESULT_FORMAT

from .buckets import make_buckets, pick_target
from .sft_data import student_messages, valid_row
from .summary import by_task, load, sh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", nargs="+", required=True, help="student rollout files of this round")
    ap.add_argument("--hint-retries", nargs="*", default=[])
    ap.add_argument("--teacher", nargs="*", default=[])
    ap.add_argument("--extra-rows", nargs="*", default=[], help="rows.jsonl of earlier rounds to carry over")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="keep assistant text as generated")
    a = ap.parse_args()
    clean = not a.raw

    student = load(a.student)
    by_id = {r["trajectory_id"]: r for r in student}
    buckets = make_buckets(student)
    # The system prompt every student rollout saw (tool list included, no hint).
    system_text = next(r for r in student if not sh(r).get("hint"))["metadata"]["conversation"][0]["content"]

    rows, picked, dropped = [], [], Counter()

    def add(rec: dict, source: str, teacher: bool = False) -> bool:
        messages = student_messages(rec, clean, DEFAULT_TOOL_RESULT_FORMAT, system_text if teacher else None)
        if not valid_row(messages):
            dropped[f"{source}:invalid_row"] += 1
            return False
        s = sh(rec)
        # Rollout ids repeat across sources ("<task>#t0"), so the dataset id carries the source and round.
        uid = f"r{a.round}:{source}:{rec['trajectory_id']}"
        rows.append({"messages": messages, "source": source, "round": a.round, "task_id": s["task_id"],
                     "trajectory_id": uid, "n_tool_calls": sum(m["role"] == "assistant" for m in messages) - 1,
                     "n_tool_calls_before_cleanup": s["n_tool_calls"]})
        picked.append({**rec, "trajectory_id": uid})
        return True

    covered = set()
    for t in buckets["tasks"]:
        if t["target_id"] and add(by_id[t["target_id"]], "efficiency" if t["bucket"] == "always" else "reliability"):
            covered.add(t["task_id"])

    for source, paths, teacher in (("repair_hint", a.hint_retries, False), ("repair_teacher", a.teacher, True)):
        if not paths:
            continue
        for task_id, group in by_task(load(paths)).items():
            if task_id in covered:
                continue
            target = pick_target([r for r in group if sh(r)["success"]], prefer_clean=True)
            if target and add(target, source, teacher):
                covered.add(task_id)

    all_tasks = {t["task_id"] for t in buckets["tasks"]}
    carried = [json.loads(line) for p in a.extra_rows for line in Path(p).open()]
    carried = [r for r in carried if r["task_id"] not in covered]  # this round's target replaces an older one
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w") as f:
        for r in rows + carried:
            f.write(json.dumps(r) + "\n")
    with (out / "trajectories.jsonl").open("w") as f:
        for rec in picked:
            f.write(json.dumps(rec, default=str) + "\n")
    stats = {
        "round": a.round,
        "tasks": len(all_tasks),
        "bucket_stats": buckets["stats"],
        "rows_this_round": dict(Counter(r["source"] for r in rows)),
        "rows_carried_from_earlier_rounds": dict(Counter(r["source"] for r in carried)),
        "rows_total": len(rows) + len(carried),
        "tasks_without_any_verified_success": len(all_tasks - covered),
        "dropped": dict(dropped),
        "mean_tool_calls_by_source": {
            s: round(sum(r["n_tool_calls"] for r in rows if r["source"] == s) / max(1, sum(r["source"] == s for r in rows)), 2)
            for s in {r["source"] for r in rows}
        },
        "clean_targets": clean,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "uncovered_ids.json").write_text(json.dumps(sorted(all_tasks - covered)))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
