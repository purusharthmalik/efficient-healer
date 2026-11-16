"""Read rollout JSONL files and summarise them per task. Pure Python, no GPU libraries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def sh(rec: dict) -> dict:
    return rec["metadata"]["selfheal"]


def by_task(recs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[sh(r)["task_id"]].append(r)
    return groups


def summarise(recs: list[dict]) -> dict:
    groups = by_task(recs)
    k = max(len(g) for g in groups.values())
    solved = {t: sum(sh(r)["success"] for r in g) for t, g in groups.items()}
    succ = [r for r in recs if sh(r)["success"]]
    return {
        "n_tasks": len(groups),
        "k": k,
        "n_rollouts": len(recs),
        "success_rate": round(mean(sh(r)["success"] for r in recs), 4),
        "strict_success_rate": round(mean(sh(r)["strict"] for r in recs), 4),
        "tasks_all_k": sum(1 for t, s in solved.items() if s == len(groups[t])),
        "tasks_some": sum(1 for t, s in solved.items() if 0 < s < len(groups[t])),
        "tasks_none": sum(1 for s in solved.values() if s == 0),
        "mean_tool_calls": round(mean(sh(r)["n_tool_calls"] for r in recs), 2),
        "mean_tool_calls_success": round(mean(sh(r)["n_tool_calls"] for r in succ), 2) if succ else None,
        "tool_failure_rate": round(
            sum(sh(r)["n_tool_failures"] for r in recs) / max(1, sum(sh(r)["n_tool_calls"] for r in recs)), 4
        ),
        "no_tool_call_share": round(mean(sh(r)["n_tool_calls"] == 0 for r in recs), 4),
        "mean_completion_tokens": round(mean(sh(r).get("completion_tokens", 0) for r in recs), 1),
        "mean_prompt_tokens": round(mean(sh(r).get("prompt_tokens", 0) for r in recs), 1),
        "mean_seconds": round(mean(sh(r)["seconds"] for r in recs), 2),
        "mean_gen_seconds": round(mean(sh(r).get("gen_seconds", 0) for r in recs), 2),
        "mean_model_calls": round(mean(sh(r).get("calls", 0) for r in recs), 2),
        "verify_rules": dict(Counter(sh(r)["verify_rule"] for r in recs)),
        "tool_call_histogram": dict(sorted(Counter(sh(r)["n_tool_calls"] for r in recs).items())),
        "errors": dict(Counter((sh(r)["error"] or "")[:80] for r in recs if sh(r)["error"])),
    }


def show(rec: dict, width: int = 700) -> None:
    s = sh(rec)
    print("=" * 100)
    print(f"{rec['trajectory_id']} | success={s['success']} rule={s['verify_rule']} gold={s['gold']} pred={s['pred']} "
          f"| calls={s['n_tool_calls']} fails={s['n_tool_failures']} | {s['seconds']}s")
    for m in rec["metadata"].get("conversation", []):
        content = m.get("content")
        content = content if isinstance(content, str) else json.dumps(content)
        print(f"--- {m.get('role')} ---")
        print(content[:width] + (" …" if len(content) > width else ""))
        if m.get("tool_calls"):
            print("tool_calls:", json.dumps(m["tool_calls"])[:width])
    if s.get("error"):
        print("ERROR:", s["error"], "\n", s.get("traceback", ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--show", type=int, default=0, help="print this many successes and failures")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = load(a.paths)
    stats = summarise(recs)
    print(json.dumps(stats, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(stats, indent=2))
    if a.show:
        for group in ([r for r in recs if sh(r)["success"]], [r for r in recs if not sh(r)["success"]]):
            for r in group[: a.show]:
                show(r)
