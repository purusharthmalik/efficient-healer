"""Print the write-up's result tables as Markdown from the evaluation outputs.

    python tools/results_tables.py runs/ops/out/eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/ops/out/eval")


def load(name: str) -> dict | None:
    p = root / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def num(x, d: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def ci(entry: dict | None, as_pct: bool = False, d: int = 2) -> str:
    if not entry or entry.get("delta") is None:
        return "n/a"
    lo, hi = entry["ci95"]
    if as_pct:
        return f"{100 * entry['delta']:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{entry['delta']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def sig(entry: dict | None, as_pct: bool = False, d: int = 3) -> str:
    if not entry or "delta" not in entry:
        return "n/a"
    lo, hi = entry["ci"]
    if as_pct:
        return f"{100 * entry['delta']:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{entry['delta']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def model_table(mode: str) -> None:
    if mode == "confirm":
        names = [("Base Tiny Aya", "confirm_base"), ("Round 1 adapter", "confirm_v1"), ("Final adapter", "confirm_final")]
    else:
        names = [("Base Tiny Aya", f"base_{mode}"), ("Round 1 adapter", f"v1_{mode}"), ("Final adapter", f"final_{mode}")]
    sums = [(label, load(f"{key}.summary.json")) for label, key in names]
    sums = [(label, s) for label, s in sums if s]
    if not sums:
        return
    print("| Metric | " + " | ".join(label for label, _ in sums) + " |")
    print("|---|" + "---|" * len(sums))
    rows = [
        ("Rollouts solved (tolerant)", lambda s: f"{round(s['success_rate'] * s['n_rollouts'])} of {s['n_rollouts']} ({pct(s['success_rate'])})"),
        ("Rollouts solved (strict)", lambda s: pct(s["strict_success_rate"])),
        ("Tasks solved in all 3 runs", lambda s: f"{s['tasks_all_k']} of {s['n_tasks']}"),
        ("Tasks solved in at least 1 run", lambda s: f"{s['tasks_all_k'] + s['tasks_some']} of {s['n_tasks']}"),
        ("Tool calls per run", lambda s: num(s["mean_tool_calls"])),
        ("Tool calls per solved run", lambda s: num(s["mean_tool_calls_success"])),
        ("Output tokens per run", lambda s: num(s["mean_completion_tokens"], 0)),
        ("Seconds per run", lambda s: num(s["mean_seconds"], 1)),
        ("Runs with no `<answer>` tag", lambda s: f"{s['verify_rules'].get('no_answer_tag', 0)} of {s['n_rollouts']}"),
    ]
    for label, fn in rows:
        print(f"| {label} | " + " | ".join(fn(s) for _, s in sums) + " |")


def comparison(title: str, name: str) -> None:
    r = load(f"cmp_{name}/report.json")
    if not r:
        return
    tl, b, c, s = r["task_level"], r["auditkit_base"], r["auditkit_cand"], r["auditkit_significance"]
    print(f"\n**{title}** (difference = second minus first, 95% paired bootstrap interval)\n")
    print("| Metric | First | Second | Difference |")
    print("|---|---|---|---|")
    print(f"| Success rate, per task | {pct(tl['success_rate']['base'])} | {pct(tl['success_rate']['cand'])} | {ci(tl['success_rate'], True)} |")
    print(f"| Tasks solved in all 3 runs | {pct(tl['solved_in_all_k']['base'])} | {pct(tl['solved_in_all_k']['cand'])} | {ci(tl['solved_in_all_k'], True)} |")
    print(f"| Tasks solved in at least 1 run | {pct(tl['solved_in_any_k']['base'])} | {pct(tl['solved_in_any_k']['cand'])} | {ci(tl['solved_in_any_k'], True)} |")
    both = tl["tool_calls_on_tasks_both_solve"]
    print(f"| Tool calls on tasks both solve (n={both['n_tasks']}) | {num(both['base'])} | {num(both['cand'])} | {ci(both)} |")
    print(f"| Tool calls per run | {num(tl['tool_calls_all_rollouts']['base'])} | {num(tl['tool_calls_all_rollouts']['cand'])} | {ci(tl['tool_calls_all_rollouts'])} |")
    print(f"| Output tokens per run | {num(tl['completion_tokens']['base'], 0)} | {num(tl['completion_tokens']['cand'], 0)} | {ci(tl['completion_tokens'], d=0)} |")
    print(f"| Seconds per run | {num(tl['seconds']['base'], 1)} | {num(tl['seconds']['cand'], 1)} | {ci(tl['seconds'], d=1)} |")
    for key, label in (("tool_call_validity", "Tool-call validity (AuditKIT)"),
                       ("redundant_tool_calls", "Redundant tool-call rate (AuditKIT)"),
                       ("agent_loop_detection", "Loop-free share (AuditKIT)")):
        print(f"| {label} | {num(b.get(key), 3)} | {num(c.get(key), 3)} | {sig(s.get(key))} |")


def gate(name: str) -> None:
    g = load(name)
    if not g:
        return
    print(f"\n**Gate, {g['label']}: {'DEPLOY' if g['deploy'] else 'HOLD'}**\n")
    for line in g["our_rule"]["reasons"]:
        print(f"- {line}")
    a = g["agenttune_deployment_gate"]
    print(f"- AgentTune DeploymentGate: {'approved' if a['approved'] else 'blocked'} ({a['reason']}; "
          f"pass rate {a['task_old']:.3f} -> {a['task_new']:.3f} on {a['num_cases']} runs)")


print("### Held-out results with the harness stop rule (100 tasks x 3 runs)\n")
model_table("stop")
print("\n### Held-out results without the harness stop rule (100 tasks x 3 runs)\n")
model_table("nostop")
comparison("Base vs final adapter, with stop rule", "overall_stop")
comparison("Base vs round 1 adapter, with stop rule", "r1_gate")
comparison("Round 1 adapter vs final adapter, with stop rule", "r2_gate")
comparison("Base vs final adapter, without stop rule", "overall_nostop")
gate("gate_round1.json")
gate("gate_round2.json")

print("\n### Confirmatory held-out set (400 fresh tasks x 3 runs, stop rule on)\n")
model_table("confirm")
comparison("Confirmatory set: round 1 adapter vs final adapter", "confirm_r2")
comparison("Confirmatory set: base vs final adapter", "confirm_overall")
comparison("Confirmatory set: base vs round 1 adapter", "confirm_r1")
gate("gate_round1_confirm.json")
gate("gate_round2_confirm.json")
