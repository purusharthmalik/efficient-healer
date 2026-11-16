"""Before/after evaluation with AuditKIT. Runs in the AuditKIT environment.

AgentTune trajectory JSONL -> ak.episodes_from_agenttune -> precomputed samples
-> AuditKIT agent metrics + our task metrics as AuditKIT scorers -> ak.compare
with a paired bootstrap. A task-level bootstrap (resampling tasks, not rollouts)
is added for the headline numbers, because rollouts of one task are not independent.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import auditkit as ak

from .summary import by_task, load, sh

REF_FREE_METRICS = ["tool_call_validity", "redundant_tool_calls", "agent_loop_detection"]


def tool_schemas(rec: dict) -> list[dict]:
    """The tool schemas exactly as the student saw them, read back from its system prompt."""
    system = rec["metadata"]["conversation"][0]["content"]
    start = system.index("You can call these tools:\n") + len("You can call these tools:\n")
    return json.JSONDecoder().raw_decode(system[start:])[0]


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="task_success")
def task_success(sample, output):
    return float(sample.metadata["success"])


@ak.scorer(direction=ak.Direction.MINIMIZE, name="tool_calls")
def n_tool_calls(sample, output):
    return float(sample.metadata["n_tool_calls"])


@ak.scorer(direction=ak.Direction.MINIMIZE, name="completion_tokens")
def completion_tokens(sample, output):
    return float(sample.metadata.get("completion_tokens", 0))


@ak.scorer(direction=ak.Direction.MINIMIZE, name="seconds")
def seconds(sample, output):
    return float(sample.metadata["seconds"])


def run_auditkit(path: str, schemas: list[dict]):
    recs = {r["trajectory_id"]: r for r in load([path])}
    episodes = ak.episodes_from_agenttune(path)
    samples = []
    for e in episodes:
        rec = recs[e.source_id]
        samples.append(
            ak.Sample(
                id=f"{sh(rec)['task_id']}#t{sh(rec)['trial']}",  # the same ids in the base and candidate runs
                input=str(rec.get("task", "")),
                actual_output=e.final_answer or "",
                actual_trace=e.to_trace(),
                tools=schemas,
                metadata=sh(rec),
            )
        )
    metrics = [ak.ToolCallValidity(), ak.RedundantToolCalls(), ak.AgentLoopDetection(),
               task_success, n_tool_calls, completion_tokens, seconds]
    return ak.evaluate(samples, "precomputed", metrics)


def task_table(recs: list[dict]) -> dict[str, dict]:
    out = {}
    for task_id, group in by_task(recs).items():
        succ = [r for r in group if sh(r)["success"]]
        out[task_id] = {
            "success_rate": len(succ) / len(group),
            "all_k": float(len(succ) == len(group)),
            "any_k": float(len(succ) > 0),
            "calls_success": mean(sh(r)["n_tool_calls"] for r in succ) if succ else None,
            "calls_all": mean(sh(r)["n_tool_calls"] for r in group),
            "tokens_all": mean(sh(r).get("completion_tokens", 0) for r in group),
            "seconds_all": mean(sh(r)["seconds"] for r in group),
        }
    return out


def paired_task_bootstrap(base: dict, cand: dict, key: str, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean difference (candidate - base) over tasks where both have a value, with a 95% interval."""
    tasks = [t for t in base if t in cand and base[t][key] is not None and cand[t][key] is not None]
    if len(tasks) < 2:
        return {"n_tasks": len(tasks), "base": None, "cand": None, "delta": None, "ci95": None}
    diffs = [cand[t][key] - base[t][key] for t in tasks]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {
        "n_tasks": len(tasks),
        "base": round(mean(base[t][key] for t in tasks), 4),
        "cand": round(mean(cand[t][key] for t in tasks), 4),
        "delta": round(mean(diffs), 4),
        "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_recs = load([a.base])
    schemas = tool_schemas(base_recs[0])
    base_run = run_auditkit(a.base, schemas)
    base_run.save(str(out_dir / "auditkit_base.json"))
    report = {"auditkit_base": base_run.headline, "n_base_rollouts": len(base_recs)}
    print("AuditKIT base headline:", json.dumps(base_run.headline, indent=1, default=str))

    if a.cand:
        cand_recs = load([a.cand])
        cand_run = run_auditkit(a.cand, schemas)
        cand_run.save(str(out_dir / "auditkit_cand.json"))
        cmp = ak.compare(base_run, cand_run)
        print(cmp.summary())
        report["auditkit_cand"] = cand_run.headline
        report["auditkit_significance"] = {}
        for m in REF_FREE_METRICS + ["task_success", "tool_calls", "completion_tokens", "seconds"]:
            try:
                report["auditkit_significance"][m] = cmp.significance(m)
            except Exception as e:  # a metric with no scored samples on one side
                report["auditkit_significance"][m] = {"error": str(e)[:200]}
        bt, ct = task_table(base_recs), task_table(cand_recs)
        report["task_level"] = {
            "success_rate": paired_task_bootstrap(bt, ct, "success_rate"),
            "solved_in_all_k": paired_task_bootstrap(bt, ct, "all_k"),
            "solved_in_any_k": paired_task_bootstrap(bt, ct, "any_k"),
            "tool_calls_on_tasks_both_solve": paired_task_bootstrap(bt, ct, "calls_success"),
            "tool_calls_all_rollouts": paired_task_bootstrap(bt, ct, "calls_all"),
            "completion_tokens": paired_task_bootstrap(bt, ct, "tokens_all"),
            "seconds": paired_task_bootstrap(bt, ct, "seconds_all"),
        }
        print(json.dumps(report["task_level"], indent=1))

    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
