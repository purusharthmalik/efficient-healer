"""The parts of AgentTune's closed loop we call directly (see its self-healing guide).

  detect   Project.heal() -> FailureDetector on EventLog.to_audit_records()
  trigger  TrainingBuffer + RetrainingTrigger, fed TrainingExample(chosen, rejected)
  gate     DeploymentGate.evaluate_decision on the held-out results
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .sft_data import student_messages
from .summary import by_task, load, sh


def agenttune_detect(paths: list[str]) -> dict:
    """AgentTune's rule-based detector (loop_collapse, tool_crash) over our trajectories."""
    from agenttune.agentic import Project
    from agenttune.agentic.events import EventLog
    from agenttune.agentic.trajectory.dataset import TrajectoryDataset

    project = Project()
    n = 0
    for p in paths:
        for traj in TrajectoryDataset.from_jsonl(p).trajectories:
            project.add_trajectory(EventLog.from_trajectory(traj))
            n += 1
    failures = project.heal()
    return {"trajectories_scanned": n, "failures_detected": len(failures),
            "by_type": dict(Counter(f.failure_type for f in failures))}


def preference_examples(rows: list[dict], student_recs: list[dict], labels: dict) -> list:
    """One AgentTune TrainingExample per repaired task: the verified run is `chosen`,
    a real failed student run of the same task is `rejected`."""
    from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOL_RESULT_FORMAT
    from agenttune.decide.closed_loop.contracts import TrainingExample

    failed = {}
    for task_id, group in by_task(student_recs).items():
        bad = [r for r in group if not sh(r)["success"]]
        if bad:
            failed[task_id] = bad[0]
    examples = []
    for row in rows:
        rec = failed.get(row["task_id"])
        if rec is None or not row["source"].startswith(("repair", "reliability")):
            continue
        rejected = student_messages(rec, clean=False, tool_result_format=DEFAULT_TOOL_RESULT_FORMAT)
        lab = labels.get(row["task_id"], {})
        examples.append(
            TrainingExample(
                trajectory_id=row["trajectory_id"],
                original_failure_type=lab.get("rule_label", "verifier_failed"),
                root_cause=lab.get("llm_label") or lab.get("rule_label") or "incomplete_reasoning",
                prompt=row["messages"][:2],
                chosen=row["messages"][2:],
                rejected=rejected[2:],
            )
        )
    return examples


def trigger_check(examples: list) -> dict:
    """Feed the examples to AgentTune's buffer and ask its trigger whether to retrain."""
    from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

    trigger = RetrainingTrigger(config=TriggerConfig(max_buffer_size=max(500, len(examples))))
    for ex in examples:
        trigger.buffer.add(ex)
    health = trigger.buffer.health()
    fired, reason = trigger.should_trigger()
    return {
        "examples_buffered": len(examples),
        "trigger_fired": fired,
        "trigger_reason": reason,
        "buffer_size": health.buffer_size,
        "dominant_failure_type": health.dominant_failure_type,
        "dominant_failure_ratio": round(health.dominant_failure_ratio or 0.0, 3),
        "drop_rate": health.drop_rate,
    }


def agenttune_gate(base_recs: list[dict], cand_recs: list[dict], task_regression_tol: float = 0.0) -> dict:
    """AgentTune's gate on the held-out rollouts: blocks if the task pass rate regresses."""
    from agenttune.decide.closed_loop.deployment_gate import DeploymentGate

    key = lambda r: f"{sh(r)['task_id']}#t{sh(r)['trial']}"  # noqa: E731
    base = {key(r): sh(r)["success"] for r in base_recs}
    cand = {key(r): sh(r)["success"] for r in cand_recs}
    ids = sorted(set(base) & set(cand))
    test_set = [{"pipeline_id": i, "input_text": i, "expected_verdict": "solved"} for i in ids]
    decision = DeploymentGate(seed=0).evaluate_decision(
        test_set,
        old_model_fn=lambda i: "solved" if base[i] else "failed",
        new_model_fn=lambda i: "solved" if cand[i] else "failed",
        task_regression_tol=task_regression_tol,
    )
    return {"approved": decision.approved, "reason": decision.reason, "task_old": decision.task_old,
            "task_new": decision.task_new, "task_delta": decision.task_delta, "num_cases": len(ids)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = load(a.student)
    rows = [json.loads(line) for line in Path(a.rows).open()]
    labels = json.loads(Path(a.failures).read_text())["tasks"]
    examples = preference_examples(rows, recs, labels)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "preference_pairs.jsonl").open("w") as f:
        for ex in examples:
            f.write(json.dumps({"trajectory_id": ex.trajectory_id, "root_cause": ex.root_cause,
                                "original_failure_type": ex.original_failure_type,
                                "prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected}) + "\n")
    report = {"agenttune_detector": agenttune_detect(a.student), "agenttune_trigger": trigger_check(examples)}
    (out / "agenttune_loop.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=2))
