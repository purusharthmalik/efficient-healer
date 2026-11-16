"""Deployment gate: decides whether a new adapter replaces the currently deployed model.

Rule ("improve, never regress"), applied once per round on the
held-out tasks, with both models paired by task id:
  1. success rate does not fall by more than 2 points,
  2. tool-call validity does not fall by more than 2 points,
  3. something really improves: the success rate, or the tool calls on tasks both
     models solve, with a 95% paired-bootstrap interval that excludes zero.
AgentTune's own DeploymentGate must also approve (it blocks a task-accuracy regression).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SUCCESS_MARGIN = 0.02
VALIDITY_MARGIN = 0.02


@dataclass(frozen=True)
class EvalSummary:
    n_tasks: int  # held-out tasks, the same ids for both models
    k: int  # rollouts per task
    success_rate: float  # mean over tasks of (successes / k)
    pass_all_k: float  # share of tasks solved in all k rollouts
    mean_steps_both_solved: float | None  # mean tool calls, only on tasks BOTH models solve
    tool_call_validity: float  # share of calls that match the tool schema
    redundant_call_rate: float  # duplicate calls / all calls


@dataclass(frozen=True)
class GateDecision:
    deploy: bool
    reasons: list[str]  # one line per check, passed or failed, for the audit log


def gate_decision(
    base: EvalSummary,
    cand: EvalSummary,
    success_delta_ci: tuple[float, float],
    steps_delta_ci: tuple[float, float] | None = None,
) -> GateDecision:
    """Both intervals are 95% paired-bootstrap intervals of (candidate - base).
    steps_delta_ci is None when fewer than two tasks are solved by both models."""
    reasons = []

    def check(name: str, passed: bool, detail: str) -> bool:
        reasons.append(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
        return passed

    no_success_drop = check(
        "success does not regress",
        cand.success_rate >= base.success_rate - SUCCESS_MARGIN,
        f"{base.success_rate:.3f} -> {cand.success_rate:.3f} (allowed drop {SUCCESS_MARGIN})",
    )
    no_validity_drop = check(
        "tool-call validity does not regress",
        cand.tool_call_validity >= base.tool_call_validity - VALIDITY_MARGIN,
        f"{base.tool_call_validity:.3f} -> {cand.tool_call_validity:.3f} (allowed drop {VALIDITY_MARGIN})",
    )
    success_up = success_delta_ci[0] > 0
    steps_down = steps_delta_ci is not None and steps_delta_ci[1] < 0
    steps_text = f"{steps_delta_ci}" if steps_delta_ci is not None else "not measurable (fewer than 2 tasks solved by both)"
    improved = check(
        "something improves",
        success_up or steps_down,
        f"success delta 95% interval {success_delta_ci} ({'above' if success_up else 'not above'} zero); "
        f"tool calls on tasks both solve, delta 95% interval {steps_text}",
    )
    return GateDecision(deploy=no_success_drop and no_validity_drop and improved, reasons=reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="report.json written by selfheal.evaluate")
    ap.add_argument("--base", required=True, help="held-out rollouts of the currently deployed model")
    ap.add_argument("--cand", required=True, help="held-out rollouts of the new adapter")
    ap.add_argument("--label", required=True, help="e.g. 'round 1: v1 vs base'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from .agenttune_loop import agenttune_gate
    from .summary import load, sh

    report = json.loads(Path(a.report).read_text())
    tl = report["task_level"]
    base_recs, cand_recs = load([a.base]), load([a.cand])
    k = max(sh(r)["trial"] for r in base_recs) + 1

    def summary(side: str, headline: dict) -> EvalSummary:
        return EvalSummary(
            n_tasks=tl["success_rate"]["n_tasks"],
            k=k,
            success_rate=tl["success_rate"][side],
            pass_all_k=tl["solved_in_all_k"][side],
            mean_steps_both_solved=tl["tool_calls_on_tasks_both_solve"][side],
            tool_call_validity=float(headline.get("tool_call_validity") or 0.0),
            redundant_call_rate=float(headline.get("redundant_tool_calls") or 0.0),
        )

    base, cand = summary("base", report["auditkit_base"]), summary("cand", report["auditkit_cand"])
    steps_ci = tl["tool_calls_on_tasks_both_solve"]["ci95"]
    ours = gate_decision(base, cand, tuple(tl["success_rate"]["ci95"]), tuple(steps_ci) if steps_ci else None)
    agenttune = agenttune_gate(base_recs, cand_recs)
    deploy = ours.deploy and agenttune["approved"]
    result = {
        "label": a.label,
        "deploy": deploy,
        "our_rule": {"deploy": ours.deploy, "reasons": ours.reasons},
        "agenttune_deployment_gate": agenttune,
        "base": asdict(base),
        "candidate": asdict(cand),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
