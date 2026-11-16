"""Run k rollouts per task through AgentTune and write one JSONL line per rollout.

Each line is AgentTune's portable trajectory record (the format AuditKIT's
`episodes_from_agenttune` reads), plus our own fields under `metadata.selfheal`.
The run is resumable: (task, trial) pairs already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

from .engine import VLLMServerEngine
from .tasks import SYSTEM_PROMPT, make_tools
from .verify import verify


# A JSON call list is written on one line (newlines inside JSON strings are escaped),
# so the first raw newline after "]" marks the end of the tool call.
STOP_STRINGS = ["]\n", "</answer>"]


def _done_pairs(path: Path) -> set[tuple[str, int]]:
    done = set()
    if path.exists():
        for line in path.open():
            s = json.loads(line)["metadata"]["selfheal"]
            done.add((s["task_id"], s["trial"]))
    return done


def run_rollouts(
    tasks: list[dict],
    k: int,
    out_path: Path,
    engine: VLLMServerEngine,
    *,
    tag: str,
    max_steps: int = 8,
    concurrency: int = 16,
    hints: dict[str, str] | None = None,
    hint_label: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_pairs(out_path)
    jobs = [(t, i) for t in tasks for i in range(k) if (t["task_id"], i) not in done]
    print(f"[rollout:{tag}] {len(jobs)} rollouts to run ({len(done)} already done), concurrency {concurrency}", flush=True)

    write_lock = threading.Lock()
    local = threading.local()

    # One shared tool set: building it reflects every table in the database, which
    # is far too slow to repeat per thread. AgentTune's own threaded path shares tools too.
    t_tools = time.time()
    tools = make_tools()
    # The database handle is created lazily on the first call and reflects every
    # table. Do that once here; otherwise every thread starts its own reflection.
    tools[0]("warmup")
    print(f"[rollout:{tag}] tools ready in {time.time() - t_tools:.1f}s", flush=True)

    def rollout_fn_for(system_prompt: str):
        # One rollout function per thread and system prompt (the capture list is per thread).
        if not hasattr(local, "fns"):
            local.fns = {}
        if system_prompt not in local.fns:
            captured: list = []
            fn = create_rollout_fn(
                rollout_engine=engine,
                tools=tools,
                max_steps=max_steps,
                system_prompt=system_prompt,
                on_trajectory_end=captured.append,
                context_length=7600,
            )
            local.fns[system_prompt] = (fn, captured)
        return local.fns[system_prompt]

    def one(task: dict, trial: int) -> dict:
        hint = (hints or {}).get(task["task_id"])
        system_prompt = SYSTEM_PROMPT + (f"\n\nHint: {hint}" if hint else "")
        fn, captured = rollout_fn_for(system_prompt)
        captured.clear()
        engine.begin()
        t0 = time.time()
        extra = {
            "task_id": task["task_id"],
            "trial": trial,
            "tag": tag,
            "gold": task["answer"],
            "company": task["company"],
            "question_type": task["question_type"],
            "hint": hint,
            "hint_label": hint_label if hint else None,
            "max_steps": max_steps,
            "harness_stop": bool(engine.stop),
        }
        try:
            fn({"prompt": task["prompt"], "answer": task["answer"]})
            traj = captured[-1]
            rec = traj.to_dict()
            v = verify(traj.final_response, task["answer"])
            extra.update(
                success=v["success"],
                strict=v["strict"],
                verify_rule=v["rule"],
                pred=v["pred"],
                n_tool_calls=int(traj.metadata.get("tool_call_count", 0)),
                n_tool_failures=int(traj.metadata.get("tool_failure_count", 0)),
                n_steps=len(traj.steps),
                error=None,
            )
        except Exception as e:  # one broken rollout must not stop the run
            rec = {"task": task["question"], "steps": [], "final_response": "", "metadata": {}}
            extra.update(
                success=False, strict=False, verify_rule="rollout_error", pred=None,
                n_tool_calls=0, n_tool_failures=0, n_steps=0,
                error=f"{type(e).__name__}: {e}"[:500], traceback=traceback.format_exc()[-1500:],
            )
        extra.update(seconds=round(time.time() - t0, 2), **engine.usage())
        rec["trajectory_id"] = f"{task['task_id']}#t{trial}"
        rec["reward"] = 1.0 if extra["success"] else 0.0
        rec["metadata"]["selfheal"] = extra
        with write_lock, out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return extra

    t_start = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, t, i) for t, i in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            e = fut.result()
            n_ok += e["success"]
            n_err += e["error"] is not None
            if n % 25 == 0 or n == len(jobs):
                mins = (time.time() - t_start) / 60
                print(f"[rollout:{tag}] {n}/{len(jobs)} done | success {n_ok} | errors {n_err} | {n / max(mins, 1e-6):.1f} rollouts/min", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="splits.json, or a JSON list of tasks")
    ap.add_argument("--split", default=None, help="key inside splits.json: pool or eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-path", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--served-model", default="student")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--hints", default=None, help="JSON file: {task_id: hint text}")
    ap.add_argument("--hint-label", default=None)
    ap.add_argument("--only", default=None, help="JSON file with a list of task_ids to keep")
    ap.add_argument("--stop", action="store_true", help="stop each turn at the end of the first call list or answer tag")
    a = ap.parse_args()

    data = json.loads(Path(a.tasks).read_text())
    tasks = data[a.split] if a.split else data
    if a.only:
        keep = set(json.loads(Path(a.only).read_text()))
        tasks = [t for t in tasks if t["task_id"] in keep]
    if a.limit:
        tasks = tasks[: a.limit]
    hints = json.loads(Path(a.hints).read_text()) if a.hints else None

    engine = VLLMServerEngine(
        a.model_path, a.served_model, a.base_url,
        temperature=a.temperature, max_new_tokens=a.max_new_tokens,
        stop=STOP_STRINGS if a.stop else None,
    )
    run_rollouts(
        tasks, a.k, Path(a.out), engine,
        tag=a.tag, max_steps=a.max_steps, concurrency=a.concurrency,
        hints=hints, hint_label=a.hint_label,
    )


if __name__ == "__main__":
    main()
