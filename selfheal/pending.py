"""List the tasks that still have no grounded, verified success in the given rollout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .buckets import pick_target
from .summary import by_task, load, sh

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--split", default="pool")
    ap.add_argument("--have", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    all_ids = [t["task_id"] for t in json.loads(Path(a.tasks).read_text())[a.split]]
    groups = by_task(load([p for p in a.have if Path(p).exists()]))
    covered = {t for t, g in groups.items() if pick_target([r for r in g if sh(r)["success"]], True)}
    pending = [t for t in all_ids if t not in covered]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(pending))
    print(f"[pending] {len(pending)} of {len(all_ids)} tasks have no grounded success yet -> {a.out}")
