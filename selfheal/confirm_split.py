"""A second, larger held-out set for a single confirmatory evaluation.

It takes questions from the companies reserved for held-out use that were never
used anywhere: not in the train pool (other companies) and not in the first
held-out set of 100. Train-split and validation-split questions both qualify,
because no question of these companies was ever trained on.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .tasks import prepare_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="out/splits.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="out/splits_confirm.json")
    ap.add_argument("--exclude", nargs="*", default=[], help="earlier fresh sets whose questions must not be reused")
    a = ap.parse_args()

    splits = json.loads(Path(a.splits).read_text())
    train, val = prepare_data()
    # Rebuild the held-out company set exactly as make_splits did (same seed, same order).
    companies = sorted({r["company"] for r in train} | {r["company"] for r in val})
    random.Random(splits["seed"]).shuffle(companies)
    held_out = set(companies[: max(1, int(len(companies) * 0.3))])
    pool_companies = {r["company"] for r in splits["pool"]}
    assert {r["company"] for r in splits["eval"]} <= held_out and not (pool_companies & held_out), "company sets disagree"

    used_ids = {r["task_id"] for r in splits["eval"]}
    used_questions = {r["question"] for r in splits["eval"]} | {r["question"] for r in splits["pool"]}
    for path in a.exclude:
        earlier = json.loads(Path(path).read_text())["confirm"]
        used_ids |= {r["task_id"] for r in earlier}
        used_questions |= {r["question"] for r in earlier}
    fresh = [r for r in train + val
             if r["company"] in held_out and r["task_id"] not in used_ids and r["question"] not in used_questions]
    seen, unique = set(), []
    for r in fresh:  # one copy of each question text
        if r["question"] not in seen:
            seen.add(r["question"])
            unique.append(r)
    random.Random(a.seed).shuffle(unique)
    confirm = unique[: a.n]
    stats = {"held_out_companies": len(held_out), "fresh_questions_available": len(unique), "confirm_size": len(confirm),
             "from_train_split": sum(r["task_id"].startswith("train-") for r in confirm),
             "from_val_split": sum(r["task_id"].startswith("val-") for r in confirm),
             "companies_in_confirm": len({r["company"] for r in confirm}),
             "shared_companies_with_pool": len({r["company"] for r in confirm} & pool_companies),
             "shared_questions_with_pool_or_first_heldout": len({r["question"] for r in confirm} & used_questions)}
    Path(a.out).write_text(json.dumps({"confirm": confirm, "stats": stats}))
    print(json.dumps(stats, indent=2))
