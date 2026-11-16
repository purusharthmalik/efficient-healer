"""FinQA tasks, tools and the train-pool / held-out split.

The split is company-disjoint: no company in the held-out set appears in the
train pool, so the agent never trains on tables it is later tested on.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from agenttune.agentic.tools.builtin.finqa_tool import (
    build_finqa_db,
    build_finqa_tools,
    calculator,
    load_finqa_datasets,
)
from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool

DATA_DIR = Path("data/data")
DB_PATH = Path("finqa.db")

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer questions about a company's financial tables "
    "by calling the tools. Call one tool at a time and wait for its result. "
    "When you know the answer, reply with only the final number inside <answer></answer> tags."
)


def prepare_data() -> tuple[list[dict], list[dict]]:
    """Download FinQA once, build the SQLite database once, return (train, val) rows."""
    train, val = load_finqa_datasets(DATA_DIR)
    if not DB_PATH.exists():
        build_finqa_db(str(DB_PATH), DATA_DIR / "company_tables")

    def rows(ds, split):
        return [
            {
                "task_id": f"{split}-{r['question_id']}",
                "prompt": r["prompt"],
                "answer": str(r["answer"]),
                "company": r["company"],
                "question_type": r["question_type"],
                "question": r["prompt"][0]["content"],
            }
            for r in ds
        ]

    return rows(train, "train"), rows(val, "val")


def make_tools() -> list:
    """A fresh tool set with its own database handle (one per worker thread)."""
    db_tool = SQLDatabaseTool(f"sqlite:///{DB_PATH.resolve()}")
    list_tables, get_table_schema, query_finqa_tables = build_finqa_tools(db_tool)
    return [list_tables, get_table_schema, query_finqa_tables, calculator]


def make_splits(n_pool: int, n_eval: int, seed: int = 0, eval_company_share: float = 0.3) -> dict:
    train, val = prepare_data()
    rng = random.Random(seed)
    companies = sorted({r["company"] for r in train} | {r["company"] for r in val})
    rng.shuffle(companies)
    eval_companies = set(companies[: max(1, int(len(companies) * eval_company_share))])

    eval_rows = [r for r in val if r["company"] in eval_companies]
    pool_rows = [r for r in train if r["company"] not in eval_companies]
    rng.shuffle(eval_rows)
    rng.shuffle(pool_rows)
    eval_rows, pool_rows = eval_rows[:n_eval], pool_rows[:n_pool]

    # Leak check: no shared company and no shared question text.
    shared_companies = {r["company"] for r in eval_rows} & {r["company"] for r in pool_rows}
    shared_questions = {r["question"] for r in eval_rows} & {r["question"] for r in pool_rows}
    assert not shared_companies and not shared_questions, "split leaks"

    return {
        "seed": seed,
        "pool": pool_rows,
        "eval": eval_rows,
        "stats": {
            "companies_total": len(companies),
            "companies_eval_only": len(eval_companies),
            "train_rows": len(train),
            "val_rows": len(val),
            "pool_size": len(pool_rows),
            "eval_size": len(eval_rows),
            "pool_question_types": dict(Counter(r["question_type"] for r in pool_rows)),
            "eval_question_types": dict(Counter(r["question_type"] for r in eval_rows)),
            "shared_companies": len(shared_companies),
            "shared_questions": len(shared_questions),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=240)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/splits.json")
    a = ap.parse_args()
    splits = make_splits(a.n_pool, a.n_eval, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(splits))
    print(json.dumps(splits["stats"], indent=2))
    ex = splits["pool"][0]
    print("--- example task ---")
    print(ex["question"])
    print("gold:", ex["answer"])
    print("gold examples:", [r["answer"] for r in splits["pool"][:25]])
