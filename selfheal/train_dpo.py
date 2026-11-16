"""DPO on step-level preference pairs, with AgentTune's agentic DPO trainer (TRL DPOTrainer + LoRA).

The pairs are conversational: `prompt` is a message list ending on a user turn, `chosen` and
`rejected` are one assistant message each, so TRL applies the model's own chat template.
The base here is the merged round 2 model; DPO adds a second, small LoRA on top of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agenttune.core.backend_factory import create_agentic_trainer
from datasets import Dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True, help="merged round 2 model directory")
    ap.add_argument("--out", default="out/dpo/adapter")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = [json.loads(line) for line in Path(a.pairs).open()]
    data = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs])
    print(f"[dpo] {len(data)} pairs; lr {a.lr}, beta {a.beta}, LoRA r={a.lora_r}", flush=True)

    trainer = create_agentic_trainer(
        algorithm="dpo",
        backend="trl",
        model=a.model,
        train_dataset=data,
        output_dir=a.out,
        peft_config={"r": a.lora_r, "lora_alpha": 2 * a.lora_r, "lora_dropout": 0.05, "task_type": "CAUSAL_LM",
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        num_train_epochs=a.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=a.lr,
        beta=a.beta,
        max_length=a.max_length,
        truncation_mode="keep_end",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        seed=a.seed,
    )
    stats = trainer.train()
    print("[dpo] train stats:", json.dumps({k: v for k, v in (stats or {}).items() if isinstance(v, (int, float, str))})[:600], flush=True)
    inner = getattr(trainer, "trainer", None)
    if inner is not None:
        inner.save_model(a.out)
        history = [h for h in inner.state.log_history if "loss" in h or "train_loss" in h]
        Path(a.out, "train_log.json").write_text(json.dumps(history, indent=1))
        print("[dpo] last log:", history[-1] if history else None, flush=True)
    print("[dpo] files:", sorted(p.name for p in Path(a.out).glob("*"))[:12], flush=True)


if __name__ == "__main__":
    main()
