"""Fine-tune one LoRA adapter on the selected trajectories with AgentTune.

AgentTune's `create_distill_trainer` drives the run (TRL SFTTrainer + PEFT LoRA,
with its fixes for Cohere chat templates). The only change is the row format:
AgentTune's default rows write tool calls as `<tool_call>{...}</tool_call>`,
which is not what its Aya prompt asks the model to produce, so we feed the
student-format conversations built in sft_data.py instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agenttune import create_distill_trainer
from agenttune.agentic.distill_trainer import _DefaultSFTTrainerFactory
from agenttune.agentic.events import EventLog
from agenttune.agentic.trajectory.dataset import TrajectoryDataset


TURN_START = "<|START_OF_TURN_TOKEN|>"


def encode_agent_turns_only(messages: list[dict], tok) -> dict:
    """Tokenise one conversation with labels only on the tokens the agent writes:
    each assistant turn's content and the end-of-turn tokens that follow it.
    Everything else (system prompt, task, tool results) is context, label -100."""
    text = tok.apply_chat_template(messages, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, pos = [], 0
    for m in messages:
        start = text.find(m["content"], pos)
        if start == -1:
            continue
        end = start + len(m["content"])
        if m["role"] == "assistant":
            nxt = text.find(TURN_START, end)
            spans.append((start, nxt if nxt != -1 else len(text)))
        pos = end
    labels = [
        tid if any(a <= s < b for a, b in spans) else -100
        for tid, (s, _e) in zip(enc["input_ids"], enc["offset_mapping"])
    ]
    return {"input_ids": enc["input_ids"], "labels": labels}


class StudentFormatFactory(_DefaultSFTTrainerFactory):
    """AgentTune's SFT factory, fed our student-format rows. With agent_turns_only,
    the rows are tokenised here with a labels column, which TRL's SFTTrainer uses as is."""

    def __init__(self, rows: list[dict], agent_turns_only: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.agent_turns_only = agent_turns_only

    def __call__(self, *, model, train_dataset, **kwargs):
        if not self.agent_turns_only:
            return super().__call__(model=model, train_dataset=self.rows, **kwargs)
        import torch
        from agenttune.agentic.distill_trainer import _DEFAULT_LORA_TARGETS
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map=self.device_map)
        encoded = [encode_agent_turns_only(r["messages"], self.tokenizer) for r in self.rows]
        n_all = sum(len(e["input_ids"]) for e in encoded)
        n_lab = sum(sum(x != -100 for x in e["labels"]) for e in encoded)
        print(f"[train] loss on agent turns only: {n_lab} of {n_all} tokens carry a label "
              f"({n_lab / n_all:.1%}, {n_lab / len(encoded):.0f} per row)", flush=True)
        e0 = encoded[0]
        shown = self.tokenizer.decode([t for t in e0["labels"] if t != -100])
        print("[train] labelled text of row 0:", repr(shown[:400]), flush=True)
        peft_config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout, bias=self.lora_bias,
            task_type=self.task_type, target_modules=self.lora_targets or _DEFAULT_LORA_TARGETS,
        )
        self.sft_trainer = SFTTrainer(
            model=self.model, args=SFTConfig(**kwargs), train_dataset=Dataset.from_list(encoded),
            processing_class=self.tokenizer, peft_config=peft_config,
        )
        return self.sft_trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="JSONL of {messages, source, task_id}")
    ap.add_argument("--trajectories", required=True, help="AgentTune trajectory JSONL of the same targets")
    ap.add_argument("--student", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--out", default="out/adapter")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=0, help="0 = the 95th-percentile row length, up to --max-length-cap")
    ap.add_argument("--max-length-cap", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["agent_turns", "all_tokens"], default="agent_turns")
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.rows).open()]
    # A row cut off at max_length loses its final answer, so over-long rows are dropped, not truncated.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.student)
    lengths = [
        len(tok(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False)["input_ids"])
        for r in rows
    ]
    srt = sorted(lengths)
    if a.max_length == 0:
        a.max_length = min(srt[int(0.95 * (len(srt) - 1))], a.max_length_cap)
    if a.max_length > 4096 and a.batch_size > 2:  # same effective batch, half the memory per step
        a.grad_accum, a.batch_size = a.grad_accum * a.batch_size // 2, 2
    kept = [r for r, n in zip(rows, lengths) if n <= a.max_length]
    print(f"[train] token lengths: median {srt[len(srt) // 2]}, p95 {srt[int(0.95 * (len(srt) - 1))]}, max {srt[-1]}; "
          f"dropped {len(rows) - len(kept)} of {len(rows)} rows longer than {a.max_length}; "
          f"batch {a.batch_size} x accumulation {a.grad_accum}", flush=True)
    kept_ids = {r["trajectory_id"] for r in kept}
    rows = kept
    train_rows = [{"messages": r["messages"]} for r in rows]
    # Iterating a TrajectoryDataset yields TRL dicts; the Trajectory objects are in .trajectories.
    logs = [EventLog.from_trajectory(t) for t in TrajectoryDataset.from_jsonl(a.trajectories).trajectories
            if t.trajectory_id in kept_ids]
    print(f"[train] {len(train_rows)} rows, {len(logs)} AgentTune event logs", flush=True)

    # 5% warm-up, given in steps (this transformers version no longer takes warmup_ratio).
    total_steps = max(1, round(len(train_rows) * a.epochs / (a.batch_size * a.grad_accum)))
    warmup_steps = max(1, round(0.05 * total_steps))
    print(f"[train] {total_steps} optimizer steps, {warmup_steps} warm-up", flush=True)
    factory = StudentFormatFactory(train_rows, agent_turns_only=a.loss == "agent_turns",
                                   lora_r=a.lora_r, lora_alpha=a.lora_alpha)
    trainer = create_distill_trainer(
        a.student,
        trajectories=logs,
        trainer_factory=factory,
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        seed=a.seed,
    )
    trainer.train()
    factory.sft_trainer.save_model(a.out)
    factory.tokenizer.save_pretrained(a.out)
    history = [h for h in factory.sft_trainer.state.log_history if "loss" in h or "train_loss" in h]
    Path(a.out, "train_log.json").write_text(json.dumps(history, indent=1))
    Path(a.out, "train_rows.json").write_text(json.dumps(
        {"rows": len(rows), "by_source": {s: sum(r["source"] == s for r in rows) for s in {r["source"] for r in rows}},
         "args": vars(a)}, indent=1))
    print("[train] done; last log:", history[-1] if history else None, flush=True)


if __name__ == "__main__":
    main()
