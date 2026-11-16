---
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- agent
- tool-calling
- self-healing
- finqa
- agenttune
- auditkit
datasets:
- Purusharth/efficient-healer-trajectories
---

# Efficient Healer: an Aya Expanse 8B LoRA for a finance database agent

A LoRA adapter (r=16) for [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b),
trained on a training set that a self-healing loop produced with a *smaller* student,
[`CohereLabs/tiny-aya-global`](https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora). It answers
the question "does the loop's data transfer to another Cohere model, and does capacity matter?".

Built with **AgentTune** and evaluated with **AuditKIT** for the Lexsi Labs hackathon, Track 3.

- Code and full write-up: [github.com/purusharthmalik/efficient-healer](https://github.com/purusharthmalik/efficient-healer)
- Data: [`Purusharth/efficient-healer-trajectories`](https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories)
  (the training rows are `round2/dataset/rows.jsonl`)

## Results

Held-out questions from 55 companies never seen in training, 400 tasks × 3 runs per set, harness stop rule on.
Success is decided by a numeric verifier, not a judge model.

| Model, trained on the same 402 rows | Set A, runs solved | Set B, runs solved | Tool calls per run (set A) | Redundant-call rate (set A) |
|---|---|---|---|---|
| Untrained Aya Expanse 8B | 0 of 300 (first 100 tasks) | — | 4.07 | — |
| Tiny Aya (3.35B), round 2 adapter | 30.4% | 26.2% | 4.47 | 0.069 |
| **Aya Expanse 8B, this adapter** | **38.9%** | **36.5%** | **4.05** | **0.019** |

Against the Tiny Aya adapter trained on identical data: +8.5 points on set A (95% paired bootstrap interval
[+4.5, +12.6]) and +10.3 points on set B ([+6.7, +13.9]).

Note that Tiny Aya's *round 3* adapter, trained on a three times larger set from the same loop, reaches
49.1% on set A. This Expanse adapter has not been through round 3.

## How it was made

The training set is round 2 of the loop described in the write-up: for each of 480 tasks, the shortest
verified run of the student where it could solve the task, otherwise a verified repair (a hinted retry, or
a run by a teacher model). Training: AgentTune `create_distill_trainer`, LoRA r=16, α=32, dropout 0.05 on
`q_proj, k_proj, v_proj, o_proj`, 3 epochs, learning rate 2e-4 cosine, effective batch 16, bf16, loss on the
agent's own turns only. 24 minutes on one RTX PRO 6000.

## How to use

Same as the Tiny Aya adapter: merge, serve with vLLM, run through AgentTune's rollout loop with the tools
and system prompt in `selfheal/tasks.py` of the code repository, passing
`--base CohereLabs/aya-expanse-8b` to `selfheal.merge`.

## Limits

- Research artifact. About six in ten held-out runs still fail; do not use it for real financial decisions.
- English FinQA-style questions over the bundled tables only. No retention or safety check was run on this adapter.
- Non-commercial: the base model is CC-BY-NC-4.0, and so is this adapter.
