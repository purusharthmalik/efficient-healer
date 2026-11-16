---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
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

# Efficient Healer: a Tiny Aya LoRA for a finance database agent

A LoRA adapter (r=16) for [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)
that turns it into a tool-calling agent for questions about company financial tables (FinQA). It was
produced by a three-round self-healing loop built on **AgentTune** and evaluated with **AuditKIT**, for the
Lexsi Labs hackathon, Track 3 (Agentic Fine-tuning).

- Code and full write-up: [github.com/purusharthmalik/efficient-healer](https://github.com/purusharthmalik/efficient-healer)
- All trajectories, labels, training sets and evaluation logs:
  [`Purusharth/efficient-healer-trajectories`](https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories)

## Which adapter is which

| Folder | What | Status |
|---|---|---|
| repository root | The adapter the deployment gate approved | = `round3/` |
| `round1/` | Round 1: teacher repairs of the tasks base Tiny Aya could not solve (480-task pool) | passed its gate against base Tiny Aya (0.5% → 25.2%) |
| `round2/` | Round 2: the round 1 adapter's own shortest verified runs plus repairs (480-task pool) | held on a 100-task set, passed on a 400-task confirmation (25.2% → 30.4%) |
| `round3/` | Round 3: the same loop from the round 2 adapter on a 1,400-task pool | passed its gate on a fresh 400-task set (26.2% → 46.4%) |
| `round2_dpo/`, `round3_dpo/` | DPO on step-level preference pairs. **Not standalone**: each is a second LoRA that goes on top of the merged round 2 or round 3 model | experiments |
| `negative_all_token_loss/` | First attempt, loss on all tokens. Solves 2% of held-out runs. Kept as a negative result | not for use |

`deployed.json` records the gate's choice. A sibling adapter for the larger student is at
[`Purusharth/aya-expanse-8b-efficient-healer-lora`](https://huggingface.co/Purusharth/aya-expanse-8b-efficient-healer-lora).

## Results

400 held-out questions from 55 companies never seen in training, 3 runs each, harness stop rule on.
Success is decided by a numeric verifier, not a judge model.

| Metric | Base Tiny Aya | Round 1 | Round 2 | Round 3 (root) |
|---|---|---|---|---|
| Runs solved | 0.5% | 25.2% | 30.4% | **49.1%** |
| Runs solved, strict exact match | 0.3% | 15.0% | 17.6% | **29.8%** |
| Tasks solved in all 3 runs | 0 of 400 | 35 of 400 | 61 of 400 | **143 of 400** |
| Tool calls per run | 2.88 | 4.65 | 4.47 | 4.32 |
| Output tokens per run | 221 | 239 | 215 | 210 |
| Redundant tool-call rate (AuditKIT) | 0.126 | 0.108 | 0.069 | **0.051** |
| Loop-free share (AuditKIT) | 0.761 | 0.805 | 0.818 | **0.893** |

- Round 2 → round 3: success +18.7 points, 95% paired bootstrap interval [+15.2, +22.4]. On a second
  held-out set of 400 fresh questions, used for the round 3 gate: 26.2% → 46.4%, +20.2 points [+16.8, +23.7].
- Retention and safety (AuditKIT, 100 samples each, round 2 adapter vs base): GSM8K 0.71 → 0.67
  (−0.04 [−0.12, +0.05]); XSTest refusal of unsafe prompts 0.94 → 0.94; over-refusal of safe prompts
  0.06 → 0.06. Multilingual ability was not measured, and round 3 was not re-checked.

Full tables, intervals and negative results are in the write-up.

## How it was made

1. Roll out the current model 4 times on each train-pool task (480 tasks in rounds 1 and 2, 1,400 in round 3;
   AgentTune rollout loop, FinQA tools).
2. Bucket each task: always solved, sometimes solved, never solved.
3. Always or sometimes solved: keep the shortest verified run.
4. Never solved: label the failure (rules + AgentTune `FailureClassifier`), retry with a hint chosen by the
   label, then ask a teacher (`nvidia/Qwen3.6-35B-A3B-NVFP4`). Only verified runs are kept.
5. Train one LoRA from the base model on the kept runs (AgentTune `create_distill_trainer`), with the loss
   on the agent's own turns only.
6. Deploy only if the gate passes on held-out tasks: success and tool-call validity must not regress, and
   success or tool calls on tasks both models solve must improve with a 95% interval that excludes zero.

Training: LoRA r=16, α=32, dropout 0.05 on `q_proj, k_proj, v_proj, o_proj`; 3 epochs; learning rate 2e-4,
cosine; effective batch 16; bf16; rows up to about 4,300 tokens.

## How to use

The adapter expects the prompt format it was trained on: AgentTune's tool-prompt fallback for Aya (tool
schemas in the system prompt, calls written as `[{"tool_name": ..., "parameters": {...}}]`, tool results
returned as user turns). The path we ran and tested is the one in the code repository: merge the adapter,
serve the merged model with vLLM, and run the agent through AgentTune's rollout loop.

```bash
# in the AgentTune environment of the code repository
hf download Purusharth/tiny-aya-efficient-healer-lora --local-dir adapter
python -m selfheal.merge --adapter adapter --out merged
# in the vLLM environment
vllm serve merged --served-model-name student --port 8000 --max-model-len 8192 --dtype bfloat16
# back in the AgentTune environment: 3 runs on each held-out task
python -m selfheal.tasks --n-pool 480 --n-eval 100 --out splits.json
python -m selfheal.rollout --tasks splits.json --split eval --k 3 --stop \
  --model-path merged --served-model student --out eval.jsonl --tag demo
python -m selfheal.summary eval.jsonl
```

The system prompt and the four tools are defined in `selfheal/tasks.py`.

## Limits

- Research artifact. About half of held-out runs still fail; do not use it for real financial decisions.
- English FinQA-style questions over the bundled tables only. Other languages, other schemas and general
  chat were not evaluated after fine-tuning, and neither was safety behaviour.
- Non-commercial: the base model is CC-BY-NC-4.0, and so is this adapter.
