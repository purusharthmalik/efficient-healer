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
produced by a two-round self-healing loop built on **AgentTune** and evaluated with **AuditKIT**, for the
Lexsi Labs hackathon, Track 3 (Agentic Fine-tuning).

- Code and full write-up: [github.com/purusharthmalik/efficient-healer](https://github.com/purusharthmalik/efficient-healer)
- All trajectories, labels, training sets and evaluation logs:
  [`Purusharth/efficient-healer-trajectories`](https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories)

## Which adapter is which

| Folder | What | Status |
|---|---|---|
| repository root | The adapter the deployment gate approved | = `round2/`. Deployed after a confirmatory evaluation on 400 held-out tasks |
| `round1/` | Round 1: trained on teacher repairs of the tasks base Tiny Aya could not solve | passed the gate against base Tiny Aya (0.5% → 25.2% success) |
| `round2/` | Round 2: trained from the base model on the round 1 adapter's own shortest verified runs plus repairs | held by the gate on a 100-task set, passed on the 400-task confirmation (25.2% → 30.4%) |
| `negative_all_token_loss/` | First attempt, loss on all tokens. Solves 2% of held-out runs. Kept as a negative result | not for use |

`deployed.json` records the gate's choice.

## Results

400 held-out questions from 55 companies never seen in training, 3 runs each, harness stop rule on.
Success is decided by a numeric verifier, not a judge model.

| Metric | Base Tiny Aya | Round 1 adapter | Final adapter |
|---|---|---|---|
| Rollouts solved (tolerant) | 6 of 1200 (0.5%) | 302 of 1200 (25.2%) | 365 of 1200 (30.4%) |
| Rollouts solved (strict) | 0.3% | 15.0% | 17.6% |
| Tasks solved in all 3 runs | 0 of 400 | 35 of 400 | 61 of 400 |
| Tasks solved in at least 1 run | 6 of 400 | 168 of 400 | 187 of 400 |
| Tool calls per run | 2.88 | 4.65 | 4.47 |
| Tool calls per solved run | 2.00 | 4.09 | 4.01 |
| Output tokens per run | 221 | 239 | 215 |
| Seconds per run | 5.1 | 5.8 | 5.5 |
| Runs with no `<answer>` tag | 505 of 1200 | 183 of 1200 | 117 of 1200 |

- Base → final adapter: success +29.9 points, 95% paired bootstrap interval [+26.2, +33.7].
- Round 1 → final adapter: success +5.2 points [+2.5, +8.1]; redundant tool-call rate 0.108 → 0.069
  (AuditKIT); output tokens per run −24 [−34, −15].
- On the first held-out set (100 tasks × 3), without any harness stop rule: base 0 of 300 runs solved,
  final adapter 75 of 300; output tokens per run 620 → 213.

Full tables, intervals and negative results are in the write-up.

## How it was made

1. Roll out the current model 4 times on each of 480 train-pool tasks (AgentTune rollout loop, FinQA tools).
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

- Research artifact. About seven in ten held-out runs still fail; do not use it for real financial decisions.
- English FinQA-style questions over the bundled tables only. Other languages, other schemas and general
  chat were not evaluated after fine-tuning, and neither was safety behaviour.
- Non-commercial: the base model is CC-BY-NC-4.0, and so is this adapter.
