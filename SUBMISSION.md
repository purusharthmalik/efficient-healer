# Submission sheet — copy these into the form at https://hackathon.lexsi.ai/submission/

Deadline: **17:29 IST today** (Nov 15, 23:59 AoE). You can submit more than once; the latest counts.
**If you already submitted the two-round version, submit again**: the title, claim and abstract changed.

| Form field | Value |
|---|---|
| Email | the email you registered with |
| Track | Track 3 — Agentic Fine-tuning |
| Problem statements | Self-healing agents; Efficient trajectories; Data and database agents; Agentic distillation (supporting) |
| Hackathon repos used | AgentTune, AuditKIT |
| Base model | `CohereLabs/tiny-aya-global` (also `CohereLabs/aya-expanse-8b` as a second student) |
| Other models | `nvidia/Qwen3.6-35B-A3B-NVFP4` (teacher and failure classifier only) |
| HF model collection | https://huggingface.co/collections/Purusharth/efficient-healer-models-6afaa8b6120d5e7cae162d33 |
| HF dataset collection | https://huggingface.co/collections/Purusharth/efficient-healer-datasets-6afaa8b70386ac5f12a8f637 |
| Write-up | https://huggingface.co/blog/Purusharth/efficient-healer (Hugging Face article; the GitHub README has the same text) |
| GitHub repo | https://github.com/purusharthmalik/efficient-healer |
| Model repo (Tiny Aya) | https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora |
| Model repo (Aya Expanse 8B) | https://huggingface.co/Purusharth/aya-expanse-8b-efficient-healer-lora |
| Dataset repo | https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories |

## Title

Efficient Healer: a Tiny Aya database agent that repairs itself, round after round

## One-line claim

A self-healing loop takes Tiny Aya from 0.5% to 49.1% task success on financial tables of companies it
never saw, in three rounds; a deployment gate decided every release, and held one adapter back until a
larger test proved it.

## Abstract (218 words, limit 500)

Small open models are cheap to host but unreliable at multi-step tool use. `CohereLabs/tiny-aya-global`
(3.35B) solves 0.5% of FinQA database-agent tasks out of the box. We improve it with a self-healing loop
built from AgentTune and AuditKIT.

Each round rolls the current model out four times per task. For tasks it can solve, we keep the shortest
verified run. For the rest, we label the failure, retry with a hint chosen by the label, and finally ask a
teacher model. Only runs that pass a numeric verifier and an evidence guard become training data. One LoRA
is trained on them, and a deployment gate decides whether it replaces the deployed model.

Three rounds move success on 400 held-out questions, from companies never seen in training, from 0.5% to
25.2%, 30.4% and 49.1%. Tasks solved in all three runs go from 0 to 143 of 400, and the redundant
tool-call rate falls from 0.126 to 0.051. Every difference carries a 95% paired bootstrap interval.

Three things mattered most: putting the loss on the agent's own turns (2% → 19.7% on the same data), more
tasks through the same loop (+18.7 points), and a gate that held one adapter back until a larger test
proved its gain. The project cost about $18 of GPU time, and all data is public.

## Hugging Face article

Published: https://huggingface.co/blog/Purusharth/efficient-healer. Its text matches the GitHub README.
