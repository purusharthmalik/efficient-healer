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
| Write-up | https://github.com/purusharthmalik/efficient-healer (README) |
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

## Abstract (380 words, limit 500)

Small open models are cheap to host but unreliable at multi-step tool use. We take
`CohereLabs/tiny-aya-global` (3.35B), which solves 0.5% of FinQA database-agent tasks out of the box, and
improve it with a self-healing loop built from AgentTune and AuditKIT.

Each round rolls the current model out four times per task and buckets tasks by how often they are solved.
For tasks the model can solve, it keeps the shortest verified run. For the rest it repairs: failures are
labelled by rules and by AgentTune's `FailureClassifier`, retried with a hint chosen by the label, and
finally handed to a teacher model. Only runs that pass a numeric verifier and an evidence guard become
training data. One LoRA is then trained from the base model with AgentTune, and a deployment gate decides
whether it replaces the deployed model.

Three rounds move success on 400 held-out questions, from companies never seen in training, from 0.5% to
25.2% (round 1, repairs only, 480 training tasks), 30.4% (round 2, same tasks, 61% of targets now the
model's own shortest verified runs) and 49.1% (round 3, the same loop on 1,400 tasks). Tasks solved in
all three runs go from 0 to 143 of 400, and the redundant tool-call rate falls from 0.126 to 0.051. Every
difference carries a 95% paired bootstrap interval from AuditKIT.

What we learned. Where the loss goes mattered most at the start: loss on all tokens gave 2% success, loss
on the agent's own turns gave 19.7% with the same data. More data through the same loop was the largest
later gain (+18.7 points). The gate earned its place: it held the round 2 adapter on a 100-task set where
its gain was not provable and released it on a pre-declared 400-task confirmation; AgentTune's default
gate would have shipped it both times. Further experiments on the same day show that the loop's data
transfers to a larger Cohere student (Aya Expanse 8B: 38.9% against Tiny Aya's 30.4% on identical data),
that DPO on step-level preference pairs adds 5 points of success but no efficiency, and that the adapter
does not measurably change GSM8K accuracy or refusal behaviour on XSTest.

The whole project cost about $18 of GPU time. All trajectories, labels, training sets, preference pairs
and evaluation logs are public.

## For a Hugging Face article (optional)

Open https://huggingface.co/new-blog while signed in as `Purusharth`, paste the whole of `README.md`, and
publish. The README is written to stand alone as the article.
