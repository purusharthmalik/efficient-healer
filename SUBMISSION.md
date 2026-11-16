# Submission sheet — copy these into the form at https://hackathon.lexsi.ai/submission/

Deadline: **17:29 IST today** (Nov 15, 23:59 AoE). You can submit more than once; the latest counts.

| Form field | Value |
|---|---|
| Email | the email you registered with |
| Track | Track 3 — Agentic Fine-tuning |
| Problem statements | Self-healing agents; Efficient trajectories; Data and database agents; Agentic distillation (supporting) |
| Hackathon repos used | AgentTune, AuditKIT |
| Base model | `CohereLabs/tiny-aya-global` |
| Other models | `nvidia/Qwen3.6-35B-A3B-NVFP4` (teacher and failure classifier only) |
| HF model collection | created when the repos go public; the link is added here |
| HF dataset collection | created when the repos go public; the link is added here |
| Write-up | https://github.com/purusharthmalik/efficient-healer (README) |
| GitHub repo | https://github.com/purusharthmalik/efficient-healer |
| Model repo | https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora |
| Dataset repo | https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories |

## Title

Efficient Healer: a Tiny Aya database agent that repairs itself, then gets shorter

## One-line claim

A two-round self-healing loop takes Tiny Aya from 0.5% to 30.4% task success on financial tables of
companies it never saw; a deployment gate held the second adapter back until a larger test proved it.

## Abstract (339 words, limit 500)

Small open models are cheap to host but unreliable at multi-step tool use. We take
`CohereLabs/tiny-aya-global` (3.35B), which solves 0.5% of FinQA database-agent tasks out of the box, and
improve it with a self-healing loop built from AgentTune and AuditKIT.

Each round rolls the current model out four times per task and buckets tasks by how often they are solved.
For tasks the model can solve, it keeps the shortest verified run. For the rest it repairs: failures are
labelled by rules and by AgentTune's `FailureClassifier`, retried with a hint chosen by the label, and
finally handed to a teacher model. Only runs that pass a numeric verifier and an evidence guard become
training data. One LoRA is then trained from the base model with AgentTune, and a deployment gate decides
whether it replaces the deployed model.

Two rounds on 480 training tasks move success on 400 held-out questions, from companies never seen in
training, from 0.5% to 25.2% (round 1, repairs only) and then to 30.4% (round 2). In round 2, 61% of the
training targets are the model's own shortest verified runs, and the teacher's share falls from 95% to
38%. Round 2 also lowers the redundant tool-call rate from 0.108 to 0.069 and output tokens per run from
239 to 215. Every difference carries a 95% paired bootstrap interval from AuditKIT.

Three things we learned. First, where the loss goes mattered more than anything else: with the loss on
all tokens the adapter reached 2% success, and with the loss on the agent's own turns, same data and
settings, 19.7%. Second, the gate earned its place: it held the round 2 adapter on a 100-task set where
its gain was not provable, and released it on a pre-declared 400-task confirmation; AgentTune's default
gate would have shipped it both times. Third, hints chosen by failure labels repair a trained model
(39 tasks) but not an untrained one (8 tasks).

The whole run cost $8.57 of GPU time. All trajectories, labels, training sets and evaluation logs are
public.

## For a Hugging Face article (optional)

Open https://huggingface.co/new-blog while signed in as `Purusharth`, paste the whole of `README.md`, and
publish. The README is written to stand alone as the article.
