---
license: cc-by-nc-4.0
language:
- en
task_categories:
- question-answering
- table-question-answering
tags:
- agent
- tool-calling
- trajectories
- self-healing
- finqa
- agenttune
- auditkit
pretty_name: Efficient Healer trajectories
---

# Efficient Healer: trajectories, failure labels, training sets and evaluation logs

Everything produced by the two-round self-healing run behind
[`Purusharth/tiny-aya-efficient-healer-lora`](https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora),
a LoRA that turns `CohereLabs/tiny-aya-global` into a finance database agent. Code and write-up:
[github.com/purusharthmalik/efficient-healer](https://github.com/purusharthmalik/efficient-healer).

Built for the Lexsi Labs hackathon, Track 3 (Agentic Fine-tuning), with AgentTune and AuditKIT.

## Headline

On 400 held-out questions from companies never seen in training (3 runs each), task success goes from
0.5% (base Tiny Aya) to 25.2% (round 1 adapter) to 30.4% (final adapter). All 15,000 student runs and
1,600 teacher runs behind that are here, successes and failures alike.

## What is here

| Path | Contents |
|---|---|
| `splits.json` | The 480 train-pool tasks and 100 held-out tasks (FinQA), company-disjoint |
| `splits_confirm.json` | 400 more held-out tasks from the same 55 held-out companies, used once for confirmation |
| `round1/student.jsonl` | Base Tiny Aya, 480 tasks × 4 rollouts |
| `round1/failures.json` | Failure labels: rule label and AgentTune `FailureClassifier` root cause |
| `round1/repair/` | Hint chosen for every failing task |
| `round1/hint_retry.jsonl` | Hinted retries by the student |
| `round1/teacher.jsonl` | Teacher rollouts (`nvidia/Qwen3.6-35B-A3B-NVFP4`), 3 per task |
| `round1/dataset/rows.jsonl` | Round 1 training set, in Tiny Aya's own conversation format |
| `round1/dataset/preference_pairs.jsonl` | The same repairs as (prompt, chosen, rejected), ready for DPO |
| `round1/dataset/agenttune_loop.json` | AgentTune `FailureDetector` and `RetrainingTrigger` output |
| `round2/…` | The same files for round 2, where the student is the round 1 adapter |
| `eval/*.jsonl` | Held-out rollouts (100 tasks × 3) for the base model and both adapters, with and without the harness stop rule |
| `eval/confirm_*.jsonl` | The confirmatory evaluation: 400 tasks × 3 for the base model and both adapters |
| `eval/cmp_*/` | AuditKIT results and paired-bootstrap comparisons for every pair of models |
| `eval/gate_round1.json`, `eval/gate_round2.json` | Gate decisions on the first held-out set (round 2: hold) |
| `eval/gate_round1_confirm.json`, `eval/gate_round2_confirm.json` | Gate decisions on the confirmatory set (round 2: deploy) |
| `negative_all_token_loss/` | The first adapter's evaluation, trained with the loss on all tokens (it failed) |
| `pilot/` | Pilot runs: Tiny Aya and Aya Expanse 8B with and without the stop rule, and the teacher |
| `worker_logs/` | Console log of every pipeline step |

## Record format

Every rollout file is JSON Lines in AgentTune's portable trajectory format (`Trajectory.to_dict()`), which
AuditKIT reads directly with `auditkit.episodes_from_agenttune(path)`:

- `task`, `steps[]` (each with `action.tool_calls[]` and `observation`), `final_response`, `reward`
- `metadata.conversation`: the exact messages the model saw and wrote
- `metadata.selfheal`: `task_id`, `trial`, `tag`, `gold`, `success` (tolerant verifier), `strict`
  (AgentTune `numerical_match_reward`), `verify_rule`, `pred`, `n_tool_calls`, `n_tool_failures`,
  `prompt_tokens`, `completion_tokens`, `seconds`, `hint`, `harness_stop`

Training rows (`rows.jsonl`) have `messages`, `source` (`efficiency`, `reliability`, `repair_hint`,
`repair_teacher`), `round`, `task_id` and `n_tool_calls`.

## Source data and licence

Questions, answers and tables come from FinQA as packaged in `rLLM/rLLM-FinQA-Dataset` and loaded through
AgentTune. Student outputs come from `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0), teacher outputs from
`nvidia/Qwen3.6-35B-A3B-NVFP4` (Apache-2.0). This dataset is released under CC-BY-NC-4.0.

Language: English. No personal data; the tables are public company filings.
