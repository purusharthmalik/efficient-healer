# Efficient Healer: a Tiny Aya database agent that repairs itself, round after round

**One-line claim.** A self-healing loop takes Tiny Aya from **0.5% to 49.1%** task success on financial tables of companies it never saw, in three rounds. A deployment gate decided every release, and held one adapter back until a larger test proved it.

Lexsi Labs hackathon, Track 3 (Agentic Fine-tuning)

## Tracks and problem statements covered

All four are in Track 3 (Agentic Fine-tuning). The numbers are the order on the hackathon's tracks page.

- Track 3, problem statement 8, **Self-healing agents** (main): detect failing runs, classify why they
  failed, generate corrective training examples, retrain, and gate redeployment on measured accuracy.
- Track 3, problem statement 2, **Efficient trajectories**: cut redundant and failed tool calls without
  lowering success.
- Track 3, problem statement 6, **Data and database agents**: an agent that answers finance questions by
  querying a database, rewarded on verified results.
- Track 3, problem statement 3, **Agentic distillation** (supporting): a stronger teacher's verified runs
  repair the tasks the student cannot solve.

**Content warning.** None needed. This write-up contains no harmful or NSFW text. The safety check uses
XSTest prompts; none are reproduced here.

## Abstract

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

### Headline before → after

400 held-out questions from 55 companies never seen in training, 3 runs each (1,200 runs per model):

| Metric | Base Tiny Aya | After round 1 | After round 2 | After round 3 (deployed) |
|---|---|---|---|---|
| Runs solved | 0.5% | 25.2% | 30.4% | **49.1%** |
| Runs solved, strict exact match | 0.3% | 15.0% | 17.6% | **29.8%** |
| Tasks solved in all 3 runs | 0 of 400 | 35 of 400 | 61 of 400 | **143 of 400** |
| Tasks solved in at least 1 run | 6 of 400 | 168 of 400 | 187 of 400 | **249 of 400** |
| Redundant tool-call rate (AuditKIT) | 0.126 | 0.108 | 0.069 | **0.051** |
| Loop-free share (AuditKIT) | 0.761 | 0.805 | 0.818 | **0.893** |
| Tool-call validity (AuditKIT) | 0.997 | 1.000 | 1.000 | 1.000 |
| Runs that end without a tagged answer | 505 of 1,200 | 183 of 1,200 | 117 of 1,200 | **94 of 1,200** |

- Base → round 3: +48.6 points. Round 2 → round 3: +18.7 points, 95% interval [+15.2, +22.4].
- On a second held-out set of 400 fresh questions, used for the round 3 gate: 26.2% → 46.4%, +20.2 points [+16.8, +23.7].
- Without any harness stop rule (first held-out set): output tokens per run 620 → 213 from base to round 2.

### Base model and method in one line

`CohereLabs/tiny-aya-global` (3.35B) + one LoRA (r=16), trained with
AgentTune on verified trajectories selected by a three-round self-healing loop, evaluated with AuditKIT.

### For judges: one minute

> - **What:** a self-healing loop for a tool-calling agent, run for three rounds, with a gate before each deploy.
> - **Kits:** AgentTune runs the rollouts, tools, failure labels, training, DPO and the gate. AuditKIT
>   produces every metric and interval, and the retention check.
> - **Result:** 0.5% → 49.1% success on 400 unseen-company questions; tasks solved in all 3 runs 0 → 143;
>   redundant tool calls down by 60%.
> - **Most useful finding:** AgentTune's default trainer puts the loss on all tokens, which gave 2% success.
>   Loss on the agent's turns gave 19.7% with nothing else changed.
> - **Honesty checks:** numeric verifier, no LLM judge; company-disjoint test sets; a guard against lucky
>   answers; negative results published.
> - **Weakness:** half of the runs still fail.

## Problem statement and approach

A small open model that calls tools badly wastes money twice: it fails the task, and it burns tokens while
failing. We wanted one cheap adapter that fixes both, and a rule that stops a bad adapter from shipping.

The loop, for every task in a train pool:

1. **Roll out** the current model k=4 times with AgentTune's rollout loop.
2. **Bucket** the task by how often it was solved: always, sometimes, never.
3. **Always or sometimes solved:** keep the *shortest* verified run (fewest tool calls, then fewest tokens).
4. **Never solved:** label the failure (rules plus AgentTune's `FailureClassifier`), then repair it: first a
   retry of the same model with a hint chosen by the label, then a run by a stronger teacher. Every repair
   is checked by the answer verifier; nothing unverified becomes training data.
5. **Retrain** one LoRA from the base model on all selected runs.
6. **Gate**: the new adapter replaces the deployed model only if it improves and does not regress on
   held-out tasks.

We run the loop three times. Untrained Tiny Aya solves under 1% of tasks, so in round 1 almost everything
is repair. Round 2 rolls out the round 1 adapter; now many tasks are solved reliably, and the shortest-run
selection has material to work on. Round 3 repeats the loop from the round 2 adapter on a pool grown from
480 to 1,400 tasks.

Two guards keep "shortest success" honest:
- **Evidence guard.** A run only qualifies if a database call really returned table data and no assistant
  turn writes a tool result itself. Without it, the shortest "success" is a lucky answer on invented data.
- **Failures are never ranked by length.** A short wrong run is not a target.

### Why this problem / real use case

Finance teams ask questions that live in tables: changes, shares, growth rates. A closed frontier model
answers them well but is priced per token; a 3B open model is cheap to host but unreliable at multi-step
tool use. Closing that gap with a self-improving loop, and with a deployment gate a risk team can read, is
the practical version of "small model, real work".

### Research questions

1. Can a self-healing loop turn a 3B model that solves almost nothing into a usable database agent?
   **Partly:** 0.5% → 49.1%, and half of the runs still fail.
2. Does training on the *shortest* verified run make the agent cheaper as well as better?
   **Partly:** repeated calls, loops and output tokens fall, but solved runs barely get shorter.
3. Does a gate with confidence intervals decide differently from a "no regression" gate?
   **Yes:** it held an adapter that AgentTune's shipped gate approved.

### Pre-registered hypotheses

Written down before the first rollout. Not registered with a third party.

| Hypothesis | Outcome |
|---|---|
| H1. On held-out tasks both models solve, tool calls per task fall by at least 15% | **Could not be tested cleanly.** The base model solved only 6 of 1,200 runs, so at most 6 of the 400 tasks are solved by both models and the comparison is too noisy to read. Between adapters, where it can be measured, the fall is 3 to 4% at best (Finding 7) |
| H2. Held-out success does not fall, and we expect it to rise | **Met.** 0.5% → 25.2% → 30.4% → 49.1% |
| H3. The redundant tool-call rate falls | **Met.** 0.126 → 0.108 → 0.069 → 0.051 |

### Related work

None of the parts is new. The table names the closest work for each part and where we differ.

| Part of our loop | Closest prior work | Where we differ |
|---|---|---|
| Train on the model's own verified outputs, in rounds | STaR ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)), rejection-sampling fine-tuning ([Yuan et al., 2023](https://arxiv.org/abs/2308.01825)), self-imitation ([Oh et al., 2018](https://arxiv.org/abs/1806.05635)), expert iteration ([Anthony et al., 2017](https://arxiv.org/abs/1705.08439)) | Tool trajectories, not rationales. Our hint never contains the answer. We keep one run per task, the shortest, and gate every round |
| Fine-tune a small model to be an agent | ReAct ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)), FireAct ([Chen et al., 2023](https://arxiv.org/abs/2310.05915)), AgentTuning ([Zeng et al., 2023](https://arxiv.org/abs/2310.12823)), ToolLLM ([Qin et al., 2023](https://arxiv.org/abs/2307.16789)), Toolformer ([Schick et al., 2023](https://arxiv.org/abs/2302.04761)); distillation ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531), [Hsieh et al., 2023](https://arxiv.org/abs/2305.02301)) | They distil from a stronger model once. We add rounds in which the student's own runs replace the teacher's (62% of targets by round 3) |
| Repair failed runs | Reflexion ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)), DAgger ([Ross et al., 2010](https://arxiv.org/abs/1011.0686)) | The repaired run is trained into the adapter, so no hint is needed at run time. Our teacher restarts the task; it does not correct the student's state |
| Label failures | MAST ([Cemri et al., 2025](https://arxiv.org/abs/2503.13657)), failure attribution ([Zhang et al., 2025](https://arxiv.org/abs/2505.00212)) | We see the same weakness: LLM labels agree with simple rules on 115 of 142 cases, and on 27 of 58 with thinking off. So labels only choose hints |
| Loss on the agent's turns only | Loss over instructions ([Shi et al., 2024](https://arxiv.org/abs/2405.14394)) | They report that loss on the prompt can help. We found the opposite: 2.0% vs 19.7%. Our extra tokens are tool results, not instructions |
| Preference pairs | DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) | Applied to single agent steps, not whole answers |
| Adapters and forgetting | LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), LoRA forgets less ([Biderman et al., 2024](https://arxiv.org/abs/2405.09673)) | Our retention check agrees, on a small sample |
| Sample k runs per task | Self-consistency ([Wang et al., 2022](https://arxiv.org/abs/2203.11171)) | We sample at training time to pick the cheapest correct run. The deployed agent runs once |
| Gate and intervals | AgentTune's [self-healing guide](https://agenttune.lexsi.ai/user-guide/self-healing/), paired bootstrap ([Koehn, 2004](https://aclanthology.org/W04-3250/)) | Our gate needs a gain whose interval excludes zero. The shipped gate only needs no regression |

Task, benchmarks and models: FinQA ([Chen et al., 2021](https://arxiv.org/abs/2109.00122)) in the
tool-calling version shipped with AgentTune; GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168));
XSTest ([Röttger et al., 2023](https://arxiv.org/abs/2308.01263)); Global-MMLU
([Singh et al., 2024](https://arxiv.org/abs/2412.03304)); Aya Expanse
([Dang et al., 2024](https://arxiv.org/abs/2412.04261)).

What we add is the combination, measured on one small model: the shortest verified run as the target,
with an evidence guard; repair directed by a failure label; and a gate that held twice when the evidence
was thin.

## Hackathon repos (kits) used

Track 3 asks for AgentTune and AuditKIT. CuratorKIT, AlignTune, SafeTune and CircuitKIT are not used.

### AgentTune (1.1.7)
- **Rollout loop**: `create_rollout_fn` runs every trajectory (prompting, the Aya tool-prompt fallback,
  tool-call parsing, tool execution). We plug in a custom `RolloutEngine` (`selfheal/engine.py`) that sends
  generation to a vLLM server, so AgentTune keeps the loop while generation is batched.
- **Task environment**: FinQA tools (`build_finqa_tools`, `calculator`), data loader and database builder.
- **Reward**: `numerical_match_reward` is our strict success rate.
- **Self-healing loop** (its user guide's closed loop): `FailureDetector` through
  `EventLog.to_audit_records()`, `FailureClassifier`, `TrainingBuffer` + `RetrainingTrigger`,
  `DeploymentGate.evaluate_decision`.
- **Training**: `create_distill_trainer` (TRL SFT + PEFT LoRA).
- **Trajectory schema**: `Trajectory.to_dict()` JSONL, the interchange format with AuditKIT.

Two pieces of AgentTune we replaced, and why:
- `TrainingExampleGenerator` emits single corrected actions for two of five root causes. We need whole
  verified trajectories, so repair is our own ladder (hinted retry, then teacher run).
- The default SFT rows write tool calls as `<tool_call>{...}</tool_call>`, while AgentTune's Aya prompt asks
  the model for `[{"tool_name": ..., "parameters": {...}}]`. We train on the format the model is prompted for.

### AuditKIT (1.1.12)
- `episodes_from_agenttune` imports our trajectory JSONL with full coverage.
- Reference-free agent metrics: `ToolCallValidity`, `RedundantToolCalls`, `AgentLoopDetection`.
- Our task metrics run as AuditKIT scorers (`@ak.scorer`): task success, tool calls, output tokens, seconds.
- `ak.compare(...).significance(...)`: paired bootstrap for every before/after difference.

### How the repos connect

```
AgentTune rollouts ──► trajectory JSONL ──► AuditKIT import ──► metrics + paired comparison
      │                      │                                         │
      │                      ▼                                         ▼
      │        buckets, failure labels (AgentTune),          gate (our rule + AgentTune
      │        repair, RetrainingTrigger (AgentTune)          DeploymentGate) ──► deploy or hold
      │                      │
      └──── new adapter ◄─── AgentTune create_distill_trainer (LoRA)
```

## Models and licenses

| Role | Model | Licence | Notes |
|---|---|---|---|
| Student (submitted) | `CohereLabs/tiny-aya-global`, 3.35B, bf16 | CC-BY-NC-4.0 | LoRA r=16 on attention projections |
| Second student (further experiment) | `CohereLabs/aya-expanse-8b`, 8.0B, bf16 | CC-BY-NC-4.0 | LoRA r=16, trained on the round 2 set |
| Teacher and failure classifier | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Apache-2.0 | Supplementary only; never fine-tuned or submitted |

### Base Aya model

`CohereLabs/tiny-aya-global`, 3.35B parameters, loaded in bf16 with no quantization, licence CC-BY-NC-4.0.
The adapter is a LoRA (r=16) on the attention projections and inherits that licence.

One further experiment uses a second Cohere student, `CohereLabs/aya-expanse-8b` (8.0B, bf16, CC-BY-NC-4.0).
Untrained, it is no better than untrained Tiny Aya (0 of 300 runs). Trained on the round 2 set it is a
stronger student (see Further experiments); its adapter is published separately.

### Teacher model

`nvidia/Qwen3.6-35B-A3B-NVFP4` (Apache-2.0), a quantized 35B model. We chose it for two reasons. It fits on
the same GPU beside the student, so one job can collect, repair and train. And it solves the task well
enough to repair it: 77.9% of its runs passed the verifier on the tasks it was given. It writes repair
runs and failure labels only. It is never fine-tuned, never submitted and never scores a result.

### Closed-model baseline

None was run. The cost section estimates what a closed API would cost on our token counts; its success
rate is not measured.

AgentTune and AuditKIT are used under the Lexsi Labs Source Available License for non-commercial research.

## Datasets, benchmarks and licenses

### Sources of data

- **Task data**: FinQA as shipped by AgentTune (`rLLM/rLLM-FinQA-Dataset`, Apache-2.0): 4,030 train and
  522 validation questions over the financial tables of 185 companies. Answers are numbers. It is an
  existing public dataset; we wrote no synthetic questions.
- **Training data**: trajectories made in this project by the student, by the student with a hint, and by
  the teacher (distillation). Every one is checked by the verifier. Published as
  `Purusharth/efficient-healer-trajectories` under CC-BY-NC-4.0.

### Languages covered

English only: the task data, the training data and every held-out set. The base model is multilingual, and
we did not test other languages (the one multilingual check we ran did not produce a valid measurement).

### Train / test splits, leak check

- **Train pool**: 480 questions from the train split in rounds 1 and 2, grown to 1,400 in round 3 (the 480
  are a prefix of the 1,400).
- **Held-out sets**, all from 55 companies reserved for testing: a first set of 100 validation questions;
  set A, 400 fresh questions used once to confirm the round 2 gate; set B, 400 more fresh questions used
  for the round 3 gate. No question appears in two sets.
- **Leak check**: the split is company-disjoint, so no held-out table is ever seen in training. Shared
  companies: 0. Shared question texts: 0, by exact match.

### General capability and safety benchmarks

These are not the focus task. They were run once, as a retention check, to see whether fine-tuning broke
something else: base Tiny Aya against the round 2 adapter, 100 samples each, in English.

| Benchmark | What it checks | Licence | Status |
|---|---|---|---|
| GSM8K | Maths word problems, numeric match | MIT | Measured |
| XSTest | Refusal of unsafe prompts, and over-refusal of safe prompts that look unsafe | CC-BY-4.0 | Measured |
| Global-MMLU, five languages | Multilingual knowledge | Apache-2.0 | Run, but not a valid measurement: the answer extraction failed |

### Tool / task environment and trajectories

The environment is AgentTune's FinQA setup: a SQLite database of company tables and four tools (see Method).
All runs are published, successes and failures alike.

| Stage | Rollouts | What they are |
|---|---|---|
| Round 1 student | 1,920 | Base Tiny Aya, 480 tasks × 4 |
| Round 1 hinted retries | 1,876 | Base Tiny Aya with a label-chosen hint, 469 tasks × 4 |
| Round 1 teacher | 1,383 | Qwen3.6-35B, 461 tasks × 3 |
| Round 2 student | 1,920 | Round 1 adapter, 480 tasks × 4 |
| Round 2 hinted retries | 1,032 | Round 1 adapter with a hint, 258 tasks × 4 |
| Round 2 teacher | 183 | Only tasks with no verified run yet, 61 × 3 |
| Round 3 student | 5,600 | Round 2 adapter, 1,400 tasks × 4 |
| Round 3 hinted retries | 2,948 | Round 2 adapter with a hint, 737 tasks × 4 |
| Round 3 teacher | 1,500 | Only tasks with no verified run yet, 500 × 3 |
| First held-out set | 1,800 | 100 tasks × 3, for base and the round 1 and 2 adapters, with and without the stop rule |
| Held-out set A | 6,000 | 400 tasks × 3, for base, rounds 1 to 3, Aya Expanse and DPO |
| Held-out set B | 4,800+ | 400 tasks × 3, for rounds 2 and 3, Aya Expanse and DPO |

Training sets (one row per task, the shortest verified run):

| Source of the target | Round 1 set | Round 2 set | Round 3 set |
|---|---|---|---|
| Pool size | 480 | 480 | 1,400 |
| Efficiency: task the student always solves | 0 | 32 | 170 |
| Reliability: task the student sometimes solves | 11 | 190 | 493 |
| Repair by hinted student retry | 8 | 39 | 87 |
| Repair by teacher | 395 | 163 | 453 |
| **Rows** | **414** | **424** | **1,203** |
| Used after dropping over-long rows | 393 | 402 | 1,142 |
| Pool tasks with no verified run from any source | 66 | 56 | 197 |

In round 1 the student's own runs supply 5% of the targets. In round 2 they supply 61% (261 of 424) and
the teacher's share falls from 95% to 38%. In round 3, with 920 tasks the student had never seen, its own
runs still supply 62% (750 of 1,203).

## Metrics, and why these

### List of metrics

| Metric | Better | Why |
|---|---|---|
| Task success rate (tolerant) | higher | The answer matches the gold number after rounding to the gold's precision or within 0.5%, with percent and fraction treated as the same number |
| Task success rate (strict) | higher | AgentTune's `numerical_match_reward`, exact first-number match |
| Solved in all k runs / in any run | higher | Reliability: a task solved once in three tries is not solved |
| Tool calls per task | lower | The direct cost of a run |
| Tool calls on tasks both models solve | lower | Efficiency without rewarding a model for only solving easy tasks |
| Redundant tool-call rate (AuditKIT) | lower | Repeated identical calls |
| Tool-call validity (AuditKIT) | higher | Calls that match the tool schema; must not get worse |
| Loop-free share (AuditKIT) | higher | No repeated-call loop |
| Output tokens and seconds per task | lower | Cost and latency |

Every difference is reported with a 95% paired bootstrap interval. Rollouts of one task are not
independent, so headline intervals resample tasks, not rollouts.

### Metrics for what should not get worse

- **Tool-call validity** (AuditKIT): part of the gate. It is 1.000 for every adapter (0.997 for the base model).
- **Strict success**, shown beside tolerant success, so a gain cannot come from the tolerance alone.
- **GSM8K accuracy and XSTest refusal and over-refusal** (AuditKIT): see Further experiments.

### Checking the automatic judge

**No LLM judge scores any result.** Success is decided by a numeric verifier.

- **The verifier.** Every table shows the strict rate beside the tolerant rate. We found one wrong rule by
  reading accepted runs, fixed it and re-scored every run. Agreement with human labels was not measured.
- **The failure labels** (AgentTune's `FailureClassifier`). Not checked against people. They agree with our
  rule labels on 115 of 142 cases in round 1 and 33 of 45 in round 2. They choose repair hints and never
  enter a metric.

### Agentic trajectory metrics

Task success, tool-call validity, redundant-call rate and loop-free share are reported for every model,
and success is split by question type. There is one task and one language, so there is no further split.

## Method / experiments

### Training setup and hyperparameters

#### AgentTune

| Part | Setting |
|---|---|
| Task and tools | Four AgentTune tools: `list_tables`, `get_table_schema`, `query_finqa_tables`, `calculator`. The answer goes inside `<answer></answer>`. At most 8 tool calls per run |
| Rollouts | k=4 per pool task, temperature 0.7, top-p 0.95, at most 384 new tokens per turn (1,024 for the teacher). vLLM 0.33 behind our `RolloutEngine`, 48 runs at a time |
| Tool format | Tiny Aya's chat template has no tool support, so AgentTune lists the tools in the system prompt and asks for calls as `[{"tool_name": ..., "parameters": {...}}]` |
| Harness stop rule | Ends a turn after the first call list or at `</answer>`. On while collecting data; evaluation is reported with and without it |
| Target of a task | Its shortest verified run that passes the evidence guard. Exact repeats and the repair hint are removed |
| Failure labels | Rules (`loop_collapse`, `no_answer_tag`, `step_cap`, `tool_errors`, `no_tool_evidence`, `wrong_number`) plus AgentTune's `FailureClassifier` (five root causes), served by the teacher |
| Repair ladder | (1) 4 student retries with a one-sentence hint chosen by the label. (2) 3 teacher runs. A repair counts only if it passes the verifier and the evidence guard |
| Training | AgentTune `create_distill_trainer` with a custom trainer factory: TRL `SFTTrainer`, LoRA r=16, α=32, dropout 0.05 on `q_proj, k_proj, v_proj, o_proj`, 3 epochs, learning rate 2e-4 with cosine decay and 5% warm-up, batch 2 × 8, bf16. **Loss on the agent's own turns only.** Rows above the 95th-percentile length are dropped |
| Gate | Deploy only if success and tool-call validity each fall by no more than 2 points, and success or tool calls on tasks both models solve improves with a 95% interval that excludes zero. AgentTune's `DeploymentGate.evaluate_decision` must also approve. Applied once per round |
| Round 3 | The same loop from the round 2 adapter on 1,400 tasks, with the classifier's thinking off |
| DPO | AgentTune `create_agentic_trainer(algorithm="dpo")` (TRL `DPOTrainer`). The prompt is the conversation up to the first agent turn where a verified run and a failed run differ; chosen and rejected are that turn. LoRA r=8 on a merged round adapter, learning rate 2e-5, β 0.1, 1 epoch, batch 1 × 4 |

#### AuditKIT

| Part | Setting |
|---|---|
| Scoring | `episodes_from_agenttune` import, then `ak.evaluate` on the `precomputed` backend. Metrics: `ToolCallValidity`, `RedundantToolCalls`, `AgentLoopDetection` at their defaults, plus our task scorers |
| Comparison | `ak.compare(...).significance(...)`. Headline intervals come from a task-level paired bootstrap, 5,000 resamples |
| Retention check | GSM8K scenario (`numeric_match`), `ak.load_xstest()` with `ak.refusal_scorer()`, `ak.load_global_mmlu()`. The model is served by vLLM through the `api:` backend, 100 samples each |

#### Compute

Hugging Face Jobs on RTX PRO 6000 Blackwell GPUs (96 GB), each holding the student server, the teacher
server and training.

### Seeds

Split seed 0, training seed 0, bootstrap seed 0. Rollouts are sampled at temperature 0.7, not seeded. Each
configuration was trained once.

### Ablations

Each row changes one thing.

| What changed | What stayed the same | Result |
|---|---|---|
| Loss on all tokens → loss on the agent's turns | Same 393 rows, same LoRA settings | Held-out success 2.0% → 19.7% |
| Harness stop rule on → off | Round 2 adapter, first held-out set | 24.0% with the rule, 25.0% without |
| Harness stop rule off → on | Base model | Output tokens halve; success stays at 0–0.3% |
| Student: Tiny Aya 3.35B → Aya Expanse 8B | Same round 2 training rows | 30.4% → 38.9% on set A |
| DPO stage added to the round 2 adapter | Same test sets | 30.4% → 35.8% on set A; 26.2% → 29.8% on set B |
| DPO stage added to the round 3 adapter | Same test set | 46.4% → 48.5% on set B; the interval includes zero |
| Repair hint on the retry → no hint | Base model, 469 failing tasks | 0.5% of retries solved with a hint, 0.8% without |

Round 2 → round 3 is not a clean ablation: the pool and the adapter that collects the runs change together.

## Results

All numbers are on held-out companies whose tables were never seen in training. Success is decided by the
numeric verifier. The detailed tables for rounds 1 and 2 are in the Appendix.

### Same model, before vs after

`CohereLabs/tiny-aya-global` without and with the deployed round 3 adapter, on held-out set A (400 tasks ×
3 runs, stop rule on):

| Metric | Before (base model) | After (round 3 adapter) |
|---|---|---|
| Runs solved | 6 of 1,200 (0.5%) | **589 of 1,200 (49.1%)** |
| Runs solved, strict exact match | 0.3% | **29.8%** |
| Tasks solved in all 3 runs | 0 of 400 | **143 of 400** |
| Tasks solved in at least 1 run | 6 of 400 | **249 of 400** |
| Redundant tool-call rate (AuditKIT) | 0.126 | **0.051** |
| Loop-free share (AuditKIT) | 0.761 | **0.893** |
| Tool-call validity (AuditKIT) | 0.997 | 1.000 |
| Tool calls per run | 2.88 | 4.32 |

Tool calls per run rise because the base model gives up or invents results, and the adapter fetches the data.

### Round 3: the same loop on 1,400 tasks

Round 3 starts from the round 2 adapter, grows the pool from 480 to 1,400 tasks, and trains one LoRA from
the base model on 1,142 verified runs. Its gate was decided on held-out set B (400 questions that no
earlier decision had touched); set A is shown so the number sits beside rounds 1 and 2.

| Metric (400 tasks × 3 runs, stop rule on) | Round 2 adapter, set B | Round 3 adapter, set B | Round 2 adapter, set A | Round 3 adapter, set A |
|---|---|---|---|---|
| Rollouts solved (tolerant) | 314 of 1,200 (26.2%) | **557 of 1,200 (46.4%)** | 365 of 1,200 (30.4%) | **589 of 1,200 (49.1%)** |
| Rollouts solved (strict) | 14.0% | 26.0% | 17.6% | 29.8% |
| Tasks solved in all 3 runs | 53 of 400 | 134 of 400 | 61 of 400 | 143 of 400 |
| Tasks solved in at least 1 run | 164 of 400 | 239 of 400 | 187 of 400 | 249 of 400 |
| Tool calls per run | 4.45 | 4.31 | 4.47 | 4.32 |
| Tool calls per solved run | 3.95 | 3.97 | 4.01 | 3.91 |
| Output tokens per run | 214 | 214 | 215 | 210 |
| Runs with no `<answer>` tag | 123 of 1,200 | 83 of 1,200 | 117 of 1,200 | 94 of 1,200 |

**Round 2 adapter vs round 3 adapter** (difference = round 3 minus round 2, 95% paired bootstrap interval)

| Metric | Set B (the gate) | Set A |
|---|---|---|
| Success rate, per task | +20.2 pts [+16.8, +23.7] | +18.7 pts [+15.2, +22.4] |
| Tasks solved in all 3 runs | +20.2 pts [+16.2, +24.5] | +20.5 pts [+15.8, +25.2] |
| Tool calls on tasks both solve | +0.01 [−0.07, +0.09] (n=153) | −0.16 [−0.25, −0.08] (n=173) |
| Tool calls per run | −0.15 [−0.25, −0.05] | −0.16 [−0.25, −0.07] |
| Redundant tool-call rate (AuditKIT) | 0.069 → 0.048 [−0.032, −0.011] | 0.069 → 0.051 [−0.027, −0.008] |
| Loop-free share (AuditKIT) | 0.826 → 0.887 [+0.038, +0.085] | 0.818 → 0.893 [+0.053, +0.098] |
| Tool-call validity (AuditKIT) | 1.000 → 1.000 | 1.000 → 1.000 |

**Gate, round 3 (set B): DEPLOY.** Success does not regress (0.262 → 0.464), validity does not regress, and
the success gain's interval is above zero. AgentTune's `DeploymentGate` also approved. The round 3 adapter
is the deployed one.

### Rounds 1 and 2: the path to round 3

Round 1 trains on repairs only. Round 2 adds the round 1 adapter's own shortest verified runs. On set A
success goes 0.5% → 25.2% → 30.4% (see the headline table). Round 2 against round 1:

- success +5.2 points [+2.5, +8.1];
- redundant tool-call rate −0.040 [−0.053, −0.028];
- output tokens per run −24 [−34, −15].

The adapters also learned to stop by themselves. Without the harness stop rule, the round 2 adapter solves
25.0% on the first held-out set (24.0% with it) and writes 213 output tokens per run, against 620 for the
base model.

### By question type

First held-out set (100 tasks × 3 runs, stop rule on): solved runs and their share of that type's runs.

| Question type | Runs | Base | Round 1 adapter | Round 2 adapter |
|---|---|---|---|---|
| Net Change | 105 | 0 (0%) | 22 (21%) | 30 (29%) |
| Financial Ratio - Composition | 57 | 1 (2%) | 10 (18%) | 12 (21%) |
| Contribution Share | 48 | 0 (0%) | 6 (12%) | 10 (21%) |
| Duration Ladder | 33 | 0 (0%) | 6 (18%) | 7 (21%) |
| Growth - YoY | 27 | 0 (0%) | 8 (30%) | 8 (30%) |
| Spread | 15 | 0 (0%) | 0 (0%) | 3 (20%) |
| Mix Shift | 12 | 0 (0%) | 4 (33%) | 1 (8%) |
| Financial Ratio - Earnings Per Share | 3 | 0 (0%) | 3 (100%) | 1 (33%) |

### What the gates decided

| Decision | Test set | Change in success [95% interval] | Gate |
|---|---|---|---|
| Round 1 vs base | First set, 100 tasks | +19.3 pts [+13.7, +25.7] | DEPLOY |
| Round 2 vs round 1 | First set, 100 tasks | +4.3 pts [−1.7, +10.7] | **HOLD** |
| Round 2 vs round 1, confirmation | Set A, 400 tasks | +5.2 pts [+2.5, +8.1] | DEPLOY |
| Round 3 vs round 2 | Set B, 400 tasks | +20.2 pts [+16.8, +23.7] | DEPLOY |
| Round 2 + DPO vs round 2 | Set A, 400 tasks | +5.3 pts [+2.4, +8.2] | DEPLOY |
| Round 3 + DPO vs round 3 | Set B, 400 tasks | +2.1 pts [−0.3, +4.3] | **HOLD** |

On 100 tasks the round 2 gain could not be told from zero, so the gate held. We did not change the rule.
We ran one pre-declared confirmation on 400 fresh questions and accepted its answer. AgentTune's own gate
approved round 2 on the 100-task set too, because it passes anything that does not regress.

### Further experiments

**Aya Expanse 8B as the student.** A LoRA on `CohereLabs/aya-expanse-8b` with the unchanged round 2
training set (402 rows, same settings).

| 400 tasks × 3 runs, stop rule on | Tiny Aya, round 2 adapter | Aya Expanse 8B, same data | Difference [95% interval] |
|---|---|---|---|
| Rollouts solved, set A | 365 of 1,200 (30.4%) | 467 of 1,200 (38.9%) | +8.5 pts [+4.5, +12.6] |
| Rollouts solved, set B | 314 of 1,200 (26.2%) | 438 of 1,200 (36.5%) | +10.3 pts [+6.7, +13.9] |
| Tool calls per run, set A | 4.47 | 4.05 | −0.42 [−0.53, −0.31] |
| Redundant tool-call rate, set A | 0.069 | 0.019 | [−0.060, −0.039] |
| Loop-free share, set A | 0.818 | 0.950 | |

Two things to read from this:

- **The gain comes from our training data, not from the model's size.** Before training, Aya Expanse 8B
  solved 0 of 300 runs.
- **More data helped more than a bigger model.** Aya Expanse 8B was trained only on the round 2 set and
  reaches 38.9%. Tiny Aya, trained on the larger round 3 set, reaches 49.1%.

**DPO on step-level preference pairs.** 887 pairs came out of round 2; 480 were used, on top of the round 2
adapter, for 11.5 minutes of training.

| 400 tasks × 3 runs, stop rule on | Round 2 adapter | Round 2 + DPO | Difference [95% interval] |
|---|---|---|---|
| Rollouts solved, set A | 365 of 1,200 (30.4%) | 429 of 1,200 (35.8%) | +5.3 pts [+2.4, +8.2] |
| Rollouts solved, set B | 314 of 1,200 (26.2%) | 358 of 1,200 (29.8%) | +3.7 pts [+1.1, +6.3] |
| Tasks solved in all 3 runs, set A | 61 of 400 | 89 of 400 | +7.0 pts [+2.8, +11.2] |
| Tool calls per run, set A | 4.47 | 4.71 | +0.24 [+0.15, +0.33] |
| Loop-free share, set A | 0.818 | 0.764 | |

DPO buys success, not efficiency: runs get slightly longer. On top of round 3 (320 of 2,158 pairs,
8 minutes) it scores 48.5% against 46.4% on set B, a gain of +2.1 points [−0.3, +4.3]. The gate says HOLD,
so round 3 stays deployed.

**Retention and safety check (AuditKIT).** Base Tiny Aya against the round 2 adapter, 100 samples each,
plain chat with no tools.

| Check | Base | Round 2 adapter | Difference [95% interval] |
|---|---|---|---|
| GSM8K, numeric match | 0.71 | 0.67 | −0.04 [−0.12, +0.05] |
| XSTest, refusal of unsafe prompts (higher is safer) | 0.94 | 0.94 | 0.00 [−0.04, +0.04] |
| XSTest, over-refusal of safe prompts (lower is better) | 0.06 | 0.06 | 0.00 [−0.04, +0.04] |

No measurable change. Global-MMLU scored 0.05 for both models, below chance, so its answer extraction
failed and we do not report it. The round 3 adapter was not re-checked.

### Findings

1. **Where the loss goes decided everything.** Same 393 rows and settings: loss on all tokens gave 2.0%
   held-out success; loss on the agent's turns gave 19.7%. The agent's turns are about 5% of a row's tokens.
2. **More data through the same loop was the next biggest lever.** Round 3 grew the pool from 480 to 1,400
   tasks and gained 18.7 points on set A and 20.2 on set B.
3. **The gate changed a decision.** It held round 2 on 100 tasks and released it on 400. AgentTune's
   shipped gate would have deployed on the thin evidence.
4. **The loop weans itself off the teacher.** Teacher-made targets fell from 95% of the training set in
   round 1 to 38% in rounds 2 and 3.
5. **Hints repair a trained model, not an untrained one.** A hint repaired 8 of 469 failing tasks for the
   base model, 39 of 258 for the round 1 adapter and 87 of 737 for the round 2 adapter.
6. **The adapter learned to stop.** The base model writes a tool call and then invents its result. The
   adapters do not, with or without the harness stop rule.
7. **Efficiency shows up as less waste, not shorter runs.** The redundant tool-call rate falls
   0.126 → 0.108 → 0.069 → 0.051. Tool calls on tasks two adapters both solve fall by 3 to 4% at best.
8. **The data transfers, and DPO adds success only.** Aya Expanse 8B reaches 38.9% on Tiny Aya's round 2
   data (Tiny Aya: 30.4%). DPO adds 4 to 5 points to round 2 and makes runs slightly longer.

### What did not work (negative results)

- **Loss on all tokens.** With AgentTune's default trainer the loss fell from 1.74 to 0.30, yet held-out
  success was 6 of 300 (2.0%).
- **A bigger untrained student.** Aya Expanse 8B solved 0 of 300 runs before training.
- **A harness stop rule alone.** It cuts the base model's output tokens from 620 to 208 per run, but
  success stays at 0–0.3%.
- **"Shortest success" without a guard.** One base-model "success" invented its tool results and answered
  `0` to a question whose answer is `0`. The evidence guard exists because of it.
- **Our first verifier rule.** It accepted `4%` for a gold answer of `0`. We fixed it and re-scored every run.
- **One void run.** The first Aya Expanse evaluation scored 0.2% because that GPU job had no FinQA
  database. We fixed the tools, reran it, and published the void runs with a label.
- **Global-MMLU.** Both models scored 5% on a four-choice test. Not a valid measurement.
- **Rushed failure labels.** With the classifier's thinking off, agreement with the rule labels fell from
  115 of 142 to 27 of 58.

### Sample inputs and outputs

One solved run by the deployed adapter on a held-out company (set A, task `train-2497`, run 2): 3 tool
calls, 113 output tokens, 4.4 seconds. Calls are shown in short form; the model writes them as
`[{"tool_name": ..., "parameters": {...}}]`. Long tool results are cut with `…`.

**Input** (the user turn; the system prompt lists the tools):

> Company: newmont
>
> Question: For company `newmont`, here is the question: What is the spread between interest cost and
> service cost for pension benefits in 2024?
>
> *(followed by AgentTune's suggested steps and the instruction to wrap the answer in `<answer>` tags)*

| Step | What the agent wrote | What came back |
|---|---|---|
| 1 | `list_tables(company_prefix="newmont")` | The names of the company's tables: `newmont_nem_CurrentLongTermEmployeeBenefitLiabilitiesTableTextBlock, …` |
| 2 | `get_table_schema(table_name="newmont_us_gaap_ScheduleOfNetBenefitCostsTableTextBlock")` | The columns and the first rows: `service cost 14 12 15 …`, `interest cost 17 17 19 …` |
| 3 | `calculator(expression="17 - 14")` | `3.0` |
| 4 | `The spread between interest cost and service cost for pension benefits in 2024 is 3.` `<answer>3</answer>` | Gold answer: `3`. Solved. |

Every run, solved or failed, is in the dataset repository; this one is in `eval/confirm_r3.jsonl`.

## Cost / ROI

### Compute used

Hugging Face Jobs, paid from the hackathon's $20 credit.

| Part | GPU | Time | Cost |
|---|---|---|---|
| Setup and first smoke test | L40S | 37 min | $1.10 |
| Rounds 1 and 2, evaluation, gates, confirmation | RTX PRO 6000 Blackwell | 2 h 43 min | $7.47 |
| Round 3, DPO on round 3 | RTX PRO 6000 Blackwell | 1 h 55 min | $5.25 |
| Aya Expanse 8B, retention check, DPO, re-tests | RTX PRO 6000 Blackwell | 1 h 32 min | $4.23 |
| **Total** | | **6 h 47 min** | **$18.05** |

That covers about 34,000 student rollouts, 3,100 teacher rollouts, five LoRA trainings (12 to 34 minutes
each), two DPO runs (8 and 12 minutes) and all evaluation.

### Cost per 1,000 agent runs vs API

A run is a whole task: about 5 model calls and 4 tool calls.

| Option | Throughput | Cost per 1,000 runs | Success | Cost per 1,000 solved tasks |
|---|---|---|---|---|
| Tiny Aya + round 3 adapter, one RTX PRO 6000 at $2.75/h | 522 runs/min, measured | $0.09 | 49.1%, measured | $0.18 |
| Aya Expanse 8B + adapter, same GPU | 391 runs/min, measured | $0.12 | 38.9%, measured | $0.30 |
| Teacher, Qwen3.6-35B NVFP4, same GPU | 148 runs/min, measured | $0.31 | 77.9%, measured on a different task set | $0.40 |
| A closed API at an assumed $3 / $15 per million input / output tokens | not measured | $54, estimated | not measured | not measured |

Throughput was measured with 48 agents running at once, tool execution included. The closed-API row is an
estimate from our own token counts (about 17,000 input and 210 output tokens per run) at assumed list
prices; we did not run a closed model. Per solved task, the Tiny Aya adapter costs about half as much as
the 35B teacher.

## Limitations

### Known weaknesses

- **Half of the held-out runs still fail**, mostly with a wrong number from the wrong table or rows.
- **One training seed, and reused test sets.** Sets A and B each decided one gate and were then reused for
  other comparisons. No training saw them, but those comparisons are not independent.
- **The adapters make more tool calls than the base model** (4.32 vs 2.88 per run), because the base model
  gives up instead of fetching data. Our efficiency claims are between adapters.
- **Retention is only partly checked:** 100 samples, the round 2 adapter only, no multilingual result.
  These adapters are research artifacts.
- **English only**, and the teacher is a larger, non-Cohere model.
- **The failure labels are not validated by people.**
- **The task prompt lists suggested steps**, as shipped by AgentTune.
- **The verifier is tolerant** (within 0.5%). The strict rate is always shown beside it.

## With more credits

### Planned approach

1. **Finish the pool.** About 1,400 train questions are still unused, and data through this loop was the
   strongest lever.
2. **Aya Expanse 8B through the full three-round loop**, then DPO on its own failures.
3. **More rounds**, until the gate holds twice in a row.
4. **A reinforcement-learning stage (GRPO)** whose reward penalises extra tool calls only on tasks the
   model already solves. This targets what the loop moved least.
5. **A proper held-out study:** several seeds, 5 runs per task, each test set used for one decision.
6. **Full retention checks** on the deployed adapter, including other languages.

### Expected before / after

A forecast, not a measurement: held-out success from 49% to 60–65%, tasks solved in all 3 runs from 36% to
about 50%, and tool calls per solved run from 3.9 toward 3.5.

## Artifacts

- **HF datasets collection:** [Efficient Healer — datasets](https://huggingface.co/collections/Purusharth/efficient-healer-datasets-6afaa8b70386ac5f12a8f637).
  It holds [`Purusharth/efficient-healer-trajectories`](https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories):
  every rollout with its tool calls, failure label, repair hint, training set, preference pair, evaluation
  log, AuditKIT report and gate decision, for all three rounds and the further experiments.
- **HF models collection:** [Efficient Healer — models](https://huggingface.co/collections/Purusharth/efficient-healer-models-6afaa8b6120d5e7cae162d33).
  It holds two repositories:
  - [`Purusharth/tiny-aya-efficient-healer-lora`](https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora):
    the gate-approved round 3 adapter at the root; every adapter trained is in its own folder (`round1/`,
    `round2/`, `round3/`, `round2_dpo/`, `round3_dpo/`, `negative_all_token_loss/`).
  - [`Purusharth/aya-expanse-8b-efficient-healer-lora`](https://huggingface.co/Purusharth/aya-expanse-8b-efficient-healer-lora):
    the Aya Expanse 8B adapter, trained on the round 2 set.
- **GitHub repo:** [`purusharthmalik/efficient-healer`](https://github.com/purusharthmalik/efficient-healer).
- **Demo, video, slides:** none.

## Documentation and reproduction

### Environment requirements

Three Python environments, because their pins differ: AgentTune (`agenttune` 1.1.7, pinned in the setup
script), vLLM (`vllm>=0.30`, the first with Cohere support) and AuditKIT
(`auditkit[transformers,interop,jsonschema]==1.1.12`). `infra/cmds/000_setup.sh` builds all three.
Hardware used: one RTX PRO 6000 Blackwell (96 GB) per job, which holds the student server, the teacher
server and training together. We did not test smaller GPUs.

### Repo layout and run instructions

```
selfheal/
  engine.py          AgentTune RolloutEngine backed by a vLLM server
  tasks.py           FinQA tasks and tools, company-disjoint split, leak check
  verify.py          tolerant answer check + AgentTune's strict reward
  rollout.py         k rollouts per task, resumable, one JSONL line per rollout
  summary.py         per-task statistics
  buckets.py         always / sometimes / never, shortest-success selection, evidence guard
  failures.py        rule labels + AgentTune FailureClassifier
  repair.py          failure label -> repair hint
  pending.py         tasks that still lack a grounded success
  sft_data.py        trajectory -> training row in the student's own format
  build_dataset.py   assembles the training set of a round
  agenttune_loop.py  AgentTune FailureDetector, RetrainingTrigger, DeploymentGate
  train.py           AgentTune create_distill_trainer (LoRA), loss on agent turns
  merge.py           merge an adapter for serving
  evaluate.py        AuditKIT import, metrics, paired comparison
  gate.py            the deployment rule
  confirm_split.py   further held-out sets from the reserved companies
  rescore.py         re-apply the verifier to stored runs
  dpo_pairs.py       step-level preference pairs
  train_dpo.py       AgentTune agentic DPO trainer
  retention.py       AuditKIT retention and safety check
  publish.py         stage and upload run artifacts
infra/
  worker.sh          runs queued steps inside one long Hugging Face Job
  cmds/              one script per pipeline step, in run order (e*.sh are the further experiments)
tools/               result tables and the publishing script for the further experiments
```

To reproduce, start `infra/worker.sh` inside a GPU job, with `OPS_REPO` set to a private Hugging Face
dataset repository of your own. Then queue the steps in the order of `infra/cmds/`, one at a time:

```bash
infra/submit.sh 000_setup infra/cmds/000_setup.sh
```

The order is: setup, baseline evaluation, round 1 (collect, repair, train), round 2 (collect, repair,
train), evaluation and gates, then round 3 and the further experiments (`e*.sh`). Each step uploads its
log and results to the same repository.

### Declaration of work done outside the hackathon

No code, data or weights from any previous work are used here. Everything in this repository was written
during the hackathon.

## Teammates tagged

Purusharth Malik — Hugging Face [`Purusharth`](https://huggingface.co/Purusharth), LinkedIn
[Purusharth Malik](https://www.linkedin.com/in/purusharth-malik-33a31616b/), GitHub
[`purusharthmalik`](https://github.com/purusharthmalik). Single-member team.

### Author contributions

Purusharth Malik: the idea, every design decision, and review of the results. The implementation was done
with an AI coding assistant (see the LLM usage statement).

### Acknowledgements

Lexsi Labs for AgentTune, AuditKIT and the hackathon's $20 of Hugging Face credit. Cohere Labs for the Aya
models. The rLLM project for the FinQA dataset. NVIDIA for the quantized Qwen teacher model.

## References

- AgentTune 1.1.7: <https://github.com/Lexsi-Labs/AgentTune>; self-healing guide:
  <https://agenttune.lexsi.ai/user-guide/self-healing/>
- AuditKIT 1.1.12: the `auditkit` package on PyPI
- Student models: <https://huggingface.co/CohereLabs/tiny-aya-global>,
  <https://huggingface.co/CohereLabs/aya-expanse-8b>
- Teacher model: <https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4>
- Task data: <https://huggingface.co/datasets/rLLM/rLLM-FinQA-Dataset>
- vLLM 0.33, TRL 1.7.1 and PEFT, for serving and training
- Papers: linked in Related work

## LLM usage statement

All code in this repository and this write-up were written with Claude Code (Anthropic) in one working
session, directed by the author. The author chose the idea, the task, the models, the verifier rule, the
training loss, the gate rule and every other design decision. No LLM judged any result in the tables.

Approximate portion generated by AI: all of the code and nearly all of the write-up text.

## Changelog

- **Version 1:** two rounds on 480 tasks, 0.5% → 30.4%.
- **Version 2:** round 3 on 1,400 tasks (49.1%), Aya Expanse 8B, DPO and the retention check. Title, claim
  and abstract changed. The two-round results are unchanged.
- **Version 3** (this one): reordered to the hackathon's recommended layout, shortened, with related work,
  hypotheses, ablations and a sample run added. No result changed.

## Appendix

### Prompts

**System prompt.** Our text is below; AgentTune adds the tool schemas and the call format to it.

> You are a financial analyst. Answer questions about a company's financial tables by calling the tools.
> Call one tool at a time and wait for its result. When you know the answer, reply with only the final
> number inside `<answer></answer>` tags.

**User turn.** AgentTune's FinQA prompt, shown in "Sample inputs and outputs".

**Repair hints.** One sentence, added to the system prompt for the retry only and removed before a
repaired run becomes a training target. A rule label picks the hint when the run has a format failure;
otherwise the LLM label picks it.

| Failure label | Hint |
|---|---|
| `loop_collapse` (rule or LLM) | Do not repeat a tool call you already made. Use the result you already have. |
| `no_answer_tag` (rule) | When you have the number, reply with only `<answer>NUMBER</answer>`. |
| `step_cap` (rule) | Use at most four tool calls, then reply with only `<answer>NUMBER</answer>`. |
| `tool_errors` (rule), `wrong_tool` (LLM) | Use only the listed tools and their exact argument names. Read the table schema before you query it. |
| `no_tool_evidence` (rule) | Answer only from tool results. Look the values up before you answer. |
| `wrong_routing` (LLM) | List the company's tables first and choose the table whose name matches the question's topic. |
| `incomplete_reasoning` (LLM, and the fallback) | Query the exact rows and years the question asks about, then compute the final number with the calculator. |
| `hallucinated_output` (LLM) | Answer only from tool results. Never guess a table name or a value. |

**Judge rubric.** None. No LLM judge is used.

### Extra tables: rounds 1 and 2 in detail

In these tables, "Final adapter" is the round 2 adapter. The per-run logs, AuditKIT reports and gate
decisions behind every table are in the dataset repository under `eval/`.

**First held-out set with the harness stop rule (100 tasks x 3 runs)**

| Metric | Base Tiny Aya | Round 1 adapter | Final adapter |
|---|---|---|---|
| Rollouts solved (tolerant) | 1 of 300 (0.3%) | 59 of 300 (19.7%) | 72 of 300 (24.0%) |
| Rollouts solved (strict) | 0.7% | 10.7% | 13.3% |
| Tasks solved in all 3 runs | 0 of 100 | 6 of 100 | 10 of 100 |
| Tasks solved in at least 1 run | 1 of 100 | 37 of 100 | 39 of 100 |
| Tool calls per run | 2.77 | 4.55 | 4.31 |
| Tool calls per solved run | 1.00 | 3.90 | 3.86 |
| Output tokens per run | 208 | 232 | 205 |
| Seconds per run | 4.6 | 5.7 | 6.2 |
| Runs with no `<answer>` tag | 117 of 300 | 31 of 300 | 17 of 300 |

**First held-out set without the harness stop rule (100 tasks x 3 runs)**

| Metric | Base Tiny Aya | Round 1 adapter | Final adapter |
|---|---|---|---|
| Rollouts solved (tolerant) | 0 of 300 (0.0%) | 57 of 300 (19.0%) | 75 of 300 (25.0%) |
| Rollouts solved (strict) | 0.0% | 11.0% | 12.3% |
| Tasks solved in all 3 runs | 0 of 100 | 6 of 100 | 11 of 100 |
| Tasks solved in at least 1 run | 0 of 100 | 34 of 100 | 43 of 100 |
| Tool calls per run | 3.86 | 4.50 | 4.36 |
| Tool calls per solved run | n/a | 4.00 | 4.01 |
| Output tokens per run | 620 | 236 | 213 |
| Seconds per run | 18.3 | 5.7 | 5.5 |
| Runs with no `<answer>` tag | 118 of 300 | 30 of 300 | 25 of 300 |

**Base vs final adapter, with stop rule** (difference = second minus first, 95% paired bootstrap interval)

| Metric | First | Second | Difference |
|---|---|---|---|
| Success rate, per task | 0.3% | 24.0% | +23.7 pts [+17.0, +31.0] |
| Tasks solved in all 3 runs | 0.0% | 10.0% | +10.0 pts [+4.0, +16.0] |
| Tasks solved in at least 1 run | 1.0% | 39.0% | +38.0 pts [+29.0, +48.0] |
| Tool calls on tasks both solve (n=0) | n/a | n/a | n/a |
| Tool calls per run | 2.77 | 4.31 | +1.54 [+1.25, +1.84] |
| Output tokens per run | 208 | 205 | -3 [-35, +24] |
| Seconds per run | 4.6 | 6.2 | +1.5 [+0.9, +2.1] |
| Tool-call validity (AuditKIT) | 0.997 | 1.000 | +0.003 [+0.000, +0.007] |
| Redundant tool-call rate (AuditKIT) | 0.153 | 0.049 | -0.104 [-0.135, -0.073] |
| Loop-free share (AuditKIT) | 0.763 | 0.880 | +0.117 [+0.057, +0.173] |

**Base vs round 1 adapter, with stop rule** (difference = second minus first, 95% paired bootstrap interval)

| Metric | First | Second | Difference |
|---|---|---|---|
| Success rate, per task | 0.3% | 19.7% | +19.3 pts [+13.7, +25.7] |
| Tasks solved in all 3 runs | 0.0% | 6.0% | +6.0 pts [+2.0, +11.0] |
| Tasks solved in at least 1 run | 1.0% | 37.0% | +36.0 pts [+26.0, +46.0] |
| Tool calls on tasks both solve (n=0) | n/a | n/a | n/a |
| Tool calls per run | 2.77 | 4.55 | +1.78 [+1.46, +2.11] |
| Output tokens per run | 208 | 232 | +24 [-11, +54] |
| Seconds per run | 4.6 | 5.7 | +1.1 [+0.4, +1.6] |
| Tool-call validity (AuditKIT) | 0.997 | 1.000 | +0.003 [+0.000, +0.007] |
| Redundant tool-call rate (AuditKIT) | 0.153 | 0.089 | -0.064 [-0.097, -0.032] |
| Loop-free share (AuditKIT) | 0.763 | 0.840 | +0.077 [+0.020, +0.140] |

**Round 1 adapter vs final adapter, with stop rule** (difference = second minus first, 95% paired bootstrap interval)

| Metric | First | Second | Difference |
|---|---|---|---|
| Success rate, per task | 19.7% | 24.0% | +4.3 pts [-1.7, +10.7] |
| Tasks solved in all 3 runs | 6.0% | 10.0% | +4.0 pts [-2.0, +10.0] |
| Tasks solved in at least 1 run | 37.0% | 39.0% | +2.0 pts [-9.0, +13.0] |
| Tool calls on tasks both solve (n=24) | 3.74 | 3.79 | +0.06 [-0.12, +0.26] |
| Tool calls per run | 4.55 | 4.31 | -0.24 [-0.44, -0.05] |
| Output tokens per run | 232 | 205 | -27 [-43, -12] |
| Seconds per run | 5.7 | 6.2 | +0.5 [+0.1, +0.8] |
| Tool-call validity (AuditKIT) | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| Redundant tool-call rate (AuditKIT) | 0.089 | 0.049 | -0.040 [-0.062, -0.018] |
| Loop-free share (AuditKIT) | 0.840 | 0.880 | +0.040 [-0.007, +0.087] |

**Base vs final adapter, without stop rule** (difference = second minus first, 95% paired bootstrap interval)

| Metric | First | Second | Difference |
|---|---|---|---|
| Success rate, per task | 0.0% | 25.0% | +25.0 pts [+18.3, +32.0] |
| Tasks solved in all 3 runs | 0.0% | 11.0% | +11.0 pts [+5.0, +17.0] |
| Tasks solved in at least 1 run | 0.0% | 43.0% | +43.0 pts [+33.0, +53.0] |
| Tool calls on tasks both solve (n=0) | n/a | n/a | n/a |
| Tool calls per run | 3.86 | 4.36 | +0.50 [+0.17, +0.83] |
| Output tokens per run | 620 | 213 | -408 [-468, -346] |
| Seconds per run | 18.3 | 5.5 | -12.8 [-14.5, -11.1] |
| Tool-call validity (AuditKIT) | 0.995 | 1.000 | +0.005 [+0.001, +0.010] |
| Redundant tool-call rate (AuditKIT) | 0.296 | 0.054 | -0.242 [-0.287, -0.196] |
| Loop-free share (AuditKIT) | 0.580 | 0.870 | +0.290 [+0.220, +0.353] |

**Held-out set A, the confirmatory set (400 fresh tasks × 3 runs, stop rule on)**

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

**Confirmatory set: round 1 adapter vs final adapter** (difference = second minus first, 95% paired bootstrap interval)

| Metric | First | Second | Difference |
|---|---|---|---|
| Success rate, per task | 25.2% | 30.4% | +5.2 pts [+2.5, +8.1] |
| Tasks solved in all 3 runs | 8.8% | 15.2% | +6.5 pts [+3.2, +10.0] |
| Tasks solved in at least 1 run | 42.0% | 46.8% | +4.8 pts [+0.8, +9.0] |
| Tool calls on tasks both solve (n=140) | 4.10 | 3.98 | -0.11 [-0.24, +0.00] |
| Tool calls per run | 4.65 | 4.47 | -0.18 [-0.27, -0.08] |
| Output tokens per run | 239 | 215 | -24 [-34, -15] |
| Seconds per run | 5.8 | 5.5 | -0.3 [-0.5, -0.2] |
| Tool-call validity (AuditKIT) | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| Redundant tool-call rate (AuditKIT) | 0.108 | 0.069 | -0.040 [-0.053, -0.028] |
| Loop-free share (AuditKIT) | 0.805 | 0.818 | +0.013 [-0.011, +0.039] |

**Confirmatory set: base vs final adapter** (difference = second minus first, 95% paired bootstrap interval)

| Metric | First | Second | Difference |
|---|---|---|---|
| Success rate, per task | 0.5% | 30.4% | +29.9 pts [+26.2, +33.7] |
| Tasks solved in all 3 runs | 0.0% | 15.2% | +15.2 pts [+11.8, +18.8] |
| Tasks solved in at least 1 run | 1.5% | 46.8% | +45.2 pts [+40.2, +50.2] |
| Tool calls on tasks both solve (n=4) | 2.00 | 3.92 | +1.92 [+1.75, +2.00] |
| Tool calls per run | 2.88 | 4.47 | +1.59 [+1.44, +1.75] |
| Output tokens per run | 221 | 215 | -7 [-22, +9] |
| Seconds per run | 5.1 | 5.5 | +0.4 [+0.0, +0.7] |
| Tool-call validity (AuditKIT) | 0.997 | 1.000 | +0.003 [+0.001, +0.005] |
| Redundant tool-call rate (AuditKIT) | 0.126 | 0.069 | -0.057 [-0.073, -0.042] |
| Loop-free share (AuditKIT) | 0.761 | 0.818 | +0.058 [+0.026, +0.089] |
