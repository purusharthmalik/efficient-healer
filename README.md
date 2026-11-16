# Efficient Healer: a Tiny Aya database agent that repairs itself, round after round

**One-line claim.** A self-healing loop takes Tiny Aya from **0.5% to 49.1%** task success on financial tables of companies it never saw, in three rounds. A deployment gate decided every release, and held one adapter back until a larger test proved it.

Lexsi Labs hackathon, Track 3 (Agentic Fine-tuning). Built on 16 November 2026.

## Tracks and problem statements covered

- Track 3, **Self-healing agents**: detect failing runs, classify why they failed, generate corrective
  training examples, retrain, and gate redeployment on measured accuracy.
- Track 3, **Efficient trajectories**: cut redundant and failed tool calls without lowering success.
- Track 3, **Data and database agents**: an agent that answers finance questions by querying a database,
  rewarded on verified results.
- Track 3, **Agentic distillation** (supporting): a stronger teacher's verified runs repair the tasks the
  student cannot solve.

## Abstract

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

## Headline before → after

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

**Base model and method in one line.** `CohereLabs/tiny-aya-global` (3.35B) + one LoRA (r=16), trained with
AgentTune on verified trajectories selected by a three-round self-healing loop, evaluated with AuditKIT.

## For judges: one minute

> - **What:** a self-healing loop for a tool-calling agent, run for three rounds, with a gate before each deploy.
> - **Kits:** AgentTune runs the rollouts, the FinQA tools, the failure classifier, the retraining trigger,
>   the LoRA training, DPO and the deployment gate. AuditKIT imports the trajectories and produces every
>   metric, every interval and the retention and safety check.
> - **Result:** 0.5% → 49.1% success on 400 unseen-company questions; tasks solved in all 3 runs 0 → 143;
>   redundant tool calls down by 60%; the model learned to stop after a tool call.
> - **Most useful findings:** AgentTune's default trainer puts the loss on all tokens, which gave 2% success;
>   loss on the agent's turns only gave 19.7% with nothing else changed. After that, data volume through
>   the same loop was the biggest lever.
> - **Honesty checks:** numeric verifier, no LLM judge; company-disjoint test sets; an evidence guard against
>   lucky answers; a gate that said HOLD when the evidence was thin; a void run caught and thrown away;
>   negative results kept and published.
> - **Also tried, same day:** Aya Expanse 8B as the student (38.9% on the round 2 data), DPO on step-level
>   preference pairs (+5 points), a retention and safety check.
> - **Weakness:** half of the runs still fail. This is a method and a measured path, not a finished agent.

## Problem and approach

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

### Why this problem

Finance teams ask questions that live in tables: changes, shares, growth rates. A closed frontier model
answers them well but is priced per token; a 3B open model is cheap to host but unreliable at multi-step
tool use. Closing that gap with a self-improving loop, and with a deployment gate a risk team can read, is
the practical version of "small model, real work".

## Hackathon repos used

### AgentTune (commit `8918cc3`)
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

## Models and licences

| Role | Model | Licence | Notes |
|---|---|---|---|
| Student (submitted) | `CohereLabs/tiny-aya-global`, 3.35B, bf16 | CC-BY-NC-4.0 | LoRA r=16 on attention projections |
| Teacher and failure classifier | `nvidia/Qwen3.6-35B-A3B-NVFP4` | Apache-2.0 | Supplementary only; never fine-tuned or submitted |

Aya Expanse 8B was measured as an alternative student and gave no gain (see Negative results).
The adapter inherits CC-BY-NC-4.0. AgentTune and AuditKIT are used under the Lexsi Labs Source Available
License for non-commercial research.

## Data

- **Source**: FinQA as shipped by AgentTune (`rLLM/rLLM-FinQA-Dataset`): 4,030 train and 522 validation
  questions over the financial tables of 185 companies, in English. Answers are numbers.
- **Train pool**: 480 questions from the train split in rounds 1 and 2, grown to 1,400 in round 3 (the 480 are a prefix of the 1,400).
- **Held-out sets**, all from 55 companies reserved for testing: a first set of 100 validation questions;
  set A, 400 fresh questions used once to confirm the round 2 gate; set B, 400 more fresh questions used
  for the round 3 gate. No question appears in two sets.
- **Leak check**: the split is company-disjoint. 55 companies are reserved for the held-out set, so no
  held-out table is ever seen in training. Shared companies: 0. Shared question texts: 0 (exact match).
- **Trajectories**: all runs are published, successes and failures alike.

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

**No LLM judge scores any result.** Success is decided by a numeric verifier. The only LLM-produced labels
are failure root causes, which choose repair hints and never enter a metric.

## Method and setup

**Task and tools.** Each task is a question about one company's financial tables. The agent has four
tools from AgentTune: `list_tables(company_prefix)`, `get_table_schema(table_name)`,
`query_finqa_tables(sql_command)` and `calculator(expression)`. It must end with the number inside
`<answer></answer>`. A run is cut off after 8 tool calls.

**Rollouts.** Temperature 0.7, top-p 0.95, at most 384 new tokens per turn for the student (1,024 for the
teacher). Tiny Aya's chat template has no tool support, so AgentTune puts the tool schemas in the system
prompt and asks for calls as `[{"tool_name": ..., "parameters": {...}}]`; tool results come back as user turns.
Generation runs on vLLM 0.33 behind our `RolloutEngine`; one trajectory per thread, 48 at a time.

**Harness stop rule.** Untrained Tiny Aya does not stop after its tool call: it writes the call and then
invents the tool's result. A stop rule ends each turn at the end of the first call list or at `</answer>`.
Training data is collected with the rule. Evaluation is reported both with and without it.

**Buckets and targets.** k=4 rollouts per pool task. The target of a task is its shortest verified run that
passes the evidence guard. In a target, a tool-call turn is only the call list; the final turn keeps the
written working and ends at `</answer>`; an exact repeat of the previous call and result is removed; the
repair hint is removed from the system prompt.

**Failure labels.** Rules read off the trajectory (`loop_collapse`, `no_answer_tag`, `step_cap`,
`tool_errors`, `no_tool_evidence`, `wrong_number`), plus AgentTune's `FailureClassifier` (five root causes)
served by the teacher model, one failed run per task. AgentTune's `FailureDetector` runs alongside.

**Repair ladder.** (1) The current student retries 4 times with a one-sentence hint chosen by the label.
(2) The teacher runs 3 times, thinking off. A repair counts only if the verifier accepts it and it passes
the evidence guard.

**Training.** AgentTune `create_distill_trainer` with a custom trainer factory: TRL `SFTTrainer`, PEFT LoRA
r=16, α=32, dropout 0.05 on `q_proj, k_proj, v_proj, o_proj`, 3 epochs, learning rate 2e-4 with cosine decay
and 5% warm-up, effective batch 16 (2 × 8 accumulation), bf16, seed 0. Rows longer than the 95th-percentile
length (4,382 tokens in round 1, 4,278 in round 2) are dropped, not truncated. **The loss covers only the agent's own turns** (its calls,
its final answer, the end-of-turn tokens); see Negative results for why.

**Gate ("improve, never regress").** On the held-out tasks, a new adapter replaces the deployed model only if
success does not fall by more than 2 points, tool-call validity does not fall by more than 2 points, and
either success or tool calls on tasks both models solve improves with a 95% interval that excludes zero.
AgentTune's `DeploymentGate.evaluate_decision` must also approve. Applied once per round.

**Round 3.** The same loop from the round 2 adapter on the 1,400-task pool. One difference: the failure
classifier ran with the teacher's thinking switched off to save time (rounds 1 and 2 had it on).

**DPO (further experiment).** AgentTune's agentic DPO trainer (`create_agentic_trainer(algorithm="dpo")`,
TRL `DPOTrainer`) on step-level preference pairs. For a task with a verified run and a failed run, the
prompt is the conversation up to the first agent turn where the two differ; chosen and rejected are that
one turn. LoRA r=8 on top of a merged round adapter, learning rate 2e-5, β 0.1, 1 epoch, batch 1 × 4.

**Compute.** Hugging Face Jobs on RTX PRO 6000 Blackwell GPUs (96 GB), each holding the student server, the
teacher server and training. Seeds: split seed 0, training seed 0; rollouts are sampled, not seeded.

## Results

All numbers are on held-out companies whose tables were never seen in training. Success is decided by the
numeric verifier, not by a judge model. "Round 1 adapter" was trained on repairs only. In the tables for
rounds 1 and 2 below, "Final adapter" means the round 2 adapter, trained from the base model on the round 1
adapter's own shortest verified runs plus repairs. Round 3 comes first because it is the deployed result;
the two-round tables after it are the path that led there.

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

### Held-out results with the harness stop rule (100 tasks x 3 runs)

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

### Held-out results without the harness stop rule (100 tasks x 3 runs)

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

### By question type (stop rule on; solved runs, share of that type's runs)

| Question type | Runs | Base | Round 1 adapter | Final adapter |
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

**Round 1 (round 1 adapter vs base): DEPLOY.** Success rose by 19.3 points, 95% interval [+13.7, +25.7].
Tool-call validity did not fall. AgentTune's `DeploymentGate` also approved.

**Round 2 (final adapter vs round 1 adapter), first held-out set: HOLD.** The final adapter is ahead on every
point estimate, but on 100 tasks the success gain of 4.3 points has a 95% interval of [−1.7, +10.7], which
includes zero, and tool calls on the 24 tasks both adapters solve did not change (+0.06 [−0.12, +0.26]).
Our rule needs one of those two to improve beyond doubt. It does not accept the gains that *are* significant
(tool calls per run −0.24, output tokens −27, redundant-call rate −0.040), because a model can lower those
by giving up earlier. AgentTune's own gate approved the same adapter, since it passes anything that does
not regress.

**Round 2, confirmatory set: DEPLOY.** Rather than change the rule, we ran one pre-declared confirmation: 400
fresh questions from the same 55 held-out companies (1,244 were available and unused), 3 runs each, the
same rule applied once, its answer accepted either way. The success gain is +5.2 points with a 95% interval
of [+2.5, +8.1], above zero, so the final adapter is the deployed one. On this set the round 1 gate also
passes again (+24.7 points [+21.4, +28.1]).

### Confirmatory held-out set (400 fresh tasks × 3 runs, stop rule on)

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

### Further experiments (same day, after the three rounds were designed)

**Aya Expanse 8B as the student.** We trained a LoRA on `CohereLabs/aya-expanse-8b` with the unchanged
round 2 training set (402 rows, same settings) to ask whether the loop's data transfers and whether
capacity limits Tiny Aya.

| 400 tasks × 3 runs, stop rule on | Tiny Aya, round 2 adapter | Aya Expanse 8B, same data | Difference [95% interval] |
|---|---|---|---|
| Rollouts solved, set A | 365 of 1,200 (30.4%) | 467 of 1,200 (38.9%) | +8.5 pts [+4.5, +12.6] |
| Rollouts solved, set B | 314 of 1,200 (26.2%) | 438 of 1,200 (36.5%) | +10.3 pts [+6.7, +13.9] |
| Tool calls per run, set A | 4.47 | 4.05 | −0.42 [−0.53, −0.31] |
| Redundant tool-call rate, set A | 0.069 | 0.019 | [−0.060, −0.039] |
| Loop-free share, set A | 0.818 | 0.950 | |

Untrained Aya Expanse 8B solves 0 of 300 runs on the first 100 tasks of set A. So the data produced by the
Tiny Aya loop lifts a second Cohere student from zero to 38.9%, and the larger student is both more
accurate and leaner on identical data. Tiny Aya with round 3's larger set (49.1%) still beats it.

**DPO on step-level preference pairs.** 887 pairs came out of round 2 (297 at the table choice, 519 at the
query step); 480 were used, on top of the round 2 adapter, for 11.5 minutes of training.

| 400 tasks × 3 runs, stop rule on | Round 2 adapter | Round 2 + DPO | Difference [95% interval] |
|---|---|---|---|
| Rollouts solved, set A | 365 of 1,200 (30.4%) | 429 of 1,200 (35.8%) | +5.3 pts [+2.4, +8.2] |
| Rollouts solved, set B | 314 of 1,200 (26.2%) | 358 of 1,200 (29.8%) | +3.7 pts [+1.1, +6.3] |
| Tasks solved in all 3 runs, set A | 61 of 400 | 89 of 400 | +7.0 pts [+2.8, +11.2] |
| Tool calls per run, set A | 4.47 | 4.71 | +0.24 [+0.15, +0.33] |
| Loop-free share, set A | 0.818 | 0.764 | |

The gate says DEPLOY on both sets, because success improves beyond doubt. But DPO buys success, not
efficiency: runs get slightly longer and loops slightly more frequent.

**DPO on top of round 3: held by the gate.** The same recipe on the round 3 adapter (320 of 2,158 pairs,
8 minutes) scores 582 of 1,200 on set B (48.5%) against round 3's 557 (46.4%). The gain is +2.1 points with
an interval of [−0.3, +4.3], which includes zero, and tool calls on tasks both solve do not change
([−0.08, +0.06]). The gate says HOLD, so round 3 stays deployed. With fewer pairs and a stronger starting
model, DPO's loss barely moved (0.693 → 0.691); it likely needs more pairs or more steps here.

**Retention and safety check (AuditKIT).** Base Tiny Aya against the round 2 adapter, 100 samples each, plain
chat with no tools, through AuditKIT's `api:` backend.

| Check | Base | Round 2 adapter | Difference [95% interval] |
|---|---|---|---|
| GSM8K, numeric match | 0.71 | 0.67 | −0.04 [−0.12, +0.05] |
| XSTest, refusal of unsafe prompts (higher is safer) | 0.94 | 0.94 | 0.00 [−0.04, +0.04] |
| XSTest, over-refusal of safe prompts (lower is better) | 0.06 | 0.06 | 0.00 [−0.04, +0.04] |

No measurable loss of maths ability and no change in refusal behaviour. We also ran Global-MMLU in five
languages; both models scored 0.05, below the 25% chance level of a four-choice test, which means the
answer extraction failed, so we do not report it as a measurement. The round 3 adapter was not re-checked.

### Findings

**1. Where the loss goes decided everything.** Same 393 rows, same LoRA settings: loss on all tokens gave
2.0% held-out success; loss on the agent's own turns gave 19.7%. The agent's decisions are about 5% of a
row's tokens (201 of roughly 3,500), so an all-token loss spends 95% of its effort on text the agent never
has to write.

**2. After that, data through the same loop was the biggest lever.** Round 3 changed nothing but the pool
(480 → 1,400 tasks) and gained 18.7 points on set A and 20.2 on set B. Tasks solved in all 3 runs more than
doubled (61 → 143 of 400).

**3. Round 2 solves more and wastes less.** Against the round 1 adapter on the confirmatory set: success
+5.2 points [+2.5, +8.1]; tasks solved in all 3 runs 8.8% → 15.2% (+6.5 points [+3.2, +10.0]); redundant
tool-call rate −0.040 [−0.053, −0.028]; output tokens per run −24 [−34, −15]. On the 140 tasks both
adapters solve, tool calls fall by 0.11, with an interval that just touches zero [−0.24, +0.00].

**4. The gate earned its place.** On 100 tasks the round 2 gain could not be told from noise, and the gate
held. On 400 tasks it could, and the gate released it. AgentTune's shipped gate approves anything that does
not regress, so it would have deployed on the thin evidence too.

**5. Hints repair a trained model, not an untrained one.** A one-sentence hint chosen by the failure label
repaired 8 of 469 failing tasks for base Tiny Aya, 39 of 258 for the round 1 adapter and 87 of 737 for the
round 2 adapter.

**6. The adapter learned to stop.** Base Tiny Aya writes a tool call and then invents the tool's result.
The adapters solve as much without the harness stop rule as with it (25.0% vs 24.0% for the final adapter
on the first held-out set), and output tokens per run without the rule fall from 620 to 213.

**7. The loop weans itself off the teacher.** Teacher-made targets were 95% of the round 1 training set and
38% of the round 2 and round 3 sets; the student's own verified runs went from 5% to 61% and 62%, even
though two thirds of round 3's tasks were new to it.

**8. The data transfers, and capacity matters.** Aya Expanse 8B trained on Tiny Aya's round 2 set reaches
38.9%, against 30.4% for Tiny Aya on the same rows, with fewer and cleaner tool calls.

**9. Preference learning adds success, not efficiency.** DPO on step-level pairs adds 4 to 5 points on top
of the round 2 adapter in under 12 minutes, at the price of slightly longer runs.

**10. Efficiency shows up as less waste.** Across rounds the redundant tool-call rate falls
0.126 → 0.108 → 0.069 → 0.051 and the loop-free share rises 0.76 → 0.89. Tool calls on tasks two adapters
both solve fall only by 0.1 to 0.2, and not on every test set.

## Negative results

- **Loss on all tokens does not teach the agent's decisions.** Our first LoRA used AgentTune's default
  trainer, which trains on every token of the conversation. Loss fell from 1.74 to 0.30, yet held-out
  success was 6 of 300 (2.0%) and success on its own training tasks was 42 of 1,920 (2.2%). It had learned
  the shape of a run but picked its training target's table only 31% of the time. A row is about 3,500
  tokens of prompt and tool results and about 150 tokens of agent turns, so the decisions carried about 4%
  of the loss. Restricting the loss to agent turns, with the same data and settings, raised held-out success to
  59 of 300 (19.7%) and train-task success to 505 of 1,920 (26.3%).
- **Hints alone do not repair an untrained model.** Label-chosen hints on 469 failing tasks: 10 of 1,876
  retries solved (0.5%), no better than without a hint (0.8%).
- **A bigger Cohere student did not help.** Untrained Aya Expanse 8B solved 0 of 40 pilot rollouts, the
  same as Tiny Aya, at twice the latency.
- **A harness stop rule alone does not help success.** It halves output tokens for the base model
  (621 → 208 per run) but success stays at 0–0.3%.
- **"Shortest success" picks lucky runs unless guarded.** One base-model "success" made a single
  `list_tables` call, invented the remaining tool results in its final turn, and answered `0` to a question
  whose gold answer is `0`. The evidence guard exists because of it.
- **A tolerant verifier needs care.** Our first percent-vs-fraction rule accepted `4%` for a gold of `0`.
  We fixed the rule and re-scored every run.
- **A void run, caught and discarded.** The first Aya Expanse evaluation scored 0.2%. The cause was ours:
  that GPU job had never built the FinQA database, so every table lookup returned "No tables found". The
  tools now build the database themselves and refuse to run on an empty one. The void runs are published
  under `exploration/void_expanse_run_without_database/` so the mistake is inspectable.
- **Global-MMLU did not measure anything.** Both models scored 5% on a four-choice test, so the answer
  extraction failed. We report the multilingual retention check as not done.
- **DPO makes runs longer.** Tool calls per run rose from 4.47 to 4.71 and the loop-free share fell from
  0.82 to 0.76, although success rose.
- **The LLM failure labels got worse when we rushed them.** With the classifier's thinking switched off in
  round 3, rule and LLM labels agreed on 27 of 58 comparable cases, against 115 of 142 in round 1.
- **Round 2 did not clearly shorten solved runs.** The efficiency arm's direct target, tool calls on tasks
  both adapters solve, moved from 4.10 to 3.98; the interval [−0.24, +0.00] does not exclude zero. The clear
  efficiency gains are in redundant calls and output tokens.
- **The fine-tuned agent makes more tool calls than the base model** (4.47 vs 2.88 per run), because the
  base model gives up or invents results instead of fetching data. Raw call count against the base model is
  not an efficiency win.

## Limitations

- **Half of the held-out runs still fail.** After valid tool calls, most failures are a wrong number: the
  wrong table or the wrong rows, on a task with about 50 look-alike tables per company.
- **One seed, and test sets that were reused.** Each configuration was trained once; rollouts are sampled.
  Set A decided the round 2 gate and was then reused to compare Expanse, DPO and round 3; set B decided the
  round 3 gate and was reused to re-test Expanse and DPO. No training ever saw either set, but several
  comparisons on the same questions are not independent tests.
- **"Efficient" is relative.** The base model makes fewer tool calls than the adapters (2.77 vs 4.31 per
  run on the first held-out set, 4.32 vs 2.88 for round 3 on set A) because it gives up or invents results
  instead of fetching data. Our efficiency claims are between adapters, and in redundant-call rate and
  output tokens, not in raw call count against the base model.
- **Retention and safety are only partly checked.** On 100 samples each, the round 2 adapter does not
  measurably change GSM8K accuracy or XSTest refusal behaviour. Multilingual ability was not measured, and
  the deployed round 3 adapter was not re-checked. AgentTune's licence asks for a safety re-check before
  real deployment; these adapters are research artifacts.
- **English only**, although the base model is multilingual.
- **The teacher is a larger, non-Cohere model.** The student's ceiling here is the teacher's verified runs.
- **The failure labels are not validated by people.** We only report agreement between the rule labels and
  AgentTune's LLM labels (115 of 142 in round 1, 33 of 45 in round 2, where the two are comparable).
- **The task prompt lists suggested steps**, as shipped by AgentTune, so the agent is not planning from scratch.
- **Tolerant verifier.** An answer within 0.5% counts. The strict rate is always reported beside it.

## With more credits

Several items from our first list are now done (a third round on more data, DPO, a retention check, a
second student). What we would do next:

**Planned approach**
1. **Finish the pool.** About 1,400 train-split questions from the non-held-out companies are still
   unused. Round 3 showed that data through this loop is the strongest lever.
2. **Combine what worked.** Aya Expanse 8B through the full three-round loop (it reached 38.9% on round 2's
   data alone), then DPO on its own failures.
3. **More rounds, until the gate holds twice in a row**, so the teacher is only used on the hard tail.
4. **GRPO on top**, with a group-relative step reward from our earlier work on length-penalised RL:
   `R = ok · (1 − c·tanh(λ·p̂²·(steps − steps_min)/c))`, which penalises extra tool calls only on tasks the
   model already solves reliably. This targets the one thing the loop moved least: tool calls per solved task.
5. **A proper held-out study.** All remaining fresh held-out-company questions, 5 runs each, several
   seeds, each test set used for one decision only.
6. **Retention checks done properly** with AuditKIT: Global-MMLU with a working answer extractor, MultiJail,
   and FinQA-style questions in other languages, on the deployed adapter.

**Expected before / after (a forecast, not a measurement).** With the full pool, Aya Expanse 8B as the
student and a DPO stage, we expect held-out success to move from 49% to 60–65%, the share of tasks solved
in all 3 runs from 36% to about 50%, and tool calls per solved run from 3.9 toward 3.5.

## Cost

**Compute used.** Hugging Face Jobs, paid from the hackathon's $20 credit.

| Part | GPU | Time | Cost |
|---|---|---|---|
| Setup and first smoke test | L40S | 37 min | $1.10 |
| Rounds 1 and 2, evaluation, gates, confirmation | RTX PRO 6000 Blackwell | 2 h 43 min | $7.47 |
| Round 3, DPO on round 3 | RTX PRO 6000 Blackwell | 1 h 55 min | $5.25 |
| Aya Expanse 8B, retention check, DPO, re-tests | RTX PRO 6000 Blackwell | 1 h 32 min | $4.23 |
| **Total** | | **6 h 47 min** | **$18.05** |

That covers about 34,000 student rollouts, 3,100 teacher rollouts, five LoRA trainings (12 to 34 minutes
each), two DPO runs (8 and 12 minutes) and all evaluation.

**Cost per 1,000 agent runs** (a run is a whole task: about 5 model calls and 4 tool calls):

| Option | Throughput | Cost per 1,000 runs | Success | Cost per 1,000 solved tasks |
|---|---|---|---|---|
| Tiny Aya + round 3 adapter, one RTX PRO 6000 at $2.75/h | 522 runs/min, measured | $0.09 | 49.1%, measured | $0.18 |
| Aya Expanse 8B + adapter, same GPU | 391 runs/min, measured | $0.12 | 38.9%, measured | $0.30 |
| Teacher, Qwen3.6-35B NVFP4, same GPU | 148 runs/min, measured | $0.31 | 77.9%, measured on a different task set | $0.40 |
| A closed API at an assumed $3 / $15 per million input / output tokens | not measured | $54, estimated | not measured | not measured |

How the numbers are made: throughput was measured with 48 agents running at once and tool execution
included, while the GPU also held other servers. The closed-API row is an estimate from our own token
counts (about 17,000 input and 210 output tokens per run, no prompt caching) at assumed list prices; we did
not run a closed model. The honest reading: per attempt the Tiny Aya adapter is about 3 times cheaper than
the 35B teacher and hundreds of times cheaper than the API estimate. Per *solved* task it was only level
with the teacher after round 2 (30% success, $0.32); after round 3 it is about half the teacher's cost. The
economics improve directly with the success rate. The student's weights are 6.7 GB in bf16; the teacher's
quantized weights are 23.5 GB.

## Artifacts

- **Model (Tiny Aya adapters):** [`Purusharth/tiny-aya-efficient-healer-lora`](https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora)
  — the gate-approved round 3 adapter at the root; every adapter trained is in its own folder (`round1/`,
  `round2/`, `round3/`, `round2_dpo/`, `round3_dpo/`, `negative_all_token_loss/`).
- **Model (Aya Expanse 8B adapter):** [`Purusharth/aya-expanse-8b-efficient-healer-lora`](https://huggingface.co/Purusharth/aya-expanse-8b-efficient-healer-lora)
  — trained on the round 2 set.
- **Dataset:** [`Purusharth/efficient-healer-trajectories`](https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories)
  — every rollout, failure label, repair hint, training set, preference pair, evaluation log, AuditKIT
  report and gate decision, for all three rounds and the further experiments.
- **Collections:** [Efficient Healer — models](https://huggingface.co/collections/Purusharth/efficient-healer-models-6afaa8b6120d5e7cae162d33) and [Efficient Healer — datasets](https://huggingface.co/collections/Purusharth/efficient-healer-datasets-6afaa8b70386ac5f12a8f637).
- **Code:** this repository. `PLAN.md` is the build log.

## Repo layout and how to reproduce

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
PLAN.md              the build log: every decision, measurement and change, with times
```

Three Python environments, because their pins differ: AgentTune (`agenttune` at commit `8918cc3`),
vLLM (`vllm>=0.30`, the first with Cohere support), AuditKIT (`auditkit[transformers,interop,jsonschema]==1.1.12`).
`infra/cmds/000_setup.sh` builds all three. The steps then run in the order of `infra/cmds/`:
setup, baseline evaluation, round 1 (collect, repair, train), round 2 (collect, repair, train), evaluation
and gates, then round 3 and the further experiments (`e*.sh`). Hardware used: RTX PRO 6000 Blackwell (96 GB).

## Declaration of work done outside the hackathon

The idea of replaying the shortest correct sample was noted in our earlier work on length-penalised
reinforcement learning for math reasoning (October 2026, a different model and codebase). No code, data or
weights from that work are used here. Everything in this repository was written on 16 November 2026.

## LLM usage statement

All code in this repository and this write-up were written with Claude Code (Anthropic) in one working
session, directed by the author. The author chose the idea, the task, the models, the verifier rule, the
training loss, the gate rule and every other design decision; `PLAN.md` logs each one with its time. The
assistant proposed options, implemented, ran and debugged. No LLM judged any result in the tables.

## Team

Purusharth Malik — Hugging Face [`Purusharth`](https://huggingface.co/Purusharth), GitHub
[`purusharthmalik`](https://github.com/purusharthmalik). Single-member team: idea, design decisions and
review.
