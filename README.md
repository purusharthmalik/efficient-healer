# Efficient Healer: a Tiny Aya database agent that repairs itself, then gets shorter

**One-line claim.** A two-round self-healing loop takes Tiny Aya from **0.5% to 30.4%** task success on financial tables of companies it never saw, and the second round, trained mostly on the model's own shortest verified runs, adds 5.2 points while cutting redundant tool calls by a third. A deployment gate held that second adapter back until a larger test proved it.

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

## Headline before → after

400 held-out questions from 55 companies never seen in training, 3 runs each (1,200 runs per model):

| Metric | Base Tiny Aya | After round 1 | After round 2 (deployed) |
|---|---|---|---|
| Runs solved | 0.5% | 25.2% | **30.4%** |
| Tasks solved in all 3 runs | 0 of 400 | 35 of 400 | **61 of 400** |
| Tasks solved in at least 1 run | 6 of 400 | 168 of 400 | **187 of 400** |
| Redundant tool-call rate (AuditKIT) | 0.126 | 0.108 | **0.069** |
| Tool-call validity (AuditKIT) | 0.997 | 1.000 | 1.000 |
| Runs that end without a tagged answer | 505 of 1,200 | 183 of 1,200 | **117 of 1,200** |
| Output tokens per run, no harness stop rule (first held-out set) | 620 | 236 | **213** |

Base → round 2 success: +29.9 points, 95% interval [+26.2, +33.7]. Round 1 → round 2: +5.2 points [+2.5, +8.1].

**Base model and method in one line.** `CohereLabs/tiny-aya-global` (3.35B) + one LoRA (r=16), trained with
AgentTune on verified trajectories selected by a two-round self-healing loop, evaluated with AuditKIT.

## For judges: one minute

> - **What:** a self-healing loop for a tool-calling agent, run for two rounds, with a gate before each deploy.
> - **Kits:** AgentTune runs the rollouts, the FinQA tools, the failure classifier, the retraining trigger,
>   the LoRA training and the deployment gate. AuditKIT imports the trajectories and produces every metric
>   and every interval.
> - **Result:** 0.5% → 30.4% success on 400 unseen-company questions; redundant tool calls down by a third
>   in round 2; the model learned to stop after a tool call (620 → 213 output tokens per run).
> - **Most useful finding:** AgentTune's default trainer puts the loss on all tokens, which gave 2% success.
>   Loss on the agent's turns only gave 19.7% with nothing else changed.
> - **Honesty checks:** numeric verifier, no LLM judge; company-disjoint test sets; an evidence guard against
>   lucky answers; a gate that said HOLD when the evidence was thin; negative results kept and published.
> - **Weakness:** seven in ten runs still fail. This is a method and a measured start, not a finished agent.

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

We run the loop twice. Untrained Tiny Aya solves under 1% of tasks, so in round 1 almost everything is
repair. Round 2 rolls out the round 1 adapter; now many tasks are solved reliably, and the shortest-run
selection has material to work on.

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
- **Train pool**: 480 questions from the train split. **Held-out set**: 100 questions from the validation split.
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
| Held-out evaluation | 1,800 | 100 tasks × 3, for base and both adapters, with and without the stop rule |
| Confirmatory evaluation | 3,600 | 400 fresh held-out-company tasks × 3, for base and both adapters |

Training sets (one row per task, the shortest verified run):

| Source of the target | Round 1 set | Round 2 set (final) |
|---|---|---|
| Efficiency: task the student always solves | 0 | 32 |
| Reliability: task the student sometimes solves | 11 | 190 |
| Repair by hinted student retry | 8 | 39 |
| Repair by teacher | 395 | 163 |
| **Rows** | **414** | **424** |
| Used after dropping over-long rows | 393 | 402 |
| Pool tasks with no verified run from any source | 66 | 56 |

In round 1 the student's own runs supply 5% of the targets. In round 2 they supply 61% (261 of 424), and
the teacher's share falls from 95% to 38%.

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

**Compute.** One Hugging Face Job on an RTX PRO 6000 Blackwell (96 GB) holding the student server, the
teacher server and training. Seeds: split seed 0, training seed 0; rollouts are sampled, not seeded.

## Results

All numbers are on held-out companies whose tables were never seen in training. Success is decided by the
numeric verifier, not by a judge model. "Round 1 adapter" was trained on repairs only; "Final adapter" is the
round 2 adapter, trained from the base model on the round 1 adapter's own shortest verified runs plus repairs.

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

### Findings

**1. Where the loss goes decided everything.** Same 393 rows, same LoRA settings: loss on all tokens gave
2.0% held-out success; loss on the agent's own turns gave 19.7%. The agent's decisions are about 5% of a
row's tokens (201 of roughly 3,500), so an all-token loss spends 95% of its effort on text the agent never
has to write.

**2. Round 2 solves more and wastes less.** Against the round 1 adapter on the confirmatory set: success
+5.2 points [+2.5, +8.1]; tasks solved in all 3 runs 8.8% → 15.2% (+6.5 points [+3.2, +10.0]); redundant
tool-call rate −0.040 [−0.053, −0.028]; output tokens per run −24 [−34, −15]. On the 140 tasks both
adapters solve, tool calls fall by 0.11, with an interval that just touches zero [−0.24, +0.00].

**3. The gate earned its place.** On 100 tasks the round 2 gain could not be told from noise, and the gate
held. On 400 tasks it could, and the gate released it. AgentTune's shipped gate approves anything that does
not regress, so it would have deployed on the thin evidence too.

**4. Hints repair a trained model, not an untrained one.** A one-sentence hint chosen by the failure label
repaired 8 of 469 failing tasks for base Tiny Aya and 39 of 258 for the round 1 adapter.

**5. The adapter learned to stop.** Base Tiny Aya writes a tool call and then invents the tool's result.
The adapters solve as much without the harness stop rule as with it (25.0% vs 24.0% for the final adapter
on the first held-out set), and output tokens per run without the rule fall from 620 to 213.

**6. The loop weans itself off the teacher.** Teacher-made targets were 95% of the round 1 training set and
38% of the round 2 set; the student's own verified runs went from 5% to 61%.

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
- **Round 2 did not clearly shorten solved runs.** The efficiency arm's direct target, tool calls on tasks
  both adapters solve, moved from 4.10 to 3.98; the interval [−0.24, +0.00] does not exclude zero. The clear
  efficiency gains are in redundant calls and output tokens.
- **The fine-tuned agent makes more tool calls than the base model** (4.47 vs 2.88 per run), because the
  base model gives up or invents results instead of fetching data. Raw call count against the base model is
  not an efficiency win.

## Limitations

- **About seven in ten held-out runs still fail.** After valid tool calls, most failures are a wrong number
  (211 of 300 runs for the final adapter on the first held-out set): the wrong table or the wrong rows. 424 examples are few for a
  3B model on a task with about 50 look-alike tables per company.
- **Small test sets, one seed.** The first held-out set has 100 tasks; the round 2 gain was not provable on
  it, which is why a confirmatory set exists. Each configuration was trained once. Rollouts are sampled.
- **"Efficient" is relative.** The base model makes fewer tool calls than the adapters (2.77 vs 4.31 per
  run) because it gives up or invents results instead of fetching data. Our efficiency claims are between
  adapters, and in redundant-call rate and output tokens, not in raw call count against the base model.
- **No retention or safety check after fine-tuning.** We did not measure whether the adapter changes
  Tiny Aya's multilingual ability, general ability or safety behaviour. AgentTune's licence asks for a
  safety re-check before real deployment; this adapter is a research artifact.
- **English only**, although the base model is multilingual.
- **The teacher is a larger, non-Cohere model.** The student's ceiling here is the teacher's verified runs.
- **The failure labels are not validated by people.** We only report agreement between the rule labels and
  AgentTune's LLM labels (115 of 142 in round 1, 33 of 45 in round 2, where the two are comparable).
- **The task prompt lists suggested steps**, as shipped by AgentTune, so the agent is not planning from scratch.
- **Tolerant verifier.** An answer within 0.5% counts. The strict rate is always reported beside it.

## With more credits

**Planned approach**
1. **More data through the same loop.** About 2,800 train-split questions from the non-held-out companies
   are unused. The teacher runs at about 150 rollouts per minute, so the whole pool is under an hour of GPU.
2. **More rounds.** Repeat collect → bucket → repair → retrain until the gate holds twice in a row, so the
   share of student-made targets keeps rising and the teacher is only used on the hard tail.
3. **A reinforcement-learning stage on top.** The published preference pairs (verified run vs real failed
   run) feed DPO directly. After that, GRPO with a group-relative step reward from our earlier work on
   length-penalised RL: `R = ok · (1 − c·tanh(λ·p̂²·(steps − steps_min)/c))`, which penalises extra tool
   calls only on tasks the model already solves reliably.
4. **A proper held-out study.** All 1,244 fresh held-out-company questions, 5 runs each, several seeds.
5. **Retention checks with AuditKIT**: GSM8K and MMLU subsets, XSTest and MultiJail, and FinQA-style
   questions in other languages.
6. **Aya Expanse 8B** as a second student through the identical loop.

**Expected before / after (a forecast, not a measurement).** With six times the data and two more rounds we
expect held-out success to move from about 24% to 40–50%, the share of tasks solved in all 3 runs from 10%
to 20–25%, and tool calls per solved run from 3.9 toward 3.5.

## Cost

**Compute used.** Hugging Face Jobs, paid from the hackathon's $20 credit: an L40S for 37 minutes of setup
($1.10) and one RTX PRO 6000 Blackwell (96 GB) for 2 h 43 min ($7.47). **Total $8.57.** That covers about
14,500 student rollouts, 1,600 teacher rollouts, three LoRA trainings of 12–16 minutes each, and all
evaluation.

**Cost per 1,000 agent runs** (a run is a whole task: about 5 model calls and 4 tool calls):

| Option | Throughput | Cost per 1,000 runs | Success | Cost per 1,000 solved tasks |
|---|---|---|---|---|
| Tiny Aya + this adapter, one RTX PRO 6000 at $2.75/h | 467 runs/min, measured | $0.10 | 30.4%, measured | $0.32 |
| Teacher, Qwen3.6-35B NVFP4, same GPU | 148 runs/min, measured | $0.31 | 77.9%, measured on a different task set | $0.40 |
| A closed API at an assumed $3 / $15 per million input / output tokens | not measured | $54, estimated | not measured | not measured |

How the numbers are made: throughput was measured with 48 agents running at once and tool execution
included, while the GPU also held other servers. The closed-API row is an estimate from our own token
counts (17,039 input and 205 output tokens per run, no prompt caching) at assumed list prices; we did not
run a closed model. The honest reading: per attempt the adapter is about 3 times cheaper than the 35B
teacher and hundreds of times cheaper than the API estimate, but per *solved* task it is only slightly
ahead of the teacher at today's 30% success. The economics improve directly with the success rate, which
is what more rounds are for. The student's weights are 6.7 GB in bf16; the teacher's quantized weights are 23.5 GB.

## Artifacts

- **Model (adapters):** [`Purusharth/tiny-aya-efficient-healer-lora`](https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora)
  — the gate-approved adapter at the root; every adapter trained is in its own folder (`round1/`, `round2/`,
  `negative_all_token_loss/`).
- **Dataset:** [`Purusharth/efficient-healer-trajectories`](https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories)
  — every rollout, failure label, repair hint, training set, preference pair, evaluation log, AuditKIT
  report and gate decision.
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
  train.py           AgentTune create_distill_trainer (LoRA)
  merge.py           merge an adapter for serving
  evaluate.py        AuditKIT import, metrics, paired comparison
  gate.py            the deployment rule
infra/
  worker.sh          runs queued steps inside one long Hugging Face Job
  cmds/              one script per pipeline step, in run order
PLAN.md              the build log: every decision, measurement and change, with times
```

Three Python environments, because their pins differ: AgentTune (`agenttune` at commit `8918cc3`),
vLLM (`vllm>=0.30`, the first with Cohere support), AuditKIT (`auditkit[transformers,interop,jsonschema]==1.1.12`).
`infra/cmds/000_setup.sh` builds all three. The steps then run in the order of `infra/cmds/`:
setup, baseline evaluation, round 1 (collect, repair, train), round 2 (collect, repair, train), final
evaluation and gates. Hardware used: one RTX PRO 6000 Blackwell (96 GB).

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
