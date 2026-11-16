# Self-healing, shorter trajectories for Tiny Aya — build plan

Lexsi hackathon, Track 3 (Agentic Fine-tuning). Written 16 Nov 2026, 08:50 IST.
**Hard deadline: 17:29 IST today** (Nov 15, 23:59 AoE). Target: submitted by ~15:00 IST.

## 0. Status and decision log (last updated 14:35 IST)

**Result.** On 400 held-out questions from companies never seen in training (3 runs each), task success is
0.5% for base Tiny Aya, 25.2% after round 1, 30.4% after round 2 and **49.1% after round 3**. Round 3 is the
deployed adapter: its gate passed on a fresh 400-task set (26.2% → 46.4%). Further experiments the same
day: Aya Expanse 8B on the round 2 data reaches 38.9%; DPO adds about 5 points to round 2 and is held by
the gate on round 3; the adapter does not measurably change GSM8K or XSTest behaviour.
Total GPU spend **$18.05 of $20**. All GPU jobs are stopped.

**Status**

| Phase | State | Notes |
|---|---|---|
| 0 Setup | done | HF Jobs worker, three environments |
| 1 Smoke and pilot | done | Base Tiny Aya near 0%; teacher 82.5% on the pilot |
| 2 Collect | done | Three rounds: 480, 480 and 1,400 tasks × 4 |
| 3 Classify and repair | done | Rules + AgentTune classifier, hinted retries, teacher |
| 4 Train | done | Five LoRAs (all-token loss failed and kept; rounds 1, 2, 3; Aya Expanse 8B) and two DPO runs |
| 5 Evaluate and gate | done | AuditKIT on three held-out sets; gates: round 1 DEPLOY, round 2 HOLD then DEPLOY on confirmation, round 3 DEPLOY, DPO on round 2 DEPLOY, DPO on round 3 HOLD |
| 6 Ship | public, resubmission pending | Everything is public and updated for round 3. **The form must be submitted again**: title, claim and abstract changed |

**What is left (14:35 IST)**

| Step | Who |
|---|---|
| Submit the form again at `hackathon.lexsi.ai/submission` from `SUBMISSION.md` (title, claim and abstract changed) | you, before 17:29 IST |
| Optional: paste `README.md` into a Hugging Face article | you |

**Decisions made by you**

| When (IST) | Decision | Choice |
|---|---|---|
| 08:44 | Training method | SFT today; RL kept for "with more credits" |
| 09:14 | Answer verifier | Tolerant: rounding to the gold's precision or within 0.5%, and percent = fraction. Strict rate reported beside it |
| 09:14 | Teacher and failure-classifier model | `nvidia/Qwen3.6-35B-A3B-NVFP4` on the same GPU |
| 09:14 | Tasks solved 1–3 of 4 times | Train on the shortest successful run |
| 09:14 | GitHub repo | `purusharthmalik/efficient-healer` (exists, private) |
| 09:24 | GPU | RTX PRO 6000 (Blackwell, 96 GB, $2.75/h). H100 is not offered by HF Jobs today |
| 09:50 | Student | Stays Tiny Aya. Aya Expanse 8B was measured as a possible student and gave no gain |
| 09:50 | Pipeline structure | **Two rounds.** Round 1: teacher repairs failing tasks, train a first LoRA. Round 2: roll out with it; always-solved and sometimes-solved tasks give shortest-success targets, still-failing tasks are repaired again. The final LoRA is trained once from the base model on both sets |
| 09:50 | Harness stop rule | Used for collecting training data. The before/after evaluation is run both with and without it, and both tables are reported |
| 10:05 | Train-pool size | 480 tasks (was 240). Teacher runs 3 times per task |
| 10:05 | Training target shape | A tool-call turn is only the call list. The final turn keeps the written working and ends at `</answer>` |
| 10:05 | Training settings | LoRA r=16, α=32, 3 epochs, lr 2e-4 cosine, effective batch 16 |
| 11:05 | Training loss | **Agent turns only.** Labels cover the agent's tool calls, final answer and end-of-turn tokens; the prompt and tool results are context. Replaces the all-token loss after the first LoRA failed |
| 11:05 | Repeated calls in targets | A tool call and its result are removed when identical to the call and result just before them (3 of 414 round 1 rows) |
| 11:26 | Public artifacts | Model repo `Purusharth/tiny-aya-efficient-healer-lora`, dataset repo `Purusharth/efficient-healer-trajectories`, two collections named "Efficient Healer", under your personal account |
| 11:26 | Going public | Everything is created private. I flip repos public only after you review the results and write-up and say so |
| 11:26 | Write-up | GitHub README in `purusharthmalik/efficient-healer` is the write-up; I also hand you the text for a Hugging Face article |
| 11:26 | Team | Purusharth Malik, single-member team. LLM usage stated plainly |
| 11:50 | Round 2 gate HOLD | One confirmatory evaluation on 400 fresh held-out-company questions; the same rule applied once; its answer stands. The rule itself is not changed |
| 12:40 | Going public | Yes: both HF repos, the two collections and the GitHub repo made public |
| 10:09 | Gate rule | "Improve, never regress": success may not fall by more than 2 points; tool-call validity may not fall by more than 2 points; and either success or tool calls on tasks both models solve must improve with a 95% interval that excludes zero. Applied per round (v1 vs base, then final vs v1). AgentTune's `DeploymentGate` must also approve |

**Changes I made while building, for your review**

| Change | Why |
|---|---|
| Teacher runs with thinking off | It already solves 87.5% of pilot rollouts at about 100 rollouts/min |
| Evidence guard tightened | A run only qualifies as a target if a `query_finqa_tables` or `get_table_schema` call really returned table data, and no assistant turn writes a tool result itself. One student "success" had invented its tool results and hit the right number by luck |
| Verifier bug fixed | The percent-vs-fraction rule combined with the rounding tolerance marked `4%` as correct for a gold of `0`. Percent scaling now applies only when the gold is a percentage or a fraction below 1, never for a gold of 0, and only within 0.5%. Everything collected so far is re-scored with the fixed rule |
| Sequence length | Rows are dropped, not truncated, if longer than the limit (a truncated row loses its answer). The limit is the 95th-percentile row length, up to 8,192 tokens instead of 4,096 if the rows turn out longer |

**How this maps onto AgentTune's self-healing guide** (`agenttune.lexsi.ai/user-guide/self-healing/`)

| Guide step | What we use |
|---|---|
| Detect | FinQA verifier (labelled signal, the guide's "construct `Failure` objects yourself" route), plus AgentTune's `FailureDetector` on `EventLog.to_audit_records()` |
| Classify | AgentTune `FailureClassifier`, served by the teacher model |
| Generate | Our repair ladder (hinted student retry, then teacher rollout, each verified). It replaces `TrainingExampleGenerator`, which only emits single corrected actions |
| Buffer and trigger | AgentTune `TrainingBuffer` + `RetrainingTrigger`; each example carries the failed run as `rejected` and the repaired run as `chosen` |
| Retrain | AgentTune `create_distill_trainer` (SFT LoRA) instead of the guide's DPO retrainer |
| Gate | AgentTune `DeploymentGate.evaluate_decision`, with our rule on top |

**Round 1 on the 480-task pool (10:22 IST):**
- Base Tiny Aya, 480 × 4 with stop rule: 15 of 1,920 rollouts solved (0.8%). Buckets: 0 always, 13 sometimes (11 pass the evidence guard), 467 never.
- AgentTune `FailureClassifier` on the teacher: 480 failed runs labelled, 0 classifier errors. Where the rule label maps to one of its root causes, the two agree on 115 of 142.
- Hinted student retries on 469 tasks × 4: 10 of 1,876 rollouts solved (0.5%). Hints alone do not rescue an untrained model, as expected.
- 461 of 480 tasks went to the teacher (3 runs each): 1,077 of 1,383 rollouts solved (77.9%, strict 39.1%); 394 of 461 tasks got at least one success, 67 got none. 148 rollouts/min.
- Failure labels on the 467 never-solved tasks. Rules: 282 wrong number, 116 no answer tag, 68 loop, 1 step cap. AgentTune classifier: 361 incomplete reasoning, 46 hallucinated output, 33 loop collapse, 25 wrong tool, 2 wrong routing.
- **Round 1 training set: 414 rows** — 395 teacher repairs (mean 4.42 tool calls), 8 hinted-retry repairs, 11 from tasks the student sometimes solved. 66 of 480 tasks have no verified success from any source and are left out.
- AgentTune's own `FailureDetector` flags 658 `loop_collapse` failures in the 1,920 base rollouts. Its `RetrainingTrigger` fired on the 414 buffered examples (reason: total failures exceeded; dominant root cause incomplete reasoning, 77%).
- Row lengths: median 3,541 tokens, 95th percentile 4,404, max 6,055. Limit set to 4,404; 21 longer rows dropped, **393 rows train LoRA v1** (batch 2 × 8 accumulation). Training started 10:33 IST after two quick fixes to my training script.
- **First LoRA (loss on all tokens) failed, 10:56 IST.** Loss 1.74 → 0.30 in 16 minutes, but held-out success only 6 of 300 (2.0%) and train-pool success 42 of 1,920 (2.2%).
  - It learned the run shape (most runs make 4 calls) but not the decisions. Compared with its own training targets on 1,580 train-pool runs, the first differing call is `get_table_schema` in 1,089 of them, and it picks the target's table only 31% of the time.
  - Cause: a row is about 3,500 tokens of prompt and tool results and about 150 tokens of agent turns, so the agent's decisions carried about 4% of the loss.
  - Kept as a negative result under `out/neg_alltokens`. Retraining with the loss on agent turns only started 11:07 IST.
- **LoRA v1 retrained with the loss on agent turns only, 11:20 IST. It works.**
  - Labels on 79,081 of 1,392,843 tokens (5.7%, about 201 per row). Loss 0.66 → 0.076 in 12 minutes (393 rows, 74 steps).
  - Held-out, stop rule, 100 tasks × 3: **59 of 300 solved (19.7%)**, strict 10.7%; 6 tasks solved in all 3 runs, 31 in some, 63 in none. Base was 1 of 300.
  - Train pool (round 2 collection), 480 × 4: 505 of 1,920 solved (26.3%). **Buckets: 32 always, 190 sometimes, 258 never.**
  - Shortest verified run: 3.66 tool calls on always-solved tasks (all their successes average 4.02); 3.89 on sometimes-solved tasks (4.04).
  - Round 2 repair and the final training started 11:21 IST.
  - v1 without the stop rule, held-out: 57 of 300 (19.0%). It solves as much with or without the harness rule, so it has learned to stop after a call by itself.
  - Round 2 labels: 258 never-solved tasks classified by AgentTune's classifier, 0 errors; rule and LLM labels agree on 33 of 45 comparable cases.
- **Round 2 repair, 11:31 IST.**
  - Hinted v1 retries on the 258 never-solved tasks × 4: 54 of 1,032 solved (5.2%); **39 tasks repaired by a hint alone.** In round 1, on the untrained model, hints repaired 8.
  - Teacher: round 1 teacher runs reused; 61 tasks still without a verified run got 183 new teacher runs, 9 solved.
  - **Final training set: 424 rows** — 32 efficiency (always-solved, shortest run, mean 3.66 calls), 190 reliability (sometimes-solved, shortest run, 3.82), 39 hint repairs (4.00), 163 teacher repairs (4.26). 56 of 480 tasks have no verified run from any source.
  - So 222 of 424 targets (52%) are now the student's own shortest verified runs; in round 1 it was 11 of 414.
  - AgentTune `RetrainingTrigger` fired on 392 buffered examples (dominant root cause incomplete reasoning, 89%).
  - Final LoRA training started 11:32 IST.
- **Final adapter and gates, 11:48 IST.** Trained on 402 rows (22 over-long rows dropped) in 12 minutes. Held-out, 100 tasks × 3:

  | Metric (stop rule on) | Base | Round 1 adapter | Final adapter |
  |---|---|---|---|
  | Rollouts solved | 1 of 300 (0.3%) | 59 of 300 (19.7%) | 72 of 300 (24.0%) |
  | Strict success | 0.7% | 10.7% | 13.3% |
  | Tasks solved in all 3 runs | 0 | 6 | 10 |
  | Tool calls per run | 2.77 | 4.55 | 4.31 |
  | Output tokens per run | 208 | 232 | 205 |
  | Redundant tool-call rate (AuditKIT) | 0.153 | 0.089 | 0.049 |
  | Loop-free share (AuditKIT) | 0.763 | 0.840 | 0.880 |
  | Tool-call validity (AuditKIT) | 0.997 | 1.000 | 1.000 |

  Without the stop rule: base 0 of 300, round 1 adapter 57 of 300 (19.0%), final adapter 75 of 300 (25.0%); output tokens per run 620 → 213; seconds per run 18.3 → 5.5; redundant-call rate 0.296 → 0.054.
  - **Gate round 1 (v1 vs base): DEPLOY.** Success +19.3 points, 95% interval [+13.7, +25.7]. AgentTune gate approved.
  - **Gate round 2 (final vs v1): HOLD.** Success +4.3 points, interval [−1.7, +10.7], includes zero. Tool calls on the 24 tasks both solve: +0.06 [−0.12, +0.26]. The final adapter is significantly better on tool calls per run (−0.24 [−0.44, −0.05]), output tokens (−27 [−43, −12]) and redundant-call rate (−0.040 [−0.062, −0.018]), but the rule does not count those. AgentTune's own gate approved (it passes anything that does not regress).
  - Both adapters and all data uploaded to the two private repos at 11:48 IST; the deployed adapter at the model repo's root is round 1.
  - **Confirmatory run started 11:50 IST (your decision):** 400 fresh questions from the held-out companies, 3 runs each, base and both adapters; the same gate rule applied once, answer accepted either way.
- **Confirmatory evaluation, 12:08 IST.** 400 fresh questions from the 55 held-out companies (1,244 were available), 3 runs each, stop rule on:

  | Metric | Base | Round 1 adapter | Final adapter |
  |---|---|---|---|
  | Rollouts solved | 6 of 1,200 (0.5%) | 302 of 1,200 (25.2%) | 365 of 1,200 (30.4%) |
  | Strict success | 0.3% | 15.0% | 17.6% |
  | Tasks solved in all 3 runs | 0 of 400 | 35 of 400 | 61 of 400 |
  | Tool calls per run | 2.88 | 4.65 | 4.47 |
  | Output tokens per run | 221 | 239 | 215 |
  | Redundant tool-call rate (AuditKIT) | 0.126 | 0.108 | 0.069 |

  - **Confirmatory gate, round 2 (final vs v1): DEPLOY.** Success +5.2 points, 95% interval [+2.5, +8.1]. Tool calls on the 140 tasks both solve: −0.11 [−0.24, +0.00].
  - Confirmatory gate, round 1 (v1 vs base): DEPLOY again, +24.7 points [+21.4, +28.1].
  - The first attempt at this run failed to start (I launched two servers at once and the second found no free memory); it cost 6.5 minutes of GPU and no data.
  - Private repos updated: the root adapter is now round 2. Model and dataset cards uploaded. Secret scan of the uploaded logs and of the code: nothing found.
  - **GPU job stopped at 12:08 IST.** RTX PRO 6000: 2 h 43 min, $7.47. L40S: 37 min, $1.10. Total $8.57.
- Evaluation self-test passed (10:31 IST): AuditKIT import, metrics, comparison and both gates run end to end. On base Tiny Aya it measures tool-call validity 0.995 and redundant-call rate 0.296 without the stop rule, 0.153 with it.

**Re-scored with the fixed verifier (10:07 IST):** held-out baseline unchanged (1 of 300 with the stop rule, 0 of 300 without). Round 1 collection 7 of 960 (was 8). Teacher pilot 33 of 40, 82.5% (was 35). Aya Expanse 8B with stop rule 0 of 40 (was 1).

**Public links (since 12:42 IST)**
- Model repo: https://huggingface.co/Purusharth/tiny-aya-efficient-healer-lora
- Dataset repo: https://huggingface.co/datasets/Purusharth/efficient-healer-trajectories
- Second model repo (Aya Expanse 8B adapter): https://huggingface.co/Purusharth/aya-expanse-8b-efficient-healer-lora
- Model collection: https://huggingface.co/collections/Purusharth/efficient-healer-models-6afaa8b6120d5e7cae162d33
- Dataset collection: https://huggingface.co/collections/Purusharth/efficient-healer-datasets-6afaa8b70386ac5f12a8f637
- Code and write-up: https://github.com/purusharthmalik/efficient-healer
- The command-queue repo `Purusharth/selfheal-ops` stays private.

**Measured so far**
- FinQA as shipped by AgentTune: 4,030 train rows, 522 val rows, 185 companies.
- Split: 240 train-pool tasks and 100 held-out tasks, company-disjoint (55 companies held out), 0 shared questions.
- Smoke on Tiny Aya, 6 tasks × 2: 0 of 12 solved. Mean 3.8 tool calls. Failures: 6 with no `<answer>` tag, 5 wrong number, 1 non-numeric. The model keeps writing after its tool call (it invents a tool result).
- Rollout speed: generation is fast (7.5 s of model time per rollout). The first wave of every run was slow (about 2 minutes per rollout) because the database handle is created lazily and reflects every table, once per thread. Fixed with one warm-up call before the threads start; real speed to be measured in the next run.
- **Why Tiny Aya fails (12 smoke trajectories read by hand):** it does not stop after its tool call. In the same turn it invents the tool result, or writes all four steps at once with guessed table names. It then answers without real data, often with no `<answer>` tag.
- **Pilot, Tiny Aya as is, 20 train-pool tasks × 2: 0 of 40 rollouts solved** (0 of 20 tasks). Mean 3.3 tool calls. Final-answer check: 19 wrong number, 15 no `<answer>` tag, 6 non-numeric. Rule labels: 15 wrong number, 13 loop, 12 no answer tag.
- **Student and harness comparison, same 20 tasks × 2 (40 rollouts each):**

  | Student | Stop rule | Solved | Mean tool calls | Mean output tokens | Seconds per rollout |
  |---|---|---|---|---|---|
  | Tiny Aya | off | 0 of 40 | 3.3 | 460 | (first-wave bug) |
  | Tiny Aya | on | 1 of 40 | 3.1 | 202 | 3.3 |
  | Aya Expanse 8B | off | 0 of 40 | 4.1 | 303 | 7.6 |
  | Aya Expanse 8B | on | 1 of 40 | 4.1 | 304 | 6.0 |

  The stop rule ends a turn at the end of the first call list or at `</answer>`. Neither Cohere model can do this task untrained, so a bigger student does not help.
- **Rollout speed after the fix: 160–390 rollouts per minute.** A full collection (960 rollouts) takes about 5 minutes, so more than one round is affordable.
- **Held-out baseline, base Tiny Aya, 100 tasks × 3 (the "before" numbers):**

  | Harness | Rollouts solved | Tasks solved in all 3 | Mean tool calls | Mean output tokens | Seconds per rollout |
  |---|---|---|---|---|---|
  | With stop rule | 1 of 300 (0.3%) | 0 of 100 | 2.77 | 208 | 4.6 |
  | Without stop rule | 0 of 300 | 0 of 100 | 3.86 | 621 | 18.3 |

- **Round 1 collection, base Tiny Aya with stop rule, 240 tasks × 4:** 8 of 960 rollouts solved (0.8%). Buckets: 0 always, 7 sometimes, 233 never. Rule labels on the 233 never-solved tasks: 142 wrong number, 58 no answer tag, 33 loop. Speed 528 rollouts/min.
- **Consequence for the idea:** with base success at 0–2.5% there is no "always solved" group in the first round. The shortening arm only has material after a first repair round.
- **Teacher pilot (Qwen3.6-35B NVFP4, thinking off), same 20 tasks × 2:** 35 of 40 rollouts solved (87.5%); 19 of 20 tasks have at least one success; 5.5 tool calls per successful run; about 100 rollouts/min. All 35 successes convert into valid rows in Tiny Aya's call format and pass the evidence guard.

**Spend (final)**: L40S 37 min = $1.10. RTX PRO 6000 from 09:26 to 12:08 IST = 2 h 43 min at $2.75/h = $7.47. Total **$8.57**; about $11.40 of credit left.

## 0b. Exploration after the submission was ready (started 12:29 IST)

You asked to use the remaining time and credit to explore, and picked all four directions. They run on two
GPU jobs side by side (same cost per minute of work, half the wall-clock time). Hard timeouts cap the
total at $10.31 of the $11.43 left.

| Job | Order | What | Status |
|---|---|---|---|
| A (2 h cap, ends 14:29 IST at the latest) | 1 | **Round 3 on a bigger pool**: pool grown from 480 to 1,400 tasks; collect with the round 2 adapter, repair, train round 3, gate it against round 2 on a second fresh held-out set (B, 400 tasks) | collecting |
| B (1 h 45 cap, ends 14:14 IST at the latest) | 1 | **Aya Expanse 8B as the student**, trained on the same round 2 training set, evaluated on the 400-task confirmatory set | training |
| B | 2 | **Retention and safety check** with AuditKIT: GSM8K, Global-MMLU in 5 languages, XSTest refusal and over-refusal; base Tiny Aya vs the deployed adapter | queued |
| B | 3 | **DPO** on 480 of 887 step-level preference pairs, on top of the round 2 adapter, with AgentTune's agentic DPO trainer. Best effort: it is the least tested path and runs last | queued |

Choices I made inside these (tell me if you want any changed):
- Round 3's failure classifier runs with the teacher's thinking off, to save about 10 minutes. Rounds 1 and 2 had it on.
- The round 3 gate uses a new fresh set (B), not the confirmatory set, so no test set is reused for a decision. Round 3 is also run on the confirmatory set so its number can sit next to the published 30.4%.
- DPO pairs are step-level: the prompt is the conversation up to the first agent turn where a failed run leaves the verified run; chosen and rejected are that one turn. 887 pairs from 392 tasks (297 at the table choice, 519 at the query step).
- **Void result, caught 13:10 IST:** the first Aya Expanse numbers on job B (untrained 0 of 300, fine-tuned 2 of 1,200) are invalid. Job B never built the FinQA database, so every `list_tables` call returned "No tables found". The tools now build the database themselves and refuse to run on an empty one. The Expanse evaluation is rerunning. Job A built its database and is not affected.
- Round 3 collection on job A (valid): the round 2 adapter solves 1,650 of 5,600 rollouts (29.5%) on the 1,400-task pool: 170 always, 493 sometimes, 737 never. On the second fresh held-out set (B) it solves 314 of 1,200 (26.2%).

**Results so far**
- **Retention and safety check (AuditKIT, 100 samples each, base vs round 2 adapter), 13:28 IST:** GSM8K 0.71 → 0.67 (−0.04 [−0.12, +0.05], no significant change). XSTest refusal of unsafe prompts 0.94 → 0.94. XSTest over-refusal of safe prompts 0.06 → 0.06. Global-MMLU in 5 languages scored 0.05 for both, below the 25% chance level, so its answer extraction failed and it is not a measurement. The check took 17 minutes because AuditKIT's API backend sends requests one at a time.
- **Round 3 repair (job A), 13:28 IST:** hinted retries repaired 87 of 737 never-solved tasks (115 of 2,948 retries solved). Teacher on the 500 tasks still without a verified run: 806 of 1,500 solved, 303 tasks covered. **Round 3 training set: 1,203 rows** — 170 efficiency (3.74 calls), 493 reliability (3.88), 87 hint repairs (3.91), 453 teacher repairs (4.24). 197 of 1,400 tasks have no verified run. Classifier: 737 labelled, 7 errors; rule and LLM labels agree on 27 of 58 comparable cases. AgentTune's trigger fired. Training started 13:28 IST.

- **Aya Expanse 8B as the student (valid rerun), 13:33 IST:** LoRA trained on the same round 2 set (402 rows, 24 minutes, loss 0.58 → 0.053). On the 400-task confirmatory set: **467 of 1,200 solved (38.9%)**, strict 21.4%, 85 tasks solved in all 3 runs, 4.05 tool calls per run. Against the Tiny Aya round 2 adapter (30.4%): +8.5 points, 95% interval [+4.5, +12.6]. Untrained Expanse on the first 100 of those tasks: 0 of 300. So the loop's data transfers to a second Cohere student, and capacity is part of Tiny Aya's limit.
- **DPO on top of the round 2 adapter, 13:52 IST:** AgentTune's agentic DPO trainer, 480 step-level pairs, 120 steps, 11.5 minutes, lr 2e-5, β 0.1, LoRA r=8; loss 0.693 → 0.64. On the 400-task confirmatory set: **429 of 1,200 solved (35.8%)**, strict 20.8%, 89 tasks solved in all 3 runs (round 2: 61). Against round 2: +5.3 points, 95% interval [+2.4, +8.2]; gate DEPLOY. Tool calls per run rose from 4.47 to 4.71, and tool calls on tasks both solve did not improve (interval [0.00, +0.20]).
- **Clean re-test on the second fresh set (B, 400 tasks × 3, never used for a decision), 14:01 IST:**

  | Model | Rollouts solved | vs round 2 adapter [95% interval] |
  |---|---|---|
  | Tiny Aya, round 2 adapter | 314 of 1,200 (26.2%) | — |
  | Tiny Aya, round 2 + DPO | 358 of 1,200 (29.8%) | +3.7 points [+1.1, +6.3]; gate DEPLOY |
  | Aya Expanse 8B, LoRA on the round 2 set | 438 of 1,200 (36.5%) | +10.3 points [+6.7, +13.9] |

  - DPO buys success, not efficiency: on both sets it uses more tool calls per run (+0.20 to +0.24), slightly more on tasks both models solve (+0.05 to +0.21 on set B), and its loop-free share drops (0.83 → 0.79 on set B).
  - Expanse is both better and leaner than Tiny Aya on the same data: 4.04 tool calls per run against 4.45 (−0.42 [−0.54, −0.30]), redundant-call rate 0.021 against 0.069, loop-free share 0.94 against 0.83.
- **Round 3 adapter (1,400-task pool), 14:07 IST.** Trained on 1,142 rows (61 over-long rows dropped) in 34 minutes. On the fresh held-out set B: **557 of 1,200 solved (46.4%)**, strict 26.0%, 134 tasks solved in all 3 runs, 4.31 tool calls per run. Against the round 2 adapter on the same set (26.2%, 53 tasks): **+20.2 points, 95% interval [+16.8, +23.7]. Gate DEPLOY.** Tool calls on tasks both solve: no change ([−0.07, +0.09]). More data through the same loop was the main bottleneck.
- Queued as job A's last step: DPO on top of round 3 (320 pairs, time-boxed to fit before the 14:29 IST timeout).
- **Job B stopped at 14:02 IST** after 92 minutes ($4.23). Its adapters (Expanse, DPO) are saved in the private ops repo.

- **Round 3 on the confirmatory set (A), 14:11 IST:** 589 of 1,200 solved (49.1%), strict 29.8%, 143 tasks solved in all 3 runs. Against round 2 on the same set: +18.7 points [+15.2, +22.4]; tool calls on the 173 tasks both solve −0.16 [−0.25, −0.08]; redundant-call rate 0.069 → 0.051; loop-free share 0.818 → 0.893.

**Your decisions on the exploration results (14:12 IST)**
- Publish everything: round 3 becomes the deployed adapter at the root of the Tiny Aya repo; the Expanse adapter gets its own public repo in the model collection; DPO adapters go in folders; all exploration data goes into the dataset repo.
- The write-up leads with round 3 (0.5% → 49.1% in three rounds); the two-round story and the gate HOLD stay in the body; Expanse, DPO and the retention check are a "further experiments" section.

- **DPO on top of round 3, 14:24 IST:** 320 of 2,158 pairs, 8 minutes, loss 0.693 → 0.691. Set B: 582 of 1,200 (48.5%) against round 3's 557 (46.4%): +2.1 points [−0.3, +4.3]. **Gate HOLD.** Round 3 stays deployed.
- **Both exploration jobs stopped.** Job A ran 1 h 55 min ($5.25), job B 1 h 32 min ($4.23). Exploration total $9.48; project total $18.05; about $1.95 of credit left.
- **Published 14:30 IST:** round 3 adapter at the root of the Tiny Aya repo (verified identical to `round3/`), DPO adapters in `round2_dpo/` and `round3_dpo/`, the Expanse adapter in its own public repo and in the model collection, 107 exploration files (810 MB) in the dataset repo, three cards updated. Secret scan of the new uploads and the code: nothing found.

Lost time: about 4 GPU-minutes on each job. The Hub rejects file paths containing `/x00`, which silently refused my bootstrap step's name; the submit script now verifies every upload and retries.

## 1. The claim we want to prove

One LoRA on `CohereLabs/tiny-aya-global` (3.35B) makes a finance database agent **solve more tasks in fewer
tool calls**, and it is only published as the deployed version if a measured gate passes.

Problem statements covered (Track 3): **Self-healing agents** (main), **Efficient trajectories**,
**Data and database agents**.

Hypotheses, fixed before any run:
- H1: on held-out tasks both models solve, mean tool calls per task drops by at least 15%.
- H2: held-out success rate does not fall (gate margin below), and we expect it to rise.
- H3: the redundant tool call rate drops.

## 2. Pipeline

```
FinQA tasks ──► k=4 rollouts each (AgentTune rollout + FinQA tools + numeric verifier)
                     │
      ┌──────────────┼──────────────────────┐
   4/4 solved     1–3 of 4 solved         0/4 solved
      │              │                       │
 shortest         shortest success      failure label (rules + AgentTune FailureClassifier)
 success          of the siblings            │
 "efficiency"     "reliability"         repair ladder, each try checked by the verifier:
      │              │                   1. student retry with a hint chosen by the label
      │              │                   2. teacher rollout (Qwen3.6-35B-A3B NVFP4)
      └──────┬───────┴───────────────────────┘  "repair"   (unrepaired tasks are dropped and counted)
             ▼
   one SFT LoRA (AgentTune create_distill_trainer) ─► candidate adapter on the Hub
             ▼
   held-out eval, base vs candidate, k=3 (AuditKIT metrics + paired bootstrap)
             ▼
   gate (AgentTune DeploymentGate + our rule) ─► promote to `main`, or hold and report why
```

What carries over from our A-DLP work (ideas only, no code; see §9):
- **Shortest finished correct sample is the length target** (`L_min`). Here: the fewest tool calls, then the
  fewest tokens, among verified successes.
- **Length pressure only where success is reliable** (the `p̂²` factor). Here: only 4/4 tasks feed the
  "efficiency" set.
- **Hitting the cap is a failure** (`TRUNC_AS_WRONG`). Here: reaching `max_steps` counts as unsolved.
- **Never rank failures by length.** A short wrong run is not a target. It would teach giving up early.
- **Gate with a margin** (`acc_ref − m`). Here: the deployment gate.

Guard against the obvious cheat: a kept trajectory must contain at least one tool call that returned
without error. Otherwise "shortest success" can select a lucky answer with no evidence.

## 3. Choices and the reasons

| Choice | Pick | Why | Fallback |
|---|---|---|---|
| Student | `tiny-aya-global` | Cheapest allowed model; AuditKIT's own smoke table rates its tool calls 3–4 of 5, so there is room to improve | `aya-expanse-8b` if pilot success is under 10% |
| Task | FinQA via AgentTune (`load_finqa_datasets`, `build_finqa_tools`): tools `list_tables`, `get_table_schema`, `query_finqa_tables`, `calculator` | Numeric answers cannot be guessed, tasks take 3–5 calls, local SQLite, no sandbox | BioGRID SQL (yes/no answers, so the evidence guard above is mandatory) |
| Verifier | Tolerant match in `selfheal/verify.py`; AgentTune's `numerical_match_reward` reported as the strict rate | Automatic, no judge model. Tolerant avoids false failures on 14.1% vs 0.141 and on rounding | — |
| Teacher and classifier model | `nvidia/Qwen3.6-35B-A3B-NVFP4` on the same GPU (your choice) | Stronger than an Aya teacher; allowed by the FAQ as a supplementary model. Its tool calls are converted into the student's format before training | Aya Expanse 8B if it cannot be served |
| Compute | One HF Jobs GPU, RTX PRO 6000 Blackwell 96 GB at $2.75/h (your choice) | On the teacher's supported hardware list; holds student and teacher together | L40S $1.80/h (not on the teacher's list) |
| Three Python envs | `.venv-tune` (AgentTune), `.venv-vllm` (vLLM server), `.venv-audit` (AuditKIT) | They only exchange files or HTTP, and their torch/transformers pins differ (AgentTune's own vLLM extra pins vLLM ≤ 0.23, too old for Aya) | — |

Sizes (cut by the Phase 1 speed test, never below the last row):

| Tier | Train pool × k | Held-out × k × 2 models | Rollouts incl. repairs |
|---|---|---|---|
| Full | 240 × 4 | 100 × 3 | ~2,200 |
| Reduced | 120 × 4 | 60 × 3 | ~1,150 |
| Minimum | 60 × 3 | 50 × 3 | ~600 |

Train pool comes from FinQA `train` (5,021 rows), held-out from `test` (689 rows). Leak check: no shared
question text and no shared source table between the two samples.

## 4. What the two libraries do, and what we must write

Read from source at AgentTune `8918cc3` and AuditKIT `1.1.12`.

**AgentTune gives us**
- Rollouts: `create_rollout_engine(backend)` + `create_rollout_fn(..., async_rollouts=True, on_trajectory_end=trajectory_writer(path, keep_reward=True))`.
  Aya chat templates ignore `tools`, so AgentTune puts the tool list in the system prompt and expects
  `[{"tool_name": ..., "parameters": {...}}]`.
- FinQA tools, data loader and numeric rewards.
- `FailureClassifier` (LLM-based, 5 labels: `wrong_tool`, `wrong_routing`, `incomplete_reasoning`,
  `hallucinated_output`, `loop_collapse`), pointed at our own endpoint with `api_base`.
- `create_distill_trainer(student, trajectories=..., lora_r=16, ...)`: TRL SFT + PEFT LoRA, with Hub push.
- `DeploymentGate.evaluate_decision(...)`.

**AuditKIT gives us**
- `ak.episodes_from_agenttune(run_dir)`: reads AgentTune trajectory JSONL directly.
- Metrics with no judge model: `tool_call_validity`, `redundant_tool_calls`, `agent_loop_detection`.
- Reliability over repeated trials: `pass_at_k`, `all_k`, `success_rate`.
- `ak.compare(base, cand).significance(metric)`: paired bootstrap by sample id, with a confidence interval.
- `RunResult.push_to_hub(...)` for the eval logs.

**We write ourselves** (the libraries do not do these)
1. k rollouts per task and per-task counts (repeat each task k times, group by task id). No agent pass@k exists.
2. Bucketing and shortest-success selection.
3. Rule-based failure labels (loop, tool error, bad call format, step cap hit, no tool evidence, wrong number).
   AgentTune's classifier turns any error into `incomplete_reasoning` with confidence 0, so we keep both
   labels and report how often they agree.
4. The repair ladder. AgentTune's `TrainingExampleGenerator` returns single corrected actions for 2 of the 5
   labels and its validator passes everything, so it cannot produce a repaired trajectory.
5. SFT examples built from the exact conversation the student saw (`metadata["conversation"]`). AgentTune's
   default SFT rows write calls as `<tool_call>{...}</tool_call>`, which is not the format its Aya prompt
   asks for. Training on a different format from inference would hurt.
6. Step count, tokens and wall-clock per task. AuditKIT has no step-count metric.
7. The gate rule (§6), wrapped around `DeploymentGate`, with our own `deploy_fn` (the default one has an
   argument mismatch).

Repair hints by label (added to the system prompt for the retry, removed from the saved training example):

| Label | Hint |
|---|---|
| `loop_collapse` | Do not repeat a call you already made. Use the result you have. |
| `wrong_tool`, bad format | Use only the listed tools and argument names. Read the table schema before you query. |
| `wrong_routing` | List the tables first and pick the one for the company and year in the question. |
| `incomplete_reasoning`, step cap | Compute the final number with the calculator from queried values, then answer. |
| `hallucinated_output`, no evidence | Answer only from tool results. |

## 5. Timeline (T0 = your go-ahead; about 5.5 hours)

| Phase | Time | Work | Done when |
|---|---|---|---|
| 0 Setup | 0:00–0:30 | HF login, accept Aya licences, GPU up, two envs, Hub repos created | `import agenttune`, `import auditkit` work on the GPU |
| 1 Smoke and pilot | 0:30–1:15 | 20 tasks × 2 with Tiny Aya. Pick the rollout route. Check the JSONL with `auditkit agent import-agenttune --inspect` | Decisions below are made |
| 2 Collect | 1:15–2:15 | Baseline held-out eval (the "before") and the train pool rollouts; bucket | Bucket counts and baseline table on the Hub |
| 3 Classify and repair | 2:15–3:15 | Labels, hinted retries, teacher runs, build the SFT set | Dataset on the Hub with counts per source |
| 4 Train | 3:15–4:00 | LoRA r=16, α=32, lr 2e-4, 3 epochs, bf16 | Adapter on the Hub as `candidate` |
| 5 Evaluate and gate | 4:00–4:45 | Held-out eval of the merged candidate, AuditKIT compare, gate decision | `gate_decision.json` and results table |
| 6 Ship | 4:45–5:45 | Write-up, collections, GitHub repo, submission form | Form submitted |

Write-up drafting runs alongside phases 2–5 while the GPU works.

Phase 1 decisions:
- **Rollout route (settled)**: a small custom engine, `selfheal/engine.py`, that implements AgentTune's
  `RolloutEngine` interface and sends generation to a vLLM server. AgentTune still runs the loop, the Aya
  tool prompt, call parsing and tool execution. One trajectory per call from our own thread pool, so
  concurrency is not capped at AgentTune's 8 threads and the run can resume.
- **Size tier** from measured rollouts per minute: budget 75 GPU-minutes of rollouts in total.
- **Difficulty**: pilot success under 10% → switch student to `aya-expanse-8b` or task to BioGRID; over 85% →
  sample harder FinQA rows (more table lookups).
- **Sequence length**: set `max_length` to the 95th percentile of conversation tokens, capped at 4,096.

## 6. Metrics and the gate

Reported for base and candidate on the same held-out tasks (n tasks × k = 3):

| Metric | Better | Source |
|---|---|---|
| Task success rate | higher | FinQA verifier |
| Solved in all k runs (`all_k`) and in any (`pass_at_k`) | higher | AuditKIT reliability |
| Tool calls per task, on tasks both models solve | lower | ours |
| Redundant tool call rate | lower | AuditKIT |
| Tool call validity | higher | AuditKIT |
| Loop-free share | higher | AuditKIT |
| Output tokens and seconds per task | lower | ours |

Each difference gets a 95% paired bootstrap interval from `ak.compare`. Steps are compared only on tasks
both models solve, so a model that solves just the easy tasks cannot look more efficient.

Gate: `selfheal/gate.py::gate_decision` (the rule is open, see the TODO there). Pass → the adapter is
promoted to `main` on the Hub. Fail → it stays on `candidate` and the write-up reports why.

## 7. Submission checklist (from the rules and the submit page)

- [ ] Base model is a Cohere Labs Aya model; other models only as teacher or judge.
- [ ] AgentTune and AuditKIT used together in one pipeline; write-up says how they connect.
- [ ] Public, un-gated HF **model collection**: the LoRA adapter (licence CC-BY-NC-4.0, same as the base).
- [ ] Public HF **dataset collection**: all rollouts, failure labels, repaired trajectories, SFT set, eval logs, gate decision.
- [ ] Before and after table with more than one metric.
- [ ] Write-up as an HF article, or the GitHub README: title and one-line claim, tracks and problem
      statements, abstract under 500 words, headline number, base model and method, kits used, models and
      licences, data sources and splits with the leak check, metrics and why, setup and hyperparameters,
      results, limitations, "with more credits", artifacts, run instructions, declaration of outside work, teammates.
- [ ] Public GitHub repo with reproduction notes.
- [ ] Submission form filled with the registered email.
- [ ] Optional: cost per 1,000 tasks vs a closed API, from our token counts and throughput.

## 8. Budget

| Item | Cost |
|---|---|
| L40S, 37 min (setup and first smoke test), stopped | $1.10 |
| RTX PRO 6000, up to 6 h (job timeout) | $16.50 at most |
| Left over at the worst case | $2.40 |

The job bills per minute while it runs, busy or idle. I stop it the moment the last phase ends.

## 9. Things only you can do, and risks

**Before T0 (all done)**: HF login works (user `Purusharth`), Aya licences accepted, credits available,
HF Jobs confirmed as the GPU route, GitHub repo named.

**Still needed from you**: the open decisions in §0, the gate rule in `selfheal/gate.py`, and a yes before
anything is made public.

**Risks**
- Rollout speed is the critical path. The size tiers and the three routes exist for this.
- Tiny Aya may fail almost all FinQA tasks. The pilot catches it in the first 75 minutes.
- Aya Expanse 8B may be a weak teacher. Repair yield is measured; unrepaired tasks are dropped, not faked.
- The gate may fail. That is a valid, reportable result ("what didn't work"), and the pipeline still stands.
- 50–100 held-out tasks give wide intervals. We report intervals, not just point values.

**Declaration for the write-up.** The idea of replaying the shortest correct sample was noted in our earlier
A-DLP length-penalty work (October 2026, math reasoning, a different model and codebase). No code or data
from it is used; everything in this repo is written during the hackathon.

**Licences.** AgentTune and AuditKIT are source-available for non-commercial use (LSAL). Tiny Aya is
CC-BY-NC-4.0. AgentTune's licence asks for a safety re-check before deploying a post-trained model; if time
allows, run a small AuditKIT XSTest sample as part of the gate.

## 10. Repo layout

```
selfheal/
  engine.py     AgentTune RolloutEngine backed by a vLLM server            (written)
  tasks.py      FinQA tasks and tools, company-disjoint split, leak check  (written, run)
  verify.py     tolerant answer check + AgentTune's strict reward          (written, run)
  rollout.py    k rollouts per task, resumable, JSONL out                  (written, run)
  summary.py    per-task statistics from rollout files                     (written, run)
  buckets.py    always / sometimes / never, shortest-success, evidence guard (written)
  failures.py   rule labels + AgentTune FailureClassifier                  (written)
  sft_data.py   trajectories → student-format SFT rows                     (written)
  train.py      AgentTune create_distill_trainer, LoRA                     (written)
  evaluate.py   AuditKIT import, metrics, compare (runs in .venv-audit)    (written)
  gate.py       gate_decision                                              (rule open)
infra/
  worker.sh     runs queued commands inside one long HF Job, keeps envs and models warm
  submit.sh     queue a command with the current code;  logs.sh  read the worker log
  cmds/         one script per queued step
```

The GPU side uses a private HF dataset repo, `Purusharth/selfheal-ops`, as a command queue and a store for
logs and results. It stays private; public artifacts get their own repos at ship time.

Sources: hackathon pages `/tracks`, `/rules`, `/structure`, `/faqs`, `/submit`, `/schedule`, `/prizes`
(they return HTTP 404 to plain fetchers but render in a browser).
