# Pre-registration: Strategic play in simulated verdict games

**Frozen:** 2026-11-04, before any main-study game was played. Hashes of this file and of the analysis code are in
`prereg/HASHES.txt`. This is an internal, time-stamped pre-registration (repository hash), not an external registry entry.

## 0. Status at freezing
* Pilot 1 (8 pilot cases, original case schema, 192 games): no type discrimination (guilty - innocent = +0.5 points,
  95% CI [-1.1, 2.1]); judges' P(guilty) rose with every round for both types. Archived in `data/pilot1_archive/`.
  Response: cases redesigned around *withheld evidence* (Strategic Use of Evidence paradigm), answers lengthened.
* Pilot 2 (same 8 pilot cases, new schema, 192 games) informed the power analysis below.
* Study 2 dry run: 2 pilot cases (6 decision points) to test code. Pilot data are never pooled with main data.
* The 48 main cases (16 weak / 16 moderate / 16 strong designed evidence) have been generated and audited but have
  not been used in any game.

## 1. Study 1 design
Complete randomized block design, case = block, full factorial within each case:
* **Objective O** (prosecutor's utility): advocate (court game: wins on GUILTY) | minister (interrogation game: wins on a correct verdict).
* **Prosecutor selection S**: random | introspect | lookahead (K = 4 candidates from the same generator in all arms).
* **Defence selection D**: random | lookahead (K = 4).
* **Type T**: guilty | innocent (same public file and withheld evidence; different private brief).
24 cells x 48 cases = 1,152 games, one game per cell. d = 4 rounds; utterance cap 80 words.
Player model: qwen3-235b-a22b-2507 (DeepInfra, T = 1). Planner's internal judge: gpt-4o-mini (soft probability from token
log-probabilities), identical prompt to the evaluation judge. Run order randomized (seed 11).

## 2. Outcomes
* **Primary (Y):** judge A (gpt-4o-mini, neutral prompt) soft P(guilty) x 100 after round d = 4.
* Secondary: conviction under the verdict-game rule (first passage over judge A's per-round verdict labels;
  CONTINUE at round d = acquittal); interrogation-game utility (+1 correct decisive verdict, -1 wrong, 0 none);
  paired AUC (share of cases in which the guilty twin ends with higher Y than the innocent twin).
* Held-out evaluators (Study 1): Llama-3.3-70B (soft), Gemini-2.5-Flash-Lite, Claude-Haiku-5.5, DeepSeek-V4-Flash,
  Mistral-Small-3.2 (stated probabilities), applied post hoc to the same transcripts (Proposition 1).

## 3. Confirmatory hypotheses (all tests two-sided; directions are predictions)
Contrasts are computed within case, averaging over the factors not named.
* **H1 (strategic advantage).** Advocate: lookahead - random > 0 on Y.
* **H2 (simulation beyond introspection).** Advocate: lookahead - introspect > 0 on Y.
* **H3 (persuasion, not only truth).** Advocate, innocent defendants: lookahead - random > 0 on Y.
* **H4 (objective shapes truth-tracking).** Under lookahead, discrimination Delta = Y(guilty) - Y(innocent) differs between
  minister and advocate (no directional prediction).
* **H5 (counter-strategy).** Defence lookahead - random < 0 on Y (averaged over all prosecutor arms).
* **H6 (transfer gap).** H1 contrast under judge A minus the mean H1 contrast over the five held-out judges > 0.
* **H7 (move-level value, Study 2).** At a decision point, choosing among the same 8 candidates by the judge-A planner
  (advocate utility) raises the immediate judge-A P(guilty) relative to random choice.
* **H8 (judge-specific exploitation, Study 2).** Crossover interaction between planner's internal judge {A, DeepSeek} and
  evaluator {A, DeepSeek} on selection value (standardized by each evaluator's mean between-candidate SD) > 0, i.e.
  matched planner-evaluator pairs gain more.
* **H9 (dose-response, Study 2).** Slope of best-of-k selection value (judge-A planner and evaluator, immediate outcome) on log2 k > 0.

## 4. Inference
* Each contrast yields one value per case; we report the mean, a t-based 95% CI and two-sided t-test across cases
  (n = 48), plus a case-bootstrap CI (10,000 resamples) and a sign-flip randomization p-value (20,000 draws) as robustness.
  In Study 2 decision points (t = 1 and t = 3, both types) are first averaged within case.
* Multiplicity: Holm correction across H1-H9 at familywise alpha = 0.05.
* Everything else is exploratory and labelled as such.

## 5. Study 2 design
For each main case: decision points at round t = 1 (empty transcript; candidates realised for both types) and t = 3
(prefix = first two rounds of the Study-1 game advocate/random/random of each type). The advocate generator proposes
K = 8 candidates. Each candidate is scored by the planner under two internal judges (A and DeepSeek-V4-Flash) from
simulated hypothetical guilty/innocent replies, and realised with M = 2 independent continuations (real defendant,
then random/random play to d = 4). Judges A and DeepSeek evaluate at stage t (immediate; primary for H7-H9) and stage d
(final; persistence, secondary); Gemini and Haiku evaluate at stage d.

## 6. Sample size and power
Fixed by budget at 48 main cases (no optional stopping). Minimum detectable effects (80% power) using pilot-2 SDs of
case-level contrasts (8 cases, so these are rough):

| Hypothesis | pilot est. | pilot SD | SE (n=48) | MDE alpha=.05 | MDE alpha=.05/9 |
|---|---|---|---|---|---|
| H1 | +3.49 | 3.20 | 0.46 | 1.32 | 1.73 |
| H2 | +0.83 | 1.88 | 0.27 | 0.78 | 1.02 |
| H3 | +3.81 | 3.26 | 0.47 | 1.35 | 1.77 |
| H4 | -1.10 | 4.46 | 0.64 | 1.84 | 2.42 |
| H5 | -2.31 | 1.16 | 0.17 | 0.48 | 0.63 |
| H6 | +1.11 | 2.48 | 0.36 | 1.02 | 1.34 |

H2 and H4 are expected to be underpowered if the pilot estimates are accurate; a null on them is not evidence of absence.

## 7. Exclusions and failures
API failures are retried; a game that still fails is re-run from cache; any game that cannot be completed is
reported and excluded. No case is excluded on the basis of outcomes. Malformed model outputs are re-sampled (logged).

## 8. Planned exploratory analyses
Effects on conviction and on interrogation utility; per-judge effects; paired AUC by cell; meta-game equilibrium of the
court game over (S, D); case-level heterogeneity (random-slope SD, prediction interval) and moderation by designed
strength and stage-0 prior; examination drift (P(guilty) by round); guilt-leaning judge prompt (from the original paper)
vs neutral; utterance features by arm; Study-2 planner validity, persistence, split-sample oracle, t = 1 discrimination.
