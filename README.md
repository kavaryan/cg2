# Verdict Games and a Strategic View of the Turing Test

This repository explores how the choice of questions influences a judge's decision.

The full paper is available at [paper/main.pdf](paper/main.pdf). Numerical results, tables, and figures are generated using [`analysis/make_report.py`](analysis/make_report.py).

This work extends [Aryan K., *Conversation Games and a Strategic View of the Turing Test* (2025)](https://arxiv.org/abs/2501.18455).

## Introduction
A **verdict game** is a conversation in which one speaker asks questions, another
answers, and a judge uses the evidence and conversation to make a decision. The
speakers choose what to say to get the outcome they want. A prosecutor seeking a
conviction may therefore ask different questions from an examiner trying to find
out what really happened.

We test this idea with LLMs playing prosecutors,
defendants and judges in made-up criminal cases. The questioner generates several
possible questions, then uses one of three methods: **random** (pick a question
without assessing it), **introspective** (ask the model which question it thinks
best serves its goal), or **lookahead** (simulate possible answers and estimate
how a judge would react before choosing).

In these experiments, lookahead made both guilty and innocent defendants seem
more guilty. Its effect depended on the judge and was reduced when the defence
also used lookahead. The paper connects this setup to the **Turing test** (using a
conversation to judge whether a respondent is human or a machine), where the
questions asked can influence the verdict as well as the answers given.

## Layout
| Path | Contents |
|---|---|
| `cg/llm.py` | OpenRouter client: response cache (exact replay and common random numbers), cost ledger, hard budget stop, provider pinning |
| `cg/game.py` | Verdict-game simulator: prompts, views (court vs public), selection rules (random / introspect / lookahead), soft judge |
| `scripts/gen_cases.py`, `audit_cases.py`, `rename_defendants.py` | Stimulus bank: 56 cases with withheld evidence, LLM audit, name reassignment |
| `scripts/run_study1.py` | Study 1: 2x3x2x2 within-case factorial (48 cases, 1,152 games) |
| `scripts/run_study2.py` | Study 2: question-selection effects (8 candidates, 2 continuations per candidate and type) |
| `scripts/judge_pass.py`, `judge_study2.py` | Retrospective judging with fixed policies (Proposition 1) |
| `scripts/annotate_questions.py`, `judge_retest.py` | Exploratory content coding and judge test-retest |
| `analysis/common.py`, `study1.py`, `study2.py` | Recorded analysis code (developed on validation data; hashes in `prereg/HASHES.txt`) |
| `analysis/make_report.py` | Writes `paper/macros.tex`, `paper/figs/*`, `paper/tables/*` |
| `prereg/` | Pre-registration (frozen 2026-11-04 19:50 UTC) and SHA-256 hashes |
| `data/cases/cases.json` | Case bank with audit scores |
| `data/raw/` | All transcripts, judgements and annotations (JSONL) |
| `data/pilot1_archive/` | Initial stimulus-validation experiment (no withheld evidence) |
| `data/cache/ledger.jsonl` | Per-call cost ledger used by the report |
| `data/cache/llm_cache.sqlite` | Local response cache (excluded from Git) |
| `requirements.txt` | Pinned direct Python dependencies |

## Reproduce
Use Python 3.13 (the version used for the existing research environment) and `uv`.
The committed transcripts, judgements, annotations and cost ledger let the analysis
and report run without an API key or the local response cache. Building the paper
also requires a LaTeX installation with `pdflatex` and `bibtex`.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -I analysis/study1.py --split main
.venv/bin/python -I analysis/study2.py --split main
.venv/bin/python -I analysis/make_report.py --s1 main --s2 main
(cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main)
```

Running the LLM experiment or judging scripts requires `OPENROUTER_API_KEY` in the
environment. The client reuses matching responses when the local SQLite cache is
present; cache misses make paid API calls. The cache is excluded from Git because
of its size, so a fresh clone does not include it. Virtual environments, bytecode,
run logs and LaTeX intermediates are also excluded.
