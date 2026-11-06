"""Shared loading and estimation utilities.

Estimation philosophy (see paper Sec. 4): every case is observed under every treatment cell, so contrasts are
computed WITHIN case and averaged ACROSS cases. Inference treats cases as a random sample from the case
population (stimulus sampling) and the simulator as a stochastic data-generating process; the case-level
contrast vector therefore carries both sources of uncertainty. We report t-intervals, case-cluster bootstrap
intervals and sign-flip randomisation p-values for each contrast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["objective", "pros_sel", "def_sel", "dtype"]
LEVELS = {"objective": ["advocate", "minister"], "pros_sel": ["random", "introspect", "lookahead"],
          "def_sel": ["random", "lookahead"], "dtype": ["guilty", "innocent"]}
JUDGE_NAMES = {"A": "GPT-4o-mini", "B": "Llama-3.3-70B", "C": "Gemini-2.5-FL", "D": "Claude-Haiku-5.5",
               "E": "DeepSeek-V4-Flash", "F": "Mistral-Small-3.2"}


def load_cases():
    return {c["case_id"]: c for c in json.loads((ROOT / "data/cases/cases.json").read_text())}


def load_games(split: str) -> list[dict]:
    p = ROOT / f"data/raw/study1_{split}.jsonl"
    games = {}
    for l in p.open():
        g = json.loads(l)
        games[g["game_id"]] = g  # last write wins (re-runs)
    return list(games.values())


def first_passage(verdicts: list[str]) -> str:
    for v in verdicts:
        if v in ("GUILTY", "NOT_GUILTY"):
            return v
    return "CONTINUE"


def thr_verdicts(ps: list[float], theta_g: float, theta_n: float) -> list[str]:
    return ["GUILTY" if p >= theta_g else ("NOT_GUILTY" if p <= theta_n else "CONTINUE") for p in ps]


def tidy(split: str, variant: str = "neutral", theta=(80.0, 10.0)) -> pd.DataFrame:
    """One row per (game, judge): stage-wise soft p, first-passage verdict outcomes, utilities."""
    games = {g["game_id"]: g for g in load_games(split)}
    jp = ROOT / f"data/raw/study1_{split}_judgments.jsonl"
    J = pd.DataFrame([json.loads(l) for l in jp.open()])
    J = J[J.variant == variant].drop_duplicates(["game_id", "judge", "t"], keep="last")
    rows = []
    for (gid, jn), grp in J.groupby(["game_id", "judge"]):
        if gid not in games:
            continue
        g = games[gid]
        d = g["d"]
        by_t = grp.set_index("t")
        rec = {k: g[k] for k in ["game_id", "case_id"] + FACTORS}
        rec["judge"] = jn
        have = set(by_t.index)
        for t in range(d + 1):
            if t in have:
                rec[f"ps{t}"] = float(by_t.loc[t, "ps"])
                rec[f"p{t}"] = float(by_t.loc[t, "p"])
                rec[f"v{t}"] = by_t.loc[t, "v"]
        rec["p_final"] = rec.get(f"ps{d}", np.nan)
        rec["p_final_verbal"] = rec.get(f"p{d}", np.nan)
        if all(t in have for t in range(1, d + 1)):
            vs = [rec[f"v{t}"] for t in range(1, d + 1)]
            out = first_passage(vs)
            rec["outcome"] = out
            rec["stop_t"] = next((i + 1 for i, v in enumerate(vs) if v != "CONTINUE"), d)
            pss = [rec[f"ps{t}"] for t in range(1, d + 1)]
            rec["outcome_thr"] = first_passage(thr_verdicts(pss, *theta))
        rows.append(rec)
    df = pd.DataFrame(rows)
    if "outcome" in df:
        df["convict"] = (df.outcome == "GUILTY").astype(float).where(df.outcome.notna())
        df["convict_thr"] = (df.outcome_thr == "GUILTY").astype(float).where(df.outcome_thr.notna())
        guilty = df.dtype == "guilty"
        # Interrogation-game utility of X (Table 1): +1 correct decisive verdict, -1 wrong, 0 no verdict.
        dec = df.outcome.isin(["GUILTY", "NOT_GUILTY"])
        correct = ((df.outcome == "GUILTY") & guilty) | ((df.outcome == "NOT_GUILTY") & ~guilty)
        df["u_interrog"] = np.where(dec, np.where(correct, 1.0, -1.0), 0.0)
        df["wrongful"] = ((df.outcome == "GUILTY") & ~guilty).astype(float)
    return df


# ----------------------------------------------------------------------------- case-level contrasts

def cell_table(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """case x (objective, pros_sel, def_sel, dtype) table of mean outcome."""
    return df.pivot_table(index="case_id", columns=FACTORS, values=y, aggfunc="mean")


def cm(tab: pd.DataFrame, **f) -> pd.Series:
    """Per-case mean over all cells matching the filter (unspecified factors are averaged)."""
    cols = [c for c in tab.columns if all(c[FACTORS.index(k)] == v for k, v in f.items())]
    return tab[cols].mean(axis=1)


def summarize(x: pd.Series, n_boot: int = 10000, n_perm: int = 20000, seed: int = 0) -> dict:
    x = x.dropna().values.astype(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm = np.abs((signs * x).mean(axis=1))
    p_perm = (1 + np.sum(perm >= abs(m) - 1e-12)) / (n_perm + 1)
    p_t = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    return {"est": m, "se": se, "ci_lo": m - tcrit * se, "ci_hi": m + tcrit * se,
            "boot_lo": np.percentile(boots, 2.5), "boot_hi": np.percentile(boots, 97.5),
            "p_t": p_t, "p_perm": p_perm, "n_cases": n, "dz": m / sd if sd > 0 else np.nan}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def paired_auc(tab: pd.DataFrame, **f) -> pd.Series:
    """Per-case indicator that the guilty version is judged guiltier than the innocent version (ties = 0.5)."""
    g = cm(tab, dtype="guilty", **f)
    i = cm(tab, dtype="innocent", **f)
    return (g > i).astype(float) + 0.5 * (g == i).astype(float)


def fmt_ci(r: dict, scale: float = 1.0, digits: int = 1) -> str:
    return (f"{r['est']*scale:+.{digits}f} [{r['ci_lo']*scale:+.{digits}f}, {r['ci_hi']*scale:+.{digits}f}]")


def nash_zero_sum(A: np.ndarray):
    """Row player's maximin mixed strategy and game value for payoff matrix A (row maximises)."""
    from scipy.optimize import linprog
    m, n = A.shape
    shift = A.min() - 1.0
    B = A - shift
    # minimise sum(x) s.t. B^T x >= 1, x >= 0  ->  value = 1/sum(x)
    res = linprog(np.ones(m), A_ub=-B.T, b_ub=-np.ones(n), bounds=[(0, None)] * m, method="highs")
    x = res.x / res.x.sum()
    v = 1.0 / res.x.sum() + shift
    res2 = linprog(-np.ones(n), A_ub=B, b_ub=np.ones(m), bounds=[(0, None)] * n, method="highs")
    y = res2.x / res2.x.sum()
    return x, y, v
