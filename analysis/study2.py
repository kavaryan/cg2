"""Study 2 analysis: value of selection rules on a common candidate set, with all potential outcomes observed.
Written on dry-run (pilot) data before the main Study-2 data existed.

Per decision point h and (for t=1) per defendant type, V[j, e] = mean over M independent continuations of judge e's
probability of guilt at round d after candidate j is asked. Selection rules map planner scores -> chosen j; ties are
resolved in expectation (average over the tied set), which equals uniform random tie-breaking in expectation.
Inference: aggregate each estimand to the case level (decision points of one case are dependent), then t-test /
bootstrap / sign-flip across cases.

Usage: python -I analysis/study2.py --split main
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import ROOT, summarize, JUDGE_NAMES  # noqa: E402

PLANNERS = {"A": "scoreA", "E": "scoreB"}  # internal judge -> score field (scoreB was computed with DeepSeek = E)


def util(sc, kind):
    if kind == "adv":
        return np.array([0.5 * (s["pG"] + s["pI"]) for s in sc])
    return np.array([s["pG"] - s["pI"] for s in sc])


def pick_value(scores: np.ndarray, values: np.ndarray, subset=None) -> float:
    idx = np.arange(len(scores)) if subset is None else np.array(subset)
    s = scores[idx]
    tied = idx[s == s.max()]
    return float(values[tied].mean())


def load(split):
    H = [json.loads(l) for l in (ROOT / f"data/raw/study2_{split}.jsonl").open()]
    J = pd.DataFrame([json.loads(l) for l in (ROOT / f"data/raw/study2_{split}_judgments.jsonl").open()])
    J = J.drop_duplicates(["cid", "judge", "stage"], keep="last")
    return H, J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main")
    args = ap.parse_args()
    H, J = load(args.split)
    judges = sorted(J.judge.unique())
    R = {"split": args.split, "n_decision_points": len(H)}
    rows = []          # one row per (decision point, type, evaluator, stage-kind): values of each rule
    cand_rows = []     # candidate-level table
    for h in H:
        k = len(h["cands"])
        jh = J[J.hid == h["hid"]]
        for ty in h["types"]:
            for e in judges:
                for kind, st in (("final", h["d"]), ("immediate", h["t"])):
                    sub = jh[(jh.type == ty) & (jh.judge == e) & (jh.stage == st)]
                    if sub.empty:
                        continue
                    piv = sub.pivot_table(index="j", columns="m", values="ps", aggfunc="mean").reindex(range(k))
                    if piv.isna().any().any():
                        continue
                    V = piv.mean(axis=1).values
                    rec = {"hid": h["hid"], "case_id": h["case_id"], "t": h["t"], "type": ty, "evaluator": e,
                           "kind": kind, "random": V.mean(), "intro": V[h["intro_idx"]],
                           "sd_cand": V.std(ddof=1)}
                    for pl, field in PLANNERS.items():
                        for uk in ("adv", "min"):
                            rec[f"look_{pl}_{uk}"] = pick_value(util(h[field], uk), V)
                    # dose-response: expected value of best-of-k under planner A (advocate utility), all subsets
                    for pl, field in PLANNERS.items():
                        u = util(h[field], "adv")
                        for kk in range(1, k + 1):
                            vals = [pick_value(u, V, s) for s in itertools.combinations(range(k), kk)]
                            rec[f"bestof{kk}_{pl}"] = float(np.mean(vals))
                    # split-sample oracle: select on continuation m=0, evaluate on m=1 (and vice versa), averaged
                    if piv.shape[1] >= 2:
                        v0, v1 = piv[0].values, piv[1].values
                        rec["oracle_split"] = 0.5 * (pick_value(v0, v1) + pick_value(v1, v0))
                        # reliability of candidate values across independent continuations
                        rec["rel_r"] = float(np.corrcoef(v0, v1)[0, 1]) if v0.std() > 0 and v1.std() > 0 else np.nan
                    rows.append(rec)
                    if kind == "final":
                        for j in range(k):
                            cand_rows.append({"hid": h["hid"], "case_id": h["case_id"], "t": h["t"], "type": ty,
                                              "evaluator": e, "j": j, "V": V[j],
                                              "uA_adv": util(h["scoreA"], "adv")[j], "uA_min": util(h["scoreA"], "min")[j],
                                              "uE_adv": util(h["scoreB"], "adv")[j], "uE_min": util(h["scoreB"], "min")[j],
                                              "q_words": len(h["cands"][j].split()),
                                              "mentions_W": int(any(w in h["cands"][j].lower() for w in
                                                                    ["record", "log", "receipt", "phone", "data", "card",
                                                                     "camera", "footage", "location", "transaction"]))})
    D = pd.DataFrame(rows)
    C = pd.DataFrame(cand_rows)
    D.to_csv(ROOT / f"results/study2_{args.split}_rules.csv", index=False)
    C.to_csv(ROOT / f"results/study2_{args.split}_candidates.csv", index=False)

    def case_level(df, col):
        return df.groupby("case_id")[col].mean()

    F = D[D.kind == "final"]
    rules = ["intro", "look_A_adv", "look_E_adv", "look_A_min", "look_E_min", "oracle_split"]

    # H7-H9 are pre-registered on the IMMEDIATE outcome (judge right after the real reply to the branched move),
    # which is what the one-step planner optimises; the FINAL outcome (round d) tests persistence (secondary).
    k = 8
    xs = np.log2(np.arange(1, k + 1))
    dose = {}
    for kind in ("immediate", "final"):
        Dk = D[D.kind == kind]
        fA = Dk[Dk.evaluator == "A"]
        R[f"H7_lookA_vs_random_evalA_{kind}"] = summarize(case_level(fA.assign(x=fA.look_A_adv - fA.random), "x"))
        if "E" in judges and not Dk[Dk.evaluator == "E"].empty:
            fE = Dk[Dk.evaluator == "E"]
            m = fA.merge(fE, on=["hid", "case_id", "type", "t"], suffixes=("_A", "_E"))
            sA, sE = m.sd_cand_A.mean(), m.sd_cand_E.mean()
            inter = ((m.look_A_adv_A - m.look_E_adv_A) / sA) - ((m.look_A_adv_E - m.look_E_adv_E) / sE)
            R[f"H8_crossover_interaction_std_{kind}"] = summarize(case_level(m.assign(x=inter), "x"))
            cells = {}
            for pl in ("A", "E"):
                for ev, sd in (("A", sA), ("E", sE)):
                    cells[f"{pl}->{ev}"] = summarize(case_level(
                        m.assign(x=(m[f"look_{pl}_adv_{ev}"] - m[f"random_{ev}"]) / sd), "x"))
                    cells[f"{pl}->{ev}_raw"] = summarize(case_level(
                        m.assign(x=(m[f"look_{pl}_adv_{ev}"] - m[f"random_{ev}"])), "x"))
            R[f"crossover_cells_{kind}"] = cells
            R[f"crossover_scale_{kind}"] = {"sd_cand_A": sA, "sd_cand_E": sE}
        slopes = fA.apply(lambda r: np.polyfit(xs, [r[f"bestof{kk}_A"] for kk in range(1, k + 1)], 1)[0], axis=1)
        R[f"H9_dose_slope_log2k_A_evalA_{kind}"] = summarize(case_level(fA.assign(x=slopes), "x"))
        for pl in PLANNERS:
            for e in judges:
                fe = Dk[Dk.evaluator == e]
                if fe.empty or f"bestof{k}_{pl}" not in fe:
                    continue
                curve = {kk: summarize(case_level(fe.assign(x=fe[f"bestof{kk}_{pl}"] - fe.random), "x"))
                         for kk in range(1, k + 1)}
                dose[f"{kind}:{pl}->{e}"] = {kk: {"est": v["est"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                                             for kk, v in curve.items()}
        R[f"selection_value_{kind}"] = {
            f"{r}-random|eval={e}": summarize(case_level(Dk[Dk.evaluator == e].assign(
                x=Dk[Dk.evaluator == e][r] - Dk[Dk.evaluator == e]["random"]), "x"))
            for e in judges for r in rules if r in Dk and not Dk[Dk.evaluator == e].empty}
        R[f"candidate_value_reliability_r_{kind}"] = {e: float(Dk[Dk.evaluator == e].rel_r.mean()) for e in judges
                                                      if not Dk[Dk.evaluator == e].empty}
        R[f"candidate_value_sd_{kind}"] = {e: float(Dk[Dk.evaluator == e].sd_cand.mean()) for e in judges
                                           if not Dk[Dk.evaluator == e].empty}
    R["dose_response"] = dose
    fA = F[F.evaluator == "A"]

    # persistence: immediate (stage t) vs final (stage d) selection value, planner A, evaluator A
    I = D[(D.kind == "immediate") & (D.evaluator == "A")]
    if not I.empty:
        R["persistence_immediate"] = summarize(case_level(I.assign(x=I.look_A_adv - I.random), "x"))
        R["persistence_final"] = R["H7_lookA_vs_random_evalA_final"]
        for t in sorted(D.t.unique()):
            It, Ft = I[I.t == t], fA[fA.t == t]
            R[f"persistence_t{t}"] = {"immediate": summarize(case_level(It.assign(x=It.look_A_adv - It.random), "x")),
                                      "final": summarize(case_level(Ft.assign(x=Ft.look_A_adv - Ft.random), "x"))}

    # planner validity: within-decision-point Spearman between planner score and realised value
    val = {}
    for pl in ("A", "E"):
        for e in judges:
            ce = C[C.evaluator == e]
            rs = ce.groupby(["hid", "type"]).apply(
                lambda g: stats.spearmanr(g[f"u{pl}_adv"], g.V).correlation if g[f"u{pl}_adv"].std() > 0 and g.V.std() > 0 else np.nan)
            rs = rs.dropna()
            z = np.arctanh(np.clip(rs.values, -0.999, 0.999))
            val[f"{pl}->{e}"] = {"mean_rho": float(np.tanh(z.mean())) if len(z) else np.nan, "n": int(len(z)),
                                 "case_level": summarize(pd.Series(z, index=rs.index.get_level_values(0)).groupby(
                                     lambda h: h.split("|")[1]).mean()) if len(z) > 2 else None}
    R["planner_validity"] = val

    # t=1 discrimination: do minister scores predict candidate-level discriminativeness V(G) - V(I)?
    disc = {}
    for e in judges:
        c1 = C[(C.t == 1) & (C.evaluator == e)].pivot_table(index=["hid", "case_id", "j"], columns="type", values="V")
        if {"guilty", "innocent"} <= set(c1.columns):
            c1 = c1.reset_index()
            c1["Dj"] = c1.guilty - c1.innocent
            sc = C[(C.t == 1) & (C.evaluator == e) & (C.type == "guilty")][["hid", "j", "uA_min", "uE_min"]]
            c1 = c1.merge(sc, on=["hid", "j"])
            per = c1.groupby("case_id")
            disc[e] = {
                "mean_disc_random": summarize(per.Dj.mean()),
                "disc_gain_lookA_min": summarize(per.apply(lambda g: pick_value(g.uA_min.values, g.Dj.values) - g.Dj.mean())),
                "disc_gain_lookE_min": summarize(per.apply(lambda g: pick_value(g.uE_min.values, g.Dj.values) - g.Dj.mean())),
                "disc_gain_oracle_upper": summarize(per.apply(lambda g: g.Dj.max() - g.Dj.mean())),
            }
    R["t1_discrimination"] = disc


    (ROOT / f"results/study2_{args.split}.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: (v if not isinstance(v, dict) or "est" not in v else
                          {kk: round(v[kk], 3) for kk in ("est", "ci_lo", "ci_hi", "p_t")})
                      for k, v in R.items() if k.startswith(("H", "n_", "candidate", "persistence_i", "persistence_f"))},
                     indent=1, default=float))
    for kind in ("immediate", "final"):
        print(kind, {k: round(v["est"], 2) for k, v in R[f"selection_value_{kind}"].items() if "eval=A" in k or "eval=E" in k})
        print(kind, "reliability", R[f"candidate_value_reliability_r_{kind}"], "sd", R[f"candidate_value_sd_{kind}"])


if __name__ == "__main__":
    main()
