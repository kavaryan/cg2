"""Study 1 confirmatory + exploratory analysis. Written and frozen on PILOT data before main data existed.

Usage: python -I analysis/study1.py --split main
Outputs: results/study1_<split>.json (all estimates), results/tables/*.tex, printed summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import (FACTORS, JUDGE_NAMES, LEVELS, ROOT, cell_table, cm, holm, load_cases, load_games,  # noqa
                             nash_zero_sum, paired_auc, summarize, tidy)

HELD_OUT = ["B", "C", "D", "E", "F"]  # every judge the planner did not use internally


def contrasts_primary(tab: pd.DataFrame) -> dict[str, pd.Series]:
    """Pre-registered confirmatory contrasts H1-H6 (per-case vectors). tab = case x cell table of primary outcome."""
    adv_look = cm(tab, objective="advocate", pros_sel="lookahead")
    adv_rand = cm(tab, objective="advocate", pros_sel="random")
    adv_intr = cm(tab, objective="advocate", pros_sel="introspect")
    c = {
        "H1_adv_lookahead_vs_random": adv_look - adv_rand,
        "H2_adv_lookahead_vs_introspect": adv_look - adv_intr,
        "H3_adv_lookahead_vs_random_innocent": (cm(tab, objective="advocate", pros_sel="lookahead", dtype="innocent")
                                                - cm(tab, objective="advocate", pros_sel="random", dtype="innocent")),
        # H4: discrimination gap Delta = Y(G) - Y(I), minister vs advocate, under lookahead
        "H4_disc_minister_vs_advocate_lookahead": (
            (cm(tab, objective="minister", pros_sel="lookahead", dtype="guilty")
             - cm(tab, objective="minister", pros_sel="lookahead", dtype="innocent"))
            - (cm(tab, objective="advocate", pros_sel="lookahead", dtype="guilty")
               - cm(tab, objective="advocate", pros_sel="lookahead", dtype="innocent"))),
        "H5_def_lookahead_vs_random": cm(tab, def_sel="lookahead") - cm(tab, def_sel="random"),
    }
    return c


def full_effects(tab: pd.DataFrame) -> dict[str, pd.Series]:
    """Exploratory: all main effects and two-way interactions of interest as per-case contrasts."""
    e = {}
    for s in ["introspect", "lookahead"]:
        for o in LEVELS["objective"]:
            for ty in LEVELS["dtype"]:
                e[f"{o}:{s}-random|{ty}"] = cm(tab, objective=o, pros_sel=s, dtype=ty) - cm(tab, objective=o, pros_sel="random", dtype=ty)
            e[f"{o}:{s}-random"] = cm(tab, objective=o, pros_sel=s) - cm(tab, objective=o, pros_sel="random")
    e["objective:minister-advocate"] = cm(tab, objective="minister") - cm(tab, objective="advocate")
    e["type:guilty-innocent"] = cm(tab, dtype="guilty") - cm(tab, dtype="innocent")
    for dl in LEVELS["def_sel"]:
        e[f"adv:lookahead-random|def={dl}"] = (cm(tab, objective="advocate", pros_sel="lookahead", def_sel=dl)
                                              - cm(tab, objective="advocate", pros_sel="random", def_sel=dl))
    e["adv:PxD_interaction"] = e["adv:lookahead-random|def=lookahead"] - e["adv:lookahead-random|def=random"]
    for o in LEVELS["objective"]:
        for s in LEVELS["pros_sel"]:
            for dl in LEVELS["def_sel"]:
                e[f"disc|{o},{s},{dl}"] = (cm(tab, objective=o, pros_sel=s, def_sel=dl, dtype="guilty")
                                          - cm(tab, objective=o, pros_sel=s, def_sel=dl, dtype="innocent"))
            e[f"disc|{o},{s}"] = cm(tab, objective=o, pros_sel=s, dtype="guilty") - cm(tab, objective=o, pros_sel=s, dtype="innocent")
    for o in LEVELS["objective"]:
        for s in ["introspect", "lookahead"]:
            e[f"disc_change|{o},{s}-random"] = e[f"disc|{o},{s}"] - e[f"disc|{o},random"]
    return e


def meta_game(tab: pd.DataFrame, objective="advocate"):
    """Court game (Eq. 1) payoff for X = 2*P(conviction)-1 surrogate from the tab; zero-sum."""
    A = np.array([[cm(tab, objective=objective, pros_sel=s, def_sel=d).mean() for d in LEVELS["def_sel"]]
                  for s in LEVELS["pros_sel"]])
    x, y, v = nash_zero_sum(A)
    return A, x, y, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main")
    ap.add_argument("--theta-g", type=float, default=80.0)
    ap.add_argument("--theta-n", type=float, default=10.0)
    args = ap.parse_args()
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    df = tidy(args.split, theta=(args.theta_g, args.theta_n))
    df.to_csv(outdir / f"study1_{args.split}_tidy.csv", index=False)
    R = {"split": args.split, "n_games": int(df.game_id.nunique()), "n_cases": int(df.case_id.nunique())}
    A = df[df.judge == "A"]

    # ---------------- descriptives
    desc = {}
    for jn, g in df.groupby("judge"):
        d = {"p_final_mean": g.p_final.mean(), "p_final_sd": g.p_final.std()}
        if "outcome" in g and g.outcome.notna().any():
            d.update({f"share_{k}": float((g.outcome == k).mean()) for k in ["GUILTY", "NOT_GUILTY", "CONTINUE"]})
            d["share_thr_GUILTY"] = float((g.outcome_thr == "GUILTY").mean())
        desc[jn] = d
    R["descriptives_by_judge"] = desc
    print("Descriptives by judge:\n", pd.DataFrame(desc).T.round(3))

    # ---------------- confirmatory (primary outcome: judge A soft p at round d)
    tab = cell_table(A, "p_final")
    conf = {k: summarize(v) for k, v in contrasts_primary(tab).items()}
    # H6: transfer gap - advocate lookahead advantage under A minus mean advantage under held-out judges
    adv_eff = {}
    for jn in sorted(df.judge.unique()):
        tj = cell_table(df[df.judge == jn], "p_final")
        adv_eff[jn] = cm(tj, objective="advocate", pros_sel="lookahead") - cm(tj, objective="advocate", pros_sel="random")
    ho = [j for j in HELD_OUT if j in adv_eff]
    if ho:
        conf["H6_transfer_gap_A_minus_heldout"] = summarize(adv_eff["A"] - pd.concat([adv_eff[j] for j in ho], axis=1).mean(axis=1))
    R["confirmatory"] = conf
    R["holm_s1_only"] = holm({k: v["p_t"] for k, v in conf.items()})
    print("\nConfirmatory (Y = judge-A soft P(guilty) at round d, points):")
    for k, v in conf.items():
        print(f"  {k:45s} est={v['est']:+6.2f}  95%CI=[{v['ci_lo']:+6.2f},{v['ci_hi']:+6.2f}]  boot=[{v['boot_lo']:+6.2f},"
              f"{v['boot_hi']:+6.2f}]  p_t={v['p_t']:.4f}  p_perm={v['p_perm']:.4f}  dz={v['dz']:+.2f}  sd={v['se']*np.sqrt(v['n_cases']):.2f}")

    # ---------------- judge-specific effects (transfer)
    R["adv_lookahead_effect_by_judge"] = {jn: summarize(v) for jn, v in adv_eff.items()}
    print("\nAdvocate lookahead-random effect by evaluator judge:")
    for jn, v in R["adv_lookahead_effect_by_judge"].items():
        print(f"  {jn} {JUDGE_NAMES.get(jn, jn):18s} {v['est']:+6.2f} [{v['ci_lo']:+6.2f},{v['ci_hi']:+6.2f}] p={v['p_t']:.4f}")

    # ---------------- exploratory effects on p_final and on conviction, per judge A
    R["exploratory_p"] = {k: summarize(v) for k, v in full_effects(tab).items()}
    if A.convict.notna().any():
        tabc = cell_table(A, "convict")
        R["exploratory_convict"] = {k: summarize(v) for k, v in full_effects(tabc).items()}
        R["confirmatory_on_convict"] = {k: summarize(v) for k, v in contrasts_primary(tabc).items()}
        tabw = cell_table(A, "u_interrog")
        R["exploratory_u_interrog"] = {k: summarize(v) for k, v in full_effects(tabw).items()}
        tabt = cell_table(A, "convict_thr")
        R["confirmatory_on_convict_thr"] = {k: summarize(v) for k, v in contrasts_primary(tabt).items()}

    # ---------------- cell means (for tables/figures)
    cells = A.groupby(FACTORS).agg(p_final=("p_final", "mean"), convict=("convict", "mean"),
                                   convict_thr=("convict_thr", "mean"), n=("p_final", "size")).reset_index()
    R["cells_A"] = cells.to_dict(orient="records")

    # ---------------- paired discrimination (paired AUC) per prosecutor cell, judge A
    pauc = {}
    for o in LEVELS["objective"]:
        for s in LEVELS["pros_sel"]:
            pauc[f"{o},{s}"] = summarize(paired_auc(tab, objective=o, pros_sel=s))
    R["paired_auc_A"] = pauc

    # ---------------- meta-game (court game) on p_final and on conviction
    Ap, x, y, v = meta_game(tab)
    R["metagame_p"] = {"A": Ap.tolist(), "x": x.tolist(), "y": y.tolist(), "value": v}
    print("\nMeta-game payoff (judge A mean p_final): rows pros_sel", LEVELS["pros_sel"], "cols def_sel", LEVELS["def_sel"])
    print(np.round(Ap, 2), "\n  NE prosecutor", np.round(x, 2), "defence", np.round(y, 2), "value", round(v, 2))

    # ---------------- heterogeneity: case-level advocate lookahead effect, random-slope model
    try:
        import statsmodels.formula.api as smf
        sub = A[(A.objective == "advocate") & (A.pros_sel.isin(["random", "lookahead"]))].copy()
        sub["look"] = (sub.pros_sel == "lookahead").astype(float)
        sub["dl"] = (sub.def_sel == "lookahead").astype(float)
        sub["gu"] = (sub.dtype == "guilty").astype(float)
        m = smf.mixedlm("p_final ~ look + dl + gu", sub, groups=sub["case_id"], re_formula="~look").fit(reml=True)
        tau = float(np.sqrt(max(m.cov_re.iloc[1, 1], 0)))
        b = float(m.fe_params["look"])
        R["heterogeneity"] = {"beta_look": b, "se": float(m.bse["look"]), "tau_case_slope": tau,
                              "pred_int_lo": b - 1.96 * tau, "pred_int_hi": b + 1.96 * tau}
        print(f"\nRandom-slope model: lookahead effect {b:+.2f} (se {m.bse['look']:.2f}), case-level SD tau={tau:.2f}, "
              f"95% prediction interval for a single case [{b-1.96*tau:+.2f}, {b+1.96*tau:+.2f}]")
    except Exception as ex:  # pragma: no cover
        print("mixedlm failed", ex)

    # ---------------- utterance features by arm (mechanism, exploratory)
    feats = []
    for g in load_games(args.split):
        for i, r in enumerate(g["rounds"]):
            q = r["q"]
            feats.append({"objective": g["objective"], "pros_sel": g["pros_sel"], "round": i + 1,
                          "q_words": len(q.split()), "a_words": len(r["a"].split()),
                          "cites_evidence": int(any(w in q for w in ["E1", "E2", "E3", "E4", "E5"]) or "evidence" in q.lower()),
                          "def_sel": g["def_sel"]})
    F = pd.DataFrame(feats)
    R["utterance_features"] = F.groupby(["objective", "pros_sel"])[["q_words", "a_words", "cites_evidence"]].mean().reset_index().to_dict(orient="records")
    print("\nUtterance features:\n", F.groupby(["objective", "pros_sel"])[["q_words", "a_words", "cites_evidence"]].mean().round(2))

    (outdir / f"study1_{args.split}.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"\nwrote results/study1_{args.split}.json")


if __name__ == "__main__":
    main()
