"""Generate figures, LaTeX tables and number macros for the paper from results/*.json and tidy CSVs.
Usage: python -I analysis/make_report.py --s1 main --s2 main
Every number quoted in the paper comes from paper/macros.tex written here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.common import ROOT, JUDGE_NAMES, holm, cell_table, cm, summarize  # noqa: E402

PAPER = ROOT / "paper"
FIGS = PAPER / "figs"
TABS = PAPER / "tables"
# Reference palette (dataviz skill): categorical slots 1-3, chrome and ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TYPE_COL = {"guilty": ORANGE, "innocent": BLUE}
SEL_LAB = {"random": "random", "introspect": "introspect", "lookahead": "lookahead"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "text.color": INK,
    "axes.titlecolor": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})

MACROS: dict[str, str] = {}


_DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
        "8": "Eight", "9": "Nine"}


def mac(name, val):
    """LaTeX macro names cannot contain digits: spell them out (H1 -> HOne, R1 -> ROne)."""
    MACROS["".join(_DIG.get(ch, ch) for ch in name)] = val


def f1(x, d=1, sign=True):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "\\ensuremath{-}")


def ci(r, d=1, sign=True):
    return f"{f1(r['est'], d, sign)} [{f1(r['ci_lo'], d, sign)}, {f1(r['ci_hi'], d, sign)}]"


def pfmt(p):
    if p < 0.001:
        return "\\ensuremath{<}0.001"
    return f"{p:.3f}"


def case_ci(series):
    x = series.dropna().values
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se


# =============================================================================================== Study 1
def study1(split):
    R = json.loads((ROOT / f"results/study1_{split}.json").read_text())
    T = pd.read_csv(ROOT / f"results/study1_{split}_tidy.csv")
    A = T[T.judge == "A"]
    tab = cell_table(A, "p_final")
    mac("SoneNgames", f"{R['n_games']:,}")
    mac("SoneNcases", str(R["n_cases"]))
    conf = R["confirmatory"]
    names = {"H1_adv_lookahead_vs_random": "One", "H2_adv_lookahead_vs_introspect": "Two",
             "H3_adv_lookahead_vs_random_innocent": "Three", "H4_disc_minister_vs_advocate_lookahead": "Four",
             "H5_def_lookahead_vs_random": "Five", "H6_transfer_gap_A_minus_heldout": "Six"}
    for k, n in names.items():
        r = conf[k]
        mac(f"H{n}Est", f1(r["est"])); mac(f"H{n}CI", ci(r)); mac(f"H{n}P", pfmt(r["p_t"]))
        mac(f"H{n}dz", f1(r["dz"], 2))
    # exploratory numbers used in text
    ex = R["exploratory_p"]
    def em(key, name):
        r = ex[key]; mac(name, ci(r)); mac(name + "Est", f1(r["est"])); mac(name + "P", pfmt(r["p_t"]))
    em("type:guilty-innocent", "TypeEffect")
    em("advocate:introspect-random", "AdvIntro")
    em("advocate:lookahead-random", "AdvLook")
    em("minister:lookahead-random", "MinLook")
    em("minister:introspect-random", "MinIntro")
    em("advocate:lookahead-random|guilty", "AdvLookG")
    em("advocate:lookahead-random|innocent", "AdvLookI")
    em("minister:lookahead-random|guilty", "MinLookG")
    em("minister:lookahead-random|innocent", "MinLookI")
    em("objective:minister-advocate", "ObjEffect")
    em("adv:PxD_interaction", "PxD")
    em("adv:lookahead-random|def=random", "AdvLookDefRand")
    em("adv:lookahead-random|def=lookahead", "AdvLookDefLook")
    for o in ("advocate", "minister"):
        for s in ("random", "introspect", "lookahead"):
            em(f"disc|{o},{s}", f"Disc{o[:3].title()}{s[:4].title()}")
        for s in ("introspect", "lookahead"):
            em(f"disc_change|{o},{s}-random", f"DiscChg{o[:3].title()}{s[:4].title()}")
    # conviction
    if "exploratory_convict" in R:
        exc = R["exploratory_convict"]
        for key, name in (("advocate:lookahead-random", "ConvAdvLook"), ("advocate:lookahead-random|innocent", "ConvAdvLookI"),
                          ("advocate:lookahead-random|guilty", "ConvAdvLookG"), ("type:guilty-innocent", "ConvType"),
                          ("minister:lookahead-random", "ConvMinLook")):
            mac(name, ci(exc[key], 1) if abs(exc[key]["est"]) > 1 else ci({k: v * 100 for k, v in exc[key].items() if k in ("est", "ci_lo", "ci_hi")}))
        cA = A.groupby("dtype").convict.mean()
        mac("ConvRateG", f"{100*cA['guilty']:.1f}"); mac("ConvRateI", f"{100*cA['innocent']:.1f}")
        mac("ConvRateAll", f"{100*A.convict.mean():.1f}")
    pa = R["paired_auc_A"]
    for k, v in pa.items():
        o, s = k.split(",")
        mac(f"PAUC{o[:3].title()}{s[:4].title()}", f"{v['est']:.2f}")
    # descriptives
    d = R["descriptives_by_judge"]
    mac("PfinalMeanA", f"{d['A']['p_final_mean']:.1f}")
    drift = A.groupby("dtype")[[f"ps{t}" for t in range(5)]].mean()
    mac("DriftFrom", f"{A['ps0'].mean():.0f}"); mac("DriftTo", f"{A['ps4'].mean():.0f}")
    # transfer by judge
    adv = R["adv_lookahead_effect_by_judge"]
    for j, v in adv.items():
        mac(f"AdvLookJudge{j}", ci(v)); mac(f"AdvLookJudge{j}P", pfmt(v["p_t"]))
    # heterogeneity
    if "heterogeneity" in R:
        h = R["heterogeneity"]
        mac("HetTau", f"{h['tau_case_slope']:.1f}"); mac("HetPI", f"[{f1(h['pred_int_lo'])}, {f1(h['pred_int_hi'])}]")
    # metagame
    mg = R["metagame_p"]
    mac("MetaValue", f"{mg['value']:.1f}")

    # ------------------------------------------------------------------ Figure: cell means
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, o in zip(axes, ["advocate", "minister"]):
        for k, ty in enumerate(["guilty", "innocent"]):
            ms, lo, hi = [], [], []
            for s in ["random", "introspect", "lookahead"]:
                m, l, h_ = case_ci(cm(tab, objective=o, pros_sel=s, dtype=ty))
                ms.append(m); lo.append(m - l); hi.append(h_ - m)
            x = np.arange(3) + (k - 0.5) * 0.12
            ax.errorbar(x, ms, yerr=[lo, hi], fmt="o-", color=TYPE_COL[ty], ms=4.5, lw=1.4, elinewidth=1.0,
                        capsize=0, mec="white", mew=0.8, label=f"{ty} defendant")
        ax.set_xticks(range(3)); ax.set_xticklabels(["random", "introspect", "lookahead"])
        ax.set_title({"advocate": "Advocate prosecutor (court game)", "minister": "Minister prosecutor (interrogation game)"}[o], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Judge P(guilty) after round 4")
    axes[1].legend(loc="lower right", handlelength=1.5)
    fig.savefig(FIGS / "s1_cells.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Figure: drift by judge (small multiples)
    js = [j for j in ["A", "B", "C", "D", "E", "F"] if j in T.judge.unique()]
    allstage = [j for j in js if T[T.judge == j][[f"ps{t}" for t in range(5)]].notna().all(axis=1).mean() > 0.9]
    fig, axes = plt.subplots(1, len(allstage), figsize=(1.65 * len(allstage), 1.9), sharey=True, squeeze=False)
    for ax, j in zip(axes[0], allstage):
        tj = T[(T.judge == j) & (T.pros_sel == "random") & (T.def_sel == "random")]
        for ty in ["guilty", "innocent"]:
            m = tj[tj.dtype == ty][[f"ps{t}" for t in range(5)]].mean()
            ax.plot(range(5), m.values, "-o", color=TYPE_COL[ty], ms=3.5, lw=1.4, mec="white", mew=0.6, label=ty)
        ax.set_title(JUDGE_NAMES[j], loc="left", fontsize=7.5)
        ax.set_xticks(range(5)); ax.set_xlabel("round")
    axes[0][0].set_ylabel("P(guilty)")
    axes[0][-1].legend(loc="lower right", handlelength=1.2)
    fig.savefig(FIGS / "s1_drift.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Figure: transfer forest
    order = [j for j in ["A", "B", "C", "D", "E", "F"] if j in adv]
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    for i, j in enumerate(order):
        v = adv[j]
        col = BLUE if j == "A" else MUTED
        ax.plot([v["ci_lo"], v["ci_hi"]], [i, i], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(v["est"], i, "o", color=col, ms=5, mec="white", mew=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([JUDGE_NAMES[j] + (" (planner's judge)" if j == "A" else "") for j in order])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Lookahead $-$ random, advocate (points)")
    fig.savefig(FIGS / "s1_transfer.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ Figure: heterogeneity caterpillar
    eff = (cm(tab, objective="advocate", pros_sel="lookahead") - cm(tab, objective="advocate", pros_sel="random")).sort_values()
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    ax.plot(range(len(eff)), eff.values, "o", color=BLUE, ms=3.2, mec="white", mew=0.5)
    ax.axhline(eff.mean(), color=INK2, lw=0.9)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.text(len(eff) - 1, eff.mean(), f"mean {eff.mean():+.1f}", color=INK2, ha="right", va="bottom", fontsize=7)
    ax.set_xlabel("case (sorted)"); ax.set_ylabel("Lookahead $-$ random (points)")
    ax.set_xticks([]); ax.grid(axis="x", visible=False)
    fig.savefig(FIGS / "s1_cases.pdf"); plt.close(fig)
    mac("CasesPositive", f"{(eff > 0).sum()}"); mac("CasesTotal", f"{len(eff)}")

    # ------------------------------------------------------------------ Table: judge panel
    rows = []
    for j in order:
        tj = T[T.judge == j]
        tj_tab = cell_table(tj, "p_final")
        disc = summarize(cm(tj_tab, dtype="guilty") - cm(tj_tab, dtype="innocent"))
        rows.append((JUDGE_NAMES[j], "soft" if j in ("A", "B") else "stated", f"{tj.p_final.mean():.1f}",
                     f"{tj.p_final.std():.1f}", ci(disc), ci(adv[j]), pfmt(adv[j]["p_t"])))
    with open(TABS / "s1_judges.tex", "w") as f:
        f.write("\\begin{tabular}{@{}llrrlll@{}}\\toprule\n")
        f.write("Judge & $p$ & Mean & SD & Guilty $-$ innocent & Lookahead $-$ random (adv.) & $p$\\\\\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    # ------------------------------------------------------------------ Table: meta-game
    A_ = np.array(mg["A"])
    with open(TABS / "s1_metagame.tex", "w") as f:
        f.write("\\begin{tabular}{@{}lccc@{}}\\toprule\n & \\multicolumn{2}{c}{Defence} & \\\\\\cmidrule(lr){2-3}\n")
        f.write("Prosecutor (advocate) & random & lookahead & NE weight\\\\\\midrule\n")
        for i, s in enumerate(["random", "introspect", "lookahead"]):
            f.write(f"{s} & {A_[i,0]:.1f} & {A_[i,1]:.1f} & {mg['x'][i]:.2f}\\\\\n")
        f.write(f"NE weight (defence) & {mg['y'][0]:.2f} & {mg['y'][1]:.2f} & \\\\\n\\bottomrule\\end{{tabular}}\n")

    # ------------------------------------------------------------------ Table: cell means (appendix)
    cells = pd.DataFrame(R["cells_A"])
    with open(TABS / "s1_cells.tex", "w") as f:
        f.write("\\begin{tabular}{@{}llllrrr@{}}\\toprule\nObjective & Prosecutor & Defence & Type & $P$(guilty) & Conviction & Conv.\\ ($\\theta$)\\\\\\midrule\n")
        for _, r in cells.iterrows():
            f.write(f"{r['objective']} & {r['pros_sel']} & {r['def_sel']} & {r['dtype']} & {r['p_final']:.1f} & {100*r['convict']:.0f}\\% & {100*r['convict_thr']:.0f}\\%\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
    return R


# =============================================================================================== Study 2
def study2(split):
    p = ROOT / f"results/study2_{split}.json"
    if not p.exists():
        return None
    R = json.loads(p.read_text())
    mac("StwoNpoints", str(R["n_decision_points"]))
    for kind in ("immediate", "final"):
        K = kind.title()
        r = R[f"H7_lookA_vs_random_evalA_{kind}"]; mac(f"HSeven{K}", ci(r, 2)); mac(f"HSeven{K}P", pfmt(r["p_t"]))
        if f"H8_crossover_interaction_std_{kind}" in R:
            r = R[f"H8_crossover_interaction_std_{kind}"]; mac(f"HEight{K}", ci(r, 2)); mac(f"HEight{K}P", pfmt(r["p_t"]))
        r = R[f"H9_dose_slope_log2k_A_evalA_{kind}"]; mac(f"HNine{K}", ci(r, 2)); mac(f"HNine{K}P", pfmt(r["p_t"]))
        rel = R[f"candidate_value_reliability_r_{kind}"]
        mac(f"Rel{K}A", f"{rel.get('A', float('nan')):.2f}"); mac(f"Rel{K}E", f"{rel.get('E', float('nan')):.2f}")
        sv = R[f"selection_value_{kind}"]
        for key, name in (("oracle_split-random|eval=A", "Oracle"), ("intro-random|eval=A", "Intro"),
                          ("look_A_min-random|eval=A", "LookAMin"), ("look_E_adv-random|eval=A", "LookEAdvOnA"),
                          ("look_E_adv-random|eval=E", "LookEAdvOnE"), ("look_A_adv-random|eval=E", "LookAAdvOnE")):
            if key in sv:
                mac(f"Sel{name}{K}", ci(sv[key], 2)); mac(f"Sel{name}{K}P", pfmt(sv[key]["p_t"]))
    sv = R["selection_value_immediate"]
    mac("SelIntroImmediateE", ci(sv["intro-random|eval=E"], 2))
    cc = R["crossover_cells_immediate"]
    for k, nm in (("A->A", "AA"), ("A->E", "AE"), ("E->A", "EA"), ("E->E", "EE")):
        mac(f"Cross{nm}", f"{cc[k]['est']:.2f}")
    for t, nm in ((1, "One"), (3, "Three")):
        mac(f"Persist{nm}Imm", ci(R[f"persistence_t{t}"]["immediate"], 2))
        mac(f"Persist{nm}Fin", ci(R[f"persistence_t{t}"]["final"], 2))
    H_ = [json.loads(l) for l in (ROOT / f"data/raw/study2_{split}.jsonl").open()]
    mac("StwoNcont", f"{sum(len(h['real']) for h in H_):,}")
    # Figure: dose-response (immediate) + crossover
    dose = R["dose_response"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    for key, col, lab in (("immediate:A->A", BLUE, "evaluated by GPT-4o-mini (planner's judge)"),
                          ("immediate:A->E", ORANGE, "evaluated by DeepSeek-V4-Flash")):
        if key in dose:
            ks = sorted(int(k) for k in dose[key])
            est = [dose[key][str(k)]["est"] for k in ks]
            lo = [dose[key][str(k)]["ci_lo"] for k in ks]; hi = [dose[key][str(k)]["ci_hi"] for k in ks]
            ax.fill_between(ks, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(ks, est, "-o", color=col, ms=3.5, lw=1.4, mec="white", mew=0.6, label=lab)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("candidates considered, $k$"); ax.set_ylabel("Selection value (points)")
    ax.set_title("Best-of-$k$ (planner: GPT-4o-mini)", loc="left")
    h0, l0 = ax.get_legend_handles_labels()
    ax.set_xticks(range(1, 9))
    # crossover (standardized) immediate
    for ax, kind in zip(axes[1:], ("immediate", "final")):
        cells = R.get(f"crossover_cells_{kind}")
        if not cells:
            continue
        for i, ev in enumerate(["A", "E"]):
            for k, (pl, col) in enumerate((("A", BLUE), ("E", ORANGE))):
                v = cells[f"{pl}->{ev}"]
                x = i + (k - 0.5) * 0.22
                ax.plot([x, x], [v["ci_lo"], v["ci_hi"]], color=col, lw=1.4, solid_capstyle="round")
                ax.plot(x, v["est"], "o", color=col, ms=4.5, mec="white", mew=0.7,
                        label=({"A": "planner: GPT-4o-mini", "E": "planner: DeepSeek"}[pl] if i == 0 else None))
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["GPT-4o-mini", "DeepSeek"]); ax.set_xlabel("evaluator")
        ax.set_title(f"Crossover, {kind} (SD units)", loc="left"); ax.grid(axis="x", visible=False)
    axes[1].set_ylabel("Selection value (SD)")
    h, l = axes[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.0, rect=(0, 0.12, 1, 1))
    fig.legend(h0, l0, loc="lower center", bbox_to_anchor=(0.20, -0.03), ncol=1, fontsize=6.5, handlelength=1.4)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.72, 0.0), ncol=2, fontsize=6.8, handlelength=1.0)
    fig.savefig(FIGS / "s2_dose_crossover.pdf"); plt.close(fig)
    return R


def extras(split):
    """Exploratory numbers cited in the text (all labelled exploratory in the paper)."""
    from analysis.common import tidy, paired_auc
    T = pd.read_csv(ROOT / f"results/study1_{split}_tidy.csv")
    A = T[T.judge == "A"]
    tab = cell_table(A, "p_final")
    # conviction (verdict-game outcome, judge A labels) in percentage points
    tc = cell_table(A, "convict")
    def pp(x, name):
        r = summarize(x * 100); mac(name, ci(r)); mac(name + "P", pfmt(r["p_t"])); mac(name + "Est", f1(r["est"]))
    pp(cm(tc, objective="advocate", pros_sel="lookahead", dtype="guilty") - cm(tc, objective="advocate", pros_sel="random", dtype="guilty"), "ConvAdvLookG")
    pp(cm(tc, objective="advocate", pros_sel="lookahead", dtype="innocent") - cm(tc, objective="advocate", pros_sel="random", dtype="innocent"), "ConvAdvLookI")
    pp(cm(tc, dtype="guilty") - cm(tc, dtype="innocent"), "ConvType")
    pp(cm(tc, def_sel="lookahead") - cm(tc, def_sel="random"), "ConvDef")
    tt = cell_table(A, "convict_thr")
    pp(cm(tt, objective="advocate", pros_sel="lookahead", dtype="innocent") - cm(tt, objective="advocate", pros_sel="random", dtype="innocent"), "ConvThrAdvLookI")
    mac("ConvRateG", f"{100*A[A.dtype=='guilty'].convict.mean():.1f}"); mac("ConvRateI", f"{100*A[A.dtype=='innocent'].convict.mean():.1f}")
    mac("ConvRateAll", f"{100*A.convict.mean():.1f}")
    mac("PAUCOverall", f"{100*summarize(paired_auc(tab))['est']:.0f}")
    # drift
    for j in ("A", "E"):
        tj = T[T.judge == j]
        for ty in ("guilty", "innocent"):
            m = tj[tj.dtype == ty]
            mac(f"Drift{j}{ty[0].upper()}0", f"{m.ps0.mean():.0f}"); mac(f"Drift{j}{ty[0].upper()}4", f"{m.ps4.mean():.1f}")
    # judge discrimination for E (largest)
    tE = cell_table(T[T.judge == "E"], "p_final")
    r = summarize(cm(tE, dtype="guilty") - cm(tE, dtype="innocent")); mac("TypeEffectE", ci(r))
    # Guilt-leaning judge instruction, judge A
    gl = tidy(split, variant="guilt_leaning"); gl = gl[gl.judge == "A"]
    tg = cell_table(gl, "p_final")
    mac("GLMean", f"{gl.p_final.mean():.1f}"); mac("GLConv", f"{100*gl.convict.mean():.0f}"); mac("NeutralConv", f"{100*A.convict.mean():.0f}")
    mac("GLConvI", f"{100*gl[gl.dtype=='innocent'].convict.mean():.0f}"); mac("GLConvG", f"{100*gl[gl.dtype=='guilty'].convict.mean():.0f}")
    mac("GLHOne", ci(summarize(cm(tg, objective="advocate", pros_sel="lookahead") - cm(tg, objective="advocate", pros_sel="random"))))
    mac("GLType", ci(summarize(cm(tg, dtype="guilty") - cm(tg, dtype="innocent"))))
    mac("GLPAUC", f"{summarize(paired_auc(tg))['est']:.2f}")
    # test-retest
    rt = ROOT / f"results/judge_retest_{split}.json"
    if rt.exists():
        rr = json.loads(rt.read_text())
        for j, v in rr.items():
            mac(f"Retest{j}R", f"{v['pearson_r']:.2f}"); mac(f"Retest{j}MAD", f"{v['mean_abs_diff']:.1f}")
            mac(f"Retest{j}Verdict", f"{100*v['verdict_agreement']:.0f}")
    # content annotations
    ap = ROOT / f"data/raw/study1_{split}_annotations.jsonl"
    if ap.exists():
        An = pd.DataFrame([json.loads(l) for l in ap.open()])
        games = {g["game_id"]: g for g in (json.loads(l) for l in (ROOT / f"data/raw/study1_{split}.jsonl").open())}
        for k in ("fabricated", "discloses_w", "leading"):
            An[k] = An.ann1.apply(lambda a: a[k] if a else np.nan)
        for f_ in ("objective", "pros_sel", "case_id"):
            An[f_] = An.game_id.map(lambda g: games[g][f_])
        sub = An[An.ann2.notna()]
        for k, nm in (("fabricated", "Fab"), ("discloses_w", "Disc"), ("leading", "Lead")):
            a = sub[k].values.astype(float); b = sub.ann2.apply(lambda x: x[k]).values.astype(float)
            po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
            mac(f"Kappa{nm}", f"{(po - pe) / (1 - pe):.2f}"); mac(f"Agree{nm}", f"{100*po:.0f}")
            mac(f"Rate{nm}", f"{100*An[k].mean():.0f}")
        r1 = An[An["round"] == 1]
        tabd = r1.pivot_table(index="case_id", columns=["objective", "pros_sel"], values="discloses_w", aggfunc="mean")
        for o in ("advocate", "minister"):
            mac(f"DiscR1Rand{o[:3].title()}", f"{100*tabd[(o, 'random')].mean():.0f}")
            for s_ in ("introspect", "lookahead"):
                r = summarize((tabd[(o, s_)] - tabd[(o, "random")]) * 100)
                mac(f"DiscR1{o[:3].title()}{s_[:4].title()}", ci(r, 0)); mac(f"DiscR1{o[:3].title()}{s_[:4].title()}P", pfmt(r["p_t"]))
        tabf = An.pivot_table(index="case_id", columns=["objective", "pros_sel"], values="fabricated", aggfunc="mean")
        r = summarize((tabf[("advocate", "lookahead")] - tabf[("advocate", "random")]) * 100); mac("FabAdvLook", ci(r)); mac("FabAdvLookP", pfmt(r["p_t"]))
        byr = An.groupby("round").fabricated.mean() * 100
        mac("FabRoundOne", f"{byr.loc[1]:.0f}"); mac("FabRoundFour", f"{byr.loc[4]:.0f}")
        fo = An.groupby("objective").fabricated.mean() * 100
        mac("FabAdv", f"{fo['advocate']:.0f}"); mac("FabMin", f"{fo['minister']:.0f}")
    # CRN realised share and provider shares
    games = [json.loads(l) for l in (ROOT / f"data/raw/study1_{split}.jsonl").open()]
    import collections
    grp = collections.defaultdict(set)
    for g in games:
        grp[(g["case_id"], g["objective"])].add(tuple(g["rounds"][0]["pros_cands"]))
    mac("CRNShare", f"{100*np.mean([len(v) == 1 for v in grp.values()]):.0f}")
    led = [json.loads(l) for l in (ROOT / "data/cache/ledger.jsonl").open()]
    q = [x for x in led if x["model"].startswith("qwen") and x["tag"] == f"study1_{split}"]
    pc = collections.Counter(x["provider"] for x in q)
    mac("ProviderText", "In Study 1, the player model was served by " + ", ".join(
        f"{k} ({100*v/len(q):.0f}\\%)" for k, v in pc.most_common()) + " (all fp8). Launch order was randomised, but differing call counts and durations mean that "
        "provider fallback need not be balanced across conditions.")


def extras2(split):
    """Exploratory Study-2 quantities: split-sample selection gain at t=1, immediate-outcome
    planner validity, exclusions."""
    path = ROOT / f"data/raw/study2_{split}.jsonl"
    if not path.exists():
        return
    H = [json.loads(l) for l in path.open()]
    J = pd.DataFrame([json.loads(l) for l in (ROOT / f"data/raw/study2_{split}_judgments.jsonl").open()])
    J = J.drop_duplicates(["cid", "judge", "stage"], keep="last")
    mac("StwoNfail", str(48 * 3 - len(H)))
    rows, val = [], []
    for h in H:
        jh = J[J.hid == h["hid"]]
        uA_min = np.array([sc["pG"] - sc["pI"] for sc in h["scoreA"]])
        uA_adv = np.array([0.5 * (sc["pG"] + sc["pI"]) for sc in h["scoreA"]])
        uE_adv = np.array([0.5 * (sc["pG"] + sc["pI"]) for sc in h["scoreB"]])
        for e in ("A", "E"):
            imm = jh[(jh.judge == e) & (jh.stage == h["t"])]
            if imm.empty:
                continue
            for ty in h["types"]:
                v = imm[imm.type == ty].groupby("j").ps.mean().reindex(range(8)).values
                if np.isnan(v).any() or v.std() == 0:
                    continue
                for pl, u in (("A", uA_adv), ("E", uE_adv)):
                    if u.std() > 0:
                        val.append({"case_id": h["case_id"], "pl": pl, "e": e,
                                    "rho": stats.spearmanr(u, v).correlation})
            if h["t"] == 1:
                piv = imm.pivot_table(index="j", columns=["type", "m"], values="ps")
                if piv.shape != (8, 4) or piv.isna().any().any():
                    continue
                D0 = (piv[("guilty", 0)] - piv[("innocent", 0)]).values
                D1 = (piv[("guilty", 1)] - piv[("innocent", 1)]).values
                Dm = 0.5 * (D0 + D1)
                split = 0.5 * ((D1[np.argmax(D0)] - D1.mean()) + (D0[np.argmax(D1)] - D0.mean()))
                best = uA_min == uA_min.max()
                rows.append({"case_id": h["case_id"], "e": e, "split_oracle": split,
                             "minister_pick": Dm[best].mean() - Dm.mean(), "base": Dm.mean()})
    V = pd.DataFrame(val)
    for (pl, e), g in V.groupby(["pl", "e"]):
        r = summarize(g.groupby("case_id").rho.mean())
        mac(f"ValImm{pl}to{e}", f"{r['est']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
    D = pd.DataFrame(rows)
    for e, g in D.groupby("e"):
        gc = g.groupby("case_id").mean(numeric_only=True)
        mac(f"DiscBaseImm{e}", ci(summarize(gc.base)))
        mac(f"DiscSplitOracle{e}", ci(summarize(gc.split_oracle)))
        mac(f"DiscMinPick{e}", ci(summarize(gc.minister_pick)))


def confirmatory_table(R1, R2):
    rows = []
    lab = {"H1_adv_lookahead_vs_random": ("H1", "Strategic advantage: advocate lookahead $-$ random"),
           "H2_adv_lookahead_vs_introspect": ("H2", "Simulation vs introspection (advocate)"),
           "H3_adv_lookahead_vs_random_innocent": ("H3", "Advocate lookahead $-$ random, innocent defendants"),
           "H4_disc_minister_vs_advocate_lookahead": ("H4", "Discrimination: minister $-$ advocate (lookahead)"),
           "H5_def_lookahead_vs_random": ("H5", "Defence lookahead $-$ random"),
           "H6_transfer_gap_A_minus_heldout": ("H6", "Transfer gap: planner's judge $-$ held-out judges")}
    pv = {}
    for k, (h, d) in lab.items():
        r = R1["confirmatory"][k]; rows.append((h, d, r, "S1")); pv[h] = r["p_t"]
    if R2:
        for h, key, d in (("H7", "H7_lookA_vs_random_evalA_immediate", "Move value: planner pick $-$ random (immediate)"),
                          ("H8", "H8_crossover_interaction_std_immediate", "Crossover: matched $-$ mismatched (SD units)"),
                          ("H9", "H9_dose_slope_log2k_A_evalA_immediate", "Dose-response slope per doubling of $k$")):
            if key in R2:
                r = R2[key]; rows.append((h, d, r, "S2")); pv[h] = r["p_t"]
    adj = holm(pv)
    for h, p in adj.items():
        mac(f"Holm{h}", pfmt(p))
    with open(TABS / "robust.tex", "w") as f:
        f.write("\\begin{tabular}{@{}lrllr@{}}\\toprule\n & Estimate & $t$ 95\\% CI & Bootstrap 95\\% CI & Sign-flip $p$\\\\\\midrule\n")
        for h, d, r, s in rows:
            f.write(f"{h} & {f1(r['est'], 2)} & [{f1(r['ci_lo'], 2)}, {f1(r['ci_hi'], 2)}] & "
                    f"[{f1(r['boot_lo'], 2)}, {f1(r['boot_hi'], 2)}] & {pfmt(r['p_perm'])}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
    # exploratory equivalence (TOST) for non-significant confirmatory contrasts: 90% CI inside +/- SESOI?
    for h, d, r, s in rows:
        se = r["se"]; n = r["n_cases"]
        tq = stats.t.ppf(0.95, n - 1)
        lo90, hi90 = r["est"] - tq * se, r["est"] + tq * se
        mac(f"Tost{h}", f"[{f1(lo90, 2)}, {f1(hi90, 2)}]")
    with open(TABS / "confirmatory.tex", "w") as f:
        f.write("\\begin{tabular}{@{}llrlrrr@{}}\\toprule\n & Contrast & Estimate & 95\\% CI & $d_z$ & $p$ & $p_{\\text{Holm}}$\\\\\\midrule\n")
        for h, d, r, s in rows:
            f.write(f"{h} & {d} & {f1(r['est'], 2)} & [{f1(r['ci_lo'], 2)}, {f1(r['ci_hi'], 2)}] & {f1(r['dz'], 2)} & "
                    f"{pfmt(r['p_t'])} & {pfmt(adj[h])}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="main"); ap.add_argument("--s2", default="main")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True); TABS.mkdir(parents=True, exist_ok=True)
    R1 = study1(args.s1)
    R2 = study2(args.s2)
    extras(args.s1)
    extras2(args.s2)
    confirmatory_table(R1, R2)
    # Stimulus-validation estimates (historical data filenames retained)
    p1 = json.loads((ROOT / "data/pilot1_archive/study1_pilot1_results.json").read_text())
    r = p1["exploratory_p"]["type:guilty-innocent"]; mac("PoneTypeEst", ci(r))
    t1 = pd.read_csv(ROOT / "data/pilot1_archive/study1_pilot_tidy.csv"); t1 = t1[t1.judge == "A"]
    mac("PoneDriftFrom", f"{t1.ps0.mean():.0f}"); mac("PoneDriftTo", f"{t1.ps4.mean():.0f}")
    pp = ROOT / "results/study1_pilot.json"
    if pp.exists():
        p2 = json.loads(pp.read_text()); mac("PtwoTypeEst", ci(p2["exploratory_p"]["type:guilty-innocent"]))
    cases = json.loads((ROOT / "data/cases/cases.json").read_text())
    K = ["ambiguity", "w_true_both", "cover_contradicted", "innocent_consistent", "statement_neutral"]
    mac("CaseAuditPass", str(sum(1 for c in cases if c.get("audit") and min(c["audit"][k] for k in K[1:4]) >= 4)))
    mac("CaseRegens", str(sum(c.get("audit_regenerations", 0) for c in cases)))
    for k in K:
        mac("Audit" + "".join(w.title() for w in k.split("_")), f"{np.mean([c['audit'][k] for c in cases]):.2f}")
    led = [json.loads(l) for l in (ROOT / "data/cache/ledger.jsonl").open()]
    mac("TotalCost", f"{sum(x['cost'] for x in led):.2f}")
    mac("TotalCalls", f"{len(led):,}")
    if R2 and R1:
        mac("MinisterSignalText", "(in Study 2, the minister's simulated scores yielded "
            + MACROS["DiscMinPickA"] + " points of discrimination gain, compared with " + MACROS["DiscSplitOracleA"]
            + " for the split-sample selection benchmark)")
        mac("StudyTwoTransferText", "(H6). Study 2 also found an interaction between planner and evaluator "
            "on the standardised selection-gain scale (H8).")
        h1 = R1["confirmatory"]["H1_adv_lookahead_vs_random"]
        mac("AbstractFindings",
            "Under the conviction-seeking objective, one-step lookahead increased the primary judge's guilt score "
            "after four rounds of forced continuation by "
            + f"{h1['est']:.1f} points on a 0--100 scale (case-level 95\\% CI {h1['ci_lo']:.1f}--{h1['ci_hi']:.1f}) "
            + "relative to baseline selection and exceeded introspective choice. It also raised scores for innocent "
            "defendants. Gains were smaller under held-out judges, and selection gains depended on the "
            "planner--evaluator pairing. Individual question-selection effects diminished under baseline "
            "continuation. A lookahead defence reduced guilt scores, but its selector queried an internal judge "
            "with access to withheld evidence. Under lookahead, a truth-seeking objective showed no clear "
            "discrimination improvement over the conviction-seeking objective. These findings concern score-guided "
            "policies within the specified simulator; their implications for human judgement and interactive AI "
            "detection remain to be tested.")
        mac("DeviationsText",
            "(1) Study 1 was restarted with more concurrent games, reusing cached responses where available. "
            "(2) " + MACROS["StwoNfail"] + " of 144 Study-2 decision points failed after retries "
            "(malformed candidate output) and were excluded under the stated failure rule. "
            "(3) Concurrent identical requests were not deduplicated. Identical first-round candidate lists "
            "were shared across all relevant cells in " + MACROS["CRNShare"]
            + "\\% of case--objective groups; the two Study-2 planners also used separately realised "
            "hypothetical replies. These departures change the coupling between runs; variance reduction "
            "is not guaranteed. (4) The broad plan included exploratory utterance features, planner validity "
            "and split-sample selection. The implemented question coding, judge test--retest, TOST intervals "
            "and detailed Study-2 discrimination and validity analyses were added after the recorded freeze "
            "and are treated as exploratory.")
    for k, default in (("AbstractFindings", "[findings]"), ("StudyTwoTransferText", "."), ("MinisterSignalText", ""),
                       ("DeviationsText", "[deviations]")):
        if k not in MACROS:
            mac(k, default)
    with open(PAPER / "macros.tex", "w") as f:
        f.write("% generated by analysis/make_report.py -- do not edit\n")
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(MACROS)} macros")


if __name__ == "__main__":
    main()
