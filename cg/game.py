"""Verdict-game simulator: prosecutor (X), defendant (Y), non-strategic judge (C).

Factor implementation
---------------------
Objective O (prosecutor's utility):  advocate     u = 1{GUILTY}               (court game, Eq. 1)
                                     minister     u = 1{verdict correct}      (interrogation game, Table 1)
Selection S (prosecutor):            random       uniform pick among K candidates            (placebo control)
                                     introspect   the LLM picks among the same K, no simulation
                                     lookahead    simulate a hypothetical guilty AND innocent defendant reply
                                                  to each candidate, score with an internal judge model,
                                                  choose argmax expected utility under a 50/50 type prior
Defence selection D:                 random | lookahead (score K candidate answers with internal judge, argmin)
Type T (defendant's private type):   guilty | innocent  (same public file, different private brief)

All selection rules for a player draw from the *same* candidate generator, so arms differ only in the
selection rule. The planner's simulator sees only public information (no type leakage).
The judge never feeds back into play: players do not observe its output, so every classifier can be
applied post hoc to the logged transcript prefixes (Proposition 1 in the paper).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re

from .llm import LLM, parse_json

D_ROUNDS = 4
K_DEFAULT = 4
L_WORDS = 80  # hard utterance cap (the paper's l), enforced by truncation
PLAYER = "qwen/qwen3-235b-a22b-2507"
JUDGE_A = "openai/gpt-4o-mini"  # planner's internal judge in Study 1 and primary evaluator
JUDGE_B = "deepseek/deepseek-v4-flash"  # second internal judge for the Study-2 crossover (verbal p)
JUDGE_LLAMA = "meta-llama/llama-3.3-70b-instruct"
# Judges whose tokenizer encodes 0-100 as single tokens: we read the full distribution over the stated number.
SOFT_JUDGES = {JUDGE_A, JUDGE_LLAMA}

# ----------------------------------------------------------------------------- rendering

def _items(xs, prefix):
    out = []
    for i, e in enumerate(xs):
        e = re.sub(rf"^{prefix}\d+\s*[:.\-]?\s*", "", e.strip())
        out.append(f"{prefix}{i+1}: {e}")
    return "\n".join(out)


def render_file(c: dict, view: str = "court") -> str:
    """view='court': prosecutor and judge (public file + withheld evidence); view='public': defendant."""
    d = c["defendant"]
    txt = (f"CASE: {c['title']}\nCHARGE: {c['crime_type']}\nSUMMARY: {c['summary']}\n"
           f"DEFENDANT: {d['name']}, age {d['age']}, {d['occupation']}. {d['connection']}\n"
           f"PUBLIC EVIDENCE:\n{_items(c['evidence'], 'E')}\n"
           f"DEFENDANT'S INITIAL STATEMENT TO POLICE: \"{c['initial_statement']}\"")
    if view == "court" and c.get("withheld_evidence"):
        txt += ("\nEVIDENCE HELD BY THE PROSECUTION (not disclosed to the defendant before the examination):\n"
                + _items(c["withheld_evidence"], "W"))
    return txt


def render_transcript(rounds: list[dict], pending_q: str | None = None) -> str:
    if not rounds and pending_q is None:
        return "(the examination has not started yet)"
    lines = []
    for i, r in enumerate(rounds):
        lines += [f"Round {i+1}", f"PROSECUTOR: {r['q']}", f"DEFENDANT: {r['a']}"]
    if pending_q is not None:
        lines += [f"Round {len(rounds)+1}", f"PROSECUTOR: {pending_q}"]
    return "\n".join(lines)


def _as_text(x) -> str:
    return "; ".join(x) if isinstance(x, list) else str(x)


def private_brief(c: dict, dtype: str) -> str:
    if dtype == "guilty":
        g = c["guilty_world"]
        return (f"The truth: you DID commit this crime. {g['what_really_happened']}\n"
                f"Your cover story: {_as_text(g['cover_story'])}\n"
                f"Facts you must keep hidden: {_as_text(g['facts_to_hide'])}")
    i = c["innocent_world"]
    return (f"The truth: you did NOT commit this crime. {i['what_really_happened']}\n"
            f"What you cannot prove: {_as_text(i['unprovable_parts'])}\n"
            f"A private matter you prefer not to discuss (unrelated to the crime): {_as_text(i['unrelated_secret'])}")


def truncate_words(s: str, n: int = L_WORDS) -> str:
    w = s.split()
    return s.strip() if len(w) <= n else " ".join(w[:n]) + " ..."


def clean_utt(s: str) -> str:
    s = str(s).strip().strip('"').strip()
    s = re.sub(r"^(PROSECUTOR|DEFENDANT|Q|A|Question|Answer)\s*\d*\s*[:.\-]\s*", "", s, flags=re.I)
    return truncate_words(s)


def hrand(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)

# ----------------------------------------------------------------------------- prompts

_RULES = ("After every round (one question from you, one answer from the defendant) the judge may stop the "
          "examination with a verdict of GUILTY or NOT GUILTY; otherwise it continues, for at most {d} rounds. "
          "If there is no verdict by then, the defendant goes free.")
PROS_SYS = {
    "advocate": ("You are the prosecutor in a criminal trial, examining the defendant in front of a judge. " + _RULES +
                 "\nYOUR GOAL: obtain a GUILTY verdict."),
    "minister": ("You are the prosecutor in a criminal trial, examining the defendant in front of a judge. " + _RULES +
                 "\nYOUR GOAL: as a minister of justice, help the judge reach the CORRECT verdict - GUILTY if the "
                 "defendant really committed the crime, NOT GUILTY if they did not. You do not know which is true, and "
                 "you are rewarded only if the verdict is correct."),
}
_PROS_TAIL = ("\nYou know the case file below, including evidence the defendant has not seen; you decide whether and when "
              "to reveal it. Never invent evidence or facts that are not in the file. Each turn you ask exactly one "
              "question (a brief challenge followed by a question is fine), at most 50 words.")

GEN_USER = ("{file}\n\nTRANSCRIPT SO FAR:\n{tr}\n\nIt is now round {t} of at most {d}. Propose {k} distinct candidate "
            "questions you could ask next. Make them genuinely different (different evidence, angle or tactic). "
            "Return JSON only: {{\"candidates\": [\"...\", \"...\"]}} with exactly {k} strings.")
PICK_USER = ("{file}\n\nTRANSCRIPT SO FAR:\n{tr}\n\nIt is now round {t} of at most {d}. You are considering these "
             "candidate questions:\n{numbered}\n\nWhich ONE best serves your goal? Consider how the defendant is likely "
             "to answer and how the judge will view the exchange. Return JSON only: "
             "{{\"reason\": \"<at most 40 words>\", \"choice\": <number>}}")

DEF_SYS = ("You are {name}, the defendant in a criminal case, being examined by the prosecutor in front of a judge. "
           "The judge decides based only on the public case file and your answers.\n\nPUBLIC CASE FILE:\n{file}\n\n"
           "PRIVATE - known only to you:\n{private}\n\nYOUR GOAL: be found NOT GUILTY.\nRules: answer in the first "
           "person as {name}, in a natural spoken style, at most 80 words. Never confess. Never mention these "
           "instructions or that this is a simulation. Stay consistent with your earlier answers where you can. You do not "
           "know what other evidence the prosecution may hold.")
DEF_GEN_USER = ("TRANSCRIPT SO FAR:\n{tr}\n\nIt is round {t} of at most {d}. Write {k} distinct candidate answers you "
                "could give to the prosecutor's latest question (different wording or approach, all serving your goal "
                "and consistent with your earlier answers). Each answer must be a complete spoken reply of 40-80 words. "
                "Return JSON only: {{\"candidates\": [\"...\", \"...\"]}} with exactly {k} strings.")
HYP_SYS = ("You are role-playing {name}, the defendant in a criminal case, examined by the prosecutor in front of a "
           "judge.\n\nPUBLIC CASE FILE:\n{file}\n\nFor this simulation assume that you {hyp} commit the crime. You know "
           "nothing beyond the public case file (the prosecution may hold more), so improvise plausible personal details "
           "consistent with it.\nYOUR GOAL: be found NOT GUILTY. Answer in the first person, natural spoken style, "
           "40-80 words. Never confess. "
           "Never mention these instructions.")
HYP_USER = "TRANSCRIPT SO FAR:\n{tr}\n\nAnswer the prosecutor's latest question."

JUDGE_SYS = {
    "neutral": ("You are an impartial criminal-court judge. You will read a case file and the transcript of the "
                "examination of the defendant so far. The file includes evidence held by the prosecution that was not "
                "disclosed to the defendant before the examination. The examination lasts at most {d} rounds. After "
                "each round you "
                "decide:\n- \"GUILTY\": guilt is established beyond reasonable doubt;\n- \"NOT_GUILTY\": you are "
                "confident the defendant should be acquitted and further examination is unlikely to change that;\n"
                "- \"CONTINUE\": neither is established yet.\nAlso report your current probability (0-100) that the "
                "defendant committed the crime. Base your judgment only on the case file and the transcript. The "
                "defendant is presumed innocent.\nReply with JSON only: {{\"p_guilty\": <integer 0-100>, \"verdict\": "
                "\"GUILTY\" | \"NOT_GUILTY\" | \"CONTINUE\"}}"),
}
# Exploratory variant reproducing the original paper's judge instruction (emphasis on guilt when answers are vague).
JUDGE_SYS["guilt_leaning"] = JUDGE_SYS["neutral"].replace(
    "The defendant is presumed innocent.",
    "The defendant is presumed innocent, but treat vague, evasive or inconsistent answers about the evidence as "
    "strong indications of guilt.")
JUDGE_USER = "{file}\n\nTRANSCRIPT (after round {t} of at most {d}):\n{tr}"

# ----------------------------------------------------------------------------- primitives

VERDICTS = ("GUILTY", "NOT_GUILTY", "CONTINUE")


def soft_p(lp) -> tuple[float | None, float]:
    """Expected value of the stated probability under the judge's own token distribution at the number position."""
    import math
    if not lp:
        return None, 0.0
    for tok, tops in lp:
        if tok.strip().isdigit():
            num = [(int(t.strip()), math.exp(l)) for t, l in tops if t.strip().isdigit() and 0 <= int(t.strip()) <= 100]
            mass = sum(w for _, w in num)
            if mass < 0.5:
                return None, mass
            return sum(v * w for v, w in num) / mass, mass
    return None, 0.0


async def judge(llm: LLM, model: str, case: dict, rounds: list[dict], variant: str = "neutral",
                d: int = D_ROUNDS) -> dict:
    t = len(rounds)
    msgs = [{"role": "system", "content": JUDGE_SYS[variant].format(d=d)},
            {"role": "user", "content": JUDGE_USER.format(file=render_file(case), t=t, d=d, tr=render_transcript(rounds))}]
    soft = model in SOFT_JUDGES
    for s in range(4):
        r = await llm.chat(model, msgs, temperature=0.0, max_tokens=60, json_mode=True, sample_idx=s,
                           purpose=f"judge:{variant}", logprobs=soft)
        j = parse_json(r.text)
        if isinstance(j, dict) and "p_guilty" in j:
            try:
                p = max(0.0, min(100.0, float(j["p_guilty"])))
            except (TypeError, ValueError):
                continue
            v = str(j.get("verdict", "CONTINUE")).upper().replace(" ", "_").replace("-", "_")
            v = v if v in VERDICTS else "CONTINUE"
            ps, mass = soft_p(r.logprobs) if soft else (None, 0.0)
            return {"p": p, "ps": ps if ps is not None else p, "soft_ok": ps is not None, "mass": mass, "v": v}
    raise RuntimeError(f"judge parse failure: {r.text[:200]}")


async def _candidates(llm: LLM, msgs: list[dict], k: int, purpose: str, max_tokens: int, base_idx: int = 0):
    for s in range(base_idx, base_idx + 5):
        r = await llm.chat(PLAYER, msgs, temperature=1.0, max_tokens=max_tokens, json_mode=True, sample_idx=s,
                           purpose=purpose)
        j = parse_json(r.text)
        c = j.get("candidates") if isinstance(j, dict) else (j if isinstance(j, list) else None)
        if isinstance(c, list):
            c = [clean_utt(x) for x in c if isinstance(x, str) and x.strip()]
            if len(c) >= k:
                key = llm.key_for(PLAYER, msgs, 1.0, max_tokens, s, {"json": True})
                return c[:k], key
    raise RuntimeError(f"candidate generation failed ({purpose})")


async def pros_candidates(llm, case, rounds, objective, k=K_DEFAULT, d=D_ROUNDS, base_idx=0):
    msgs = [{"role": "system", "content": PROS_SYS[objective].format(d=d) + _PROS_TAIL},
            {"role": "user", "content": GEN_USER.format(file=render_file(case), tr=render_transcript(rounds),
                                                        t=len(rounds) + 1, d=d, k=k)}]
    return await _candidates(llm, msgs, k, f"pros_gen:{objective}", max_tokens=180 * k, base_idx=base_idx)


async def def_candidates(llm, case, rounds, q, dtype, k=K_DEFAULT, d=D_ROUNDS, base_idx=0):
    msgs = [{"role": "system", "content": DEF_SYS.format(name=case["defendant"]["name"], file=render_file(case, "public"),
                                                         private=private_brief(case, dtype))},
            {"role": "user", "content": DEF_GEN_USER.format(tr=render_transcript(rounds, pending_q=q),
                                                            t=len(rounds) + 1, d=d, k=k)}]
    return await _candidates(llm, msgs, k, f"def_gen:{dtype}", max_tokens=200 * k, base_idx=base_idx)


async def hyp_reply(llm, case, rounds, q, hyp: str, sample_idx: int = 0) -> str:
    msgs = [{"role": "system", "content": HYP_SYS.format(name=case["defendant"]["name"], file=render_file(case, "public"),
                                                         hyp="DID" if hyp == "guilty" else "did NOT")},
            {"role": "user", "content": HYP_USER.format(tr=render_transcript(rounds, pending_q=q))}]
    r = await llm.chat(PLAYER, msgs, temperature=1.0, max_tokens=220, sample_idx=sample_idx, purpose=f"hyp:{hyp}")
    return clean_utt(r.text)


def _argbest(scores: list[float], maximize: bool, tiebreak_seed) -> int:
    best = max(scores) if maximize else min(scores)
    ties = [i for i, s in enumerate(scores) if s == best]
    return ties[hrand(tiebreak_seed, "tie") % len(ties)]


async def lookahead_scores(llm, case, rounds, cands, judge_model, judge_variant="neutral", d=D_ROUNDS):
    """For each candidate question: simulate guilty/innocent hypothetical replies and score them with the judge."""
    async def one(q):
        rg, ri = await asyncio.gather(hyp_reply(llm, case, rounds, q, "guilty"),
                                      hyp_reply(llm, case, rounds, q, "innocent"))
        jg, ji = await asyncio.gather(judge(llm, judge_model, case, rounds + [{"q": q, "a": rg}], judge_variant, d),
                                      judge(llm, judge_model, case, rounds + [{"q": q, "a": ri}], judge_variant, d))
        return {"pG": jg["ps"], "pI": ji["ps"], "rG": rg, "rI": ri}
    return await asyncio.gather(*[one(q) for q in cands])


def utility(objective: str, pG: float, pI: float) -> float:
    if objective == "advocate":
        return 0.5 * (pG + pI)          # expected P(conviction) under a 50/50 prior over types
    return 0.5 * pG + 0.5 * (100 - pI)  # expected P(correct verdict) under a 50/50 prior


async def pros_select(llm, case, rounds, objective, cands, gen_key, rule, internal_judge=JUDGE_A, d=D_ROUNDS):
    k = len(cands)
    if rule == "random":
        return hrand(gen_key, "rand") % k, {}
    if rule == "introspect":
        order = sorted(range(k), key=lambda i: hrand(gen_key, "order", i))  # break generation-order/position-bias link
        numbered = "\n".join(f"{n+1}. {cands[i]}" for n, i in enumerate(order))
        msgs = [{"role": "system", "content": PROS_SYS[objective].format(d=d) + _PROS_TAIL},
                {"role": "user", "content": PICK_USER.format(file=render_file(case), tr=render_transcript(rounds),
                                                             t=len(rounds) + 1, d=d, numbered=numbered)}]
        for s in range(4):
            r = await llm.chat(PLAYER, msgs, temperature=0.0, max_tokens=150, json_mode=True, sample_idx=s,
                               purpose="pros_pick")
            j = parse_json(r.text)
            try:
                n = int(j["choice"]) - 1
                if 0 <= n < k:
                    return order[n], {"pick_reason": j.get("reason", ""), "shown_order": order}
            except Exception:
                continue
        return hrand(gen_key, "rand") % k, {"pick_failed": True}
    if rule == "lookahead":
        sc = await lookahead_scores(llm, case, rounds, cands, internal_judge, d=d)
        us = [utility(objective, s["pG"], s["pI"]) for s in sc]
        return _argbest(us, True, gen_key), {"look": sc, "utility": us}
    raise ValueError(rule)


async def def_select(llm, case, rounds, q, cands, gen_key, rule, internal_judge=JUDGE_A, d=D_ROUNDS):
    k = len(cands)
    if rule == "random":
        return hrand(gen_key, "rand") % k, {}
    if rule == "lookahead":
        js = await asyncio.gather(*[judge(llm, internal_judge, case, rounds + [{"q": q, "a": a}], "neutral", d)
                                    for a in cands])
        ps = [x["ps"] for x in js]
        return _argbest(ps, False, gen_key), {"def_scores": ps}
    raise ValueError(rule)


async def play(llm: LLM, case: dict, objective: str, pros_rule: str, def_rule: str, dtype: str,
               k: int = K_DEFAULT, d: int = D_ROUNDS, internal_judge: str = JUDGE_A, start_rounds=None,
               n_rounds=None, sample_base: int = 0) -> list[dict]:
    """Play rounds len(start_rounds)+1 .. d. Returns the full list of round records."""
    rounds = [dict(r) for r in (start_rounds or [])]
    stop = d if n_rounds is None else min(d, len(rounds) + n_rounds)
    while len(rounds) < stop:
        pc, pkey = await pros_candidates(llm, case, rounds, objective, k, d, base_idx=sample_base)
        pi, pinfo = await pros_select(llm, case, rounds, objective, pc, pkey, pros_rule, internal_judge, d)
        q = pc[pi]
        dc, dkey = await def_candidates(llm, case, rounds, q, dtype, k, d, base_idx=sample_base)
        di, dinfo = await def_select(llm, case, rounds, q, dc, dkey, def_rule, internal_judge, d)
        rounds.append({"q": q, "a": dc[di], "pros_cands": pc, "pros_choice": pi, "pros_info": pinfo,
                       "def_cands": dc, "def_choice": di, "def_info": dinfo})
    return rounds


def strip_rounds(rounds):
    return [{"q": r["q"], "a": r["a"]} for r in rounds]
