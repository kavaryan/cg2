"""Study 2: move-level potential outcomes.

At a decision point h (case, round t, transcript prefix) the prosecutor's generator proposes K=8 candidate
questions. For EVERY candidate we (i) compute the planner's lookahead scores under two internal judge models
(A, B) from simulated hypothetical replies, and (ii) realise M independent continuations of the real game
(real defendant of the given type, then baseline random/random play to round d). Because the realised outcomes
use fresh draws independent of the planner's simulations, the value of any selection rule (random, best-of-k,
judge-A planner, judge-B planner, oracle) is estimated without winner's-curse bias, all on the same candidates.

t = 1: empty prefix; one candidate set per case, realised for BOTH types (the prosecutor cannot see the type).
t = 3: prefix = first two rounds of the Study-1 (advocate, random, random, type) game; realised for that type.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cg.game import (PLAYER, JUDGE_A, JUDGE_B, D_ROUNDS, def_candidates, hrand, lookahead_scores, play,  # noqa: E402
                     pros_candidates, pros_select, strip_rounds)
from cg.llm import LLM, BudgetExceeded  # noqa: E402

K8 = 8
M = 2


async def decision_point(llm, case, t, prefix, types, out, lock):
    hid = f"S2|{case['case_id']}|t{t}|" + ("both" if len(types) == 2 else types[0])
    cands, gkey = await pros_candidates(llm, case, prefix, "advocate", k=K8)
    scA, scB = await asyncio.gather(lookahead_scores(llm, case, prefix, cands, JUDGE_A),
                                    lookahead_scores(llm, case, prefix, cands, JUDGE_B))
    # Introspective pick among all 8 (advocate objective), for comparison with simulation-based selection.
    intro_idx, intro_info = await pros_select(llm, case, prefix, "advocate", cands, gkey, "introspect")

    async def realise(j, ty, m):
        q = cands[j]
        dc, dkey = await def_candidates(llm, case, prefix, q, ty, base_idx=10 * m)
        a = dc[hrand(dkey, "rand") % len(dc)]
        rounds = prefix + [{"q": q, "a": a}]
        rest = await play(llm, case, "advocate", "random", "random", ty, start_rounds=rounds, sample_base=10 * m + 1)
        return {"j": j, "type": ty, "m": m, "rounds": strip_rounds(rest)}

    real = await asyncio.gather(*[realise(j, ty, m) for j in range(K8) for ty in types for m in range(M)])
    rec = {"hid": hid, "case_id": case["case_id"], "t": t, "types": types, "prefix": strip_rounds(prefix),
           "cands": cands, "scoreA": scA, "scoreB": scB, "intro_idx": intro_idx,
           "intro_reason": intro_info.get("pick_reason", ""), "real": real, "player": PLAYER, "d": D_ROUNDS}
    async with lock:
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return hid


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main")
    ap.add_argument("--stages", default="1,3")
    ap.add_argument("--max-spend", type=float, default=8.0)
    ap.add_argument("--n-cases", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    cases = [c for c in json.loads((ROOT / "data/cases/cases.json").read_text()) if c["split"] == args.split]
    if args.n_cases:
        cases = cases[:args.n_cases]
    s1 = {}
    s1_path = ROOT / f"data/raw/study1_{args.split}.jsonl"
    if s1_path.exists():
        for l in s1_path.open():
            g = json.loads(l)
            if g["objective"] == "advocate" and g["pros_sel"] == "random" and g["def_sel"] == "random":
                s1[(g["case_id"], g["dtype"])] = g["rounds"]
    out = ROOT / f"data/raw/study2_{args.split}.jsonl"
    done = {json.loads(l)["hid"] for l in out.open()} if out.exists() else set()
    jobs = []
    for c in cases:
        for t in map(int, args.stages.split(",")):
            if t == 1:
                hid = f"S2|{c['case_id']}|t1|both"
                if hid not in done:
                    jobs.append((c, 1, [], ["guilty", "innocent"]))
            else:
                for ty in ("guilty", "innocent"):
                    hid = f"S2|{c['case_id']}|t{t}|{ty}"
                    if hid not in done and (c["case_id"], ty) in s1:
                        jobs.append((c, t, strip_rounds(s1[(c["case_id"], ty)][:t - 1]), [ty]))
    print(f"{len(jobs)} decision points to run")
    llm = LLM(budget_usd=min(30.0, LLM._ledger_total() + args.max_spend), concurrency=64, tag=f"study2_{args.split}")
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    n = 0

    async def run(job):
        nonlocal n
        async with sem:
            try:
                await decision_point(llm, *job, out, lock)
                n += 1
                print(f"{n}/{len(jobs)} done, spend {llm.spent_session:.3f} (total {llm.total_spent:.3f}), "
                      f"{time.time()-t0:.0f}s", flush=True)
            except BudgetExceeded:
                raise
            except Exception as e:
                print("FAIL", job[0]["case_id"], job[1], repr(e)[:200], flush=True)

    try:
        await asyncio.gather(*[run(j) for j in jobs])
    except BudgetExceeded as e:
        print("BUDGET STOP", e)
    print(f"finished; session spend {llm.spent_session:.4f}; total {llm.total_spent:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
