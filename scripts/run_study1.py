"""Study 1: 2 (objective) x 3 (prosecutor selection) x 2 (defence selection) x 2 (type) factorial,
complete randomized block design with case as block. One game per cell per case.

Usage: python -I scripts/run_study1.py --split pilot|main [--max-spend 10]
"""
import argparse
import asyncio
import itertools
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cg.game import play, PLAYER, JUDGE_A, D_ROUNDS, K_DEFAULT  # noqa: E402
from cg.llm import LLM, BudgetExceeded  # noqa: E402

OBJ = ["advocate", "minister"]
PSEL = ["random", "introspect", "lookahead"]
DSEL = ["random", "lookahead"]
TYPES = ["guilty", "innocent"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--max-spend", type=float, default=12.0, help="session spend cap (USD)")
    ap.add_argument("--games-concurrency", type=int, default=40)
    ap.add_argument("--only", default="", help="comma list of case ids")
    args = ap.parse_args()
    cases = [c for c in json.loads((ROOT / "data/cases/cases.json").read_text()) if c["split"] == args.split]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["case_id"] in keep]
    out = ROOT / f"data/raw/study1_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            done.add(json.loads(line)["game_id"])
    jobs = []
    for c, o, ps, ds, ty in itertools.product(cases, OBJ, PSEL, DSEL, TYPES):
        gid = f"S1|{c['case_id']}|{o}|{ps}|{ds}|{ty}"
        if gid not in done:
            jobs.append((gid, c, o, ps, ds, ty))
    random.Random(11).shuffle(jobs)  # randomized run order: guards against provider drift over time
    print(f"{len(jobs)} games to run ({len(done)} done)")
    llm = LLM(budget_usd=min(30.0, LLM._ledger_total() + args.max_spend), concurrency=96, tag=f"study1_{args.split}")
    gsem = asyncio.Semaphore(args.games_concurrency)
    lock = asyncio.Lock()
    n_ok = n_fail = 0
    t0 = time.time()

    async def run(job):
        nonlocal n_ok, n_fail
        gid, c, o, ps, ds, ty = job
        async with gsem:
            try:
                rounds = await play(llm, c, o, ps, ds, ty)
            except BudgetExceeded:
                raise
            except Exception as e:  # logged and re-run later; never silently dropped
                n_fail += 1
                print("FAIL", gid, repr(e)[:200], flush=True)
                return
        rec = {"game_id": gid, "case_id": c["case_id"], "objective": o, "pros_sel": ps, "def_sel": ds, "dtype": ty,
               "player": PLAYER, "internal_judge": JUDGE_A, "d": D_ROUNDS, "k": K_DEFAULT, "rounds": rounds}
        async with lock:
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"{n_ok} ok, {n_fail} fail, spend {llm.spent_session:.3f} (total {llm.total_spent:.3f}), "
                      f"{time.time()-t0:.0f}s, calls {llm.calls}, hits {llm.cache_hits}", flush=True)

    try:
        await asyncio.gather(*[run(j) for j in jobs])
    except BudgetExceeded as e:
        print("BUDGET STOP:", e)
    print(f"finished: {n_ok} ok, {n_fail} fail, session spend {llm.spent_session:.4f}, total {llm.total_spent:.4f}, "
          f"{time.time()-t0:.0f}s")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
