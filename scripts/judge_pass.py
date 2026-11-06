"""Post-hoc judging of logged transcripts (valid because players never observe the judge; Prop. 1).

Usage: python -I scripts/judge_pass.py --file data/raw/study1_pilot.jsonl --judges A,B --stages all --variant neutral
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cg.game import judge  # noqa: E402
from cg.llm import LLM, BudgetExceeded  # noqa: E402

PANEL = {
    "A": "openai/gpt-4o-mini",                # planner's internal judge (soft p via logprobs)
    "B": "meta-llama/llama-3.3-70b-instruct", # second soft judge
    "C": "google/gemini-2.5-flash-lite",
    "D": "anthropic/claude-haiku-5.5",
    "E": "deepseek/deepseek-v4-flash",
    "F": "mistralai/mistral-small-3.2-24b-instruct",
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--judges", default="A,B")
    ap.add_argument("--stages", default="all", help="all | final | comma list")
    ap.add_argument("--variant", default="neutral")
    ap.add_argument("--max-spend", type=float, default=4.0)
    args = ap.parse_args()
    src = ROOT / args.file
    games = [json.loads(l) for l in src.open()]
    cases = {c["case_id"]: c for c in json.loads((ROOT / "data/cases/cases.json").read_text())}
    out = src.with_name(src.stem + "_judgments.jsonl")
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            done.add((r["game_id"], r["judge"], r["variant"], r["t"]))
    llm = LLM(budget_usd=min(30.0, LLM._ledger_total() + args.max_spend), concurrency=64, tag="judge_pass")
    jobs = []
    for g in games:
        d = g["d"]
        n = len(g["rounds"])
        if args.stages == "all":
            ts = range(0, n + 1)
        elif args.stages == "final":
            ts = [n]
        else:
            ts = [int(x) for x in args.stages.split(",") if int(x) <= n]
        for jn in args.judges.split(","):
            for t in ts:
                if (g["game_id"], jn, args.variant, t) not in done:
                    jobs.append((g, jn, t))
    print(f"{len(jobs)} judge calls to make")
    lock = asyncio.Lock()
    rows = []
    t0 = time.time()

    async def one(g, jn, t):
        try:
            r = await judge(llm, PANEL[jn], cases[g["case_id"]], g["rounds"][:t], args.variant, g["d"])
        except BudgetExceeded:
            raise
        except Exception as e:
            print("FAIL", g["game_id"], jn, t, repr(e)[:150])
            return
        rec = {"game_id": g["game_id"], "judge": jn, "model": PANEL[jn], "variant": args.variant, "t": t,
               "p": r["p"], "ps": r.get("ps", r["p"]), "soft_ok": r.get("soft_ok", False), "v": r["v"]}
        async with lock:
            rows.append(rec)
            if len(rows) % 500 == 0:
                print(f"{len(rows)} done, spend {llm.spent_session:.3f}, {time.time()-t0:.0f}s", flush=True)
                with out.open("a") as f:
                    for x in rows[-500:]:
                        f.write(json.dumps(x) + "\n")

    try:
        await asyncio.gather(*[one(*j) for j in jobs])
    except BudgetExceeded as e:
        print("BUDGET STOP", e)
    rem = len(rows) % 500
    if rem:
        with out.open("a") as f:
            for x in rows[-rem:]:
                f.write(json.dumps(x) + "\n")
    print(f"wrote {len(rows)} rows; session spend {llm.spent_session:.4f}; total {llm.total_spent:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
