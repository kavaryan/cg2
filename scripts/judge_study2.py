"""Judge every realised continuation of Study 2 at the branch stage t (immediate effect) and at stage d (final).
Usage: python -I scripts/judge_study2.py --split main --judges A,E --stages both ; --judges C,D --stages final
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cg.game import judge  # noqa: E402
from cg.llm import LLM, BudgetExceeded  # noqa: E402
from judge_pass import PANEL  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main")
    ap.add_argument("--judges", default="A,E")
    ap.add_argument("--stages", default="both", help="both | final")
    ap.add_argument("--max-spend", type=float, default=3.0)
    args = ap.parse_args()
    src = ROOT / f"data/raw/study2_{args.split}.jsonl"
    out = ROOT / f"data/raw/study2_{args.split}_judgments.jsonl"
    cases = {c["case_id"]: c for c in json.loads((ROOT / "data/cases/cases.json").read_text())}
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            done.add((r["cid"], r["judge"], r["stage"]))
    jobs = []
    for l in src.open():
        h = json.loads(l)
        for r in h["real"]:
            cid = f"{h['hid']}|j{r['j']}|{r['type']}|m{r['m']}"
            stages = [h["t"], h["d"]] if args.stages == "both" else [h["d"]]
            for jn in args.judges.split(","):
                for st in stages:
                    if (cid, jn, st) not in done:
                        jobs.append((h, r, cid, jn, st))
    print(f"{len(jobs)} judge calls")
    llm = LLM(budget_usd=min(30.0, LLM._ledger_total() + args.max_spend), concurrency=96, tag="judge_study2")
    rows, lock, t0 = [], asyncio.Lock(), time.time()

    async def one(h, r, cid, jn, st):
        try:
            j = await judge(llm, PANEL[jn], cases[h["case_id"]], r["rounds"][:st], "neutral", h["d"])
        except BudgetExceeded:
            raise
        except Exception as e:
            print("FAIL", cid, jn, st, repr(e)[:120])
            return
        async with lock:
            rows.append({"hid": h["hid"], "cid": cid, "j": r["j"], "type": r["type"], "m": r["m"], "judge": jn,
                         "stage": st, "p": j["p"], "ps": j["ps"], "v": j["v"]})
            if len(rows) % 1000 == 0:
                print(f"{len(rows)} done, spend {llm.spent_session:.3f}, {time.time()-t0:.0f}s", flush=True)

    try:
        await asyncio.gather(*[one(*x) for x in jobs])
    except BudgetExceeded as e:
        print("BUDGET STOP", e)
    with out.open("a") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"wrote {len(rows)}; session spend {llm.spent_session:.4f}; total {llm.total_spent:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
