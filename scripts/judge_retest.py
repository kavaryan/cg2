"""Exploratory measurement check: test-retest reliability of judges at temperature 0.
Re-queries a random sample of (game, stage) prefixes with a new sample index (forcing a fresh API call) and compares.
Usage: python -I scripts/judge_retest.py --split main --n 200 --judges A,E
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cg.game import JUDGE_SYS, JUDGE_USER, SOFT_JUDGES, render_file, render_transcript, soft_p  # noqa: E402
from cg.llm import LLM, parse_json  # noqa: E402
from judge_pass import PANEL  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--judges", default="A,E")
    args = ap.parse_args()
    games = {json.loads(l)["game_id"]: json.loads(l) for l in (ROOT / f"data/raw/study1_{args.split}.jsonl").open()}
    J = [json.loads(l) for l in (ROOT / f"data/raw/study1_{args.split}_judgments.jsonl").open()]
    cases = {c["case_id"]: c for c in json.loads((ROOT / "data/cases/cases.json").read_text())}
    llm = LLM(budget_usd=min(30.0, LLM._ledger_total() + 0.5), concurrency=48, tag="judge_retest")
    rng = random.Random(5)
    out = {}
    for jn in args.judges.split(","):
        pool = [r for r in J if r["judge"] == jn and r["variant"] == "neutral" and r["t"] >= 1 and r["game_id"] in games]
        sample = rng.sample(pool, min(args.n, len(pool)))
        model = PANEL[jn]

        async def one(r):
            g = games[r["game_id"]]
            msgs = [{"role": "system", "content": JUDGE_SYS["neutral"].format(d=g["d"])},
                    {"role": "user", "content": JUDGE_USER.format(file=render_file(cases[g["case_id"]]), t=r["t"], d=g["d"],
                                                                  tr=render_transcript(g["rounds"][:r["t"]]))}]
            resp = await llm.chat(model, msgs, temperature=0.0, max_tokens=60, json_mode=True, sample_idx=1000,
                                  purpose="judge_retest", logprobs=model in SOFT_JUDGES)
            j = parse_json(resp.text) or {}
            try:
                p = float(j.get("p_guilty"))
            except (TypeError, ValueError):
                return None
            ps = soft_p(resp.logprobs)[0] if model in SOFT_JUDGES else None
            return (r["ps"], ps if ps is not None else p, r["v"], str(j.get("verdict", "")).upper())

        res = [x for x in await asyncio.gather(*[one(r) for r in sample]) if x]
        a = np.array([x[0] for x in res]); b = np.array([x[1] for x in res])
        out[jn] = {"n": len(res), "pearson_r": float(np.corrcoef(a, b)[0, 1]), "mean_abs_diff": float(np.abs(a - b).mean()),
                   "identical_share": float(np.mean(np.abs(a - b) < 1e-6)),
                   "verdict_agreement": float(np.mean([x[2] == x[3] for x in res]))}
        print(jn, out[jn])
    (ROOT / f"results/judge_retest_{args.split}.json").write_text(json.dumps(out, indent=1))
    print(f"spend {llm.spent_session:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
