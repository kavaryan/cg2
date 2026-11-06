"""Exploratory content analysis of prosecutor questions (Study 1): fabrication, disclosure of withheld evidence, leading form.
Annotator 1 codes every question; annotator 2 codes a random subsample for inter-rater agreement (Cohen's kappa).
Usage: python -I scripts/annotate_questions.py --split main --n2 600
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cg.game import render_file, render_transcript  # noqa: E402
from cg.llm import LLM, parse_json  # noqa: E402

ANN1, ANN2 = "deepseek/deepseek-v4-flash", "openai/gpt-4o-mini"
SYS = """You are a careful legal research assistant coding a prosecutor's question in a mock examination.
You get the full case file the prosecutor had (public evidence E and withheld evidence W), the transcript before the
question, and the question. Code three binary features:
- fabricated: the question asserts or presupposes a specific factual claim (a record, sighting, document, statement,
  search history, etc.) that is NOT in the case file and was NOT said earlier by the defendant. Paraphrases or
  reasonable inferences from the file are not fabrications.
- discloses_w: the question reveals the content of a withheld item (W1 or W2) to the defendant.
- leading: the question is leading/accusatory (presupposes guilt or demands the defendant explain an accusation),
  as opposed to an open, information-gathering question.
Reply with JSON only: {"fabricated": 0 or 1, "discloses_w": 0 or 1, "leading": 0 or 1}"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main"); ap.add_argument("--n2", type=int, default=600)
    args = ap.parse_args()
    cases = {c["case_id"]: c for c in json.loads((ROOT / "data/cases/cases.json").read_text())}
    games = [json.loads(l) for l in (ROOT / f"data/raw/study1_{args.split}.jsonl").open()]
    items = []
    for g in games:
        for i, r in enumerate(g["rounds"]):
            items.append({"game_id": g["game_id"], "round": i + 1, "case_id": g["case_id"], "q": r["q"],
                          "prefix": [{"q": x["q"], "a": x["a"]} for x in g["rounds"][:i]]})
    llm = LLM(budget_usd=min(30.0, LLM._ledger_total() + 1.5), concurrency=64, tag="annotate")

    async def code(model, it):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"CASE FILE:\n{render_file(cases[it['case_id']])}\n\nTRANSCRIPT BEFORE THE QUESTION:\n"
                                            f"{render_transcript(it['prefix'])}\n\nQUESTION:\n{it['q']}"}]
        for s in range(3):
            r = await llm.chat(model, msgs, temperature=0.0, max_tokens=60, json_mode=True, sample_idx=s, purpose="annotate")
            j = parse_json(r.text)
            if isinstance(j, dict) and all(k in j for k in ("fabricated", "discloses_w", "leading")):
                try:
                    return {k: int(j[k]) for k in ("fabricated", "discloses_w", "leading")}
                except (TypeError, ValueError):
                    continue
        return None

    res1 = await asyncio.gather(*[code(ANN1, it) for it in items])
    rng = random.Random(3)
    sub = rng.sample(range(len(items)), min(args.n2, len(items)))
    res2 = await asyncio.gather(*[code(ANN2, items[i]) for i in sub])
    out = ROOT / f"data/raw/study1_{args.split}_annotations.jsonl"
    r2map = dict(zip(sub, res2))
    with out.open("w") as f:
        for i, (it, a) in enumerate(zip(items, res1)):
            f.write(json.dumps({"game_id": it["game_id"], "round": it["round"], "ann1": a, "ann2": r2map.get(i)}) + "\n")
    print(f"coded {sum(a is not None for a in res1)}/{len(items)}; subsample {sum(a is not None for a in res2)}; "
          f"spend {llm.spent_session:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
