import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cg.game import judge
from cg.llm import LLM
cases = json.load(open(ROOT/"data/cases/cases.json"))
async def main():
    llm = LLM(budget_usd=30, concurrency=16, tag="debug")
    for m in ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.5-flash-lite"]:
        res = await asyncio.gather(*[judge(llm, m, c, []) for c in cases[:6]])
        print(m, [(r["p"], round(r["ps"], 1), r["v"], round(r["mass"], 2)) for r in res])
    print(f"spend {llm.spent_session:.4f}")
    await llm.aclose()
asyncio.run(main())
