import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cg.game import play, judge, JUDGE_A
from cg.llm import LLM
async def main():
    c = [x for x in json.load(open(ROOT/"data/cases/cases.json")) if x["split"]=="pilot"][0]
    llm = LLM(budget_usd=30, concurrency=32, tag="debug")
    rounds = await play(llm, c, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    for i, r in enumerate(rounds):
        print(f"--- round {i+1}  pros_choice={r['pros_choice']} def_choice={r['def_choice']}")
        print("Q:", r["q"]); print("A:", r["a"])
        if r["pros_info"].get("utility"): print("  utilities:", r["pros_info"]["utility"])
        if r["def_info"].get("def_scores"): print("  def scores:", r["def_info"]["def_scores"])
    for t in range(0, len(rounds)+1):
        print("stage", t, await judge(llm, JUDGE_A, c, rounds[:t]))
    print(f"spend {llm.spent_session:.4f} calls {llm.calls}")
    await llm.aclose()
asyncio.run(main())
