import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cg.llm import LLM, parse_json
MODELS = ["qwen/qwen3-235b-a22b-2507", "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.5-flash-lite",
          "openai/gpt-4.1-nano", "mistralai/mistral-small-3.2-24b-instruct", "deepseek/deepseek-v4-flash",
          "anthropic/claude-haiku-5.5", "openai/gpt-4o-mini"]
CASE = ("Case: A laptop worth $2,400 vanished from a locked office overnight. Defendant: Dana Ruiz, office manager, "
        "holds one of three keys. Evidence: E1 badge log shows Dana entered at 21:40 and left at 21:55; E2 the laptop "
        "was later listed on a resale site from an account created at a cafe near Dana's home; E3 Dana had complained "
        "about unpaid overtime. Transcript: PROSECUTOR: Why were you in the office at 21:40? DEFENDANT: I forgot my "
        "phone charger and went back for it. I was there ten minutes and the laptop was on the desk when I left.")
SYS = ("You are an impartial judge. Estimate the probability (0-100) that the defendant committed the crime, "
       "based only on the case file and transcript. Reply with JSON only: {\"p_guilty\": <integer>, "
       "\"verdict\": \"GUILTY\"|\"NOT_GUILTY\"|\"CONTINUE\"}")
async def main():
    llm = LLM(budget_usd=30, concurrency=8, tag="smoke")
    async def one(m):
        t = time.time()
        try:
            r = await llm.chat(m, [{"role": "system", "content": SYS}, {"role": "user", "content": CASE}],
                               temperature=0, max_tokens=60, json_mode=True, purpose="smoke", sample_idx=1)
            return m, round(time.time() - t, 1), r.provider, r.prompt_tokens, r.completion_tokens, f"{r.cost:.6f}", parse_json(r.text), r.text[:80]
        except Exception as e:
            return m, "ERR", str(e)[:200]
    for row in await asyncio.gather(*[one(m) for m in MODELS]):
        print(row)
    print("session spend", llm.spent_session)
    await llm.aclose()
asyncio.run(main())
