import asyncio, json, os, sys, httpx
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cg.game import JUDGE_SYS, JUDGE_USER, render_file, render_transcript
c = json.load(open(ROOT/"data/cases/cases.json"))[0]
msgs = [{"role":"system","content":JUDGE_SYS["neutral"].format(d=4)},
        {"role":"user","content":JUDGE_USER.format(file=render_file(c), t=0, d=4, tr=render_transcript([]))}]
async def one(cl, m):
    body = {"model": m, "messages": msgs, "temperature": 0, "max_tokens": 30, "logprobs": True, "top_logprobs": 20,
            "provider": {"require_parameters": True}, "usage": {"include": True}}
    r = await cl.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    d = r.json()
    if "choices" not in d: return m, "ERR", str(d)[:200]
    ch = d["choices"][0]; lp = (ch.get("logprobs") or {}).get("content")
    if not lp: return m, d.get("provider"), "no logprobs", ch["message"]["content"]
    toks = [x["token"] for x in lp]
    num = next((x for x in lp if x["token"].strip().isdigit()), None)
    tops = [(t["token"], round(t["logprob"], 2)) for t in num["top_logprobs"][:8]] if num else None
    return m, d.get("provider"), toks[:12], tops
async def main():
    async with httpx.AsyncClient(timeout=60) as cl:
        for row in await asyncio.gather(*[one(cl, m) for m in ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct",
                     "mistralai/mistral-small-3.2-24b-instruct", "deepseek/deepseek-v4-flash", "qwen/qwen3-235b-a22b-2507", "google/gemma-3-27b-it", "openai/gpt-4.1-nano"]]):
            print(row)
asyncio.run(main())
