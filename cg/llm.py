"""Async OpenRouter client with deterministic response cache, cost ledger and hard budget guard.

Design notes (these matter for the experiments):
* Every call is keyed by sha256(model, messages, temperature, max_tokens, sample_idx).
  Identical simulator states therefore receive identical draws (common random numbers),
  and a crashed run can be resumed without paying twice. Independent draws of the same
  prompt are requested by changing ``sample_idx``.
* The serving provider is logged for every call (providers differ in quantisation).
* A hard budget cap stops the process before spend exceeds the configured limit.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "cache" / "llm_cache.sqlite"
LEDGER_PATH = ROOT / "data" / "cache" / "ledger.jsonl"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Provider pinning: keeps quantisation/serving stack constant within a model.
PROVIDERS = {
    "qwen/qwen3-235b-a22b-2507": {"order": ["DeepInfra"], "allow_fallbacks": True},
    # Llama is used only as a judge with logprobs; DeepInfra lacks logprobs and AkashML returns malformed ones.
    "meta-llama/llama-3.3-70b-instruct": {"order": ["Parasail"], "allow_fallbacks": True, "require_parameters": True,
                                          "ignore": ["AkashML", "DeepInfra"]},
    "mistralai/mistral-small-3.2-24b-instruct": {"order": ["DeepInfra", "Mistral"], "allow_fallbacks": True},
    "deepseek/deepseek-v4-flash": {"order": ["DeepInfra", "GMICloud"], "allow_fallbacks": True},
}
# Models whose optional reasoning must be switched off (cost + comparability).
NO_REASONING = {"google/gemini-2.5-flash-lite", "deepseek/deepseek-v4-flash", "anthropic/claude-haiku-5.5",
                "openai/gpt-6-luna"}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Resp:
    text: str
    cost: float
    prompt_tokens: int
    completion_tokens: int
    provider: str
    cached: bool
    logprobs: list | None = None


class LLM:
    def __init__(self, budget_usd: float = 30.0, concurrency: int = 32, tag: str = "default"):
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.budget = budget_usd
        self.sem = asyncio.Semaphore(concurrency)
        self.tag = tag
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE_PATH, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()
        self.spent_session = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=20.0),
                                        limits=httpx.Limits(max_connections=concurrency + 8))
        self.total_spent = self._ledger_total()

    @staticmethod
    def _ledger_total() -> float:
        if not LEDGER_PATH.exists():
            return 0.0
        tot = 0.0
        with open(LEDGER_PATH) as f:
            for line in f:
                try:
                    tot += json.loads(line)["cost"]
                except Exception:
                    pass
        return tot

    @staticmethod
    def key_for(model, messages, temperature, max_tokens, sample_idx, extra=None) -> str:
        blob = json.dumps({"m": model, "msg": messages, "t": temperature, "mt": max_tokens,
                           "s": sample_idx, "x": extra}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _get(self, k):
        row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO c (k, v) VALUES (?, ?)", (k, json.dumps(v, ensure_ascii=False)))
        self.db.commit()

    async def chat(self, model: str, messages: list[dict], temperature: float = 1.0, max_tokens: int = 300,
                   sample_idx: int = 0, json_mode: bool = False, purpose: str = "", logprobs: bool = False,
                   reasoning: dict | None = None) -> Resp:
        extra = {"json": json_mode, "lp": True} if logprobs else {"json": json_mode}
        if reasoning:
            extra["reasoning"] = reasoning
        k = self.key_for(model, messages, temperature, max_tokens, sample_idx, extra)
        hit = self._get(k)
        if hit is not None:
            self.cache_hits += 1
            return Resp(hit["text"], 0.0, hit.get("pt", 0), hit.get("ct", 0), hit.get("provider", ""), True,
                        hit.get("lp"))
        if self.total_spent >= self.budget:
            raise BudgetExceeded(f"budget {self.budget} reached (spent {self.total_spent:.3f})")
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "usage": {"include": True}}
        # Seeds make the draw reproducible where the provider honours them; the cache makes it exact.
        body["seed"] = int(k[:8], 16) % (2**31 - 1)
        if model in PROVIDERS:
            body["provider"] = PROVIDERS[model]
        if reasoning:
            body["reasoning"] = reasoning
        elif model in NO_REASONING:
            body["reasoning"] = {"enabled": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "conversation-games-study"}
        async with self.sem:
            delay = 2.0
            last_err = None
            for attempt in range(7):
                try:
                    r = await self.client.post(API_URL, json=body, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504, 408):
                        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                        await asyncio.sleep(delay + random.random())
                        delay = min(delay * 2, 60)
                        continue
                    if r.status_code != 200:
                        last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                        if r.status_code == 400 and json_mode and "response_format" in body:
                            body.pop("response_format")  # some providers reject json mode; fall back to parsing
                            continue
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60)
                        continue
                    d = r.json()
                    if "error" in d or not d.get("choices"):
                        last_err = str(d.get("error", d))[:300]
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60)
                        continue
                    msg = d["choices"][0]["message"]
                    text = msg.get("content") or ""
                    lp = None
                    if logprobs:
                        content = (d["choices"][0].get("logprobs") or {}).get("content") or []
                        lp = [[x.get("token", ""), [[t.get("token", ""), t.get("logprob", -99.0)]
                                                    for t in (x.get("top_logprobs") or [])]] for x in content]
                    u = d.get("usage", {}) or {}
                    cost = float(u.get("cost") or 0.0)
                    pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
                    prov = d.get("provider", "")
                    self.total_spent += cost
                    self.spent_session += cost
                    self.calls += 1
                    with open(LEDGER_PATH, "a") as f:
                        f.write(json.dumps({"t": time.time(), "tag": self.tag, "purpose": purpose, "model": model,
                                            "provider": prov, "pt": pt, "ct": ct, "cost": cost}) + "\n")
                    if not text.strip():
                        last_err = "empty completion"
                        empties = locals().get("empties", 0) + 1
                        if empties >= 2:
                            break
                        await asyncio.sleep(1)
                        continue
                    self._put(k, {"text": text, "pt": pt, "ct": ct, "provider": prov, "model": model, "lp": lp})
                    return Resp(text, cost, pt, ct, prov, False, lp)
                except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError) as e:
                    last_err = repr(e)
                    await asyncio.sleep(delay + random.random())
                    delay = min(delay * 2, 60)
            raise RuntimeError(f"LLM call failed after retries ({model}, {purpose}): {last_err}")

    async def aclose(self):
        await self.client.aclose()
        self.db.close()


def parse_json(text: str):
    """Lenient JSON extraction: first {...} or [...] block."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None
