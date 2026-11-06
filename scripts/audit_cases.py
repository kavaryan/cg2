"""Stimulus validation: an independent model audits each case against the design constraints; failing cases are
regenerated (new sample) up to 3 times and the best-scoring version is kept. Audit scores are stored with the case
and reported in the paper as a stimulus manipulation check."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cg.llm import LLM, parse_json  # noqa: E402
import gen_cases as G  # noqa: E402

AUDIT_MODEL = "deepseek/deepseek-v4-flash"
AUDIT_REASONING = None
AUDIT_SYS = """You audit synthetic criminal cases for a research simulation based on the Strategic Use of Evidence paradigm.
Design: public evidence (E) is known to everyone. Withheld evidence (W) is known to the prosecution and judge but NOT to
the defendant. W items are records of the defendant's own movements/contacts; they are INTENDED to be literally true in
both worlds (both narratives include those facts). In the guilty world the W facts are part of the crime, so the guilty
defendant's cover story denies or misstates them; in the innocent world the same facts have ordinary reasons that the
defendant would admit. W items being somewhat incriminating is fine. Score each criterion 1 (fails) to 5 (fully met):
- ambiguity: the public evidence alone does not settle guilt or innocence (it may lean one way).
- w_true_both: both narratives explicitly include every W fact as true (no lookalikes, misread records, or framing).
- cover_contradicted: the guilty cover story makes a specific claim that W1 and/or W2 directly contradict, AND denying
  those facts makes sense for a guilty person (the W facts are incriminating in the guilty world, not an alibi).
- innocent_consistent: the innocent account explains every E and W item with ordinary innocent reasons, coherently.
- statement_neutral: the initial statement is vague and does not already admit or deny the W facts.
First check each criterion briefly, then score. Reply with JSON only: {"checks": "<at most 80 words>", "ambiguity": n,
"w_true_both": n, "cover_contradicted": n, "innocent_consistent": n, "statement_neutral": n, "problem": "<at most 25 words>"}"""
KEYS = ["ambiguity", "w_true_both", "cover_contradicted", "innocent_consistent", "statement_neutral"]


CORE = ["w_true_both", "cover_contradicted", "innocent_consistent"]


def passes(a):
    return bool(a) and min(a[k] for k in CORE) >= 4 and min(a[k] for k in KEYS) >= 3


async def audit(llm, c):
    view = {k: c[k] for k in ["summary", "evidence", "withheld_evidence", "initial_statement", "guilty_world",
                              "innocent_world"]}
    for s in range(3):
        r = await llm.chat(AUDIT_MODEL, [{"role": "system", "content": AUDIT_SYS},
                                         {"role": "user", "content": json.dumps(view, ensure_ascii=False)}],
                           temperature=0.0, max_tokens=600, json_mode=True, sample_idx=s, purpose="audit_case",
                           reasoning=AUDIT_REASONING)
        a = parse_json(r.text)
        if isinstance(a, dict) and all(isinstance(a.get(k), (int, float)) for k in KEYS):
            return a
    return None


async def main():
    path = ROOT / "data/cases/cases.json"
    cases = json.loads(path.read_text())
    llm = LLM(budget_usd=30, concurrency=16, tag="audit_cases")

    async def fix(c):
        d = {"crime": c["crime_type"], "setting": c["setting"], "strength": c["strength"], "gender": c["gender"]}
        a = await audit(llm, c)
        best, best_a, tries = c, a, 0
        while not passes(best_a) and tries < 5:
            tries += 1
            r = await llm.chat(G.GEN_MODEL, [{"role": "system", "content": G.SYSTEM},
                                             {"role": "user", "content": G.USER_TMPL.format(**d)}],
                               temperature=0.9, max_tokens=12000, json_mode=True, sample_idx=200 + tries, reasoning=G.GEN_REASONING,
                               purpose="gen_case_v2_regen")
            n = parse_json(r.text)
            if not (isinstance(n, dict) and len(n.get("evidence", [])) >= 4 and len(n.get("withheld_evidence", [])) >= 2
                    and "guilty_world" in n and "innocent_world" in n and "defendant" in n):
                continue
            na = await audit(llm, n)
            score = lambda x: (passes(x), min(x[k] for k in CORE), sum(x[k] for k in KEYS)) if x else (False, 0, 0)
            if score(na) > score(best_a):
                for k in ["case_id", "crime_type", "setting", "strength", "gender", "split"]:
                    n[k] = c[k]
                best, best_a = n, na
        best["audit"] = best_a
        best["audit_regenerations"] = tries
        return best

    out = await asyncio.gather(*[fix(c) for c in cases])
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    import collections
    print("pass:", sum(passes(c["audit"]) for c in out), "/", len(out))
    print("regenerations:", collections.Counter(c["audit_regenerations"] for c in out))
    for k in KEYS:
        print(k, sum(c["audit"][k] for c in out if c["audit"]) / len(out))
    print(f"spend {llm.spent_session:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
