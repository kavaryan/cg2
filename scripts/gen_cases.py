"""Generate the stimulus bank: N criminal cases, each with ONE public file and TWO private briefs.

Matched-pair design: the guilty and innocent versions of a case share an identical public file,
so the judge's information differs only through the conversation. Case features (crime type,
setting, designed evidence strength, defendant gender) are assigned by a balanced design, not
left to the generator, so they can be used as pre-treatment moderators.
"""
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cg.llm import LLM, parse_json  # noqa: E402

GEN_MODEL = "anthropic/claude-haiku-5.5"
GEN_REASONING = {"max_tokens": 2000}
OUT = Path(__file__).resolve().parents[1] / "data" / "cases" / "cases.json"

CRIMES = ["burglary of a home", "embezzlement from an employer", "arson of a commercial building",
          "hit-and-run causing injury", "assault outside a bar", "insurance fraud", "theft of company data",
          "poisoning a neighbour's dog", "credit-card fraud", "forging a will", "bribery of a council official",
          "theft of a valuable artwork", "vandalism of a rival's business", "theft of prescription drugs from a clinic"]
SETTINGS = ["a small coastal town", "a large city", "a university campus", "a rural village"]
STRENGTHS = ["weak", "moderate", "strong"]

SYSTEM = """You design realistic, legally plausible criminal cases for a research simulation of courtroom examinations,
following the logic of the Strategic Use of Evidence (SUE) interviewing paradigm.
Each case has: a PUBLIC file (known to everyone, including the defendant), WITHHELD evidence (known to the prosecution
and the judge, but NOT disclosed to the defendant before the examination), and two alternative private worlds for the
defendant: one in which the defendant really did it, and one in which the defendant did not.
Hard constraints:
1. Public evidence (E1-E4) is circumstantial and fits both worlds; strength varies by design.
2. Each withheld item (W1, W2) is a NEUTRAL RECORD OF THE DEFENDANT'S OWN presence, movement, purchase or contact
   (e.g. a parking-garage record, phone location data, a shop receipt, a call log, a door-entry log), NOT forensic proof
   of the crime and NOT a sighting of an unidentified person. It is literally true in both worlds.
3. Guilty world: the W facts are part of how the defendant committed the crime. The defendant does not know the police
   have them, and their prepared cover story DENIES or MISSTATES each W fact (wrong place, wrong time, denied contact).
   A judge who compares the cover story with W1/W2 would see a specific contradiction.
4. Innocent world: the very same W facts have ordinary innocent reasons (an errand, a visit, a hobby), which the
   defendant knows and would readily admit if asked. Do NOT rely on lookalikes, framing, misread records or mystery
   third parties to explain W items.
5. The initial statement to police is short and vague, identical in both worlds, and does not mention the W facts.
6. No item is conclusive on its own. Avoid graphic violence. Use ordinary realistic names (not famous people).
Output valid JSON only."""

USER_TMPL = """Write one case with these fixed design features:
- Charge: {crime}
- Setting: {setting}
- Defendant gender: {gender}
- Designed strength of the public evidence: {strength} (weak = thin circumstantial evidence; moderate = several suggestive items;
  strong = a lot points to the defendant, but an innocent explanation still exists for every item)

JSON schema:
{{
 "title": "short case title",
 "summary": "2-4 sentences: what happened, when, where (public knowledge)",
 "defendant": {{"name": "...", "age": <int>, "occupation": "...", "connection": "one sentence on their link to the victim/scene"}},
 "evidence": ["E1 ...", "E2 ...", "E3 ...", "E4 ..."],
 "withheld_evidence": ["W1 ...", "W2 ..."],
 "initial_statement": "1-2 vague sentences the defendant told police (identical in both worlds)",
 "guilty_world": {{
    "what_really_happened": "3-5 sentences: how the defendant actually committed the crime, including the facts behind W1 and W2",
    "cover_story": "the false account the defendant will give; it must make a specific false claim that W1 and W2 would contradict",
    "facts_to_hide": "the specific true facts the defendant must hide (do not mention that the police have W1/W2)"
 }},
 "innocent_world": {{
    "what_really_happened": "3-5 sentences: the defendant's true, innocent account, which explains every E and W item innocently",
    "unprovable_parts": "which parts of the true account cannot be independently verified, and why",
    "unrelated_secret": "an embarrassing but NON-criminal secret that makes the defendant guarded or evasive on one topic"
 }}
}}
Evidence items must be concrete observations (records, sightings, forensic traces, messages), never conclusions.
The true culprit in the innocent world must not be identified in any file."""


async def main(n_cases: int = 56, seed: int = 20251104):
    rng = random.Random(seed)
    combos = [(c, s) for c in CRIMES for s in SETTINGS]
    rng.shuffle(combos)
    combos = combos[:n_cases]
    design = []
    for i, (crime, setting) in enumerate(combos):
        design.append({"crime": crime, "setting": setting, "strength": STRENGTHS[i % 3],
                       "gender": ["female", "male"][(i // 3) % 2]})
    llm = LLM(budget_usd=30, concurrency=16, tag="gen_cases")

    async def make(i, d):
        for attempt in range(4):
            r = await llm.chat(GEN_MODEL, [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": USER_TMPL.format(**d)}],
                               temperature=0.9, max_tokens=12000, json_mode=True, sample_idx=attempt, reasoning=GEN_REASONING,
                               purpose="gen_case_v2")
            c = parse_json(r.text)
            ok = (isinstance(c, dict) and len(c.get("evidence", [])) >= 4 and len(c.get("withheld_evidence", [])) >= 2
                  and "guilty_world" in c
                  and "innocent_world" in c and "defendant" in c)
            if ok:
                c.update({"case_id": f"C{i:02d}", "crime_type": d["crime"], "setting": d["setting"],
                          "strength": d["strength"], "gender": d["gender"]})
                return c
        raise RuntimeError(f"case {i} failed")

    cases = await asyncio.gather(*[make(i, d) for i, d in enumerate(design)])
    # Pilot / main split: stratified by designed strength so that the main set is exactly balanced (16/16/16).
    pilot = set()
    for s, n_pilot in (("weak", 3), ("moderate", 3), ("strong", 2)):
        ids = [i for i, c in enumerate(cases) if c["strength"] == s]
        rng.shuffle(ids)
        pilot.update(ids[:n_pilot])
    for i, c in enumerate(cases):
        c["split"] = "pilot" if i in pilot else "main"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"wrote {len(cases)} cases; spend {llm.spent_session:.4f}; total {llm.total_spent:.4f}")
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
