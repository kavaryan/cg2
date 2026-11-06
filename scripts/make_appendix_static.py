"""Write paper/app_prompts.tex and paper/app_case.tex verbatim from the code and case bank (no hand-copying)."""
import json, sys, textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cg import game as G

def verb(title, s, width=118):
    lines = []
    for para in s.split("\n"):
        lines += textwrap.wrap(para, width) or [""]
    body = "\n".join(lines).replace("\\end{Verbatim}", "")
    return f"\\paragraph{{{title}}}\n\\begin{{Verbatim}}[fontsize=\\scriptsize,frame=single]\n{body}\n\\end{{Verbatim}}\n"

out = []
out.append(verb("Prosecutor system prompt (advocate objective)", G.PROS_SYS["advocate"].format(d=4) + G._PROS_TAIL))
out.append(verb("Prosecutor objective sentence (minister objective; rest identical)", G.PROS_SYS["minister"].format(d=4).split("\n")[-1]))
out.append(verb("Prosecutor candidate-generation user prompt", G.GEN_USER.format(file="<court view of the case file>", tr="<transcript>", t="t", d=4, k=4)))
out.append(verb("Introspective selection user prompt", G.PICK_USER.format(file="<court view>", tr="<transcript>", t="t", d=4, numbered="1. ...\n2. ...")))
out.append(verb("Defendant system prompt", G.DEF_SYS.format(name="<name>", file="<public view of the case file>", private="<private brief: guilty or innocent world>")))
out.append(verb("Defendant candidate-generation user prompt", G.DEF_GEN_USER.format(tr="<transcript ending with the prosecutor's question>", t="t", d=4, k=4)))
out.append(verb("Planner's hypothetical-defendant simulator (system prompt)", G.HYP_SYS.format(name="<name>", file="<public view>", hyp="DID / did NOT")))
out.append(verb("Judge system prompt (neutral; primary)", G.JUDGE_SYS["neutral"].format(d=4)))
out.append(verb("Judge sentence in the guilt-leaning variant (replaces the presumption sentence)",
                G.JUDGE_SYS["guilt_leaning"].format(d=4).split("Base your judgment only on the case file and the transcript. ")[1].split("\nReply")[0]))
out.append(verb("Judge user prompt", G.JUDGE_USER.format(file="<court view>", t="t", d=4, tr="<transcript>")))
(ROOT / "paper/app_prompts.tex").write_text("\n".join(out))

cases = json.loads((ROOT / "data/cases/cases.json").read_text())
c = next(x for x in cases if x["case_id"] == "C12")
txt = G.render_file(c, "court") + "\n\nGUILTY BRIEF:\n" + G.private_brief(c, "guilty") + "\n\nINNOCENT BRIEF:\n" + G.private_brief(c, "innocent")
(ROOT / "paper/app_case.tex").write_text(verb(f"Case {c['case_id']} (designed strength: {c['strength']}; audit scores: " +
    ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in c["audit"].items() if k in ('ambiguity','w_true_both','cover_contradicted','innocent_consistent','statement_neutral')) + ")", txt))
print("ok")
