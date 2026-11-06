"""Replace mode-collapsed generated defendant names with unique, gender-matched, diverse names.
Names are constant within a case, so they cannot confound within-case treatment contrasts."""
import json
import random
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "data" / "cases" / "cases.json"
F = ["Priya Nair", "Amara Okafor", "Sofia Moreno", "Mei Chen", "Hannah Kowalski", "Fatima Haddad", "Grace Mensah",
     "Ingrid Larsen", "Yuki Tanaka", "Leila Rahimi", "Rosa Delgado", "Chloe Dubois", "Aisha Bello", "Niamh Brennan",
     "Ana Petrovic", "Zainab Yusuf", "Maria Rossi", "Keisha Daniels", "Olga Ivanova", "Farah Siddiqui",
     "Lucia Ferreira", "Naomi Goldberg", "Esther Adeyemi", "Jiwoo Park", "Camila Torres", "Sarah Whitfield",
     "Helena Novak", "Tamar Levi", "Ines Carvalho", "Rebecca Ashworth"]
M = ["Arjun Mehta", "Kwame Asante", "Diego Alvarez", "Wei Zhang", "Tomasz Nowak", "Omar Khalil", "Samuel Okonkwo",
     "Lars Eriksen", "Kenji Watanabe", "Reza Karimi", "Mateo Silva", "Julien Moreau", "Ibrahim Diallo", "Conor Walsh",
     "Marko Horvat", "Yusuf Demir", "Marco Bianchi", "Darnell Brooks", "Dmitri Volkov", "Hamza Qureshi",
     "Rafael Costa", "David Rosen", "Chidi Eze", "Minjun Kim", "Andres Castillo", "James Pemberton", "Pavel Dvorak",
     "Avi Cohen", "Joao Mendes", "George Holloway"]


def sub_all(obj, mapping):
    if isinstance(obj, str):
        for old, new in mapping:
            obj = re.sub(rf"\b{re.escape(old)}\b", new, obj)
        return obj
    if isinstance(obj, list):
        return [sub_all(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: sub_all(v, mapping) for k, v in obj.items()}
    return obj


def main():
    cases = json.loads(P.read_text())
    rng = random.Random(7)
    f, m = F[:], M[:]
    rng.shuffle(f)
    rng.shuffle(m)
    for c in cases:
        old = c["defendant"].get("generated_name", c["defendant"]["name"])
        new = (f if c["gender"] == "female" else m).pop()
        of, *_, ol = old.split()
        nf, nl = new.split()
        mapping = [(old, new), (of, nf), (ol, nl)]
        gen_name = old
        c2 = sub_all({k: v for k, v in c.items() if k not in ("case_id", "split")}, mapping)
        c.update(c2)
        c["defendant"]["generated_name"] = gen_name
        c["defendant"]["name"] = new
    P.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print([c["defendant"]["name"] for c in cases])
    # leakage check: generated names should no longer appear
    blob = json.dumps([{k: v for k, v in c.items() if k != "defendant"} for c in cases])
    left = {c["defendant"]["generated_name"] for c in cases if c["defendant"]["generated_name"] in blob}
    print("residual old names:", left)


if __name__ == "__main__":
    main()
