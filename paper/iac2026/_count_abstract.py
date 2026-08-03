import re
from pathlib import Path

t = Path("main.tex").read_text(encoding="utf-8")
rest = t[t.index("\\IACmaketitle") :]
parts = []
depth = 0
start = None
for idx, ch in enumerate(rest):
    if ch == "{":
        if depth == 0:
            start = idx + 1
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0 and start is not None:
            parts.append(rest[start:idx])
            start = None
            if len(parts) >= 5:
                break

abs_text = parts[3].lstrip("%").strip()
kws = parts[4].strip()
words = re.findall(r"[A-Za-z0-9']+", abs_text)
kw_list = [k.strip() for k in kws.split(";") if k.strip()]
print("abstract_words", len(words))
print("keywords", kw_list)
print("kw_count", len(kw_list))
assert len(words) <= 400, len(words)
assert len(kw_list) <= 6, len(kw_list)
