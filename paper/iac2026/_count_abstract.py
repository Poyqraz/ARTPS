import re
import sys
from pathlib import Path


def _strip_tex_comments(text: str) -> str:
    """Remove TeX comments: unescaped % to end of line. Keep \\% intact."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "%" and (i == 0 or text[i - 1] != "\\"):
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _find_main_tex() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "main.tex",
        Path.cwd() / "main.tex",
        Path.cwd() / "paper" / "iac2026" / "main.tex",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(
        "main.tex not found. Run from paper/iac2026 or repo root "
        "(expected paper/iac2026/main.tex)."
    )


def _extract_maketitle_args(src: str) -> list[str]:
    marker = "\\IACmaketitle"
    pos = src.find(marker)
    if pos < 0:
        raise RuntimeError("\\IACmaketitle not found in main.tex")

    rest = src[pos + len(marker) :]
    parts: list[str] = []
    depth = 0
    start: int | None = None
    i = 0
    while i < len(rest) and len(parts) < 5:
        ch = rest[i]
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise RuntimeError("Unbalanced braces in \\IACmaketitle arguments (depth < 0)")
            if depth == 0:
                if start is None:
                    raise RuntimeError("Closing brace without open group in \\IACmaketitle")
                parts.append(rest[start:i])
                start = None
        i += 1

    if depth != 0:
        raise RuntimeError("Unbalanced braces in \\IACmaketitle arguments (unclosed group)")
    if len(parts) < 5:
        raise RuntimeError(
            f"Expected 5 \\IACmaketitle arguments (title, authors, affiliation, "
            f"abstract, keywords); found {len(parts)}"
        )
    return parts


def main() -> int:
    path = _find_main_tex()
    raw = path.read_text(encoding="utf-8")
    cleaned = _strip_tex_comments(raw)
    parts = _extract_maketitle_args(cleaned)

    abs_text = parts[3].strip()
    kws = parts[4].strip()
    words = re.findall(r"[A-Za-z0-9']+", abs_text)
    kw_list = [k.strip() for k in kws.split(";") if k.strip()]

    print(f"main_tex: {path}")
    print(f"abstract_words: {len(words)}")
    print(f"keyword_count: {len(kw_list)}")
    print(f"keywords: {kw_list}")

    if len(words) > 400:
        raise RuntimeError(f"Abstract has {len(words)} words (limit 400)")
    if len(kw_list) > 6:
        raise RuntimeError(f"Found {len(kw_list)} keywords (limit 6)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
