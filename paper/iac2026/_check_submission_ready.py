"""Submission-readiness checks for paper/iac2026 (separate from abstract word count)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = (
    "introduction.tex",
    "methods.tex",
    "experiments.tex",
    "results.tex",
    "discussion.tex",
    "conclusion.tex",
    "declaration.tex",
)

REQUIRED_NUMBERS = ("0.894", "0.847", "0.823", "0.856", "28.1")
PAPER_CODE = "IAC-26,A3,IP,109,x109221"
EMAIL_PLACEHOLDERS = ("CORRESPONDING_EMAIL_TBD", r"CORRESPONDING\_EMAIL\_TBD")

PROXY_ABSTRACT_PATTERNS = (
    re.compile(r"proxy\s+ablation", re.I),
    re.compile(r"shadow-dense\s+false\s+positive", re.I),
    re.compile(r"field-scale\s+false-positive", re.I),
    re.compile(r"size.?distance\s+policy\s+proxy", re.I),
)

FORBIDDEN_AI_IMPLICATIONS = (
    re.compile(r"AI[- ]generated\s+(experiments|results|content|science)", re.I),
    re.compile(r"AI[- ]assisted\s+coding", re.I),
    re.compile(r"produced\s+by\s+(generative\s+)?AI", re.I),
)

FORBIDDEN_MANUSCRIPT_STRINGS = (
    "Sydney",
    "IAC-26,A3,IP,xxx",
    "guaranteed safety",
    "flight-ready system",
)


def _paper_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        Path.cwd(),
        Path.cwd() / "paper" / "iac2026",
    ]
    for path in candidates:
        if (path / "main.tex").is_file() and (path / "iac2026.sty").is_file():
            return path
    raise RuntimeError(
        "paper/iac2026 not found. Run from paper/iac2026 or repository root."
    )


def _strip_tex_comments(text: str) -> str:
    out: list[str] = []
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


def _extract_abstract(main_tex: str) -> str:
    marker = "\\IACmaketitle"
    pos = main_tex.find(marker)
    if pos < 0:
        raise RuntimeError("\\IACmaketitle not found in main.tex")
    rest = main_tex[pos + len(marker) :]
    parts: list[str] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(rest):
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise RuntimeError("Unbalanced braces in \\IACmaketitle")
            if depth == 0 and start is not None:
                parts.append(rest[start:i])
                start = None
                if len(parts) >= 5:
                    break
    if len(parts) < 5:
        raise RuntimeError(f"Expected 5 \\IACmaketitle args; found {len(parts)}")
    return parts[3]


def _fail(errors: list[str]) -> None:
    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-email-placeholder",
        action="store_true",
        help="Allow CORRESPONDING_EMAIL_TBD (planning milestone only).",
    )
    args = parser.parse_args()

    root = _paper_dir()
    errors: list[str] = []
    ok: list[str] = []

    main_path = root / "main.tex"
    refs = root / "references.bib"
    decl = root / "sections" / "declaration.tex"
    sty = root / "iac2026.sty"

    for path in (main_path, refs, decl, sty):
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(root)}")

    for name in REQUIRED_SECTIONS:
        p = root / "sections" / name
        if not p.is_file():
            errors.append(f"missing section: sections/{name}")

    if errors:
        _fail(errors)

    main_raw = main_path.read_text(encoding="utf-8")
    main_clean = _strip_tex_comments(main_raw)

    # Rendered manuscript sources only (comment-stripped). No markdown/policy docs.
    source_paths = [main_path, sty] + [
        root / "sections" / n for n in REQUIRED_SECTIONS
    ]
    pack_text = "\n".join(
        _strip_tex_comments(p.read_text(encoding="utf-8")) for p in source_paths
    )

    if PAPER_CODE not in main_raw:
        errors.append(f"paper code {PAPER_CODE} not found in main.tex")
    else:
        ok.append(f"paper_code={PAPER_CODE}")

    for needle in FORBIDDEN_MANUSCRIPT_STRINGS:
        if needle in pack_text:
            errors.append(
                f"forbidden string {needle!r} found in rendered .tex/.sty sources"
            )

    abstract = _extract_abstract(main_clean)
    for num in REQUIRED_NUMBERS:
        if num not in abstract:
            errors.append(f"accepted abstract number missing from abstract block: {num}")
    if not any("accepted abstract number missing" in e for e in errors):
        ok.append("accepted_abstract_numbers_in_abstract")

    has_email_ph = any(ph in main_raw for ph in EMAIL_PLACEHOLDERS)
    if has_email_ph and not args.allow_email_placeholder:
        errors.append(
            "CORRESPONDING_EMAIL_TBD still present; replace before camera-ready "
            "(or pass --allow-email-placeholder for planning builds)"
        )
    elif has_email_ph:
        ok.append("email_placeholder_allowed")
    else:
        ok.append("email_placeholder_absent")

    for pat in PROXY_ABSTRACT_PATTERNS:
        if pat.search(abstract):
            errors.append(f"proxy-related phrasing found in abstract: /{pat.pattern}/")
    if not any("proxy-related" in e for e in errors):
        ok.append("abstract_has_no_proxy_ablation")

    decl_text = decl.read_text(encoding="utf-8")
    decl_l = re.sub(r"\s+", " ", decl_text.lower())
    for needle in ("language verification", "grammar", "readability"):
        if needle not in decl_l:
            errors.append(f"declaration.tex missing required boundary phrase: {needle}")
    if "produced and verified by the author" not in decl_l:
        errors.append("declaration.tex missing author-produced/verified wording")
    for pat in FORBIDDEN_AI_IMPLICATIONS:
        if pat.search(decl_text):
            errors.append(f"declaration implies AI-generated science: /{pat.pattern}/")
    if not any("declaration" in e for e in errors):
        ok.append("declaration_language_only_ok")

    if refs.is_file():
        ok.append("references.bib_present")
    ok.append(f"sections_present={len(REQUIRED_SECTIONS)}")

    if errors:
        _fail(errors)

    print(f"paper_dir: {root}")
    print("submission_ready_checks: OK")
    for line in ok:
        print(f"  - {line}")
    if args.allow_email_placeholder:
        print("mode: planning (--allow-email-placeholder)")
    else:
        print("mode: strict")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
