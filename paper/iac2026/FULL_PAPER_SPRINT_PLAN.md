# Full-paper sprint plan (IAC 2026)

Timeline: **5 August – 14 September 2026** (prose-first full-paper v0.1 → camera-ready hardening).

**Explicitly excluded from this sprint:** funding applications, grant writing, sponsorship language, budget prose. The manuscript must remain free of financing content.

## Goals

| Window | Deliverable |
|--------|-------------|
| Aug 5–12 | Full-paper v0.1 prose: Intro–Conclusion + Limitations; Results evidence-gated stubs |
| Aug 13–26 | Close `independent_eval_v1` data readiness (annotations / frozen split) if available; still no invented metrics |
| Aug 27–Sep 7 | Fill Results only when ledger support allows; figures from real artifacts |
| Sep 8–14 | Format polish, AI declaration re-check, submission-ready email, forbid-string greps |

## Hard freezes (do not reopen casually)

- Abstract numbers: **0.894 / 0.847 / 0.823 / 0.856 / 28.1 FPS**
- Claim supports: C05/C06/C07 `accepted_abstract_reproduction_pending`; `IND_EVAL_V1` `protocol_defined_pending_data`; C08 `planned`
- `sections/declaration.tex` language-only AI use — no edit without author decision
- No funding / grant / sponsorship / budget language

## Out of scope this sprint

- New scientific experiments invented for the paper
- Fabricated Results cells
- Flight-certification claims
- Archaeology dump of 32p full PDF into abstract

## Pointers

- Manuscript: [`main.tex`](main.tex)
- Section plan: [`PLAN.md`](PLAN.md)
- Figures/tables: [`FIGURE_TABLE_PLAN.md`](FIGURE_TABLE_PLAN.md)
- Claims: [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md)
