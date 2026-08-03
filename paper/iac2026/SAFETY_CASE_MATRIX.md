# Safety-case evidence matrix

Operational safety wording for Discussion. Align with [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md) support levels.

Separate clearly:

- **implemented mechanism** — code present; not by itself a performance claim
- **proxy evidence** — curated pseudo-GT; not human bbox GT
- **planned quantitative validation** — protocol exists; run pending
- **unsupported flight qualification** — must not be stated as current claim

| ID | Safety / ops claim | Evidence class | Gap | Allowed manuscript wording |
|----|--------------------|----------------|-----|----------------------------|
| S01 | Rover-body FP suppression | implemented + proxy (C09) | Proxy ≠ human bbox | “implemented gate; preliminary proxy on curated set” |
| S02 | Boundary-shadow FP suppression | implemented + proxy (C09) | same | same |
| S03 | Object-in-shadow not erased | proxy (C10) | small n; needs GT | “preliminary proxy shadow-rock loss; GT confirmation planned” |
| S04 | Priority Buffer reduces FN from diversity | implemented (C04) | quantitative study planned | “implemented mechanism; quantitative buffer study planned” |
| S05 | Soft similarity limits redundant targets | implemented (C03) | ranking study planned | Methods description; Results only if `measured` |
| S06 | Relative depth not metric range | disclaimer (C15 unsupported) | — | Required; never claim metres |
| S07 | Size–distance uses apparent size | implemented + proxy / software_verification (C11–C12) | — | image-relative bands only |
| S08 | Edge / onboard screening | accepted_abstract_reproduction_pending (C07) + Jetson planned (C14) | no Jetson run | “workstation timing pending reproduction; Jetson planned; not flight-qualified” |
| S09 | Flight qualification | unsupported as current claim | needs HW qualification, independent V&V, mission safety evidence | Prefer: “flight-readiness-oriented development path” / “safety-aware onboard prioritization architecture”; state **not flight-qualified** |
| S10 | Real-ESRGAN safety/quality | unsupported (C13) | no ablation | Methods optional path only |

## Flight-readiness-oriented path (not a current claim)

Keep the long-term goal visible without over-claiming:

- ARTPS is developed as a **flight-readiness-oriented development path** and a **safety-aware onboard prioritization architecture**.
- The system is **not flight-qualified**.
- Closing that gap requires **hardware qualification**, **independent V&V**, and **mission-specific safety evidence**.
- Do **not** claim absolute/universal safety guarantees or present the system as flight-certified in the present tense.

## Discussion checklist

- [ ] Separate `implemented` / `proxy` / `accepted_abstract_reproduction_pending` / `planned` / `unsupported`
- [ ] Failure cases: textureless terrain, severe illumination, depth fallback CNN
- [ ] Explicit relative-depth limit (no metric distance)
- [ ] No present-tense flight-qualified language
