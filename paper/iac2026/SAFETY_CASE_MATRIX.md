# Safety-case evidence matrix

Operational safety and flight-readiness wording for Discussion. Claims must match [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md) support levels.

| ID | Safety / ops claim | Evidence | Gap | Allowed manuscript wording |
|----|--------------------|----------|-----|----------------------------|
| S01 | Rover-body FP suppression | Code + shadow proxy (C09) | Proxy ≠ human bbox | “proxy reduction of rover/shadow FPs on curated set” |
| S02 | Boundary-shadow FP suppression | C09 | same | same |
| S03 | Object-in-shadow not erased | C10 rock-loss proxy | small n | “proxy shadow-rock loss; needs GT confirmation” |
| S04 | Priority Buffer reduces FN from diversity | Code path (C04) | no dedicated measured table yet | “mechanism described; quantitative buffer study planned” |
| S05 | Soft similarity limits redundant targets | Code (C03) | ranking study planned | Methods description; Results only if measured |
| S06 | Relative depth not metric range | depth_semantics + DEPTH todo | — | Required disclaimer; never claim metres |
| S07 | Size–distance uses apparent size | C11–C12 | proxy / software_verification | image-relative bands only |
| S08 | Edge / onboard screening | C07 workstation; C14 Jetson planned | no Jetson run; not flight HW | “workstation / planned Jetson profiles; not flight certification” |
| S09 | Flight-readiness | — | no flight V&V | **Do not claim** flight-ready; discuss suitability limits only |
| S10 | Real-ESRGAN safety/quality | C13 unsupported | no ablation | Methods optional path only |

## Discussion checklist

- [ ] Separate measured / proxy / planned in prose
- [ ] Failure cases: textureless terrain, severe illumination, depth fallback CNN
- [ ] Explicit relative-depth limit paragraph
- [ ] No flight certification language
