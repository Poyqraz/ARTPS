# Reference cleanup plan

Working bibliography: [`references.bib`](references.bib).

Camera-ready (this PR): four cited keys only; metadata verified from publisher / DBLP / CaltechAUTHORS / CVF / Springer. No unused keys. No fabricated DOIs.

## Cited keys

| Key | Status | Source |
|-----|--------|--------|
| `estlin2012` | **Verified.** ACM TIST 3(3):50:1–50:19 (2012), DOI `10.1145/2168752.2168764`. Cite key kept; entry replaced (old title was ICRA 2007 OASIS, mis-labeled IEEE Aerospace 2012). | CaltechAUTHORS, DBLP, DOI |
| `defard2020padim` | **Verified.** LNCS 12664, ICPR Workshops, pp. 475–489, DOI `10.1007/978-3-030-68799-1_35` (2021 proceedings). | Springer / HAL / Crossref |
| `roth2022patchcore` | **Verified.** CVPR 2022, pp. 14318–14328, DOI `10.1109/CVPR52688.2022.01392`. | CVF Open Access, IEEE |
| `ranftl2021dpt` | **Verified.** ICCV 2021, pp. 12179–12188, DOI `10.1109/ICCV48922.2021.01196`. | IEEE |

## UNVERIFIED_METADATA (not added)

| Candidate | Why deferred |
|-----------|----------------|
| Estlin et al., *Automated Targeting for the MSL Rover ChemCam Spectrometer*, ISAIRAS 2014 | JPL PDF exists; no DOI / page range established. TIST 2012 is the peer-reviewed AEGIS primary used in Related Work. |

## Drop / do not re-add from 32p dump

Unused in ARTPS method path (examples): BERT, LSTM-as-core-method, generic GAN surveys, industrial laundry citations not cited in text, duplicate MiDaS papers if DPT citation covers relative depth.

## Process

1. Cite only keys that appear in `\cite{...}` / `\citep{...}`.
2. Unused-bib check: currently 0 unused.
3. Do not cite sources solely to inflate Related Work.
4. Prefer numbered appearance order (`unsrtnat`) per IAC-style guidelines.
