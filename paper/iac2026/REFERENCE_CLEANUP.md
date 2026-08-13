# Reference cleanup plan

Working bibliography: [`references.bib`](references.bib).

Camera-ready literature strengthening: **12** cited keys; metadata verified from Crossref / IEEE / Wiley / Springer / Science. No unused keys. No fabricated DOIs.

## Cited keys (legacy)

| Key | Status | Source |
|-----|--------|--------|
| `estlin2012` | **Verified.** ACM TIST 3(3):50:1–50:19 (2012), DOI `10.1145/2168752.2168764`. | CaltechAUTHORS, DBLP, DOI |
| `francis2017aegis` | **Verified.** Science Robotics 2(7):eaan4582 (2017), DOI `10.1126/scirobotics.aan4582`. | Science |
| `francis2019utility` | **Verified.** Planetary and Space Science 170:52–60 (2019), DOI `10.1016/j.pss.2019.03.007`. | Elsevier |
| `gaines2020srr` | **Verified.** Journal of Field Robotics 37(7):1171–1196 (2020), DOI `10.1002/rob.21979`. | Wiley |
| `defard2020padim` | **Verified.** LNCS 12664, ICPR Workshops, pp. 475–489, DOI `10.1007/978-3-030-68799-1_35` (2021 proceedings). | Springer / Crossref |
| `roth2022patchcore` | **Verified.** CVPR 2022, pp. 14318–14328, DOI `10.1109/CVPR52688.2022.01392`. | CVF Open Access, IEEE |
| `ranftl2021dpt` | **Verified.** ICCV 2021, pp. 12179–12188, DOI `10.1109/ICCV48922.2021.01196`. | IEEE |

## Cited keys (literature strengthening)

| Key | Status | Source |
|-----|--------|--------|
| `bergmann2019mvtec` | **Verified.** CVPR 2019, pp. **9584–9592**, DOI `10.1109/CVPR.2019.00982` (IEEE/Crossref; not CVF HTML page scrape). | Crossref / IEEE |
| `ranftl2022robust` | **Verified.** IEEE TPAMI 44(3):1623–1637 (**2022** bibliographic year), DOI `10.1109/TPAMI.2020.3019967`. | Crossref / IEEE |
| `malin2017mastcam` | **Verified.** Earth and Space Science 4(8):506–539 (2017), DOI `10.1002/2016EA000252`. First author normalized to Michael C. Malin (Crossref “Michal” typo corrected). | Crossref / PMC / Wiley |
| `bell2021mastcamz` | **Verified.** Space Science Reviews 217:24 (2021), DOI `10.1007/s11214-020-00755-x`. | Crossref / Springer |
| `verma2023perseverance` | **Verified.** Science Robotics 8(80):eadi3099 (2023), DOI `10.1126/scirobotics.adi3099`. | Crossref / Science |

## UNVERIFIED_METADATA (not added)

| Candidate | Why deferred |
|-----------|----------------|
| Estlin et al., *Automated Targeting for the MSL Rover ChemCam Spectrometer*, ISAIRAS 2014 | JPL PDF exists; no DOI / page range established. TIST 2012 is the peer-reviewed AEGIS primary used in Related Work. |

## Drop / do not re-add from 32p dump

Unused in ARTPS method path (examples): BERT, LSTM-as-core-method, generic GAN surveys, industrial laundry citations not cited in text, duplicate MiDaS papers if DPT citation covers relative depth.

## Process

1. Cite only keys that appear in `\cite{...}` / `\citep{...}`.
2. Unused-bib check: currently 0 unused (12/12 cited).
3. Do not cite sources solely to inflate Related Work.
4. Prefer numbered appearance order (`unsrtnat`) per IAC-style guidelines.
5. Do not add a 13th reference without author approval (`OPTIONAL_REFERENCE_13`: no).
