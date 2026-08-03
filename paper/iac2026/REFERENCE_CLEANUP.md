# Reference cleanup plan

Working bibliography: [`references.bib`](references.bib).

**Scope of this milestone:** `references.bib` is a **minimal skeleton bibliography for stub builds only**. It is **not** a camera-ready reference list.

## Keep (methods-relevant skeleton keys)

| Key | Why |
|-----|-----|
| `estlin2012` (and/or Estlin ISAIRAS 2014 when added) | Rover autonomy / target prioritization context |
| `ranftl2021dpt` | Monocular dense depth / DPT lineage |
| `defard2020padim` | PaDiM baseline |
| `roth2022patchcore` | PatchCore baseline |

## TODO before camera-ready (do not treat as final)

- [ ] **`estlin2012` is not camera-ready:** current entry uses “Tara and others” with incomplete metadata. Expand full author list, pages/venue details, and DOI or report URL before final.
- [ ] Complete author lists, pages, and DOI/report metadata for every cited key.
- [ ] Drop any key not cited in the prose.
- [ ] Prefer numbered appearance order (`unsrtnat`) per IAC-style guidelines.

## Drop / do not re-add from 32p dump

Unused in ARTPS method path (examples): BERT, LSTM-as-core-method, generic GAN surveys, industrial laundry citations not cited in text, duplicate MiDaS papers if DPT citation covers relative depth.

## Process

1. When prose lands, cite only keys that appear in `\cite{...}`.
2. Run an unused-bib check before camera-ready.
3. Do not cite sources solely to inflate Related Work.
