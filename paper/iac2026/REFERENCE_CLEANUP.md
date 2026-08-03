# Reference cleanup plan

Working bibliography: [`references.bib`](references.bib).

## Keep (methods-relevant)

| Key | Why |
|-----|-----|
| `estlin2012` (and/or Estlin ISAIRAS 2014 when added) | Rover autonomy / target prioritization context |
| `ranftl2021dpt` | Monocular dense depth / DPT lineage |
| `defard2020padim` | PaDiM baseline |
| `roth2022patchcore` | PatchCore baseline |

## Drop / do not re-add from 32p dump

Unused in ARTPS method path (examples): BERT, LSTM-as-core-method, generic GAN surveys, industrial laundry citations not cited in text, duplicate MiDaS papers if DPT citation covers relative depth.

## Process

1. When prose lands, cite only keys that appear in `\cite{...}`.
2. Run a unused-bib check before camera-ready.
3. Prefer numbered appearance order (`unsrtnat`) per IAC-style guidelines.
4. Do not cite sources solely to inflate Related Work.
