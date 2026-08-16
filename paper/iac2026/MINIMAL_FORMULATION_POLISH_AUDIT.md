# Minimal formulation polish audit

Scope: symbol gloss / notation clarity only. No code, coefficient, metric, figure, or abstract changes.

## Entropy \(p_{ik}\)

- Manuscript formula kept (specification path).
- \(\varepsilon>0\) defined as a small stabilization constant.
- **Construction of \(p_{ik}\) / index \(k\):** UNKNOWN in shipped Python (`src/`, `ARTPS/src/` have no Shannon entropy fusion). No histogram or spatial-probability definition invented.

## Fixed-coefficient fusion

- Coefficients unchanged (`0.50/0.30/0.20/0.08/0.12`, proximity mix, `0.5+0.5`).
- \(A_{\mathrm{pre}}\) product written with \(\odot\); one “All map operations are element-wise.” sentence.
- Matches `src/artps_detection_core.py` element-wise multiply + clip.

## Candidate scoring

- \(V_r\), \(A_r\), \(S_r\) algebra unchanged.
- Gloss added for \(P^{\mathrm{PaDiM}}_r\), \(P^{\mathrm{PC}}_r\), \(D_r\) (higher = farther), \(Q_r\), \(K\) (normalized predicted-class index / 4; not a probability).
- Layer~B PaDiM/PatchCore terms fixed to zero (maps `None` → pool `0`).
- **Coefficient renormalization:** not claimed (code does not renormalize when PaDiM/PC are zero).

## Curiosity

- Formula aligned with `CuriosityScorer` weight-sum normalization + \([0,1]\) clip.
- Letter \(w_i\) reused vs entropy weights but scoped to this subsection (min-diff; no global rename).

## Similarity / Priority Buffer

- \(S'_r\) and buffer predicate unchanged; \(s(r)\) remains maximum cosine similarity.
- No negative-cosine clamp or code-semantics change.
