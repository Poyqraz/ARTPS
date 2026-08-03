# ARTPS: Autonomous Rover Target Prioritization System

[![DOI](https://zenodo.org/badge/DOI/10.13140/RG.2.2.12215.18088.svg)](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ARTPS (Autonomous Rover Target Prioritization System) is a hybrid AI system that combines monocular relative depth ordering, anomaly detection, and learnable curiosity scoring for autonomous exploration of planetary surfaces.

**Author:** Poyraz BAYDEMİR  
**Affiliation:** Selçuk University  
**Published:** July 25, 2025

### Key Features
- **Convolutional Autoencoder**: Compresses and reconstructs Mars rock images
- **DPT_Large relative depth ordering**: Monocular within-image depth maps (local weights supported; not metric distance)
- **Anomaly Detection**: PaDiM, PatchCore, and reconstruction-error scoring
- **Curiosity Score**: Combines exploitation and exploration signals
- **Streamlit UI**: TR/EN bilingual web interface at repository root

## Installation

### Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Setup
```bash
git clone https://github.com/Poyqraz/ARTPS.git
cd ARTPS
pip install -r requirements.txt
```

### Local DPT_Large weights (optional, not in git)
Place MiDaS DPT_Large state_dict at `raw_models/dpt_large_384.pt` (~1.3 GB).  
See [`raw_models/README.md`](raw_models/README.md). Requires `timm` (listed in `requirements.txt`).  
Without this file, relative depth falls back to a lightweight CNN.

### Depth output semantics
ARTPS uses monocular relative depth, not metric distance. In the UI, `0=Near` and `1=Far` only describe ordering within the same image. They should not be interpreted as calibrated meters or compared directly across unrelated images. When the app falls back to the lightweight CNN depth model, near/far cues remain usable as a rough prior but are lower-confidence than `DPT_Large`. Size–distance proposal features use **apparent size** and **image-relative near/far categorization** (`size_distance_band`), not metric size or absolute range.

Trained project weights (`results/*.pth`) and image datasets (`mars_images/`) are also kept locally and excluded from git.

## Usage

Run all commands from the **repository root** (where `app.py` lives).

### Main application (Streamlit)
```bash
streamlit run app.py
```

### Tests and demos
```bash
python test_working_autoencoder.py
python demo_artps.py
python scripts/verify_i18n.py
```

## Project Structure
```
├── app.py                 # Active Streamlit application (root)
├── src/                   # Models, UI (i18n, theme), data utilities
├── assets/                # UI theme assets
├── raw_models/            # Local DPT weights (gitignored; README only in git)
├── results/               # Trained weights and generated outputs (gitignored)
├── scripts/               # Paper figures, benchmarks, i18n verification
├── ARTPS/                 # Legacy duplicate copy (not the active app)
└── README.md
```

## Publications

- **ResearchGate**: [DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
- **Zenodo**: [Archive](https://zenodo.org/records/16943794)
- **ArXiv**: Preprint on hold

## Keywords

*Mars rover*, *Autonomous Exploration*, *relative depth ordering*, *apparent size*, *Vision Transformers*, *planetary surfaces*, *machine learning*, *Anomaly Detection*, *Computer Vision*

## Citation

```bibtex
@article{baydemir2025artps,
  title={ARTPS: Depth-Enhanced Hybrid Anomaly Detection and Learnable Curiosity Score for Autonomous Rover Target Prioritization},
  author={Baydemir, Poyraz},
  year={2025},
  doi={10.13140/RG.2.2.12215.18088}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
