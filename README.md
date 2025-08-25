# ARTPS: Autonomous Rover Target Prioritization System

[![DOI](https://zenodo.org/badge/DOI/10.13140/RG.2.2.12215.10000.svg)](https://doi.org/10.13140/RG.2.2.12215.10000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ARTPS (Autonomous Rover Target Prioritization System) is a novel hybrid AI system that combines depth estimation, anomaly detection, and learnable curiosity scoring for autonomous exploration of planetary surfaces.

## Key Features

- **Hybrid AI System**: Combines depth estimation with anomaly detection
- **Vision Transformers**: Monocular depth estimation using ViT
- **Multi-component Fusion**: Image and depth cue integration
- **Learnable Curiosity Score**: Balancing novelty and known value
- **Real-time Performance**: Optimized for edge computing constraints

## Performance

- **AUROC**: 0.894
- **AUPRC**: 0.847  
- **F1-Score**: 0.823
- **False Positive Rate**: 0.089

## Publications

- **Paper**: [PDF](docs/paper.pdf) | [LaTeX](docs/paper.tex)
- **ResearchGate**: [DOI](https://doi.org/10.13140/RG.2.2.12215.10000)
- **Zenodo**: [Archive](https://zenodo.org/record/XXXXXXX)

## Installation

```bash
git clone https://github.com/yourusername/project_mars.git
cd project_mars
pip install -r requirements.txt
```

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
