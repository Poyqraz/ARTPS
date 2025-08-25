# ARTPS: Autonomous Rover Target Prioritization System

[![DOI](https://zenodo.org/badge/DOI/10.13140/RG.2.2.12215.18088.svg)](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ARTPS (Autonomous Rover Target Prioritization System) is a novel hybrid AI system that combines depth estimation, anomaly detection, and learnable curiosity scoring for autonomous exploration of planetary surfaces.

**Author:** Poyraz BAYDEMİR  
**Affiliation:** Selçuk University  
**Published:** July 24, 2025

### 🎯 Key Features
- ✅ **Convolutional Autoencoder Model**: Compresses and reconstructs Mars rock images
- ✅ **Anomaly Detection**: Detects abnormal targets using reconstruction error
- ✅ **Feature Extraction**: Extracts color, texture, and histogram features
- ✅ **Curiosity Score**: Combines exploitation and exploration scores
- ✅ **Visualization**: Presents results visually
- ✅ **Model Persistence**: Saves and loads trained models

## System Architecture

### Curiosity Score
The heart of the system is the "Curiosity Score" calculated for each potential target:

1. **Known Value Score (Exploitation)**: Similarity to targets previously proven scientifically valuable
2. **Anomaly/Discovery Score (Exploration)**: Detection of previously unseen, unusual targets

### Technical Roadmap
- ✅ **Data Collection**: Perseverance and Curiosity data from NASA PDS
- ✅ **Target Detection**: Automatic segmentation (tested with synthetic data)
- ✅ **Feature Extraction**: Color, texture, shape vectors
- ✅ **Modeling**: Classification and Anomaly Detection (with Autoencoder)
- ✅ **Scoring**: Combination of two modules

### 🔄 Next Steps
- [ ] Integration of real NASA PDS data
- [ ] Advanced segmentation algorithms
- [ ] More sophisticated exploitation scoring
- [ ] Real-time rover integration

## Installation

### Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Installation Steps
```bash
# 1. Clone the project
git clone <repository-url>
cd ARTPS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test
python test_working_autoencoder.py
```

## Usage

### 1. Model Training and Testing
```bash
python test_working_autoencoder.py
```

### 2. Running Demo
```bash
python demo_artps.py
```

### 3. Manual Model Training
```bash
python src/models/working_autoencoder.py
```

## Project Structure
```
project_mars/
├── src/
│   ├── models/
│   │   ├── working_autoencoder.py    # Working autoencoder model
│   │   └── __init__.py
│   ├── utils/
│   │   ├── data_utils.py             # Data processing and feature extraction
│   │   └── __init__.py
│   └── __init__.py
├── data/
│   └── mars_rocks/                   # Mars rock images
├── results/                          # Trained models and results
├── test_working_autoencoder.py       # Model test script
├── demo_artps.py                     # Demo script
├── requirements.txt                  # Project dependencies
└── README.md
```

## Publications

- **ResearchGate**: [DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
- **Zenodo**: [Archive](https://zenodo.org/records/16943794)
- **ArXiv**: [Preprint](https://arxiv.org/abs/XXXX.XXXXX)

## Keywords

*Mars rover*, *Autonomous Exploration*, *depth estimation*, *Vision Transformers*, *planetary surfaces*, *machine learning*, *Anomaly Detection*, *Computer Vision*

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
