# ARTPS: Autonomous Rover Target Prioritization System

[![DOI](https://zenodo.org/badge/DOI/10.13140/RG.2.2.12215.18088.svg)](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://doi.org/10.5281/zenodo.16947598"><img src="https://zenodo.org/badge/1044467612.svg" alt="DOI"></a>


## Overview

ARTPS (Autonomous Rover Target Prioritization System) is a novel hybrid AI system that combines depth estimation, anomaly detection, and learnable curiosity scoring for autonomous exploration of planetary surfaces.

**Author:** Poyraz BAYDEMİR  
**Affiliation:** Selçuk University  
**Published:** July 25, 2025

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
- [ ] Web interface development

## Installation

### Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Installation Steps
```bash
# 1. Clone the project
git clone <repository-url>

# 2. Navigate to ARTPS folder
cd ARTPS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test
python test_working_autoencoder.py
```

## Usage

**Note:** All commands should be run from the `ARTPS/` folder.

### 1. Model Training and Testing
```bash
cd ARTPS
python test_working_autoencoder.py
```

### 2. Running Demo
```bash
cd ARTPS
python demo_artps.py
```

### 3. 🎯 **Run Main Application (Streamlit Web UI)**
```bash
cd ARTPS
streamlit run app.py
```

### 4. Manual Model Training
```bash
cd ARTPS
python src/models/working_autoencoder.py
```

### 5. Available Scripts
- **🎯 Main Application:** `app.py` - Complete Streamlit web interface
- **Test Scripts:** `test_*.py` (6 files)
- **Training Scripts:** `train_*.py` (1 file)
- **Demo Scripts:** `demo_artps.py`

## Project Structure
```

├── ARTPS/                           # Main source code repository
│   ├── src/                         # Source code package
│   │   ├── models/                  # AI models (7 models)
│   │   │   ├── working_autoencoder.py
│   │   │   ├── depth_estimation.py
│   │   │   ├── optimized_autoencoder.py
│   │   │   ├── anomaly/             # Anomaly detection models
│   │   │   └── __init__.py
│   │   ├── utils/                   # Utility functions
│   │   │   ├── data_utils.py
│   │   │   └── __init__.py
│   │   ├── core/                    # Core system components
│   │   └── __init__.py
│   ├── data/                        # Data storage
│   ├── results/                     # Model outputs and results
│   ├── requirements.txt             # Python dependencies
│   ├── app.py                       # 🎯 MAIN APPLICATION (Streamlit Web UI)
│   ├── test_*.py                    # Test scripts (6 files)
│   ├── demo_artps.py               # Demo script
│   ├── train_*.py                   # Training scripts (1 file)
│   └── README.md                    # Code documentation
└── README.md                        # Main project documentation
```

## Publications

- **ResearchGate**: [DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
- **Zenodo**: [Archive](https://zenodo.org/records/16943794)
- **ArXiv**: [Preprint](on Hold)

## Code Repository

The complete source code is available in the [`ARTPS/`](ARTPS/) folder, which includes:

- **🎯 Main Application:** `app.py` - Complete Streamlit web interface with hybrid AI models
- **AI Models:** 7 different autoencoder and depth estimation models
- **Utility Functions:** Data processing and feature extraction
- **Core Components:** Curiosity scoring and anomaly detection systems
- **Test Scripts:** Comprehensive testing suite (6 files)
- **Training Scripts:** Model training and optimization (1 file)
- **Demo Application:** Interactive demonstration system
- **Documentation:** Detailed code documentation

See [`ARTPS/README.md`](ARTPS/README.md) for detailed code documentation and usage instructions.

### Sample Figures
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/bf05e884-7187-4969-857c-967cd154867f" width="300"/><br>
      <sub>Anomaly Detection Result</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5312923e-0356-4840-a335-76da61569478" width="300"/><br>
      <sub>Curiosity: Ripple and Outcrop Depth Map</sub>
    </td>
  </tr>
</table>

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
