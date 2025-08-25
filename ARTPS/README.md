# ARTPS Code Repository

This folder contains the source code for the ARTPS (Autonomous Rover Target Prioritization System) project.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python test_working_autoencoder.py
```

### 3. Run Demo
```bash
python demo_artps.py
```

### 4. 🎯 **Run Main Application (Streamlit Web UI)**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
ARTPS/
├── src/
│   ├── models/          # AI models (7 models)
│   │   ├── autoencoder.py              # Basic autoencoder
│   │   ├── working_autoencoder.py      # Working autoencoder
│   │   ├── simple_autoencoder.py       # Simple autoencoder
│   │   ├── optimized_autoencoder.py    # Optimized autoencoder
│   │   ├── depth_estimation.py         # Depth estimation (DPT_Large)
│   │   ├── depth_enhanced_classifier.py # Depth-enhanced classifier
│   │   └── anomaly/                     # Anomaly detection models
│   │       ├── base.py                  # Base anomaly class
│   │       ├── padim.py                 # PaDiM model
│   │       └── patchcore.py             # PatchCore model
│   ├── utils/           # Utility functions
│   │   ├── data_utils.py               # Data processing
│   │   └── image_enhancement.py        # Image enhancement
│   ├── core/            # Core system components
│   │   └── curiosity_scorer.py         # Curiosity scoring system
│   └── __init__.py      # Package initialization
├── data/                # Data storage
├── results/             # Model outputs and results
├── requirements.txt     # Python dependencies
├── app.py              # 🎯 MAIN APPLICATION (Streamlit Web UI)
├── demo_artps.py       # Demo script
├── test_*.py           # Test scripts (6 files)
├── train_*.py          # Training scripts (1 file)
└── README.md           # This file
```

## 🎯 Main Application: `app.py`

### **What is `app.py`?**
`app.py` is the **main Streamlit web application** that provides a complete interactive interface for the ARTPS system.

### **Key Features:**
- **🚀 Interactive Web UI** - User-friendly interface
- **🤖 Hybrid AI Models** - Autoencoder + Depth + Classifier
- **📊 Real-time Analysis** - Instant Mars image analysis
- **🎛️ Parameter Control** - Adjustable curiosity scoring
- **📈 Visualization** - 3D depth maps, anomaly detection
- **🔄 Model Management** - Load and manage trained models

### **How to Use `app.py`:**

#### **1. Launch the Application:**
```bash
cd ARTPS
streamlit run app.py
```

#### **2. Main Interface Features:**
- **📁 Image Upload** - Upload Mars rover images
- **🎛️ Parameter Control** - Adjust α (alpha) and β (beta) weights
- **🔍 Hybrid Analysis** - Run complete anomaly + depth analysis
- **📊 Results Display** - View curiosity scores and visualizations
- **🎮 Demo Mode** - Test with sample images

#### **3. Key Parameters:**
- **α (Alpha)**: Known value weight (0.0-1.0)
- **β (Beta)**: Anomaly weight (0.0-1.0)
- **Anomaly Threshold**: Detection sensitivity
- **Depth Analysis**: 3D surface analysis

#### **4. Analysis Results:**
- **Anomaly Score**: Reconstruction error-based detection
- **Known Value Score**: Scientific value classification
- **Curiosity Score**: Combined weighted score
- **Depth Visualization**: 3D surface maps
- **Anomaly Maps**: Detection heatmaps

## 🧠 AI Models

### **Core Models:**
- **Convolutional Autoencoder**: Anomaly detection (17M parameters)
- **Depth-Enhanced Classifier**: Dynamic value scoring
- **DPT_Large**: High-accuracy depth estimation
- **PaDiM**: Patch-based anomaly detection
- **PatchCore**: Advanced anomaly detection

### **Model Performance:**
- **Anomaly Detection**: 95%+ accuracy
- **Classification**: 74% accuracy
- **Depth Estimation**: High-accuracy DPT_Large
- **Real-time**: <1 second analysis time

## 🔧 Technical Requirements

### **System Requirements:**
- Python 3.8+
- CUDA-enabled GPU (recommended)
- 8GB+ RAM
- 2GB+ free disk space

### **Dependencies:**
- **PyTorch 2.0+** - Deep learning framework
- **OpenCV 4.5+** - Computer vision
- **Streamlit 1.28+** - Web interface
- **Matplotlib/Plotly** - Visualization
- **NumPy/SciPy** - Numerical computing

## 📚 Usage Examples

### **Basic Testing:**
```bash
# Test autoencoder
python test_working_autoencoder.py

# Run demo
python demo_artps.py

# Launch main app
streamlit run app.py
```

### **Model Training:**
```bash
# Train curiosity model
python train_curiosity.py
```

### **Advanced Analysis:**
```bash
# Test extended model
python test_extended_model.py

# System testing
python test_artps_system.py
```

### **Individual Model Testing:**
```bash
# Test different autoencoder variants
python test_simple_autoencoder.py
python test_autoencoder.py

# Test depth estimation
python test_artps_system.py
```

## 🌟 Key Features

- **✅ Hybrid AI System** - Multiple model fusion
- **✅ Real-time Analysis** - Instant results
- **✅ Interactive UI** - User-friendly interface
- **✅ 3D Visualization** - Depth analysis
- **✅ Parameter Control** - Adjustable scoring
- **✅ Model Persistence** - Save/load trained models
- **✅ Comprehensive Testing** - Full test suite

## 🎮 Interactive Features

### **Streamlit Web Interface:**
- **Real-time Image Analysis** - Upload and analyze Mars images instantly
- **Parameter Tuning** - Adjust curiosity scoring weights interactively
- **3D Visualization** - Explore depth maps and surface analysis
- **Anomaly Detection** - View detection heatmaps and scores
- **Model Status** - Monitor loaded models and their performance

### **Demo Mode:**
- **Sample Images** - Test with pre-loaded Mars rover images
- **Quick Analysis** - One-click analysis with default parameters
- **Result Comparison** - Compare different parameter settings

## 🔬 Technical Details

### **Curiosity Score Calculation:**
```
Curiosity Score = α × Known Value Score + β × Anomaly Score
```

### **Model Architecture:**
- **Input**: RGB Mars images (256×256×3)
- **Autoencoder**: Convolutional layers with 17M parameters
- **Depth Estimation**: DPT_Large transformer model
- **Classifier**: RGB + Depth feature fusion
- **Output**: Anomaly scores, depth maps, curiosity scores

### **Performance Metrics:**
- **Inference Time**: <1 second per image
- **Memory Usage**: ~2GB GPU memory
- **Accuracy**: 95%+ anomaly detection, 74% classification

## 🚀 Next Steps

- [ ] Real NASA PDS data integration
- [ ] Advanced segmentation algorithms
- [ ] Real-time rover integration
- [ ] Multi-rover support
- [ ] Space station integration

## 📄 License

MIT License - see main repository for details.

## 🔗 Related Files

- **Main App**: `app.py` - Complete Streamlit application
- **Demo**: `demo_artps.py` - Simple demonstration
- **Tests**: `test_*.py` - Comprehensive testing suite
- **Models**: `src/models/` - AI model implementations
- **Utils**: `src/utils/` - Utility functions
- **Core**: `src/core/` - Core system components

## 📞 Support

For questions or issues:
1. Check the main repository README.md
2. Review the test scripts for examples
3. Ensure all dependencies are installed
4. Verify model files are in the results/ folder
