# ARTPS Code Repository

This folder contains the source code for the ARTPS (Autonomous Rover Target Prioritization System) project.

## Structure

```
ARTPS/
├── src/
│   ├── models/          # AI models (autoencoder, etc.)
│   ├── utils/           # Utility functions
│   └── __init__.py      # Package initialization
├── data/                # Data storage
├── results/             # Model outputs and results
├── requirements.txt     # Python dependencies
├── test_*.py           # Test scripts
├── demo_*.py           # Demo scripts
└── README.md           # This file
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
python test_working_autoencoder.py
```

3. Run demo:
```bash
python demo_artps.py
```

## Main Components

- **Models**: Convolutional autoencoder for anomaly detection
- **Utils**: Data processing and feature extraction functions
- **Scripts**: Training, testing, and demonstration scripts

## License

MIT License - see main repository for details.
