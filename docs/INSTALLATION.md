# Installation Guide

## System Requirements

- Python 3.8 or higher
- CoppeliaSim EDU (optional, for real simulation)
- 8GB RAM minimum
- Multi-core processor recommended

## Python Dependencies

### Core Requirements
```bash
pip install numpy>=1.20.0
pip install matplotlib>=3.5.0  
pip install pandas>=1.3.0
pip install scipy>=1.7.0
pip install scikit-learn>=1.0.0
pip install seaborn>=0.11.0
pip install gymnasium>=0.26.0
```

### CoppeliaSim Integration (Optional)
```bash
pip install coppeliasim-zmqremoteapi-client>=2.0.0
```

## CoppeliaSim Setup (Optional)

1. Download CoppeliaSim EDU from: https://coppeliarobotics.com/
2. Install following the platform-specific instructions
3. Load the provided scene file: `ur5WithRg2Grasping-python_30.08.2025.ttt`

## Verification

Run the system test to verify installation:
```bash
python src/test_morl_system.py
```

Expected output: "ALL TESTS PASSED - READY FOR PAPER VALIDATION"

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies using pip
2. **CoppeliaSim connection fails**: Ensure CoppeliaSim is running and scene is loaded
3. **Visualization errors**: Update matplotlib to latest version
4. **Memory issues**: Close other applications during large experiments

### Platform-Specific Notes

#### Windows
- Use PowerShell or Command Prompt
- Ensure Python is in PATH
- May require Visual C++ redistributables

#### Linux/macOS
- Use terminal
- May require `python3` instead of `python`
- Install system dependencies if needed