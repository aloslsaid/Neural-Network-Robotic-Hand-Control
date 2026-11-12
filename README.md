# Neural Network Robotic Hand Control

A PyTorch regression experiment that maps six PUMA560 joint angles to an end-effector position (x, y, z).

## Repository files
- `robot hand`: Python training script (extensionless filename in the current repository).
- `puma560_dataset.csv`: source dataset with joint-angle inputs and Cartesian position targets.
- `requirements.txt`: Python dependencies.
- `artifacts/`: generated model weights and test predictions (created when the script runs).

## Setup

Python 3.10+ is recommended.

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python "robot hand"
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python "robot hand"
```

The script selects CUDA when PyTorch detects a compatible GPU; otherwise it uses CPU. It trains a feed-forward network, keeps the best validation checkpoint, evaluates on a held-out test split, and writes `artifacts/robot_nn.pth` and `artifacts/robot_predictions.csv`.

## Data and evaluation
Inputs are `joint_1_deg` through `joint_6_deg`; targets are `x_m`, `y_m`, and `z_m`. Feature and target scalers are fitted on training data only. The reported MAE/RMSE are calculated after inverse-transforming predictions to meters. Re-run the script to reproduce metrics for the current dependency versions and dataset.

## Safety
This is an offline learning experiment, not a validated robot controller. Do not connect the generated model directly to physical hardware. Real-world use requires independent kinematic validation, joint/velocity/torque limits, collision checks, emergency stops, and controlled testing.

## Limitations
The training script is a standalone experiment rather than a reusable inference package. The saved state dictionary must be loaded with the same model architecture and preprocessing scalers; this script currently saves the weights but not the fitted scalers.
