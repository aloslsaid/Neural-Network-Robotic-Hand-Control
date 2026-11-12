# Neural Network Robotic Hand Control

A PyTorch regression experiment that maps six PUMA 560 joint-angle inputs to three Cartesian position outputs. The repository includes the `puma560_dataset.csv` dataset and a maintained training entry point.

## Model

- Six input joint angles: `joint_1_deg` through `joint_6_deg`.
- Three output coordinates: `x_m`, `y_m`, `z_m`.
- Feed-forward neural network with ReLU activations.
- Train/validation/test separation.
- Standardization fitted only on training data.
- Validation-based checkpoint selection and early stopping.
- CPU or CUDA execution when available.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

## Outputs

Generated locally in `artifacts/`:

- `robot_hand_model.pt`: model checkpoint and feature metadata.
- `scalers.joblib`: fitted input and output scalers.
- `test_predictions.csv`: held-out actual/predicted coordinates.

The printed MAE and RMSE are measured in metres on the held-out test set. Do not publish a performance claim without the dataset version and actual run output.

## Safety

This is a dataset-based learning experiment, not a validated robot controller. Never connect the model directly to physical actuators or a prosthetic device. Real hardware requires independent kinematic validation, bounded commands, collision checks, fail-safe behavior, emergency stop controls and supervised testing.

## Reproducibility

The maintained entry point fixes the random seed and stores the preprocessing scalers with the checkpoint. Generated artifacts are excluded from Git.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
