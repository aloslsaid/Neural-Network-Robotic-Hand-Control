# Neural Network Robotic Hand Control

> A deep-learning experiment that learns the mapping from six robotic joint angles to a three-dimensional end-effector position using PUMA 560 data.

## Project objective
Robotic systems need to translate joint configurations into predictable physical positions. This project explores a neural-network approximation of the forward-kinematics mapping: six joint-angle inputs are used to predict the robot end-effector's X, Y, and Z coordinates.

## Dataset
The repository includes `puma560_dataset.csv`. The training script expects six input columns:
- `joint_1_deg` through `joint_6_deg`

And three output columns:
- `x_m`, `y_m`, `z_m`

The dataset is split into training, validation, and test subsets. Input and output scalers are fitted using training data and then applied to validation/test data.

## Neural-network design
The PyTorch model is intended to be a feed-forward multilayer perceptron:
- Input layer: 6 joint angles.
- Hidden layers: 128, 64, and 32 neurons with ReLU activations.
- Output layer: 3 predicted Cartesian coordinates.

Training uses Adam optimization and mean-squared error. Validation loss is tracked to retain the best model state. Final predictions are transformed back to the original coordinate scale and evaluated using MAE and RMSE in metres.

## Installation
Python 3.10+ is recommended. Install the main dependencies:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn torch
```
For GPU acceleration, install a PyTorch build compatible with the local CUDA environment. CPU execution is also supported by the script's device selection.

## Run
The training script is named `robot hand`:
```bash
python "robot hand"
```
Run it from the repository root so it can locate `puma560_dataset.csv`.

## Current source note
The checked-in script should be reviewed before execution: its model constructor is currently written as `init` rather than Python's required `__init__`. This README intentionally does not modify source code or claim that the present revision has passed a runtime test. The script also saves a model file to a Google Colab-style path; adjust that output path in a local working copy if needed.

## Evaluation and scope
The intended evaluation is positional prediction error on unseen test examples. A project-reported error should always be accompanied by its test split, unit, and evaluation procedure. This is a simulation/data-driven control research exercise; it does not directly control physical hardware and must not be connected to a real robot without independent safety validation.

## Skills demonstrated
PyTorch · Feed-forward neural networks · Robotics · Forward-kinematics approximation · Regression · Feature/target scaling · Validation-based model selection
