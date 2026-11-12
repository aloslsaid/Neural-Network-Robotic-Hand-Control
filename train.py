from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "puma560_dataset.csv"
OUTPUT_DIR = ROOT / "artifacts"
SEED = 42
EPOCHS = 600
PATIENCE = 50
BATCH_SIZE = 256


class RobotNN(nn.Module):
    def __init__(self, input_size: int = 6, output_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [f"joint_{i}_deg" for i in range(1, 7)]
    target_cols = ["x_m", "y_m", "z_m"]
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[feature_cols + target_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are required.")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_cols].to_numpy(dtype=np.float32)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=SEED
    )

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    y_train = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val_scaled).to(device)

    model = RobotNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:03d} | validation MSE: {val_loss:.6f}")
        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()
    predictions = y_scaler.inverse_transform(pred_scaled)
    mae = float(np.mean(np.abs(predictions - y_test)))
    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "input_size": len(feature_cols),
        "output_size": len(target_cols),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "seed": SEED,
        "validation_mse_scaled": best_loss,
    }, OUTPUT_DIR / "robot_hand_model.pt")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, OUTPUT_DIR / "scalers.joblib")
    pd.DataFrame({
        "actual_x": y_test[:, 0], "predicted_x": predictions[:, 0],
        "actual_y": y_test[:, 1], "predicted_y": predictions[:, 1],
        "actual_z": y_test[:, 2], "predicted_z": predictions[:, 2],
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"Device: {device} | Test MAE: {mae:.6f} m | Test RMSE: {rmse:.6f} m")
    print(f"Artifacts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
