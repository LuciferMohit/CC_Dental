import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- CONFIG ----------------

DATA_FILE = "data/clinic_implant_level.xlsx"
MODELS_DIR = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------

class SurvivalNet(nn.Module):
    def __init__(self, input_dim):
        super(SurvivalNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- DATA ----------------

def load_data():

    df = pd.read_excel(DATA_FILE)

    TARGET = 'survival years predicted'

    df = df.dropna(subset=[TARGET])

    ignore = [
        'implant_id',
        'scan_id',
        'patient_name',
        'file',
        'label',
        'dimensions',
        'spacing',
        'slice_count',
        'StudyDate_ISO',
        'roi_path',
        'x_mm',
        'y_mm',
        'z_mm',
        TARGET
    ]

    X = df.drop(columns=[c for c in ignore if c in df.columns])
    y = df[TARGET]

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        torch.tensor(X_test, dtype=torch.float32).to(device),
        y_test.values,
        X.shape[1]
    )

# ---------------- EVALUATION ----------------

def evaluate_models():

    X_test, y_true, input_dim = load_data()

    mlflow.set_experiment("Dental_Implant_Models")

    for model_file in os.listdir(MODELS_DIR):

        if model_file.endswith(".pth"):

            model_path = os.path.join(MODELS_DIR, model_file)

            print(f"\nEvaluating: {model_file}")

            checkpoint = torch.load(model_path, map_location=device)

            print(checkpoint.keys())

            break

            model.eval()

            with torch.no_grad():

                predictions = model(X_test)

                y_pred = predictions.cpu().numpy()

                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)

            with mlflow.start_run(run_name=model_file):

                mlflow.log_metric("R2 Score", float(r2))
                mlflow.log_metric("MAE", float(mae))

                mlflow.log_param("model_name", model_file)

                mlflow.log_artifact(model_path)

                print(f"R2 Score: {r2:.4f}")
                print(f"MAE: {mae:.4f}")

# ---------------- MAIN ----------------

if __name__ == "__main__":
    evaluate_models()
