import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- CONFIG ----------------

CLINIC_FILE = "data/clinic_implant_level.xlsx"
RADIOMICS_FILE = "data/radiomics_features.xlsx"

MODELS_DIR = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- ARCHITECTURES ----------------

class ClinicalNet(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

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


class RadiomicsNet(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- DATA LOADERS ----------------

def load_clinical_data():

    df = pd.read_excel(CLINIC_FILE)

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

    X = pd.get_dummies(X)

    MODEL_COLUMNS = [
        'Gender', 'Smoking', 'Alcohol', 'Diabetes', 'Hypertension',
        'FOV_full jaw', 'FOV_lower jaw', 'FOV_lower jaw only', 'FOV_narrowFOV',
        'FOV_partial jaw – right side', 'FOV_partial mandible (posterior)', 'FOV_posterior',
        'FOV_upper jaw', 'FOV_upper jaw only',
        'years_placed_ 1 YEAR ', 'years_placed_ 1.5 YEARS ', 'years_placed_ 10 MONTHS ',
        'years_placed_ 11 MONTHS ', 'years_placed_ 3 MONTHS ', 'years_placed_ 4 MONTHS ',
        'years_placed_ 5 MONTHS ', 'years_placed_ 6 MONTHS ', 'years_placed_ 6 MONTHS ',
        'years_placed_ 7 MONTHS ', 'years_placed_ 8 MONTHS ', 'years_placed_ 9 MONTHS ',
        'years_placed_1 YEAR',
        'bone density Misch_D3'
    ]

    X = X.reindex(columns=MODEL_COLUMNS, fill_value=0)

    X = X.fillna(X.mean())

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    _, X_test, _, y_test = train_test_split(
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


def load_hybrid_data():

    df_c = pd.read_excel(CLINIC_FILE)
    df_r = pd.read_excel(RADIOMICS_FILE)

    df_c['implant_id'] = df_c['implant_id'].astype(str)
    df_r['implant_id'] = df_r['implant_id'].astype(str)

    df = pd.merge(df_c, df_r, on='implant_id', how='inner')

    TARGET = 'survival years predicted'

    df = df.dropna(subset=[TARGET])

    ignore_meta = [
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

    clinical_cols = [
        c for c in df.columns
        if not c.startswith('original_')
        and c not in ignore_meta
    ]

    golden_radiomics = [
        'original_glcm_ClusterShade',
        'original_firstorder_Skewness',
        'original_glszm_SmallAreaLowGrayLevelEmphasis',
        'original_glszm_ZoneEntropy',
        'original_shape_MinorAxisLength'
    ]

    valid_radiomics = [
        c for c in golden_radiomics
        if c in df.columns
    ]

    final_features = clinical_cols + valid_radiomics

    X = df[final_features]

    y = df[TARGET]

    X = pd.get_dummies(X)

    HYBRID_COLUMNS = list(X.columns)

    X = X.reindex(columns=HYBRID_COLUMNS, fill_value=0)
    X = X.fillna(X.mean())

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    _, X_test, _, y_test = train_test_split(
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


def load_radiomics_data():

    df_c = pd.read_excel(CLINIC_FILE)
    df_r = pd.read_excel(RADIOMICS_FILE)

    df_c['implant_id'] = df_c['implant_id'].astype(str)
    df_r['implant_id'] = df_r['implant_id'].astype(str)

    df = pd.merge(df_c, df_r, on='implant_id', how='inner')

    TARGET = 'survival years predicted'

    df = df.dropna(subset=[TARGET])

    radiomics_cols = [
        c for c in df.columns
        if c.startswith('original_')
    ]

    X = df[radiomics_cols]

    y = df[TARGET]

    X = X.fillna(X.mean())

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    _, X_test, _, y_test = train_test_split(
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

def evaluate_model(model_name):

    if model_name == "best_clinical_only_model.pth":

        X_test, y_true, input_dim = load_clinical_data()

        model = ClinicalNet(input_dim).to(device)

    elif model_name == "best_hybrid_elite_model.pth":

        X_test, y_true, input_dim = load_hybrid_data()

        model = ClinicalNet(input_dim).to(device)

    elif model_name == "best_radiomics_only_model.pth":

        X_test, y_true, input_dim = load_radiomics_data()

        model = RadiomicsNet(input_dim).to(device)

    else:

        print(f"Skipping unsupported model: {model_name}")

        return

    model_path = os.path.join(MODELS_DIR, model_name)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    with torch.no_grad():

        predictions = model(X_test)

        y_pred = predictions.cpu().numpy()

        r2 = r2_score(y_true, y_pred)

        mae = mean_absolute_error(y_true, y_pred)

    with mlflow.start_run(run_name=model_name):

        mlflow.log_metric("R2 Score", float(r2))

        mlflow.log_metric("MAE", float(mae))

        mlflow.log_param("model_name", model_name)

        mlflow.log_artifact(model_path)

        print(f"\n{model_name}")

        print(f"R2 Score: {r2:.4f}")

        print(f"MAE: {mae:.4f}")


def main():

    mlflow.set_experiment("Dental_Implant_Models")

    models = [

        "best_clinical_only_model.pth",

        "best_hybrid_elite_model.pth",

        "best_radiomics_only_model.pth"
    ]

    for model_name in models:

        evaluate_model(model_name)


if __name__ == "__main__":
    main()
