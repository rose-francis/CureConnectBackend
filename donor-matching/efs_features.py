# ============================================================
# efs_features.py
# Feature list and encoding for the event-free survival model,
# shared by training (train_cibmtr.py) and prediction (predict_core.py)
# so both always build the exact same matrix.
# Column names and values follow the CIBMTR dataset.
# ============================================================

import numpy as np
import pandas as pd

FEATURES = [
    # Patient
    'age_at_hct', 'prim_disease_hct', 'karnofsky_score', 'comorbidity_score', 'dri_score',
    # Donor / pairing
    'donor_age', 'cmv_status', 'sex_match', 'graft_type', 'donor_related',
    'hla_match_a_high', 'hla_match_b_high', 'hla_match_c_high',
    'hla_match_drb1_high', 'hla_match_dqb1_high', 'hla_high_res_10',
]

CATEGORICAL = ['prim_disease_hct', 'dri_score', 'cmv_status', 'sex_match',
               'graft_type', 'donor_related']


def to_matrix(df: pd.DataFrame, categories: dict) -> np.ndarray:
    """
    Numeric columns pass through; categorical columns become the index of
    the value in `categories` (learned at training time). Unknown or
    missing values become NaN, which the model handles natively.
    """
    X = pd.DataFrame(index=df.index)
    for col in FEATURES:
        if col in CATEGORICAL:
            codes = pd.Categorical(df[col], categories=categories[col]).codes
            X[col] = np.where(codes < 0, np.nan, codes)
        else:
            X[col] = pd.to_numeric(df[col], errors='coerce')
    return X.to_numpy(dtype=float)


def categorical_mask():
    return [col in CATEGORICAL for col in FEATURES]
