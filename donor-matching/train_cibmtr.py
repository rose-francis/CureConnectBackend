# ============================================================
# train_cibmtr.py
# Trains the event-free survival (EFS) model on the CIBMTR dataset
# and saves it to model_efs.pkl.
#
# Data: CIBMTR - Equity in post-HCT Survival Predictions (Kaggle).
# The competition rules forbid redistributing the data, so it is not
# in this repo. Download train.csv from Kaggle and put it in
# donor-matching/cibmtr-data/ (gitignored), or pass --data <path>.
#
# Citation: Tushar Deshpande, Deniz Akdemir, Walter Reade, Ashley Chow,
# Maggie Demkin, and Yung-Tsi Bolon. CIBMTR - Equity in post-HCT Survival
# Predictions. https://kaggle.com/competitions/equity-post-HCT-survival-predictions. Kaggle.
#
# Run: python train_cibmtr.py [--data path/to/train.csv]
# ============================================================

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from efs_features import FEATURES, CATEGORICAL, to_matrix, categorical_mask

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))


def build_model():
    # Shallow trees + low learning rate: enough capacity for ~29k rows
    # without memorising noise
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        categorical_features=categorical_mask(), random_state=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(_HERE, 'cibmtr-data', 'train.csv'))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    # efs: 1 = event (relapse, graft failure or death), 0 = event-free at last follow-up
    y = df['efs'].astype(int).to_numpy()
    print(f"[1] Loaded {len(df)} rows from {args.data}")
    print(f"[1] Event rate: {y.mean():.3f}  (majority baseline accuracy {max(y.mean(), 1 - y.mean()):.3f})")

    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    X = to_matrix(df, categories)

    # Honest evaluation: every score comes from rows the model never saw
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    r = cross_validate(build_model(), X, y, cv=cv,
                       scoring=['roc_auc', 'accuracy', 'balanced_accuracy'])
    print(f"\n[2] 5-fold cross-validation ({len(FEATURES)} features)")
    for name in ['roc_auc', 'accuracy', 'balanced_accuracy']:
        s = r[f'test_{name}']
        print(f"    {name:18s} {s.mean():.3f} +/- {s.std():.3f}")

    model = build_model().fit(X, y)
    joblib.dump({'model': model, 'features': FEATURES, 'categories': categories},
                os.path.join(_HERE, 'model_efs.pkl'))
    print("\n[3] Trained on all rows and saved model_efs.pkl")


if __name__ == '__main__':
    main()
