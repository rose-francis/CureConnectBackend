# ============================================================
# predict_core.py
# Builds the model input for one donor-patient pair and runs the
# event-free survival model trained by train_cibmtr.py.
# ============================================================

import math
import os

import joblib
import pandas as pd

from compatibility import compute_compatibility_score
from efs_features import to_matrix

# Resolve paths relative to this file, not the caller's working directory —
# this module gets imported by api.py, which may run from a different folder.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Load the model once when this module is imported
_bundle    = joblib.load(os.path.join(_HERE, 'model_efs.pkl'))
m_efs      = _bundle['model']
categories = _bundle['categories']


def _sign(cmv):
    return '+' if cmv == 'present' else '-'


def _sex(gender):
    g = str(gender).strip().lower()
    return 'F' if g.startswith('f') else 'M' if g.startswith('m') else None


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def build_and_predict(donor: dict, patient: dict, hla_matches: dict):
    """
    Builds the CIBMTR-format feature row and runs the EFS model.
    hla_matches comes from compute_hla.compute_locus_matches.
    Returns compat, derived, and the probability of an event
    (relapse, graft failure or death).
    """
    compat  = compute_compatibility_score(donor, patient)
    derived = compat['derived_fields']

    d_sex, r_sex = _sex(donor['donor_gender']), _sex(patient['recipient_gender'])
    row = {
        'age_at_hct':        patient['recipient_age'],
        'prim_disease_hct':  patient['prim_disease_hct'],
        'karnofsky_score':   patient['karnofsky_score'],
        'comorbidity_score': patient['comorbidity_score'],
        'dri_score':         patient['dri_score'],
        'donor_age':         donor['donor_age'],
        # Dataset format is donor/recipient, e.g. '+/-'
        'cmv_status':        f"{_sign(donor['donor_CMV'])}/{_sign(patient['recipient_CMV'])}",
        'sex_match':         f"{d_sex}-{r_sex}" if d_sex and r_sex else None,
        'graft_type':        donor['graft_type'],
        'donor_related':     donor['donor_related'],
        **hla_matches,
    }
    row = {k: (None if _missing(v) else v) for k, v in row.items()}

    X_input   = to_matrix(pd.DataFrame([row]), categories)
    event_prob = float(m_efs.predict_proba(X_input)[0][1])

    return compat, derived, event_prob
