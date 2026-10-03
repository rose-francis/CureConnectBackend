# ============================================================
# predict_core.py
# Shared prediction logic and report printing
# Used by both Option A and Option B
# ============================================================

import pandas as pd
import joblib, json
import os

# Resolve paths relative to this file, not the caller's working directory —
# this module gets imported by api.py, which may run from a different folder.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Load models once when this module is imported
m_survival = joblib.load(os.path.join(_HERE, 'model_survival.pkl'))
m_relapse  = joblib.load(os.path.join(_HERE, 'model_relapse.pkl'))
m_gvhd     = joblib.load(os.path.join(_HERE, 'model_gvhd.pkl'))
encoders   = joblib.load(os.path.join(_HERE, 'encoders.pkl'))
with open(os.path.join(_HERE, 'feature_cols.json')) as f:
    feature_cols = json.load(f)

from compatibility import compute_compatibility_score
from compute_hla import encode_for_model

CATEGORICALS = [
    'donor_ABO', 'donor_CMV', 'recipient_ABO', 'recipient_rh',
    'recipient_CMV', 'disease', 'disease_group', 'risk_group',
    'stem_cell_source', 'tx_post_relapse'
]


def build_and_predict(donor: dict, patient: dict, antigen: int, allel: int):
    """
    Builds feature vector and runs all 3 RF models.
    antigen / allel are mismatch counts out of 10 (from compute_hla).
    Returns compat, derived, and all 3 model predictions.
    """
    compat  = compute_compatibility_score(donor, patient)
    derived = compat['derived_fields']
    antigen_enc, allel_enc = encode_for_model(antigen, allel)

    row = {
        'donor_age':           donor['donor_age'],
        'CD34_x1e6_per_kg':    donor['CD34_x1e6_per_kg'],
        'CD3_x1e8_per_kg':     donor['CD3_x1e8_per_kg'],
        'CD3_to_CD34_ratio':   donor['CD3_to_CD34_ratio'],
        'recipient_age':       patient['recipient_age'],
        'recipient_body_mass': patient['recipient_body_mass'],
        'HLA_match_score':     {'10/10':4,'9/10':3,'8/10':2,'7/10':1}.get(
                                   derived['HLA_match'], 1),
        'CMV_status':          derived['CMV_status'],
        'ABO_match_binary':    1 if derived['ABO_match'] == 'matched' else 0,
        'gender_risk':         1 if derived['gender_match'] == 'female_to_male' else 0,
        'donor_age_risk':      1 if donor['donor_age'] >= 35 else 0,
        'total_HLA_diff':      antigen_enc + allel_enc,
        'antigen':             antigen_enc,
        'allel':               allel_enc,
        'donor_ABO':           donor['donor_ABO'],
        'donor_CMV':           donor['donor_CMV'],
        'recipient_ABO':       patient['recipient_ABO'],
        'recipient_rh':        patient['recipient_rh'],
        'recipient_CMV':       patient['recipient_CMV'],
        'disease':             patient['disease'],
        'disease_group':       patient['disease_group'],
        'risk_group':          patient['risk_group'],
        'stem_cell_source':    donor['stem_cell_source'],
        'tx_post_relapse':     patient['tx_post_relapse'],
    }

    df_row = pd.DataFrame([row])
    for col in CATEGORICALS:
        le  = encoders[col]
        val = str(df_row[col].iloc[0])
        df_row[col + '_enc'] = le.transform([val]) if val in le.classes_ else [0]

    X_input = df_row[feature_cols].astype(float)

    s_pred = m_survival.predict(X_input)[0]
    s_prob = m_survival.predict_proba(X_input)[0]
    r_pred = m_relapse.predict(X_input)[0]
    r_prob = m_relapse.predict_proba(X_input)[0]
    g_pred = m_gvhd.predict(X_input)[0]
    g_prob = m_gvhd.predict_proba(X_input)[0]

    return compat, derived, s_pred, s_prob, r_pred, r_prob, g_pred, g_prob


def print_report(compat, derived, s_pred, s_prob,
                 r_pred, r_prob, g_pred, g_prob):
    """Prints the full matching report for one donor-patient pair"""
    score = compat['compatibility_score']
    grade = compat['grade']
    grade_icon = {
        'Excellent':'🌟','Good':'✅','Moderate':'🟡','Poor':'🔴'
    }.get(grade,'')

    print("\n\n" + "="*55)
    print("     BONE MARROW TRANSPLANT MATCHING REPORT")
    print("="*55)
    print(f"\n  COMPATIBILITY SCORE : {score}/100  —  {grade} {grade_icon}")

    print(f"\n  {'Factor':<22} {'Score':>5}   {'Bar':<12}  Notes")
    print(f"  {'─'*22}  {'─'*5}   {'─'*12}  {'─'*22}")
    for factor, detail in compat['breakdown'].items():
        filled = int((detail['score'] / detail['max']) * 12)
        bar    = '█' * filled + '░' * (12 - filled)
        print(f"  {factor:<22} {detail['score']:>2}/{detail['max']:<2}"
              f"   {bar:<12}  {detail['label']}")

    print(f"\n  Auto-computed fields:")
    print(f"    HLA Match     : {derived['HLA_match']}")
    print(f"    HLA Mismatch  : {derived['HLA_mismatch']}")
    print(f"    ABO Match     : {derived['ABO_match']}")
    print(f"    CMV Status    : {derived['CMV_status']}  (0=best, 3=worst)")
    print(f"    Gender Match  : {derived['gender_match']}")

    print(f"\n  {'─'*55}")
    print(f"  PREDICTIONS")
    print(f"  {'─'*55}")

    def bar20(prob):
        return '█' * int(prob * 20)

    s_icon = '✅' if s_pred == 0 else '⚠️ '
    r_icon = '✅' if r_pred == 0 else '⚠️ '
    g_icon = '✅' if g_pred == 0 else '⚠️ '

    print(f"\n  SURVIVAL PROBABILITY     {s_icon} "
          f"{'Alive' if s_pred==0 else 'At Risk'}")
    print(f"    Alive    {s_prob[0]*100:5.1f}%  {bar20(s_prob[0])}")
    print(f"    At Risk  {s_prob[1]*100:5.1f}%  {bar20(s_prob[1])}")

    print(f"\n  RELAPSE RISK             {r_icon} "
          f"{'Low Risk' if r_pred==0 else 'High Risk'}")
    print(f"    Low      {r_prob[0]*100:5.1f}%  {bar20(r_prob[0])}")
    print(f"    High     {r_prob[1]*100:5.1f}%  {bar20(r_prob[1])}")

    print(f"\n  GvHD RISK (Stage III/IV) {g_icon} "
          f"{'Low Risk' if g_pred==0 else 'High Risk'}")
    print(f"    Low      {g_prob[0]*100:5.1f}%  {bar20(g_prob[0])}")
    print(f"    High     {g_prob[1]*100:5.1f}%  {bar20(g_prob[1])}")

    print(f"\n  {'='*55}")
    if score >= 70 and s_pred == 0:
        print("  VERDICT  :  ✅  RECOMMENDED MATCH")
    elif score >= 50:
        print("  VERDICT  :  🟡  POSSIBLE MATCH — REVIEW CAREFULLY")
    else:
        print("  VERDICT  :  ⚠️   NOT RECOMMENDED — SEEK BETTER DONOR")
    print(f"  {'='*55}\n")