# ============================================================
# top5_matcher.py
# Score all donors for a patient and return the top 5 matches
# ============================================================

import pandas as pd
from compute_hla import extract_hla_profile, compute_hla_differences, compute_locus_matches
from predict_core import build_and_predict





def score_donor_against_patient(donor_row: dict, patient_row: dict):
    donor_hla   = extract_hla_profile(donor_row,   'donor')
    patient_hla = extract_hla_profile(patient_row, 'patient')
    antigen_diff, allel_diff = compute_hla_differences(donor_hla, patient_hla)
    hla_matches = compute_locus_matches(donor_hla, patient_hla)

    donor = {
        'donor_age':     donor_row['donor_age'],
        'donor_ABO':     donor_row['donor_ABO'],
        'donor_CMV':     donor_row['donor_CMV'],
        'donor_gender':  donor_row['donor_gender'],
        'graft_type':    donor_row['graft_type'],
        'donor_related': donor_row['donor_related'],
        'antigen':       antigen_diff,
        'allel':         allel_diff,
    }

    compat, derived, event_prob = build_and_predict(donor, patient_row, hla_matches)

    return {
        'donor_id':            int(donor_row['donor_id']),
        'donor_age':           donor['donor_age'],
        'donor_ABO':           donor['donor_ABO'],
        'donor_CMV':           donor['donor_CMV'],
        'donor_gender':        donor['donor_gender'],
        'graft_type':          donor['graft_type'],
        'antigen_diff':        antigen_diff,
        'allel_diff':          allel_diff,
        'hla_match':           derived['HLA_match'],
        'abo_match':           derived['ABO_match'],
        'compatibility_score': compat['compatibility_score'],
        'grade':               compat['grade'],
        'event_free_survival': round((1 - event_prob) * 100, 1),
    }





def find_top5_donors(patient: dict, donor_db: pd.DataFrame):

    results = []

    for _, donor_row in donor_db.iterrows():
        try:
            result = score_donor_against_patient(
                donor_row.to_dict(),
                patient
            )
            results.append(result)
        except Exception:
            continue

    results.sort(
        key=lambda x: (x['compatibility_score'], x['event_free_survival']),
        reverse=True
    )

    top5 = results[:5]

    return top5