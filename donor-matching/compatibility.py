# ============================================================
# Compatibility Score — mirror this logic in React Native too
# ============================================================

from compute_hla import get_hla_match_label

def compute_compatibility_score(donor: dict, patient: dict) -> dict:
    score = 0
    breakdown = {}

    # HLA Match (40 pts) — antigen + allel are mismatch counts out of 10
    total_diff = donor.get('antigen', 0) + donor.get('allel', 0)
    hla_table = {0: (40, '10/10 Perfect'), 1: (30, '9/10'),
                 2: (18, '8/10'), 3: (5, '7/10 or below')}
    hla_score, hla_label = hla_table.get(min(total_diff, 3), (5, '7/10'))
    score += hla_score
    breakdown['HLA Match'] = {'score': hla_score, 'max': 40, 'label': hla_label}

    # ABO Match (20 pts)
    abo_ok = donor.get('donor_ABO') == patient.get('recipient_ABO')
    abo_score = 20 if abo_ok else 8
    score += abo_score
    breakdown['ABO Match'] = {
        'score': abo_score, 'max': 20,
        'label': 'Matched' if abo_ok else 'Mismatched'
    }

    # CMV Compatibility (20 pts)
    cmv_map = {
        ('absent','absent'):  (20, 'Optimal'),
        ('present','present'): (15, 'Both positive'),
        ('absent','present'):  (10, 'Moderate risk'),
        ('present','absent'):  (5,  'High risk'),
    }
    cmv_key = (donor.get('donor_CMV','absent'), patient.get('recipient_CMV','absent'))
    cmv_score, cmv_label = cmv_map.get(cmv_key, (10, 'Unknown'))
    score += cmv_score
    breakdown['CMV'] = {'score': cmv_score, 'max': 20, 'label': cmv_label}

    # Donor Age (10 pts)
    age = donor.get('donor_age', 40)
    if age < 35:    age_score, age_label = 10, 'Optimal (<35)'
    elif age < 45:  age_score, age_label = 6,  'Acceptable (35-45)'
    else:           age_score, age_label = 2,  'Suboptimal (>45)'
    score += age_score
    breakdown['Donor Age'] = {'score': age_score, 'max': 10, 'label': age_label}

    # Gender Match (10 pts)
    f2m = (donor.get('donor_gender') == 'female' and
           patient.get('recipient_gender') == 'male')
    gender_score = 5 if f2m else 10
    score += gender_score
    breakdown['Gender'] = {
        'score': gender_score, 'max': 10,
        'label': 'Female→Male (GvHD risk)' if f2m else 'Compatible'
    }

    # Auto-derive fields for model input
    derived = {
        'HLA_match':   get_hla_match_label(donor.get('antigen', 0), donor.get('allel', 0)),
        'HLA_mismatch': 'matched' if total_diff == 0 else 'mismatched',
        'ABO_match':    'matched' if abo_ok else 'mismatched',
        # Same coding as the training data's CMV_status (donor, recipient)
        'CMV_status':  {('absent','absent'):0, ('present','absent'):1,
                        ('absent','present'):2, ('present','present'):3}.get(cmv_key, 0),
        'gender_match': 'female_to_male' if f2m else 'other',
    }

    return {
        'compatibility_score': round(score, 2),
        'breakdown': breakdown,
        'derived_fields': derived,
        'grade': ('Excellent' if score >= 85 else 'Good' if score >= 70
                  else 'Moderate' if score >= 50 else 'Poor')
    }