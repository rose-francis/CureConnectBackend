# ============================================================
# test_feature_consistency.py
# Checks the CMV and HLA coding used by compatibility scoring and
# the per-locus HLA matches fed to the EFS model.
# Run: python -m pytest test_feature_consistency.py
# ============================================================

import os
import pandas as pd

from compatibility import compute_compatibility_score
from compute_hla import compute_hla_differences, compute_locus_matches, get_hla_match_label

_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, 'bone_marrow.csv'), na_values='?')


def test_cmv_status_matches_training_coding():
    known = df.dropna(subset=['donor_CMV', 'recipient_CMV', 'CMV_status'])
    for (d, r), group in known.groupby(['donor_CMV', 'recipient_CMV']):
        derived = compute_compatibility_score(
            {'donor_CMV': d}, {'recipient_CMV': r})['derived_fields']
        # A few rows in the source data disagree; compare with the usual value
        assert derived['CMV_status'] == group['CMV_status'].mode()[0], (d, r)


def test_hla_match_label_matches_training_coding():
    known = df.dropna(subset=['antigen', 'allel', 'HLA_match'])
    for _, row in known.drop_duplicates(['antigen', 'allel', 'HLA_match']).iterrows():
        ag, al = int(row['antigen']), int(row['allel'])
        # bone_marrow.csv stores a mismatch count as count + 1
        counts = (0, 0) if ag + al == 0 else (ag - 1, al - 1)
        assert get_hla_match_label(*counts) == row['HLA_match']


def _profile(**overrides):
    hla = {'HLA-A': ['01:01', '02:01'], 'HLA-B': ['07:02', '08:01'],
           'HLA-C': ['07:01', '07:02'], 'HLA-DRB1': ['15:01', '03:01'],
           'HLA-DQB1': ['06:02', '02:01']}
    hla.update(overrides)
    return hla


def test_hla_differences():
    patient = _profile()
    assert compute_hla_differences(_profile(), patient) == (0, 0)
    # Allele order within a locus does not matter
    assert compute_hla_differences(_profile(**{'HLA-A': ['02:01', '01:01']}), patient) == (0, 0)
    # Same antigen group, different allele -> allele mismatch
    assert compute_hla_differences(_profile(**{'HLA-A': ['01:01', '02:05']}), patient) == (0, 1)
    # Different antigen group -> antigen mismatch, even on a class II locus
    assert compute_hla_differences(_profile(**{'HLA-DRB1': ['15:01', '04:01']}), patient) == (1, 0)
    # A single-allele mismatch counts (previously read as a match)
    assert compute_hla_differences(_profile(**{'HLA-B': ['07:02', '44:02']}), patient) == (1, 0)
    # Both alleles at a locus mismatched -> 2 mismatches
    assert compute_hla_differences(_profile(**{'HLA-C': ['04:01', '05:01']}), patient) == (2, 0)
    # A third field is a synonymous variant, not a mismatch
    assert compute_hla_differences(_profile(**{'HLA-A': ['01:01:01', '02:01']}), patient) == (0, 0)


def test_locus_matches():
    patient = _profile()
    full = compute_locus_matches(_profile(), patient)
    assert full['hla_high_res_10'] == 10
    assert all(full[f'hla_match_{l}_high'] == 2 for l in ['a', 'b', 'c', 'drb1', 'dqb1'])
    # One allele mismatch at A, both alleles mismatched at C
    m = compute_locus_matches(_profile(**{'HLA-A': ['01:01', '02:05'], 'HLA-C': ['04:01', '05:01']}), patient)
    assert (m['hla_match_a_high'], m['hla_match_c_high'], m['hla_high_res_10']) == (1, 0, 7)
