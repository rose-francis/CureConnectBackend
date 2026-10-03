# Computes antigen and allel differences from raw HLA profiles
#
# Typings are two-field values like '02:01'. The first field is the
# antigen group, the second the specific allele. Across the 10 alleles
# (2 each at A, B, C, DRB1, DQB1):
#   - different first field                -> antigen mismatch
#   - same first field, different second   -> allele mismatch

from itertools import permutations

LOCI = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRB1', 'HLA-DQB1']

def extract_hla_profile(row: dict, prefix: str) -> dict:
    """
    Extracts HLA profile from a dataframe row.
    prefix is 'donor' or 'patient' (matches column names in CSV)
    """
    # Column names like HLA_A_1, HLA_A_2, HLA_B_1 etc.
    return {
        'HLA-A':    [row[f'HLA_A_1'],    row[f'HLA_A_2']],
        'HLA-B':    [row[f'HLA_B_1'],    row[f'HLA_B_2']],
        'HLA-C':    [row[f'HLA_C_1'],    row[f'HLA_C_2']],
        'HLA-DRB1': [row[f'HLA_DRB1_1'], row[f'HLA_DRB1_2']],
        'HLA-DQB1': [row[f'HLA_DQB1_1'], row[f'HLA_DQB1_2']],
    }

def _fields(typing):
    if typing is None or (isinstance(typing, float) and typing != typing):
        raise ValueError('missing HLA typing')
    fields = str(typing).strip().split('*')[-1].split(':')
    return fields[0], ':'.join(fields[1:])

def _compare(d, p):
    """Returns (antigen_mismatch, allele_mismatch) for one allele pair."""
    d_group, d_allele = _fields(d)
    p_group, p_allele = _fields(p)
    if d_group != p_group:
        return 1, 0
    if d_allele != p_allele:
        return 0, 1
    return 0, 0

def compute_hla_differences(donor_hla: dict, patient_hla: dict):
    """
    Returns (antigen_diff, allel_diff): the number of antigen-level and
    allele-level mismatches across all 10 alleles (total 0-10).
    At each locus the two alleles are paired up the way that gives the
    fewest mismatches.
    """
    antigen_diff = allel_diff = 0
    for locus in LOCI:
        d1, d2 = donor_hla[locus]
        pairings = []
        for p1, p2 in permutations(patient_hla[locus]):
            ag1, al1 = _compare(d1, p1)
            ag2, al2 = _compare(d2, p2)
            pairings.append((ag1 + ag2, al1 + al2))
        # Fewest mismatches; on a tie prefer allele- over antigen-level
        ag, al = min(pairings, key=lambda x: (x[0] + x[1], x[0]))
        antigen_diff += ag
        allel_diff   += al
    return antigen_diff, allel_diff

def encode_for_model(antigen_diff, allel_diff):
    """
    Converts mismatch counts to the dataset's 'antigen'/'allel' encoding:
    a full match is (0, 0); otherwise each count is stored as count + 1.
    e.g. one antigen mismatch -> (2, 1), one allele mismatch -> (1, 2).
    Clipped to the ranges seen in training (antigen 0-3, allel 0-4).
    """
    if antigen_diff + allel_diff == 0:
        return 0, 0
    return min(antigen_diff + 1, 3), min(allel_diff + 1, 4)

def get_hla_match_label(antigen_diff, allel_diff):
    total = antigen_diff + allel_diff
    return ['10/10', '9/10', '8/10', '7/10'][min(total, 3)]
