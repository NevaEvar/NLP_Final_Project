import numpy as np 
import pandas as pd 
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from statsmodels.stats.multitest import multipletests

#implemented so due to just the existing proportionality matrix created for earlier sections
#otherwise could be just done through count matrix but cause lazy decided to simply do it like this
def MatrixStatistics(matrix, categoryCounts):
    """
    Tests given matrices statistical values with chi2
    Gives back statistical values shown below
    chi2: Strength of the observed departure from the null model, on the chi-square scale
    dof: degrees of freedom
    p-value: probability value against equal stop word distribution across categories
    p-adjusted: corrected p-value for testing multiple stop words (more important)
    cramers v: the assosiation of stop words and categories != correlation
    proportion range: difference for highest and lowest proportions
    min expected: min expected count
    significant: binary classification if p adjusted is below alpha (in this case 0.05)
    """
    a = 0.05
    totals = pd.Series(categoryCounts).reindex(matrix.index)
    stats = []

    for word in matrix.columns:
        # proportions to counts
        values = matrix[word]
        withWord = values * totals
        withoutWord = totals - withWord

        table = np.rint(np.column_stack([withWord, withoutWord])).astype(int)

        chi2, pValue, dof, expected = chi2_contingency(table)
        cramersV = association(table.astype(int), method = 'cramer')
        proportions = matrix[word]

        stats.append({
            'stopword': word, 
            'chi2': chi2, 
            'dof': dof,
            'p_value': pValue,
            'cramers_v': cramersV,
            'min_expected': expected.min(),
            'proportions_range': (
                proportions.max() - proportions.min()
            )
        })
    stats = pd.DataFrame(stats)

    #significance test and binary labeling according to alpha
    stats['p_adjusted'] = multipletests(stats['p_value'], method = 'fdr_bh')[1]
    stats['significant'] = stats['p_adjusted'] < a # assigns True if condition met

    return stats.sort_values('p_adjusted').reset_index(drop = True)