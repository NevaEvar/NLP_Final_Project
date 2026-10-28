import numpy as np 
import pandas as pd 
from scipy.stats import chi2_contingency, norm
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

        # reshaping solved with co-pilot
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


# co-pilot was used to suggest different methods and also used to solve some logic issues
def MatrixAssociationStatistics(matrix, categoryCounts):
    """
    Tests given matrices statistical values with chi2
    Gives back overall and per-cell statistical values shown below
    chi2: Strength of the observed departure from the null model, on the chi-square scale
    dof: degrees of freedom
    p-value: probability value against equal token distribution across categories
    cramers v: the association of tokens and categories != correlation
    min expected: minimum expected count
    expected below 5 percent: percentage of expected counts below 5
    pearson residual: difference between observed and expected counts
    p-adjusted: corrected p-value for testing multiple cells
    significant: binary classification if p adjusted is below 0.05
    """
    a = 0.05
    totals = pd.Series(categoryCounts).reindex(matrix.index)
    tokenCounts = matrix.mul(totals, axis = 0).round().astype(int)
    tokenCounts = tokenCounts.loc[:, tokenCounts.sum(axis = 0) > 0]
    tokenCounts['other_tokens'] = totals - tokenCounts.sum(axis = 1)
    observed = tokenCounts.to_numpy()

    chi2, pValue, dof, expected = chi2_contingency(observed, correction = False)
    cramersV = association(observed, method = 'cramer')
    overall = {
        'chi2': chi2,
        'dof': dof,
        'p_value': pValue,
        'cramers_v': cramersV,
        'min_expected': expected.min(),
        'expected_below_5_percent': (expected < 5).mean() * 100,
        'significant': pValue < a,
    }

    pearsonResiduals = (observed - expected) / np.sqrt(expected)
    cells = []
    for categoryIndex, category in enumerate(tokenCounts.index):
        for tokenIndex, token in enumerate(tokenCounts.columns):
            residual = pearsonResiduals[categoryIndex, tokenIndex]
            cellPValue = 2 * norm.sf(abs(residual))

            cells.append({
                'category': category,
                'token': token,
                'observed': observed[categoryIndex, tokenIndex],
                'expected': expected[categoryIndex, tokenIndex],
                'pearson_residual': residual,
                'p_value': cellPValue,
            })
    cells = pd.DataFrame(cells)

    # significance test and binary labeling according to alpha
    cells['p_adjusted'] = multipletests(cells['p_value'], method = 'fdr_bh')[1]
    cells['significant'] = cells['p_adjusted'] < a

    return {
        'overall': overall,
        'cells': cells.sort_values('p_adjusted').reset_index(drop=True),
    }
