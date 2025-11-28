import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def calc_metrics(gb, sample_ids=None, pos_tr=0.05):
    if gb.empty:
        return {
            'spearman': np.nan,
            'spearman_pos': np.nan,
            'pearson_pos': np.nan,
            'ap': np.nan,
            'auroc': np.nan,
        }
    pos = gb['median_x'] >= pos_tr

    if sample_ids is not None:
        gb = gb[gb.isin(sample_ids)]
    
    spr = stats.spearmanr(gb['median_x'], gb['median_y']).statistic
    spr_pos = stats.spearmanr(gb['median_x'][pos], gb['median_y'][pos]).statistic
    prs_pos = stats.pearsonr(gb['median_x'][pos], gb['median_y'][pos]).statistic
    ap = average_precision_score(pos, gb['median_y'])
    auroc = roc_auc_score(pos, gb['median_y'])

    return {
        'spearman': spr,
        'spearman_pos': spr_pos,
        'pearson_pos': prs_pos,
        'ap': ap,
        'auroc': auroc,
    }