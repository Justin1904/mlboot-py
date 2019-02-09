import numpy as np
import scipy
import math
from pdb import set_trace

from sklearn.utils import resample

# CI_TYPES = {'bca': bca, 'percentile': percentile, 'paired': paired}

def bca(scores, confidence_level):
    raise NotImplementedError

def percentile(preds, labels, score_func, confidence_level, sample_size, num_bootstrap):
    scores = []
    while len(scores) < num_bootstrap:
        subset_preds, subset_labels = resample(preds, labels, n_samples=sample_size)
        subset_scores = score_func(subset_preds, subset_labels)
        scores.append(subset_scores)
    scores = sorted(scores)
    lower = np.quantile(scores, (1-confidence_level)/2)
    upper = np.quantile(scores, (1+confidence_level)/2)
    return lower, upper

def paired_percentile(preds1, preds2, labels, score_func, confidence_level, sample_size, num_bootstrap):
    scores = []
    while len(scores) < num_bootstrap:
        subset_preds1, subset_preds2, subset_labels = resample(preds1, preds2, labels, n_samples=sample_size)
        subset_scores1 = score_func(subset_preds1, subset_labels)
        subset_scores2 = score_func(subset_preds2, subset_labels)
        subset_scores = subset_scores2 - subset_scores1
        scores.append(subset_scores)
    lower = np.quantile(scores, (1-confidence_level)/2)
    upper = np.quantile(scores, (1+confidence_level)/2)
    set_trace()
    return lower, upper
