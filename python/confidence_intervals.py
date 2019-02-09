import numpy as np
import scipy
import math

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
    lower = int(math.ceil(num_bootstrap * (1-confidence_level)))
    upper = int(math.floor(num_bootstrap * confidence_level))
    return scores[lower], scores[upper]

def paired(scores1, scores2, confidence_level):
    raise NotImplementedError
