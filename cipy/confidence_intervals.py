import numpy as np
import scipy
import math
from pdb import set_trace

from sklearn.utils import resample


def bin_search(arr, val, start=0, end=None):
    '''
    Finds the first element in the list that is not less than val.
    '''
    if end is None:
        end = len(arr) - 1
    while start < end:
        mid = start + (end - start) // 2
        if arr[mid] < val:
            start = mid + 1
        else:
            end = mid
    return start


def bca_accel_param(preds, labels, score_func):
    # estimate the acceleration: 1) jackknife sampling the scores; 2) compute statistic
    jack_scores = []
    for i in range(len(preds)):
        if isinstance(preds, list):
            subset_preds = preds[:i] + preds[i+1:]
            subset_labels = labels[:i] + labels[i+1:]
        elif isinstance(preds, np.ndarray):
            subset_preds = np.concatenate((preds[:i], preds[i+1:]), axis=0)
            subset_labels = np.concatenate((labels[:i], labels[i+1:]), axis=0)
        else:
            raise TypeError("Make sure your predictions are either lists or numpy arrays.")
        subset_scores = score_func(subset_preds, subset_labels)
        jack_scores.append(subset_scores)
    jack_scores = np.array(jack_scores)
    jack_est = np.mean(jack_scores)
    num = ((jack_est - jack_scores) ** 3).sum()
    den = ((jack_est - jack_scores) ** 2).sum()
    ahat = num / (6 * den ** (3 / 2))
    return ahat


def paired_bca_accel_param(preds1, preds2, labels, score_func):
    # estimate the acceleration: 1) jackknife sampling the scores; 2) compute statistic
    jack_scores = []
    for i in range(len(preds1)):
        if isinstance(preds1, list):
            subset_preds1 = preds1[:i] + preds1[i+1:]
            subset_preds2 = preds2[:i] + preds2[i+1:]
            subset_labels = labels[:i] + labels[i+1:]
        elif isinstance(preds1, np.ndarray):
            subset_preds1 = np.concatenate((preds1[:i], preds1[i+1:]), axis=0)
            subset_preds2 = np.concatenate((preds2[:i], preds2[i+1:]), axis=0)
            subset_labels = np.concatenate((labels[:i], labels[i+1:]), axis=0)
        else:
            raise TypeError("Make sure your predictions are either lists or numpy arrays.")
        subset_scores1 = score_func(subset_preds1, subset_labels)
        subset_scores2 = score_func(subset_preds2, subset_labels)
        subset_scores = subset_scores2 - subset_scores1
        jack_scores.append(subset_scores)
    jack_scores = np.array(jack_scores)
    jack_est = np.mean(jack_scores)
    num = ((jack_est - jack_scores) ** 3).sum()
    den = ((jack_est - jack_scores) ** 2).sum()
    ahat = num / (6 * den ** (3 / 2))
    return ahat


def bca_bias_correction(scores, full_score):
    # estimate the bias-correction
    # entries_less = bin_search(scores, full_score)
    scores = np.array(scores)
    entries_less = (scores < full_score).sum()
    prop_less = entries_less / scores.shape[0]
    normal = scipy.stats.norm()
    bias_correction = normal.ppf(prop_less)
    return bias_correction


def compute_bca_CI(confidence_level, bias_correction, accel):
    alpha = 1 - confidence_level
    normal = scipy.stats.norm()
    
    zl = bias_correction + normal.ppf(alpha / 2)
    alpha1 = normal.cdf(bias_correction + zl / (1 - accel * zl))

    zu = bias_correction + normal.ppf(1 - alpha / 2)
    alpha2 = normal.cdf(bias_correction + zu / (1 - accel * zu))
    return alpha1, alpha2


def bca(preds, labels, score_func, confidence_level, sample_size, num_bootstrap):
    '''
    Bias-corrected accelerated confidence interval estimation.
    '''
    scores = []
    while len(scores) < num_bootstrap:
        subset_preds, subset_labels = resample(preds, labels, n_samples=sample_size)
        subset_scores = score_func(subset_preds, subset_labels)
        scores.append(subset_scores)
    scores = sorted(scores)

    # estimate the bias correction and acceleration
    full_score = score_func(preds, labels)
    bias_correction = bca_bias_correction(scores, full_score)
    accel = bca_accel_param(preds, labels, score_func)

    # calculate the BCa CI
    a, b = compute_bca_CI(confidence_level, bias_correction, accel)
    lower = np.quantile(scores, a)
    upper = np.quantile(scores, b)
    return lower, upper, scores


def paired_bca(preds1, preds2, labels, score_func, confidence_level, sample_size, num_bootstrap):
    scores = []
    while len(scores) < num_bootstrap:
        subset_preds1, subset_preds2, subset_labels = resample(preds1, preds2, labels, n_samples=sample_size)
        subset_scores1 = score_func(subset_preds1, subset_labels)
        subset_scores2 = score_func(subset_preds2, subset_labels)
        subset_scores = subset_scores2 - subset_scores1
        scores.append(subset_scores)

    # calculate bias-correction and acceleration
    full_score1 = score_func(preds1, labels)
    full_score2 = score_func(preds2, labels)
    full_score = full_score2 - full_score1
    bias_correction = bca_bias_correction(scores, full_score)
    accel = paired_bca_accel_param(preds1, preds2, labels, score_func)

    # calculate bca interval
    a, b = compute_bca_CI(confidence_level, bias_correction, accel)

    # calculate the BCa CI
    a, b = compute_bca_CI(confidence_level, bias_correction, accel)
    lower = np.quantile(scores, a)
    upper = np.quantile(scores, b)
    return lower, upper, scores


def percentile(preds, labels, score_func, confidence_level, sample_size, num_bootstrap):
    scores = []
    while len(scores) < num_bootstrap:
        subset_preds, subset_labels = resample(preds, labels, n_samples=sample_size)
        subset_scores = score_func(subset_preds, subset_labels)
        scores.append(subset_scores)
    scores = sorted(scores)
    lower = np.quantile(scores, (1-confidence_level)/2)
    upper = np.quantile(scores, (1+confidence_level)/2)
    return lower, upper, scores


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
    return lower, upper, scores

