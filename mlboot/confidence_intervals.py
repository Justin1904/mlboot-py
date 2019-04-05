import numpy as np
import pandas as pd
import scipy
import math

from tqdm import tqdm
from functools import reduce
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

def log(metric, lo, hi, alpha, n_samples, n_boot, method, comment=None):
    '''
    Logging the CI info
    '''
    print("="*80)
    if comment is not None:
        print(comment)
    print(f"Confidence Interval: [{round(lo, 4)}, {round(hi, 4)}], metric value: {metric}")
    print(f"confidence level: {1-alpha}")
    print(f"Number of samples in each bootstrap: {n_samples}\nNumber of total bootstrap runs: {n_boot}")
    print(f"Confidence interval type: {method}")
    print("="*80)


def groupby(a, axis=0):
    '''
    Group the rows with identical value in the specified value together
    '''
    a_ = pd.DataFrame(a).groupby(axis).apply(np.array)
    return a_.to_numpy()


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
    jack_scores1 = []
    jack_scores2 = []
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
        subset_scores1 = score_func(subset_labels, subset_preds1)
        subset_scores2 = score_func(subset_labels, subset_preds2)
        subset_scores = subset_scores2 - subset_scores1
        jack_scores.append(subset_scores)
        jack_scores1.append(subset_scores1)
        jack_scores2.append(subset_scores2)
    jack_scores = np.array(jack_scores)
    jack_scores1 = np.array(jack_scores1)
    jack_scores2 = np.array(jack_scores2)
    jack_est = np.mean(jack_scores)
    jack_est1 = np.mean(jack_scores1)
    jack_est2 = np.mean(jack_scores2)
    num1 = ((jack_est1 - jack_scores1) ** 3).sum()
    den1 = ((jack_est1 - jack_scores1) ** 2).sum()
    ahat1 = num1 / (6 * den1 ** (3 / 2))
    num2 = ((jack_est2 - jack_scores2) ** 3).sum()
    den2 = ((jack_est2 - jack_scores2) ** 2).sum()
    ahat2 = num2 / (6 * den2 ** (3 / 2))
    num = ((jack_est - jack_scores) ** 3).sum()
    den = ((jack_est - jack_scores) ** 2).sum()
    ahat = num / (6 * den ** (3 / 2))
    return ahat1, ahat2, ahat


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


def bca(preds, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap):
    '''
    Bias-corrected accelerated confidence interval estimation.
    '''
    if cluster is not None:
        print("Non-clustered confidence interval method chosen, ignoring provided cluster information.")
    scores = []
    for _ in tqdm(range(num_bootstrap)):
        subset_preds, subset_labels = resample(preds, labels, n_samples=sample_size)
        subset_scores = score_func(subset_labels, subset_preds)
        scores.append(subset_scores)
    scores = sorted(scores)

    # estimate the bias correction and acceleration
    full_score = score_func(labels, preds)
    bias_correction = bca_bias_correction(scores, full_score)
    accel = bca_accel_param(preds, labels, score_func)

    # calculate the BCa CI
    a, b = compute_bca_CI(confidence_level, bias_correction, accel)
    lower = np.quantile(scores, a)
    upper = np.quantile(scores, b)
    log(full_score, lower, upper, 1-confidence_level, sample_size, num_bootstrap, 'BCa')
    return lower, upper, scores


def paired_bca(preds1, preds2, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap):
    if cluster is not None:
        print("Non-clustered confidence interval method chosen, ignoring provided cluster information.")
    scores = []
    scores1 = []
    scores2 = []
    for _ in tqdm(range(num_bootstrap)):
        subset_preds1, subset_preds2, subset_labels = resample(preds1, preds2, labels, n_samples=sample_size)
        subset_scores1 = score_func(subset_labels, subset_preds1)
        subset_scores2 = score_func(subset_labels, subset_preds2)
        subset_scores = subset_scores2 - subset_scores1
        scores.append(subset_scores)
        scores1.append(subset_scores1)
        scores2.append(subset_scores2)

    # calculate bias-correction and acceleration
    full_score1 = score_func(labels, preds1)
    full_score2 = score_func(labels, preds2)
    full_score = full_score2 - full_score1
    bias_correction = bca_bias_correction(scores, full_score)
    bias_correction1 = bca_bias_correction(scores1, full_score1)
    bias_correction2 = bca_bias_correction(scores2, full_score2)
    accel, accel1, accel2 = paired_bca_accel_param(preds1, preds2, labels, score_func)

    # calculate bca interval
    a, b = compute_bca_CI(confidence_level, bias_correction, accel)
    a1, b1 = compute_bca_CI(confidence_level, bias_correction1, accel1)
    a2, b2 = compute_bca_CI(confidence_level, bias_correction2, accel2)

    # calculate the BCa CI
    lower = np.quantile(scores, a)
    upper = np.quantile(scores, b)
    lower1 = np.quantile(scores1, a1)
    upper1 = np.quantile(scores1, b1)
    lower2 = np.quantile(scores2, a2)
    upper2 = np.quantile(scores2, b2)
    log(full_score2 - full_score1, lower, upper, 1-confidence_level, sample_size, num_bootstrap, 'paired BCa', comment="Confidence interval for the different between two model (model2 - model1)")
    log(full_score1, lower1, upper1, 1-confidence_level, sample_size, num_bootstrap, 'paired BCa', comment="Confidence interval for model1")
    log(full_score2, lower2, upper2, 1-confidence_level, sample_size, num_bootstrap, 'paired BCa', comment="Confidence interval for model2")
    return lower, upper, scores


def percentile(preds, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap):
    if cluster is not None:
        print("Non-clustered confidence interval method chosen, ignoring provided cluster information.")
    scores = []
    for _ in tqdm(range(num_bootstrap)):
        subset_preds, subset_labels = resample(preds, labels, n_samples=sample_size)
        subset_scores = score_func(subset_labels, subset_preds)
        scores.append(subset_scores)
    scores = sorted(scores)
    lower = np.quantile(scores, (1-confidence_level)/2)
    upper = np.quantile(scores, (1+confidence_level)/2)
    log(score_func(labels, preds), lower, upper, 1-confidence_level, sample_size, num_bootstrap, 'percentile')
    return lower, upper, scores


def paired_percentile(preds1, preds2, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap):
    if cluster is not None:
        print("Non-clustered confidence interval method chosen, ignoring provided cluster information.")
    scores = []
    scores1 = []
    scores2 = []
    for _ in tqdm(range(num_bootstrap)):
        subset_preds1, subset_preds2, subset_labels = resample(preds1, preds2, labels, n_samples=sample_size)
        subset_scores1 = score_func(subset_labels, subset_preds1)
        subset_scores2 = score_func(subset_labels, subset_preds2)
        subset_scores = subset_scores2 - subset_scores1
        scores.append(subset_scores)
        scores1.append(subset_scores1)
        scores2.append(subset_scores2)
    lower = np.quantile(scores, (1-confidence_level)/2)
    upper = np.quantile(scores, (1+confidence_level)/2)
    lower1 = np.quantile(scores1, (1-confidence_level)/2)
    upper1 = np.quantile(scores1, (1+confidence_level)/2)
    lower2 = np.quantile(scores2, (1-confidence_level)/2)
    upper2 = np.quantile(scores2, (1+confidence_level)/2)
    log(score_func(labels, preds2) - score_func(labels, preds1), lower, upper, 1-confidence_level, sample_size, num_bootstrap, 'paired percentile', comment="Confidence interval for the difference between the two model (model2 - model1).")
    log(score_func(labels, preds1), lower1, upper1, 1-confidence_level, sample_size, num_bootstrap, 'paired percentile', comment="Confidence interval for model1.")
    log(score_func(lables, preds2), lower2, upper2, 1-confidence_level, sample_size, num_bootstrap, 'paired percentile', comment="Confidence interval for model2.")
    return lower, upper, scores


def cluster_percentile(preds, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap):
    cluster_scores = []
    dmat = np.stack((cluster, preds, labels), axis=-1)
    dmat = groupby(dmat)
    for cluster in dmat:
        cluster_preds, cluster_labels = cluster[:, 1], cluster[:, 2]
        cluster_score = score_func(cluster_labels, cluster_preds)
        cluster_scores.append(cluster_score)

    boot_scores = resample(cluster_scores, n_samples=sample_size)
    boot_scores = sorted(boot_scores)
    lower = np.quantile(boot_scores, (1-confidence_level)/2)
    upper = np.quantile(boot_scores, (1+confidence_level)/2)
    log(score_func(labels, preds), lower, upper, 1-confidence_level, sample_size, num_bootstrap, 'cluster percentile')
    return lower, upper, boot_scores
