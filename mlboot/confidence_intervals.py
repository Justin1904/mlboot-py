import numpy as np
import scipy
import math

from pdb import set_trace
from sklearn.utils import resample


def bootstrap_avg(samples, n_bootstrap, n_samples):
    """
    Bootstrap `n_bootstrap` sub-samples from `samples` each with sample size `n_samples`,
    average them and return.
    """
    avg_score_samples = []
    while len(avg_score_samples) < n_bootstrap:
        sample_scores = resample(samples, n_samples=n_samples)
        avg_score_samples.append(sample_scores.mean())
    return avg_score_samples


def jackknife_avg(samples):
    """
    Jackknife (leave-one-out resampling) sub-samples from `samples`,
    comptue their average and return.
    """
    jackknife_samples = []
    total_sum = samples.sum()
    div_factor = samples.size - 1
    for sample_var in samples:
        jackknife_samples.append((total_sum - sample_var) / div_factor)
    return jackknife_samples


def bca_bias_correction(scores, full_score):
    scores = np.array(scores)
    entries_less = (scores < full_score).sum()
    prop_less = entries_less / scores.shape[0]
    normal = scipy.stats.norm()
    bias_correction = normal.ppf(prop_less)
    return bias_correction


def bca_accel_param(samples):
    jackknife_scores = jackknife_avg(samples)
    jackknife_avg_scores = jackknife_scores.mean()
    accel_numerator = ((jackknife_avg_scores - jackknife_scores) ** 3).sum()
    accel_denominator = ((jackknife_avg_scores - jackknife_scores) ** 2).sum()
    accel_param = accel_numerator / (6 * accel_denominator ** (3 / 2))
    return accel_param


def compute_bca_quantiles(confidence_level, bias_correction, accel_param):
    alpha = 1 - confidence_level
    normal = scipy.stats.norm()
    
    zl = bias_correction + normal.ppf(alpha / 2)
    alpha1 = normal.cdf(bias_correction + zl / (1 - accel_param * zl))

    zu = bias_correction + normal.ppf(1 - alpha / 2)
    alpha2 = normal.cdf(bias_correction + zu / (1 - accel_param * zu))
    return alpha1, alpha2


def bca_estimator(pointwise_scores, confidence_level, n_bootstrap, n_samples):
    """
    Bias-corrected-accelerated estimator. More accurate and sample efficient.
    """
    sample_scores = bootstrap_avg(pointwise_scores, n_bootstrap, n_samples)
    global_avg_score = pointwise_scores.mean()
    bias_correction = bca_bias_correction(sample_scores, global_avg_score)
    accel_param = bca_accel_param(pointwise_scores)
    quantile_lo, quantile_hi = compute_bca_quantiles(confidence_level, bias_correction, accel_param)
    lo = np.quantile(sample_scores, quantile_lo)
    hi = np.quantile(sample_scores, quantile_hi)
    return lo, hi, sample_scores


def percentile_estimator(pointwise_scores, confidence_level, n_bootstrap, n_samples):
    """
    Vanilla percentile estimator. Takes the quantile position's value in sorted scores.
    """
    sample_scores = bootstrap_avg(pointwise_scores, n_bootstrap, n_samples)
    lo = np.quantile(sample_scores, (1 - confidence_level)/2)
    hi = np.quantile(sample_scores, (1 + confidence_level)/2)
    return lo, hi, sample_scores


# def cluster_percentile(preds, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap):
#     cluster_scores = []
#     dmat = np.stack((cluster, preds, labels), axis=-1)
#     dmat = groupby(dmat)
#     for cluster in dmat:
#         cluster_preds, cluster_labels = cluster[:, 1], cluster[:, 2]
#         cluster_score = score_func(cluster_labels, cluster_preds)
#         cluster_scores.append(cluster_score)

#     boot_scores = resample(cluster_scores, n_samples=sample_size)
#     boot_scores = sorted(boot_scores)
#     lower = np.quantile(boot_scores, (1-confidence_level)/2)
#     upper = np.quantile(boot_scores, (1+confidence_level)/2)
#     log(score_func(labels, preds), lower, upper, 1-confidence_level, sample_size, num_bootstrap, 'cluster percentile')
#     return lower, upper, boot_scores, score_func(labels, preds)
