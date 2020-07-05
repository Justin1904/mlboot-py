from mlboot.utils import resample_and_average, resample_and_score
import numpy as np
import scipy
from joblib import Parallel, delayed


def bootstrap_avg_score(samples, n_bootstrap, n_samples, n_workers):
    """
    Bootstrap `n_bootstrap` sub-samples from `samples` each with sample size `n_samples`,
    average them and return.
    """
    avg_score_samples = Parallel(n_jobs=n_workers)(
        delayed(resample_and_average)(samples, n_samples=n_samples)
        for _ in range(n_bootstrap)
    )
    return avg_score_samples


def bootstrap_compute_score(y_pred, y_true, score_fn, n_bootstrap, n_samples, n_workers):
    """
    Bootstrap 'n_bootstrap' sub-samples from `y_pred` and 'y_true`, each with sample size `n_samples`,
    use `score_fn` to compute a single-scalar score for each sub-sample and return.
    """
    computed_score_samples = Parallel(n_jobs=n_workers)(
        delayed(resample_and_score)(y_pred, y_true, score_fn, n_samples=n_samples)
        for _ in range(n_bootstrap)
    )
    return computed_score_samples


def bootstrap_diff_score(y_pred, y_pred_baseline, y_true, score_fn, n_bootstrap, n_samples, n_workers):
    """
    Bootstrap 'n_bootstrap' sub-samples from `y_pred`, `y_pred_baseline` and 'y_true`, each with sample size `n_samples`,
    use `score_fn` to compute a single-scalar score for each sub-sample, take their difference and return.
    """
    diff_score_samples = Parallel(n_jobs=n_workers)(
        delayed(resample_and_score)(y_pred, y_true, score_fn, n_samples=n_samples) -
        delayed(resample_and_score)(y_pred_baseline, y_true, score_fn, n_samples=n_samples)
        for _ in range(n_bootstrap)
    )
    return diff_score_samples


def jackknife_avg_score(samples):
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


def jackknife_compute_score(y_pred, y_true, score_fn, n_workers):
    """
    Jackknife (leave-one-out resampling) sub-samples from `y_pred` and `y_true`,
    use `score_fn` to comptue the scores and return.
    """
    n_samples = y_true.shape[0]
    jackknife_samples = Parallel(n_jobs=n_workers)(
        delayed(score_fn)(y_pred[np.arange(n_samples) != i], y_true[np.arange(n_samples) != i])
        for i in range(n_samples)
    )
    return jackknife_samples


def jackknife_diff_score(y_pred, y_pred_baseline, y_true, score_fn, n_workers):
    """
    Jackknife (leave-one-out resampling) sub-samples from `y_pred`, `y_pred_baseline` and `y_true`,
    use `score_fn` to comptue the scores for `y_pred` and `y_true`, take their difference and return.
    """
    n_samples = y_true.shape[0]
    jackknife_samples = Parallel(n_jobs=n_workers)(
        delayed(score_fn)(y_pred[np.arange(n_samples) != i], y_true[np.arange(n_samples) != i]) -
        delayed(score_fn)(y_pred_baseline[np.arange(n_samples) != i], y_true[np.arange(n_samples) != i])
        for i in range(n_samples)
    )
    return jackknife_samples


def bca_bias_correction(scores, full_score):
    scores = np.array(scores)
    entries_less = (scores < full_score).sum()
    prop_less = entries_less / scores.shape[0]
    normal = scipy.stats.norm()
    bias_correction = normal.ppf(prop_less)
    return bias_correction


def bca_accel_param(jackknife_scores):
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


def bca_estimator(confidence_level, n_bootstrap, n_samples, n_workers, pointwise_scores=None, y_pred=None, y_pred_baseline=None, y_true=None, score_fn=None):
    """
    Bias-corrected-accelerated estimator. More accurate and sample efficient.
    """
    if pointwise_scores is not None:
        sample_scores = bootstrap_avg_score(pointwise_scores, n_bootstrap, n_samples, n_workers)
        global_score = pointwise_scores.mean()
        jackknife_scores = jackknife_avg_score(pointwise_scores)
    elif y_pred_baseline is None:
        sample_scores = bootstrap_compute_score(y_pred, y_true, score_fn, n_bootstrap, n_samples, n_workers)
        global_score = score_fn(y_pred, y_true)
        jackknife_scores = jackknife_compute_score(y_pred, y_true, score_fn, n_workers)
    else:
        sample_scores = bootstrap_diff_score(y_pred, y_pred_baseline, y_true, score_fn, n_bootstrap, n_samples, n_workers)
        global_score = score_fn(y_pred, y_true) - score_fn(y_pred_baseline, y_true)
        jackknife_scores = jackknife_diff_score(y_pred, y_pred_baseline, y_true, score_fn, n_workers)

    bias_correction = bca_bias_correction(sample_scores, global_score)
    accel_param = bca_accel_param(jackknife_scores)
    quantile_lo, quantile_hi = compute_bca_quantiles(confidence_level, bias_correction, accel_param)
    lo = np.quantile(sample_scores, quantile_lo)
    hi = np.quantile(sample_scores, quantile_hi)
    return lo, hi, sample_scores


def percentile_estimator(confidence_level, n_bootstrap, n_samples, n_workers, pointwise_scores=None, y_pred=None, y_pred_baseline=None, y_true=None, score_fn=None):
    """
    Vanilla percentile estimator. Takes the quantile position's value in sorted scores.
    """
    if pointwise_scores is not None:
        sample_scores = bootstrap_avg_score(pointwise_scores, n_bootstrap, n_samples, n_workers)
        global_score = pointwise_scores.mean()
    elif y_pred_baseline is None:
        sample_scores = bootstrap_compute_score(y_pred, y_true, score_fn, n_bootstrap, n_samples, n_workers)
        global_score = score_fn(y_pred, y_true)
    else:
        sample_scores = bootstrap_diff_score(y_pred, y_pred_baseline, y_true, score_fn, n_bootstrap, n_samples, n_workers)
        global_score = score_fn(y_pred, y_true) - score_fn(y_pred_baseline, y_true)
    lo = np.quantile(sample_scores, (1 - confidence_level) / 2)
    hi = np.quantile(sample_scores, (1 + confidence_level) / 2)
    return lo, hi, sample_scores, global_score


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
