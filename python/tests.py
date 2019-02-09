import numpy as np
import timeit
import time

from sklearn.metrics import mean_absolute_error
from confidence_intervals import percentile, paired_percentile

def generate_data(n):
    labels = np.random.randn(n, 1) * 3 + 1.63
    preds1 = labels + np.random.randn(n, 1) * 0.05 + 0.05
    preds2 = labels + np.random.randn(n, 1) * 0.05 + 0.03
    return preds1, preds2, labels

def test_percentile(preds1, labels):
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = 0.95
    num_bootstrap = 2000
    a, b = percentile(preds1, labels, score_func, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval: [{a}, {b}].")
    return True

def test_paired_percentile(preds1, preds2, labels):
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = .95
    num_bootstrap = 2000
    a, b = paired_percentile(preds1, preds2, labels, score_func, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval of difference in model: [{a}, {b}]")
    return True


if __name__ == "__main__":
    # timeit.timeit('test_percentile()', number=1000)
    n = 12000
    n_runs = 100

    preds1, preds2, labels = generate_data(n)

    # test percentile
    # start = time.time()
    # for _ in range(n_runs): test_percentile(preds1, labels)
    # gap = time.time() - start
    # print(f"Running percentile test on {n} data points on average takes {gap / n_runs} seconds")

    # test paird percentile
    start = time.time()
    for _ in range(n_runs): test_paired_percentile(preds1, preds2, labels)
    gap = time.time() - start
    print(f"Running paired percentile test on {n} data points on average takes {gap / n_runs} seconds")
