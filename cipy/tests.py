import numpy as np
import timeit
import time

from sklearn.metrics import mean_absolute_error
from confidence_intervals import percentile, paired_percentile, bca, paired_bca, cluster_percentile
from mlboot import BootstrapCI

def generate_data(n):
    labels = np.random.randn(n) * 3 + 1.63
    preds1 = labels + np.random.randn(n) * 0.05 + 0.05
    preds2 = labels + np.random.randn(n) * 0.05 + 0.03
    return preds1, preds2, labels

def test_percentile(preds1, labels):
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = 0.95
    num_bootstrap = 2000
    cluster = np.arange(len(labels))
    a, b, scores = percentile(preds1, labels, score_func, np.arange(len(preds1)), confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval: [{a}, {b}].")
    return True

def test_bca(preds1, labels):
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = 0.95
    num_bootstrap = 2000
    cluster = np.arange(len(labels))
    a, b, scores = bca(preds1, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval: [{a}, {b}].")
    return True

def test_paired_percentile(preds1, preds2, labels):
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = .95
    num_bootstrap = 2000
    cluster = np.arange(len(labels))
    a, b, scores = paired_percentile(preds1, preds2, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval of difference in model: [{a}, {b}]")
    return True

def test_paired_bca(preds1, preds2, labels):
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = .95
    num_bootstrap = 2000
    cluster = np.arange(len(labels))
    a, b, scores = paired_bca(preds1, preds2, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval of difference in model: [{a}, {b}]")
    return True

def test_cluster_percentile(preds1, labels):
    score_func = mean_absolute_error
    sample_size = 1000
    confidence_level = .95
    num_bootstrap = 2000
    cluster = np.random.randint(0, 1000, (len(preds1)))
    a, b, scores = cluster_percentile(preds1, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval estimated with cluster percentile: [{a}, {b}]")

if __name__ == "__main__":
    # timeit.timeit('test_percentile()', number=1000)
    n = 12000
    n_runs = 1

    preds1, preds2, labels = generate_data(n)

    # test percentile
    start = time.time()
    for _ in range(n_runs): test_percentile(preds1, labels)
    gap = time.time() - start
    print(f"Running percentile test on {n} data points on average takes {gap / n_runs} seconds")

    # test paired percentile
    start = time.time()
    for _ in range(n_runs): test_paired_percentile(preds1, preds2, labels)
    gap = time.time() - start
    print(f"Running paired percentile test on {n} data points on average takes {gap / n_runs} seconds")
    
    # test bca
    start = time.time()
    for _ in range(n_runs): test_bca(preds1, labels)
    gap = time.time() - start
    print(f"Running BCa test on {n} data points on average takes {gap / n_runs} seconds")

    # test paired BCa
    start = time.time()
    for _ in range(n_runs): test_paired_bca(preds1, preds2, labels)
    gap = time.time() - start
    print(f"Running paired BCa test on {n} data points on average takes {gap / n_runs} seconds")
    
    start = time.time()
    for _ in range(n_runs): test_cluster_percentile(preds1, labels)
    gap = time.time() - start
    print(f"Running paired cluster percentile test on {n} data points on average takes {gap / n_runs} seconds")
    
    # test the unified API
    lo, hi, scores = BootstrapCI(preds1, labels, "mean_absolute_error")
    print(f"Running the unified API with BCa yields [{lo}, {hi}] interval.")
