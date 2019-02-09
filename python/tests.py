import numpy as np
import timeit
import time

from sklearn.metrics import mean_absolute_error
from confidence_intervals import percentile

def test_percentile():
    n = 2000
    labels = np.random.randn(2000, 1) * 3 + 1.63
    preds = labels + np.random.randn(2000, 1) * 0.05
    score_func = mean_absolute_error
    sample_size = len(labels)
    confidence_level = 0.95
    num_bootstrap = 2000
    a, b = percentile(preds, labels, score_func, confidence_level, sample_size, num_bootstrap)
    print(f"Confidence interval: [{a}, {b}].")
    return True

if __name__ == "__main__":
    # timeit.timeit('test_percentile()', number=1000)
    start = time.time()
    for _ in range(20): test_percentile()
    gap = time.time() - start
    print(f"Running percentile test on 100 data points on average takes {gap / 100} seconds")
