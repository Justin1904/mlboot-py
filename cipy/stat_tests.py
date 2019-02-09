import numpy as np
import scipy
import sklearn

from sklearn.utils import resample
from utils import get_ci, get_metric

def SignificanceTest(pred1, labels, score_func, pred2=None, type_of_ci='bca', confidence_level=0.95, sample_size=None, num_bootstrap=2000):
    
    # check the validity of arguments
    assert len(pred1) == len(labels), f"There are {len(pred1)} predictions but {len(labels)} ground truth entries."

    # this could be dealt with inside tests
    if pred2 is not None:
        assert len(pred1) == len(pred2), f"There are {len(pred1)} predictions from model 1 but {len(pred2)} predictions from model 2."

    assert 0.0 < confidence_level < 1.0, "Confidence level must be within range of [0.0, 1.0]"

    # get the score function if it is supported by sklearn
    if isinstance(score_func, str):
        try:
            score_func = get_metric(score_func)
        except AttributeError:
            print(f"Specified metric \"{score_func}\" is not supported. Please refer to the documentation for available metrics or build your own.")
            exit(0)

    # get the bootstrap sample size if not specified
    if sample_size is None:
        sample_size = len(labels)

    if pred2 is None:
        preds = (pred1,)
    else:
        preds = (pred1, pred2)

    # run the statistical test
    ci_func = get_ci(type_of_ci)
    lo, hi, scores = ci_func(*preds, labels, score_func, confidence_level, sample_size, num_bootstrap)
    return lo, hi, scores
