import numpy as np
import scipy

from sklearn.utils import resample
from sklearn.metrics import get_scorer
from confidence_intervals import CI_TYPES

def SignificanceTest(pred1, labels, score_func, pred2=None, type_of_ci='bca', confidence_level=0.95, sample_size=None, num_bootstrap=2000):
    
    # check the validity of arguments
    assert len(pred1) == len(labels), f"There are {len(pred1)} predictions but {len(labels)} ground truth entries."

    # this could be dealt with inside tests
    if pred2 is not None:
        assert len(pred1) == len(pred2), f"There are {len(pred1)} predictions from model 1 but {len(pred2)} predictions from model 2."

    assert type_of_ci in CI_TYPES, f"Unsupported confidene interval type. Supported types are :{', '.join(CI_TYPES)}."

    assert 0.0 < confidence_level < 1.0, "Confidence level must be within range of [0.0, 1.0]"

    # get the score function if it is supported by sklearn
    if isinstance(score_func, str):
        score_func = get_scorer(score_func)

    # get the bootstrap sample size if not specified
    if sample_size is None:
        sample_size = len(labels)

    # calculate the scores across entire set
    #scores1 = score_func(pred1, labels)
    #if pred2 is not None:
    #    scores2 = score_func(pred2, labels)
    #    scores = (score1, scores2)
    #else:
    #    scores = (score1,)

    if pred2 is None:
        preds = (pred1,)
    else:
        preds = (pred1, pred2)

    # run the statistical test
    ci_func = CI_TYPES[type_of_ci]
    test_measure = ci_func(*scores, labels, score_func, confidence_level, sample_size, num_bootstrap)
    return test_measure
