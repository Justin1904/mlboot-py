import numpy as np
import scipy
import sklearn

from sklearn.utils import resample
from mlboot.utils import get_ci, get_metric
from pdb import set_trace

def BootstrapCI(pred1, labels, score_func, pred2=None, cluster=None, type_of_ci='bca', confidence_level=0.95, sample_size=None, num_bootstrap=2000):

    # ensure all input are converted into numpy for convenience reasons
    pred1 = np.array(pred1)
    labels = np.array(labels)

    if pred2 is not None:
        pred2 = np.array(pred2)

    if cluster is not None:
        cluster = np.array(cluster)

    # check the validity of arguments
    assert len(pred1) == len(labels), f"There are {len(pred1)} predictions but {len(labels)} ground truth entries."

    # check if the second model has same number of outputs
    if pred2 is not None:
        assert len(pred1) == len(pred2), f"There are {len(pred1)} predictions from model 1 but {len(pred2)} predictions from model 2."

    # check if we are using the correct ci method
    if type_of_ci.startswith("paired") and pred2 is None:
        raise ValueError("Predictions from a second model is required to compute paired confidence intervals.")

    if not type_of_ci.startswith("paired") and pred2 is not None:
        raise ValueError("Non-paired confidence intervals cannot be applied to a pair of model outputs.")

    assert 0.0 < confidence_level < 1.0, "Confidence level must be within range of [0.0, 1.0]"

    if cluster is None and type_of_ci.startswith('cluster'):
        raise ValueError("If no clustering info is provided please use non-clustered CI methods for better performance.")

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
    lower, upper, scores, *full_score = ci_func(*preds, labels, score_func, cluster, confidence_level, sample_size, num_bootstrap)
    return lower, upper, scores, full_score


