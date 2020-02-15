import numpy as np
import scipy
import sklearn

from sklearn.utils import resample
from mlboot.utils import get_ci, get_metric
from pdb import set_trace

def BootstrapCI(pred1, labels1, score_func, pred2=None, labels2=None, cluster=None, type_of_ci='bca', confidence_level=0.95, sample_size=None, num_bootstrap=2000):

    # ensure all input are converted into numpy for convenience reasons
    pred1 = np.array(pred1)
    labels1 = np.array(labels1)

    if pred2 is not None:
        pred2 = np.array(pred2)

    if labels2 is not None:
        labels2 = np.array(labels2)

    if pred2 is None and labels2 is not None:
        raise ValueError("Second set of labels cannot be applied when there is no second set of input.")

    if cluster is not None:
        cluster = np.array(cluster)

    # check the validity of arguments
    assert len(pred1) == len(labels1), f"There are {len(pred1)} predictions but {len(labels1)} ground truth entries."

    # check if the second model has same number of outputs
    if pred2 is not None:
        assert len(pred1) == len(pred2), f"There are {len(pred1)} predictions from model 1 but {len(pred2)} predictions from model 2."

    if labels2 is not None:
        assert len(pred2) == len(labels2), f"There are {len(pred2)} predictions for model 2 but only {len(labels2)} ground truth entries for it."

    # check if we are using the correct ci method
    if type_of_ci.startswith("paired") and pred2 is None:
        raise ValueError("Predictions from a second model is required to compute paired confidence intervals.")

    if not type_of_ci.startswith("paired") and (pred2 is not None or labels2 is not None):
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
        sample_size = len(labels1)

    if pred2 is None:
        preds = (pred1,)
    else:
        preds = (pred1, pred2)

    if labels2 is None:
        labels = (labels1, None)
    else:
        labels = (labels1, labels2)

    # run the statistical test
    ci_func = get_ci(type_of_ci)
    lower, upper, scores, *full_score = ci_func(*preds, *labels, score_func, cluster, confidence_level, sample_size, num_bootstrap)
    return lower, upper, scores, full_score


CI_METHOD_DICT = {
    'percentile': percentile_estimator,
    'bca': bias_corrected_accelerated_estimator,
}


class ConfidenceIntervalEstimator:
    """
    Base class for confidence interval estimator. Provides some common utils as well as the basic functionality of estimating
    confidence interval of performance of a single model.

    Args:
         - score_fn [Callable]: a scoring function that scores each prediction and each ground truth. An important assumption is that 
                                the overall score of a set of predictions will be the average of scores of each prediction.
         - method [str]: 'bca'/'percentile', the method to choose for estimating the confidence interval, default to 'bca'.
         - confidence_level [float]: the confidence level of the estimation. The more confidence, the wider the interval. Default to 0.95.
         - n_bootstrap [int]: the number of Bootstrap sampling to do for estimating the confidence interval. Default to 2000.
         - n_samples [int]: the sample size for each Bootstrap sampling. Default to None, will use y_pred.shape[0] as n_samples in this case.

    Input:
         - y_pred [np.ndarray]: an array of predictions, batch dimension must be on the first dimension.
         - y_true [np.ndarray]: an array with identically-sized batch dimension as y_pred.
         - return_samples [bool]: whether or not to return the scores on Bootstrap-sampled subsets.

    Return:
         - global_avg_score: the average score over the entire set of predictions.
         - (ci_lo, ci_hi): a tuple of lower end and higher end of the confidence interval.
         - avg_score_samples (Optional): the average scores on the Bootstrap-sampled subsets.
    """
    def __init__(self, score_fn, method='bca', confidence_level=0.95, n_bootstrap=2000, n_samples=None):
        self.score_fn = score_fn
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.n_samples = n_samples  # if set to None, will be dynamically set as it processes input of different sizes
        self.method = method
        self.ci_estimator = CI_METHOD_DICT[method]

    def _compute_scores(self, y_pred, y_true):
        pointwise_scores = self.score_fn(y_pred, y_true)
        assert isinstance(pointwise_scores, np.ndarray), f"score_fn output must be of type np.ndarray, but received type {type(pointwise_scores)} instead"
        assert pointwise_scores.shape[0] == len(y_pred), f"score_fn output has {pointwise_scores.shape[0]} entries, yet y_pred has {y_pred.shape[0]} entries"
        global_avg_score = pointwise_scores.mean()
        return pointwise_scores, global_avg_score

    def _sample_avg_scores(self, scores):
        # TODO: Figure out a more efficient method for this function
        avg_score_samples = []
        n_samples = scores.shape[0] if self.n_samples is None else self.n_samples
        while len(avg_score_samples) < self.n_bootstrap:
            score_samples = resample(scores, n_samples=n_samples)
            avg_score_samples.append(score_samples.mean())
        return avg_score_samples

    def __call__(self, y_pred, y_true, return_samples=False):
        pointwise_scores, global_avg_score = self._compute_scores(y_pred, y_true)
        avg_score_samples = self._sample_avg_scores(pointwise_scores)
        ci_lo, ci_hi = self.ci_estimator(avg_score_samples, self.confidence_level, return_samples=return_samples)
        if return_samples:
            return global_avg_score, (ci_lo, ci_hi), avg_score_samples
        else:
            return global_avg_score, (ci_lo, ci_hi)


class PairedConfidenceIntervalEstimator(ConfidenceIntervalEstimator):
    """
    Class for paired confidence interval estimator. This is used when you want to compare the differences between two models.

    Args:
         - score_fn [Callable]: a scoring function that scores each prediction and each ground truth. An important assumption is that 
                                the overall score of a set of predictions will be the average of scores of each prediction.
         - method [str]: 'bca'/'percentile', the method to choose for estimating the confidence interval, default to 'bca'.
         - confidence_level [float]: the confidence level of the estimation. The more confidence, the wider the interval. Default to 0.95.
         - n_bootstrap [int]: the number of Bootstrap sampling to do for estimating the confidence interval. Default to 2000.
         - n_samples [int]: the sample size for each Bootstrap sampling. Default to None, will use y_pred.shape[0] as n_samples in this case.

    Input:
         - y_pred [np.ndarray]: an array of predictions, batch dimension must be on the first dimension.
         - y_true [np.ndarray]: an array with identically-sized batch dimension as y_pred.
         - return_samples [bool]: whether or not to return the scores on Bootstrap-sampled subsets.

    Return:
         - delta_global_avg_score: the average score delta over the entire set of predictions. Positive means score_<ours> > score_<baseline>.
         - (ci_lo, ci_hi): a tuple of lower end and higher end of the confidence interval.
         - avg_delta_score_samples (Optional): the average score deltas on the Bootstrap-sampled subsets.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, y_pred_ours, y_pred_baseline, y_true, return_samples=False):
        pointwise_scores_ours, global_avg_score_ours = self._compute_scores(y_pred, y_true_ours)
        pointwise_scores_baseline, global_avg_score_baseline = self._compute_scores(y_pred_baseline, y_true)
        delta_pointwise_scores = pointwise_scores_ours - pointwise_scores_baseline
        delta_global_avg_score = global_avg_score_ours - global_avg_score_baseline
        avg_delta_score_samples = self._sample_avg_scores(delta_pointwise_scores)
        ci_lo, ci_hi = self.ci_estimator(delta_pointwise_scores, self.confidence_level, return_samples=return_samples)
        if return_samples:
            return delta_global_avg_score, (ci_lo, ci_hi), avg_delta_score_samples
        else:
            return delta_global_avg_score, (ci_lo, ci_hi)


class ClusteredConfidenceIntervalEstimator(ConfidenceIntervalEstimator):
    """
    Class for clustered confidence interval estimator. This applies to cases where the predictions are clustered/stratified. A normal Bootstrap
    estimate may cause some bias in this case whereas a clustered version would correct for these biases.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
