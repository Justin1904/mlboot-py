import numpy as np
from mlboot.confidence_intervals import (
    percentile_estimator,
    bca_estimator,
)


CI_METHOD_DICT = {
    "percentile": percentile_estimator,
    "bca": bca_estimator,
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
         - n_workers [int]: the number of workers in any multi-processing scenarios. Defaults to 4.
         - reducible [bool]: when set to True, there are some optimizations to make internal computations faster. CAVEATS APPLY:

                             NOTE: reducible=True can only work correctly when:
                             1) the population-wide score can be represented as an average over score-per-datapoint
                                (e.g. accuracy is the average of 0-1 score, MSE is the average of squared error for each datapoint);
                             2) score_fn should return the score-per-datapoint in an 1-D array, rather than directly
                                returning the population-wide score.

                             Violating 2) would trigger an error, while violating 1) would simply give you incorrect results.
                             If you are unsure what this means, keep reducible=False to ensure correctness. Note that not all
                             metrics are reducible, examples are recall, precision, F1, where the population-wide metric is not
                             in the form of an average over score-per-datapoint.

    Input:
         - y_pred [np.ndarray]: an array of predictions, batch dimension must be on the first dimension.
         - y_true [np.ndarray]: an array with identically-sized batch dimension as y_pred.
         - return_samples [bool]: whether or not to return the scores on Bootstrap-sampled subsets.

    Return:
         - global_score: the average score over the entire set of predictions.
         - (ci_lo, ci_hi): a tuple of lower end and higher end of the confidence interval.
         - avg_score_samples (Optional): the average scores on the Bootstrap-sampled subsets.
    """

    def __init__(
        self,
        score_fn,
        method="bca",
        confidence_level=0.95,
        n_bootstrap=2000,
        n_samples=None,
        n_workers=4,
        reducible=False,
    ):
        self.score_fn = score_fn
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.n_samples = n_samples  # if set to None, will be dynamically set as it processes input of different sizes
        self.n_workers = n_workers
        self.reducible = reducible
        self.method = method
        self.estimator = CI_METHOD_DICT[method]

    def _compute_scores(self, y_pred, y_true):
        pointwise_scores = self.score_fn(y_pred, y_true)
        assert isinstance(
            pointwise_scores, np.ndarray
        ), f"score_fn output must be of type np.ndarray, but received type {type(pointwise_scores)} instead"
        assert pointwise_scores.shape[0] == len(
            y_pred
        ), f"score_fn output has {pointwise_scores.shape[0]} entries, yet y_pred has {y_pred.shape[0]} entries"
        return pointwise_scores

    def __call__(self, y_pred, y_true, return_samples=False):
        if self.reducible:
            pointwise_scores = self._compute_scores(y_pred, y_true)
        else:
            pointwise_scores = None

        n_samples = y_pred.shape[0] if self.n_samples is None else self.n_samples
        ci_lo, ci_hi, avg_score_samples, global_score = self.estimator(
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            n_samples=n_samples,
            pointwise_scores=pointwise_scores,
            y_pred=y_pred,
            y_true=y_true,
            score_fn=self.score_fn,
        )

        if return_samples:
            return global_score, (ci_lo, ci_hi), avg_score_samples
        else:
            return global_score, (ci_lo, ci_hi)


class PairedConfidenceIntervalEstimator(ConfidenceIntervalEstimator):
    """
    Class for paired confidence interval estimator. This is used when you want to compare the differences between two models.

    Input:
         - y_pred_ours [np.ndarray]: an array of predictions from our model, batch dimension must be on the first dimension.
         - y_pred_baseline [np.ndarray]: an array of predictions from a baseline model, batch dimension must be on the first dimension.
         - y_true [np.ndarray]: an array with identically-sized batch dimension as y_pred.
         - return_samples [bool]: whether or not to return the scores on Bootstrap-sampled subsets.

    Return:
         - delta_global_score: the average score delta over the entire set of predictions. Positive means score_<ours> > score_<baseline>.
         - (ci_lo, ci_hi): a tuple of lower end and higher end of the confidence interval.
         - avg_delta_score_samples (Optional): the average score deltas on the Bootstrap-sampled subsets.
    """

    def __call__(self, y_pred_ours, y_pred_baseline, y_true, return_samples=False):
        pointwise_scores_ours = self._compute_scores(
            y_pred_ours, y_true
        )
        pointwise_scores_baseline = self._compute_scores(
            y_pred_baseline, y_true
        )

        n_samples = y_pred_ours.shape[0] if self.n_samples is None else self.n_samples
        delta_pointwise_scores = pointwise_scores_ours - pointwise_scores_baseline
        ci_lo, ci_hi, avg_delta_score_samples, delta_global_score = self.estimator(
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            n_samples=n_samples,
            pointwise_scores=delta_pointwise_scores,
            y_pred=y_pred_ours,
            y_pred_baseline=y_pred_baseline,
            y_true=y_true,
            score_fn=self.score_fn,
        )
        if return_samples:
            return delta_global_score, (ci_lo, ci_hi), avg_delta_score_samples
        else:
            return delta_global_score, (ci_lo, ci_hi)


class ClusteredConfidenceIntervalEstimator(ConfidenceIntervalEstimator):
    """
    Class for clustered confidence interval estimator. This applies to cases where the predictions are clustered/stratified. A normal Bootstrap
    estimate may cause some bias in this case whereas a clustered version would correct for these biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError
