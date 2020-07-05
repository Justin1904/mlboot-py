from sklearn.utils import resample


def resample_and_average(scores, n_samples):
    return resample(scores, n_samples=n_samples).mean()


def resample_and_score(y_pred, y_true, score_fn, n_samples):
    return score_fn(resample(y_pred, y_true, n_samples=n_samples))
