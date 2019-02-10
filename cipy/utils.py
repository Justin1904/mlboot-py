import confidence_intervals as cis
from sklearn import metrics

def get_metric(metric_name):
    metric = metrics.__getattribute__(metric_name)
    return metric

def get_ci(ci_name):
    ci = cis.__getattribute__(ci_name)
    return ci

