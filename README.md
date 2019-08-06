#### this repo is currently undergoing some changes and the code may not run as expected, please come back later!

# mlboot

The goal of *mlboot* is to provide a powerful, flexible, and
user-friendly way to estimate and compare the performance of machine
learning (and other predictive) models using bootstrap resampling. It
was created in collaboration by [Jeffrey Girard](https://jmgirard.com/)
and [Zhun Liu](http://justin1904.github.io/); this version for Python is
maintained by Zhun Liu and a similar version for R is
maintained by Jeff Girard [here](https://github.com/jmgirard/mlboot).

## Installation

First clone this repo:

``` bash
git clone https://github.com/Justin1904/mlboot-py.git
```

then add the path to the `mlboot` folder of the cloned repo to your `PYTHONPATH` environment variable. For example, on Ubuntu you could add the following line to your `.bashrc` file:

``` bash
export PYTHONPATH="/path/to/mlboot-py/mlboot:$PYTHONPATH"
```

just replace `/path/to/mlboot-py/mlboot` with the actual path you cloned the repo to.

(This is a early version and was a prototype developed within the duration of a Hackathon. There are quite some redundancy in code and some low-hanging fruit in performance optimization. We will be working on perfecting it in the future, but meanwhile feel free to contribute if you have great suggestions and ideas on how to improve it.)

## Usage


### Getting Started

This package aims to provide easy to use confidence interval estimations for model performances. To start with, we generate some toy regression results and compute the overall mean absolute error

``` python
import numpy as np
np.random.seed(1)

labels = np.random.randn(3000) * 0.66 + 1.32
predictions = labels + np.random.randn(3000) * 0.15 + 0.2

mae = np.abs(labels - predictions).mean()
print(mae)
```

on my machine this yields `0.2143` mean absolute error. However, what if we want to know how confident we are about this performance metric? This can be easily achieved through `mlboot`.


``` python
from mlboot import BootstrapCI

lo, hi, scores = BootstrapCI(predictions, labels, 'mean_absolute_error')
```

a report will be printed to console automatically:

``` 
========================================
Confidence Interval: [0.2097, 0.2191], confidence level: 0.95
Number of samples in each bootstrap: 3000
Number of total bootstrap runs: 2000
Confidence interval type: BCa
========================================
```

it shows the ***confidence*** interval, the ***confidence level*** of this interval, as well as the bootstrap settings such as ***sample sizes*** and ***number of bootstrap runs***. It also shows the ***method*** used for computing confidence interval. Here (by default) it is `BCa`, bias-corrected accelerated confidence interval. As you may have guessed, each of these mentioned parameters are optionally tweakable arguments for `BootstrapCI`. The API is as follows:

```
def BootstrapCI(pred1, labels, score_func, pred2=None, cluster=None, type_of_ci='bca', confidence_level=0.95, sample_size=None, num_bootstrap=2000)

--pred1: the predictions of the model you want to analyze.

--labels: the ground truth labels.

--score_func: a score function that evaluates your predictions against the true labels. Could be a string matching the name of functions provided by sklearn.metrics, or it could be a custom evaluation function.

--pred2: the predictions of a different model you want to compare to. Can only be used with paired bootstrap confidence intervals.

--cluster: an array of same length as input, indicating the clustering of your data. Can only be used with cluster bootstrap confidence intervals.

--type_of_ci: a string indicating the type of confidence interval method to use here. Currently support "bca", "paired_bca", "percentile", "paired_percentile", "cluster_percentile".

--confidence_level: the confidence level you want for your interval. The higher the confidence, the wider the interval will be.

--sample_size: defaults to the length of input.

--num_bootstrap: the number of bootstrap runs to execute. Usually need to be larger than 1,000.
```

### Paired Bootstrap

In some scenarios you want to compare the results from two models to see if they are statistically significantly different. In event of this, `mlboot` offers paired Bootstrap confidence intervals to examine that. We can do that by passing in the predictions of the other model via argument `pred2` and specify the `type_of_ci` to be any of the paired methods.

``` python
'''
Assume you have two sets of predictions 'pred1', 'pred2' and ground truth is in 'labels'
'''

lo, hi, scores = BootstrapCI(pred1, labels, 'accuracy_score', pred2=pred2, type_of_ci="paired_percentile")

```


### Score functions

The score functions `score_func` can be passed in as a string as long as `sklearn.metrics` contains a scoring function with the same name. However, in some cases such as QA, you might want to have more complex evaluation such as BLEU or ROUGE. In this case you could pass in a custome scoring function. For example if you are working on a machine translation system, it is possible to do

``` python
'''
Assume you have translations from models in a list named 'preds', and true translations in 'labels'
'''
from nltk.translate.bleu_score import corpus_bleu

lo, hi, scores = BootstrapCI(preds, labels, corpus_bleu)
```


## Code of Conduct

Please note that the ‘mlboot’ project is released with a [Contributor
Code of Conduct](.github/CODE_OF_CONDUCT.md). By contributing to this
project, you agree to abide by its terms.

## References

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the
bootstrap.* New York, NY: Chapman and Hall.

Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data.
*Journal of the Royal Statistical Society: Series B (Statistical
Methodology), 69*(3), 369–390. <https://doi.org/10/cqwx5p>

Ren, S., Lai, H., Tong, W., Aminzadeh, M., Hou, X., & Lai, S. (2010).
Nonparametric bootstrapping for hierarchical data. *Journal of Applied
Statistics, 37*(9), 1487–1498. <https://doi.org/10/dvfzcn>
