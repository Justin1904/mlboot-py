# mlboot

The goal of *mlboot* is to provide a powerful, flexible, and
user-friendly way to estimate and compare the performance of machine
learning (and other predictive) models using bootstrap resampling. It
was created in collaboration by [Jeffrey Girard](https://jmgirard.com/)
and [Zhun Liu](http://justin1904.github.io/); this version for Python is
maintained by Zhun Liu and a similar version for R is
maintained by Jeff Girard.

## Installation

First clone this repo:

``` bash
git clone https://github.com/Justin1904/mlboot-py.git
```

then add the path to the `cipy` folder of the cloned repo to your `PYTHONPATH` environment variable. For example, on Ubuntu you could add the following line to your `.bashrc` file:

``` bash
export PYTHONPATH="/path/to/mlboot-py/cipy:$PYTHONPATH"
```

just replace `/path/to/mlboot-py/cipy` with the actual path you cloned the repo to.

## Usage

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

it shows the ***confidence*** interval, the ***confidence level*** of this interval, as well as the bootstrap settings such as ***sample sizes*** and ***number of bootstrap runs***. It also shows the ***method*** used for computing confidence interval. Here by default it is `BCa`, bias-corrected accelerated confidence interval. As you may have guessed, each of these mentioned parameters are optionally tweakable arguments for `BootstrapCI`


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
