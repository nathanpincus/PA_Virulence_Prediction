"""
Function for plotting learning curves
Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
with the following changes:
        Changed default training sizes to 0.25 to 1 in 10 steps (from .1 to 1 in 5 steps)
        Set shuffle=True (important when training data is ordered - e.g. all 1's first)
        Set error_score=np.nan
        Added variable to set random state for repeatable results
        Added variable for scoring (default = None) so that I can score by
            parameters other than accuracy (e.g. f1) if desired
        Added option to change y label if desired (e.g. if different score considered)
        Changed error fill from std to 95% CI of the mean
        Changed default to not plot a title
"""

#For data inport/proccessing
import numpy as np
import pandas as pd
import scipy.stats

#For machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

#For plotting
import matplotlib.pyplot as plt


def plot_learning_curve(estimator, X, y, rs, title=None, scoring=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.25, 1.0, 10), ylab="Score"):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    if title is not None:
        plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(ylab)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, error_score=np.nan, scoring=scoring, shuffle=True, random_state=rs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_sem = scipy.stats.sem(train_scores, axis=1)
    train_scores_CI_height = train_scores_sem * scipy.stats.t.ppf((1 + 0.95) / 2, np.size(train_scores, 1) - 1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_sem = scipy.stats.sem(test_scores, axis=1)
    test_scores_CI_height = test_scores_sem * scipy.stats.t.ppf((1 + 0.95) / 2, np.size(test_scores, 1) - 1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_CI_height,
                     train_scores_mean + train_scores_CI_height, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_CI_height,
                     test_scores_mean + test_scores_CI_height, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
