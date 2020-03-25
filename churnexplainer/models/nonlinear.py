import itertools as it
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def make_clf(*args, **kwargs):
    clf = make_pipeline(FunctionTransformer(crossterm),
                        LogisticRegressionCV())
    return clf


def crossterm(X):
    return np.vstack([x*y for x, y in it.combinations(X.T, 2)]).T
