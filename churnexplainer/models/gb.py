import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def make_clf(*args, **kwargs):
    clf = GradientBoostingClassifier()
    param_dist = {"learning_rate": list(np.power(10.0, np.arange(-3, 2, 10))),
                  "n_estimators": [50, 100, 500, 1000],
                  "loss": ["deviance", "exponential"],
                  "max_depth": list(np.arange(2, 5))}
    return GridSearchCV(clf, param_dist, verbose=10, n_jobs=-2)
