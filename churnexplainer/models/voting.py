import numpy as np
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


def make_clf(num_features, *args, **kwargs):
    voters = (
        ('lr', LogisticRegression()),
        ('nn', MLPClassifier()),
        ('rf', RandomForestClassifier(n_estimators=100))
    )
    vclf = VotingClassifier(voters, voting='soft')
    lr_param_dist = {"C": list(np.power(10.0, np.arange(-5, 5)))}
    rf_param_dist = {"max_depth": [6, None],
                     "max_features": sp_randint(1, num_features),
                     "min_samples_split": sp_randint(2, 50),
                     "min_samples_leaf": sp_randint(1, 50),
                     "bootstrap": [True, False],
                     "criterion": ["gini", "entropy"]}
    nn_param_dist = {"alpha": [1e-5, 1e-4, 1e-6],
                     "hidden_layer_sizes": [(15,), (100,), (100, 100),
                                            (100, 100, 100, 100),
                                            (20, 20, 20, 20)],
                     "early_stopping": [True, False],
                     "solver": ["adam"],
                     "activation": ["relu", "tanh", "logistic"]}
    param_distributions = {}
    for (prefix, _), clf_param_dist in zip(voters, (lr_param_dist,
                                                    nn_param_dist,
                                                    rf_param_dist)):
        param_distributions.update({'__'.join((prefix, k)): v
                                    for k, v in clf_param_dist.items()})

    return RandomizedSearchCV(vclf, param_distributions, n_iter=300,
                              verbose=10, n_jobs=-2)
