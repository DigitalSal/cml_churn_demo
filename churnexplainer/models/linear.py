from sklearn.linear_model import LogisticRegressionCV

def make_clf(*args, **kwargs):
    clf = LogisticRegressionCV()
    return clf
