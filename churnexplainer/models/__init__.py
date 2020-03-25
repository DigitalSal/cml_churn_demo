import importlib
import os


def make_clf(*args, **kwargs):
    model_type = os.environ.get('CHURN_MODEL_TYPE', 'linear')
    return (importlib
            .import_module('churnexplainer.models.' + model_type)
            .make_clf(*args, **kwargs))
