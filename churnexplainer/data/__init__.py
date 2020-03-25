import os
import importlib

dataset = os.environ.get('CHURN_DATASET', 'telco')
load_dataset = (importlib
                .import_module('churnexplainer.data.' + dataset)
                .load_dataset)
