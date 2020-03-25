import pandas as pd

from churnexplainer.data import load_dataset

customers, labels = load_dataset()


def test_data_format():
    assert isinstance(customers, pd.DataFrame)
