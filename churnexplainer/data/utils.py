import os
import dill

import pandas as pd

from churnexplainer.utils import data_dir


def categorize(df, cols):
    catcols = (c for c, iscat in cols if iscat)
    for col in catcols:
        df[col] = pd.Categorical(df[col])
    return df


def sanitize_column_names(df):
    '''Replace all spaces with underscores and "%" with "fraction_".'''
    return (df
            .rename(columns=lambda x: x.replace('%', 'fraction'))
            .rename(columns=lambda x: x.replace(' ', '_')))


def drop_non_features(df, cols):
    return df[[c for c, _ in cols]]


def booleanize_cols(df, boolcols):
    for col in boolcols:
        df[col] = df[col].astype(bool)
    return df


def splitdf(df, label):
    return df.drop(label, axis=1), df[label]


def make_pkl_path(name):
    return os.path.join(data_dir, 'processed', name + '.pkl')


def save_processed_dataset(df, name):
    pkl_path = make_pkl_path(name)
    print('Saving to', pkl_path)
    with open(pkl_path, 'wb') as f:
        dill.dump(df, f)


def load_processed_dataset(name):
    pkl_path = make_pkl_path(name)
    print('Loading', pkl_path)
    with open(pkl_path, 'rb') as f:
        return dill.load(f)
