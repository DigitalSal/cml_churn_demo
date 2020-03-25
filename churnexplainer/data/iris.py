import os
import pandas as pd

from churnexplainer.utils import data_dir
from churnexplainer.data import utils

iriscsvpath = os.path.join(data_dir, 'raw', 'iris.csv')

idcol = 'flower_id'
labelcol = 'Species'
cols = (
    ('Sepal.Length', False),
    ('Sepal.Width', False),
    ('Petal.Length', False),
    ('Petal.Width', False),
)


def load_dataset():
    '''Return Iris data and a binary label (not Virginica=0, Virginica=1).'''
    df = pd.read_csv(iriscsvpath)
    df = df.rename(columns={'Unnamed: 0': idcol})
    df[labelcol] = (df[labelcol] == 'virginica')
    df = df.rename(columns={'Species': 'virginica'})
    df = df.set_index(idcol)

    features, labels = utils.splitdf(df, 'virginica')
    return features, labels
