import os
import pandas as pd

from churnexplainer.utils import data_dir
from churnexplainer.data import utils

breastcancercsvpath = os.path.join(data_dir, 'raw', 'breast-cancer-wisconsin.data.csv')

idcol = 'id'
labelcol = 'diagnosis'


def load_dataset():
    df = pd.read_csv(breastcancercsvpath)
    df = df.set_index(idcol)
    df[labelcol] = (df[labelcol] == 'M')  # Malignant == True
    return utils.splitdf(df, labelcol)

