import os
import numpy as np
import pandas as pd
import glob

from churnexplainer.utils import data_dir
from churnexplainer.data import utils

if len(glob.glob("/home/cdsw/raw/telco-data/*.csv")) == 1:
  telco_data_path = glob.glob("/home/cdsw/raw/telco-data/*.csv")[0]
else:
  telco_data_path = os.path.join(data_dir, 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

idcol = 'customerID'
labelcol = 'Churn'
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))


def drop_missing(df):
    '''Remove rows with missing values'''
    return df.replace(r'^\s$', np.nan, regex=True).dropna()


def clean(df):
    # Make target variable a true boolean column
    # Drop unpredictive column
    df.drop(['customerID'], axis=1)


def booleanize_senior_citizen(df):
    '''Make SeniorCitizen 'Yes'/'No' like other columns in this dataset.'''
    return df.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})


def load_dataset():
    '''Return IBM customers and labels.'''
    df = pd.read_csv(telco_data_path)
    df = drop_missing(df).reset_index()
    df.index.name = 'id'
    features, labels = utils.splitdf(df, labelcol)
    features = booleanize_senior_citizen(features)
    features = utils.drop_non_features(features, cols)
    features = utils.categorize(features, cols)
    labels = (labels == 'Yes')
    return features, labels

