import os, datetime, subprocess, glob
import dill
import pandas as pd
import numpy as np
import cdsw

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

from lime.lime_tabular import LimeTabularExplainer

from churnexplainer import ExplainedModel,CategoricalEncoder

data_dir = '/home/cdsw' #os.environ.get('CHURN_DATA_DIR', '/mnt/c/Users/Jeff/tmp/cml_churn_demo')
#dataset = os.environ.get('CHURN_DATASET', 'telco')

if len(glob.glob("raw/telco-data/*.csv")) == 1:
  telco_data_path = glob.glob("raw/telco-data/*.csv")[0]
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


import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

df = spark.sql("SELECT * FROM default.telco_churn").toPandas()


#df = pd.read_csv(telco_data_path)
df = df.replace(r'^\s$', np.nan, regex=True).dropna().reset_index()
df.index.name = 'id'
data, labels = df.drop(labelcol, axis=1), df[labelcol]
data = data.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})
data = data[[c for c, _ in cols]]

catcols = (c for c, iscat in cols if iscat)
for col in catcols:
    data[col] = pd.Categorical(data[col])
labels = (labels == 'Yes')


ce = CategoricalEncoder()
X = ce.fit_transform(data)
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ct = ColumnTransformer(
    [('ohe', OneHotEncoder(), list(ce.cat_columns_ix_.values()))],
    remainder='passthrough'
)


### Experiments options
# If you are running this as an experiment, pass the cv, solver and max_iter values
# as arguments in that order. e.g. `5 lbfgs 100`.

if len (sys.argv) == 4:
  try:
    cv = int(sys.argv[1])
    solver = str(sys.argv[2])
    max_iter = int(sys.argv[3])
  except:
    sys.exit("Invalid Arguments passed to Experiment")
else:
    cv = 5
    solver = 'lbfgs' # one of newton-cg, lbfgs, liblinear, sag, saga
    max_iter = 100

clf = LogisticRegressionCV(cv=cv,solver=solver,max_iter=max_iter)
pipe = Pipeline([('ct', ct),
                 ('scaler', StandardScaler()),
                 ('clf', clf)])

pipe.fit(X_train, y_train)
train_score = pipe.score(X_train, y_train)
test_score = pipe.score(X_test, y_test)
print("train",train_score)
print("test", test_score)    
print(classification_report(y_test, pipe.predict(X_test)))
data[labels.name + ' probability'] = pipe.predict_proba(X)[:, 1]


# Create LIME Explainer
feature_names = list(ce.columns_)
categorical_features = list(ce.cat_columns_ix_.values())
categorical_names = {i: ce.classes_[c]
                     for c, i in ce.cat_columns_ix_.items()}
class_names = ['No ' + labels.name, labels.name]
explainer = LimeTabularExplainer(ce.transform(data),
                                 feature_names=feature_names,
                                 class_names=class_names,
                                 categorical_features=categorical_features,
                                 categorical_names=categorical_names)    


# Create and save the combined Logistic Regression and LIME Explained Model.
explainedmodel = ExplainedModel(data=data, labels=labels, model_name='telco_linear',
                                categoricalencoder=ce, pipeline=pipe,
                                explainer=explainer,data_dir=data_dir)
explainedmodel.save()


# If running as as experiment, this will 
cdsw.track_metric("train_score",round(train_score,2))
cdsw.track_metric("test_score",round(test_score,2))
cdsw.track_metric("model_path",explainedmodel.model_path)
cdsw.track_file(explainedmodel.model_path)
