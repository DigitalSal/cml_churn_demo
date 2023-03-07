# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

# Part 4: Model Training

# This script is used to train an Explained model using the Jobs feature
# in CML and the Experiments feature to facilitate model tuning

# If you haven't yet, run through the initialization steps in the README file and Part 1.
# In Part 1, the data is imported into the table you specified in Hive.
# All data accesses fetch from Hive.
#
# To simply train the model once, run this file in a workbench session.
#
# There are 2 other ways of running the model training process
#
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model
# training process as a job, create a new job by going to the Project window and clicking _Jobs >
# New Job_ and entering the following settings:
# * **Name** : Train Model
# * **Script** : 4_train_models.py
# * **Arguments** : _Leave blank_
# * **Kernel** : Python 3
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 2 GiB
# The rest can be left as is. Once the job has been created, click **Run** to start a manual
# run for that job.

# ***Experiments with mlflow***
#
# Training a model for use in production requires testing many combinations of model parameters
# and picking the best one based on one or more metrics.
# In order to do this in a *principled*, *reproducible* way, Experiments (based on mlflow tracking) record **input parameters**, and **output artifacts**.
# This is a very useful feature for testing a large number of hyperparameters in parallel on elastic cloud resources.

# In this instance it would be used for hyperparameter optimisation. To learn more about mlflow tracking, visit https://mlflow.org/docs/latest/tracking.html


from pyspark.sql.types import *
from pyspark.sql import SparkSession
import sys
import os
import pandas as pd
import numpy as np
import cdsw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from lime.lime_tabular import LimeTabularExplainer

#MLFlow and additional model type for experimentation
import mlflow
from sklearn.svm import SVC

try:
  os.chdir("code")
except:
  pass
from churnexplainer import ExplainedModel, CategoricalEncoder

hive_database = "default"
hive_table = "telco_churn"
hive_table_fq = hive_database + "." + hive_table

data_dir = "/home/cdsw"
labelcol = "Churn"


# This is a fail safe incase the hive table did not get created in the last step.
try:
    spark = SparkSession.builder.appName("PythonSQL").master("local[*]").getOrCreate()

    if spark.sql("SELECT count(*) FROM " + hive_table_fq).collect()[0][0] > 0:
        df = spark.sql("SELECT * FROM " + hive_table_fq).toPandas()
except:
    print("Hive table has not been created")
    df = pd.read_csv(os.path.join("../raw", "WA_Fn-UseC_-Telco-Customer-Churn-.csv"))


# Clean and prep the dataframe
df = (df
      .replace(r"^\s$", np.nan, regex=True).dropna().reset_index()
      # drop unnecessary and personally identifying information
      .drop(columns=['index', 'customerID'])
     )
try:
    # when loading from external data source, this column has str dtype
    df.replace({"SeniorCitizen": {"1": "Yes", "0": "No"}}, inplace=True)
except:
    # when loading from local data source, this column has int dtype 
    df.replace({"SeniorCitizen": {1: "Yes", 0: "No"}}, inplace=True)
  
df['TotalCharges'] = df['TotalCharges'].astype('float')
df.index.name='id'


# separate target variable column from feature columns
datadf, labels = df.drop(labelcol, axis=1), df[labelcol]

# recast all columns that are "object" dtypes to Categorical
for colname, dtype in zip(datadf.columns, datadf.dtypes):
  if dtype == "object":
    datadf[colname] = pd.Categorical(datadf[colname])

  
# Prepare data for Sklearn model and create train/test split
ce = CategoricalEncoder()
X = ce.fit_transform(datadf)
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ct = ColumnTransformer(
    [("ohe", OneHotEncoder(categories='auto'), list(ce.cat_columns_ix_.values()))],
    remainder="passthrough"
)

# Instantiate a new set of experiments (mlflow experiment object)
mlflow.set_experiment("Churn Model Tuning")
mlflow.autolog(log_input_examples=True)

# Define a search grid
kernel = ["linear", "rbf"]
max_iter = [1, 10, 100, 1000, 10000]

# Iterate over the grid, re-training the model every time and recording train and test score as the metrics
for k in kernel:
  for i in max_iter:
    
    # Start experiment run
    mlflow.start_run()
    mlflow.log_param("Kernel", k)
    mlflow.log_param("Max_iter", i)
    
    # Define and fit model pipeline
    svc = SVC(kernel = k, random_state = 0, max_iter=i, probability=True)
    svc_pipe = Pipeline([("ct", ct), ("scaler", StandardScaler()), ("svc_fit", svc)])
    svc_pipe.fit(X_train, y_train)
    
    # Capture train and test set scores
    train_score2 = svc_pipe.score(X_train, y_train)
    test_score2 = svc_pipe.score(X_test, y_test)
    datadf[labels.name + " probability"] = svc_pipe.predict_proba(X)[:, 1]
    
    mlflow.log_metric("train_score", round(train_score2, 2))
    mlflow.log_metric("test_score", round(test_score2, 2))
    mlflow.end_run()


# Create LIME Explainer
feature_names = list(ce.columns_)
categorical_features = list(ce.cat_columns_ix_.values())
categorical_names = {i: ce.classes_[c] for c, i in ce.cat_columns_ix_.items()}
class_names = ["No " + labels.name, labels.name]
explainer = LimeTabularExplainer(
    ce.transform(datadf),
    feature_names=feature_names,
    class_names=class_names,
    categorical_features=categorical_features,
    categorical_names=categorical_names,
)

# Create and save the combined Logistic Regression and LIME Explained Model.
explainedmodel = ExplainedModel(data=datadf, labels=labels, model_name='telco_linear',
                                categoricalencoder=ce, pipeline=svc_pipe,
                                explainer=explainer,data_dir=data_dir)
explainedmodel.save()
