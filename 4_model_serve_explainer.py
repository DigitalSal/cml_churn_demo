import os
import sys
from collections import ChainMap
from pandas.io.json import dumps as jsonify
import numpy

sys.path.append("/home/cdsw") 

from churnexplainer.utils import log_environment
from churnexplainer.explainedmodel import ExplainedModel

#import new SDK
import cdsw

em = ExplainedModel(os.getenv('CHURN_MODEL_NAME', 'test_model'))

@cdsw.model_metrics
def explain(args):
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)

    #Do the prediction and provide weights for the reasoning
    probability, explanation = em.explain_dct(data)
    
    #NEW! Track our inputs
    for key in data:
      if isinstance(data[key], numpy.int64) or isinstance(data[key], numpy.float64):
        cdsw.track_metric(key, data[key].item())
      else:
        cdsw.track_metric(key, data[key])
    
    #NEW! Track our prediction
    cdsw.track_metric('probability', probability)
    
    #NEW! Track explanation
    cdsw.track_metric('explanation', explanation)
    
    return {'data': dict(data),
                    'probability': probability,
                    'explanation': explanation}

# Preheat the model to get data into the metrics store - this runs at deployment
s = [
      {"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Male","Contract":"Month-to-month","TotalCharges":1597.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":40.15,"PhoneService":"Yes","PaperlessBilling":"No","Partner":"No","OnlineBackup":"Yes","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":75.35,"PhoneService":"Yes","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1317.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.65,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":10.35,"PhoneService":"Yes","PaperlessBilling":"Yes","Partner":"Yes","OnlineBackup":"Yes","gender":"Male","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":40.25,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1357.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"Yes","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"Yes","Partner":"No","OnlineBackup":"No","gender":"Male","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.32,"PhoneService":"Yes","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":417.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"Yes","gender":"Male","Contract":"Month-to-month","TotalCharges":1967.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":71.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"Yes","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":97.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":72.95,"PhoneService":"Yes","PaperlessBilling":"Yes","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Male","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":73.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Male","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"Yes","OnlineBackup":"Yes","gender":"Male","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":148.35,"PhoneService":"Yes","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":17.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":1.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"Yes","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
      {"StreamingTV":"No","MonthlyCharges":79.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"},
]  

for x in s:
  explain(x)