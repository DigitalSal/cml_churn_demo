#import os, sys
from collections import ChainMap
#from pandas.io.json import dumps as jsonify

from churnexplainer import ExplainedModel

em = ExplainedModel(model_name='telco_linear',data_dir='/home/cdsw')

def explain(args):
    #data = dict(ChainMap(request.args, em.default_data))
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    return {
        'data': dict(data),
        'probability': probability,
        'explanation': explanation
        }

#test
#x={"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}
#explain(x)