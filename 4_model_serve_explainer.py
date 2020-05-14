import os
import sys
from collections import ChainMap
from pandas.io.json import dumps as jsonify

#sys.path.append("/home/cdsw") 

from churnexplainer.utils import log_environment
from churnexplainer.explainedmodel import ExplainedModel

em = ExplainedModel(os.getenv('CHURN_MODEL_NAME', 'test_model'))

def explain(args):
    #data = dict(ChainMap(request.args, em.default_data))
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    return jsonify({'data': dict(data),
                    'probability': probability,
                    'explanation': explanation})

#test
#x={"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}
#explain(x)