## Explained Model Serving
# The `explain` function will load the trained model and calculate a new explained prediction from a single data row.

from collections import ChainMap

from churnexplainer import ExplainedModel

#Load the model save earlier.
em = ExplainedModel(model_name='telco_linear',data_dir='/home/cdsw')

#This is the main function used for serving the model. It will take in the JSON formatted arguments , calculate the probablity of churn and create a LIME explainer explained instance and return that as JSON.
def explain(args):
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    return {
        'data': dict(data),
        'probability': probability,
        'explanation': explanation
        }

#To test this is a session, uncomment and run the two rows below.
#x={"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}
#explain(x)