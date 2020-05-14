## Run this file to auto deploy the model, run a job and deploy the application.

# Install the requirements
!pip3 install -r requirements.txt

# Create the directories and upload data

from cmlbootstrap import CMLBootstrap
from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET
import datetime

# Import SparkSQL for making tables for Atlas Integration
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")


HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

# set the S3 bucket variable
try : 
  s3_bucket=os.environ["STORAGE"]
except:
  tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
  root = tree.getroot()
    
  for prop in root.findall('property'):
    if prop.find('name').text == "hive.metastore.warehouse.dir":
        s3_bucket = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  storage_environment_params = {"STORAGE":s3_bucket}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = s3_bucket

!hdfs dfs -mkdir -p $STORAGE/datalake
!hdfs dfs -mkdir -p $STORAGE/datalake/data
!hdfs dfs -mkdir -p $STORAGE/datalake/data/churn
!hdfs dfs -copyFromLocal /home/cdsw/raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv $STORAGE/datalake/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv

# Get User Details
user_details = cml.get_user({})
user_obj = {"id": user_details["id"], "username": "vdibia",
            "name": user_details["name"],
            "type": user_details["type"],
            "html_url": user_details["html_url"],
            "url": user_details["url"]
            }

# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]


# Create Training Job
create_jobs_params = {"name": "Train Model " + run_time_suffix,
                      "type": "manual",
                      "script": "3_train_models.py",
                      "timezone": "America/Los_Angeles",
                      "environment": {},
                      "kernel": "python3",
                      "cpu": 1,
                      "memory": 2,
                      "nvidia_gpu": 0,
                      "include_logs": True,
                      "notifications": [
                          {"user_id": user_obj["id"],
                           "user":  user_obj,
                           "success": False, "failure": False, "timeout": False, "stopped": False
                           }
                      ],
                      "recipients": {},
                      "attachments": [],
                      "include_logs": True,
                      "report_attachments": [],
                      "success_recipients": [],
                      "failure_recipients": [],
                      "timeout_recipients": [],
                      "stopped_recipients": []
                      }

new_job = cml.create_job(create_jobs_params)
new_job_id = new_job["id"]
print("Created new job with jobid", new_job_id)

##
# Start the Training Job
job_env_params = {}
start_job_params = {"environment": job_env_params}
job_id = new_job_id
job_status = cml.start_job(job_id, start_job_params)
print("Training Job started")

# Stop a job
#job_dict = cml.start_job(job_id, start_job_params)
#cml.stop_job(job_id, start_job_params)

# Run experiment
run_experiment_params = {
    "size": {
        "id": 1,
        "description": "1 vCPU / 2 GiB Memory",
        "cpu": 1,
        "memory": 2,
        "route": "engine-profiles",
        "reqParams": None,
        "parentResource": {
            "route": "site",
            "parentResource": None
        },
        "restangularCollection": False
    },
    "script": "3_train_models.py",
    "arguments": "linear telco",
    "kernel": "python3",
    "cpu": 1,
    "memory": 2,
    "project": str(project_id)
}
#
#cml.run_experiment(run_experiment_params)

#{"size":{"id":1,"description":"1 vCPU / 2 GiB Memory","cpu":1,"memory":2,"route":"engine-profiles","reqParams":null,"parentResource":{"route":"site","parentResource":null},"restangularCollection":false},"script":"3_train_models.py","arguments":"linear telco","kernel":"python3","cpu":1,"memory":2,"project":"212"}

run_params = {"username":"jfletcher","size":{"id":1,"description":"1 vCPU / 2 GiB Memory","cpu":1,"memory":2,"route":"engine-profiles","reqParams":"null","parentResource":{"route":"site","parentResource":"null"},"restangularCollection":"false"},"script":"3_train_models.py","arguments":"linear telco","kernel":"python3","cpu":1,"memory":2,"project":"212"}

def run_experiment(params):
    res = requests.post(
        "http://ml-e493d729-039.demo-aws.ylcu-atmi.cloudera.site/api/altus-ds-1/ds/run",
        headers={"Content-Type": "application/json"},
        auth=("d06k1unrmxo9kq8gizimfn8kwa2t3xxl", ""),
        data=json.dumps(run_params)
    )
    response = res.status_code
    if (res.status_code != 201):
        logging.error(response["message"])
        logging.error(response)
    else:
        logging.debug("Experiment Run Started")
    return response

# Get Default Engine Details
default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]

spark.sql("CREATE DATABASE IF NOT EXISTS telco_churn").show()

create_table_statement = '''CREATE TABLE IF NOT EXISTS telco_churn.historical_data (
  customerID STRING, 
  gender STRING, 
  SeniorCitizen STRING, 
  Partner STRING, 
  Dependents STRING, 
  tenure DOUBLE, 
  PhoneService STRING, 
  MultipleLines STRING, 
  InternetService STRING,
  OnlineSecurity STRING, 
  OnlineBackup STRING, 
  DevbiceProtection STRING, 
  TechSupport STRING, 
  StreamingTV STRING, 
  StreamingMovies STRING, 
  Contract STRING, 
  PaperlessBilling STRING, 
  PaymentMethod STRING, 
  MontlyCharges DOUBLE,
  TotalCharges  DOUBLE,
  Churn STRING)'''

create_view_statement = '''CREATE TABLE IF NOT EXISTS telco_churn.training_data AS 
SELECT customerID, 
  SeniorCitizen, 
  Partner, 
  Dependents, 
  tenure, 
  PhoneService, 
  MultipleLines, 
  InternetService, 
  OnlineSecurity, 
  OnlineBackup, 
  DevbiceProtection, 
  TechSupport, 
  StreamingTV, 
  StreamingMovies, 
  Contract, 
  PaperlessBilling, 
  PaymentMethod, 
  MontlyCharges,
  TotalCharges,
  Churn
FROM telco_churn.historical_data'''

spark.sql(create_table_statement).show()
# TODO: this often fails with permissions problems. Should be used to show more lineage. 
# spark.sql(create_view_statement).show()

# Create Model
example_model_input = {"StreamingTV": "No", "MonthlyCharges": 70.35, "PhoneService": "No", "PaperlessBilling": "No", "Partner": "No", "OnlineBackup": "No", "gender": "Female", "Contract": "Month-to-month", "TotalCharges": 1397.475,
                       "StreamingMovies": "No", "DeviceProtection": "No", "PaymentMethod": "Bank transfer (automatic)", "tenure": 29, "Dependents": "No", "OnlineSecurity": "No", "MultipleLines": "No", "InternetService": "DSL", "SeniorCitizen": "No", "TechSupport": "No"}

create_model_params = {
    "projectId": project_id,
    "name": "Model Explainer " + run_time_suffix,
    "description": "Explain a given model prediction",
    "visibility": "private",
    "enableAuth": False,
    "targetFilePath": "4_model_serve_explainer.py",
    "targetFunctionName": "explain",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": {}
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

# Complete our YAML file
yaml_text = \
"""Model Explainer {}":
  hive_table_qualified_names:                # this is a predefined key to link to training data
    - "telco_churn.historical_data@cm"         # the qualifiedName of the hive_table object representing                
  metadata:                                  # this is a predefined key for additional metadata
    query: "select * from historical_data"   # suggested use case: query used to extract training data
    training_file: "3_train_models.py"       # suggested use case: training file used
""".format(run_time_suffix)

with open('lineage.yml', 'w') as lineage: lineage.write(yaml_text)

# Deploy the model
new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]  # todo check for bad response
model_id = new_model_details["id"]

print("New model created with access key", access_key)

cml.set_model_auth({"id": model_id, "enableAuth": False})

#Wait for the model to deploy.
is_deployed = False
while is_deployed == False:
  model = cml.get_model({"id": str(new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
  if model["latestModelDeployment"]["status"] == 'deployed':
    print("Model is deployed")
    break
  else:
    print ("Deploying Model.....")
    time.sleep(10)

# Set CRN for deployment
crn_env_params = {"DEPLOYMENT_CRN":new_model_details["modelDeployment"]["crn"]}
crn_env = cml.create_environment_variable(crn_env_params)

# Create Accuracy Job
create_calc_jobs_params = {"name": "Calculate Accuracy " + run_time_suffix,
                      "type": "manual",
                      "script": "6_calculate_accuracy.py",
                      "timezone": "America/Los_Angeles",
                      "environment": {},
                      "kernel": "python3",
                      "cpu": 1,
                      "memory": 2,
                      "nvidia_gpu": 0,
                      "include_logs": True,
                      "notifications": [
                          {"user_id": user_obj["id"],
                           "user":  user_obj,
                           "success": False, "failure": False, "timeout": False, "stopped": False
                           }
                      ],
                      "recipients": {},
                      "attachments": [],
                      "include_logs": True,
                      "report_attachments": [],
                      "success_recipients": [],
                      "failure_recipients": [],
                      "timeout_recipients": [],
                      "stopped_recipients": []
                      }

new_calc_job = cml.create_job(create_calc_jobs_params)
new_calc_job_id = new_calc_job["id"]
print("Created new job with jobid", new_calc_job_id)

# Start the accuracy job
job_calc_env_params = {}
start_job_calc_params = {"environment": job_env_params}
job_calc_id = new_calc_job_id
job_calc_status = cml.start_job(job_calc_id, start_job_calc_params)
print("Accuracy Job 1 Started")

# Do it again for two data points for the graph
job_calc_env_params = {}
start_job_calc_params = {"environment": job_env_params}
job_calc_id = new_calc_job_id
job_calc_status = cml.start_job(job_calc_id, start_job_calc_params)
print("Accuracy Job 2 Started")

# Change the line in the flask/single_view.html file.
import subprocess
subprocess.call(["sed", "-i",  's/const\saccessKey.*/const accessKey = "' + access_key + '";/', "/home/cdsw/flask/single_view.html"])

# Create Application
create_application_params = {
    "name": "Explainer App",
    "subdomain": run_time_suffix[:],
    "description": "Explainer web application",
    "type": "manual",
    "script": "5_application.py", "environment": {},
    "kernel": "python3", "cpu": 1, "memory": 2,
    "nvidia_gpu": 0
}

new_application_details = cml.create_application(create_application_params)
application_url = new_application_details["url"]
application_id = new_application_details["id"]

# print("Application may need a few minutes to finish deploying. Open link below in about a minute ..")
print("Application created, deploying at ", application_url)

#Wait for the application to deploy.
is_deployed = False
while is_deployed == False:
#Wait for the application to deploy.
  app = cml.get_application(str(application_id),{})
  if app["status"] == 'running':
    print("Application is deployed")
    break
  else:
    print ("Deploying Application.....")
    time.sleep(10)

HTML("<a href='{}'>Open Application UI</a>".format(application_url))
