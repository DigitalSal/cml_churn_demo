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


