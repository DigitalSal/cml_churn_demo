 # Calculates accuracy and submits back to MLOps for further analysis

import cdsw, time, os
import random

import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

# Define our uqique model deployment id
model_deployment_crn = os.environ["DEPLOYMENT_CRN"]

current_timestamp_ms = int(round(time.time() * 1000))

known_metrics = cdsw.read_metrics(model_deployment_crn, 0, current_timestamp_ms) 

df = pd.io.json.json_normalize(known_metrics["metrics"])
df

# Do some conversions & Calculations
df['startTimeStampMs'] = pd.to_datetime(df['startTimeStampMs'], unit='ms')
df['endTimeStampMs'] = pd.to_datetime(df['endTimeStampMs'], unit='ms')
df["processing_time"] = (df["endTimeStampMs"] - df["startTimeStampMs"]).dt.microseconds * 1000

#df.plot(kind='line',x='endTimeStampMs',y='metrics.MonthlyCharges',color='red')

cdsw.track_aggregate_metrics({"accuracy": random.random()}, current_timestamp_ms , current_timestamp_ms, model_deployment_crn=model_deployment_crn)