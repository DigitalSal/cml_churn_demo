 # Perfoms custom analytics on desired metrics. 

import cdsw, time, os

import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

# Define our uqique model deployment id
model_deployment_crn = "GET THIS FROM MODEL SCREEN IN CML"

current_timestamp_ms = int(round(time.time() * 1000))

# NEW: Get our metrics
known_metrics = cdsw.read_metrics(model_deployment_crn, 0, current_timestamp_ms) 

df = pd.io.json.json_normalize(known_metrics["metrics"])
df.tail()

# Do some conversions & Calculations
df['startTimeStampMs'] = pd.to_datetime(df['startTimeStampMs'], unit='ms')
df['endTimeStampMs'] = pd.to_datetime(df['endTimeStampMs'], unit='ms')
df["processing_time"] = (df["endTimeStampMs"] - df["startTimeStampMs"]).dt.microseconds * 1000

non_agg_metrics = df.dropna(subset=['metrics.probability'])
non_agg_metrics.plot(kind='line',x='predictionUuid',y='metrics.MonthlyCharges',color='red')


agg_metrics = df.dropna(subset=["metrics.accuracy"])
agg_metrics.plot(kind='line', x='endTimeStampMs', y='metrics.accuracy', color='blue')