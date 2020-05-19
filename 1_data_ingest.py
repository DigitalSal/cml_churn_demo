## Data Ingest
# This script grabs the CSV file from the Cloud Storage location set in step 0 into a Spark DataFrame. Its adds schema and then 
# writes the dataframe to a hive table.

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

# Add the following config if you want to run on the k8s cluster and remove `local[*]`
#    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-2//")\
    
# Since we know the data already, we can add schema upfront. This is good practice as Spark will read *all* the Data if you try infer
# the schema.

schema = StructType(
  [
    StructField("customerID", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("SeniorCitizen", StringType(), True),
    StructField("Partner", StringType(), True),
    StructField("Dependents", StringType(), True),
    StructField("tenure", DoubleType(), True),
    StructField("PhoneService", StringType(), True),
    StructField("MultipleLines", StringType(), True),
    StructField("InternetService", StringType(), True),
    StructField("OnlineSecurity", StringType(), True),
    StructField("OnlineBackup", StringType(), True),
    StructField("DeviceProtection", StringType(), True),
    StructField("TechSupport", StringType(), True),
    StructField("StreamingTV", StringType(), True),
    StructField("StreamingMovies", StringType(), True),
    StructField("Contract", StringType(), True),
    StructField("PaperlessBilling", StringType(), True),
    StructField("PaymentMethod", StringType(), True),
    StructField("MonthlyCharges", DoubleType(), True),
    StructField("TotalCharges", DoubleType(), True),
    StructField("Churn", StringType(), True)
  ]
)

storage = os.environ['STORAGE']

# Read the CSV into a Spark DataFrame    
telco_data = spark.read.csv(
  "{}/datalake/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv".format(storage),
  header=True,
  schema=schema,
  sep=',',
  nullValue='NA'
)

telco_data.show()

telco_data.printSchema()

# Write the Spark DataFrame to the local cdsw file system as a single file.
telco_data.coalesce(1).write.csv(
  "file:/home/cdsw/raw/telco-data/",
  mode='overwrite',
  header=True
)

spark.sql("show databases").show()

spark.sql("show tables in default").show()

# This is here to create the table in Hive used be the other parts of the project.
# If the table already exists, it does not. 

if ('telco_churn' not in list(spark.sql("show tables in default").toPandas()['tableName'])):
  print("creating the telco_churn database")
  telco_data\
    .write.format("parquet")\
    .mode("overwrite")\
    .saveAsTable(
      'default.telco_churn'
  )

# Show the data in the hive table
spark.sql("select * from default.telco_churn").show()

# To get detailed information about the hive table run.
spark.sql("describe formatted default.telco_churn").toPandas()
