import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]") \
    .getOrCreate()

## Add the following config if you want to run on the k8s cluster and remove `local[*]`
#    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
#    .config("spark.yarn.access.hadoopFileSystems","s3a://jfletcher-cdp-bucket/")\
    

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
    
telco_data = spark.read.csv(
  "s3a://cdp-cldr-virginia/cdp-cldr-virginia-dl/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv",
  header=True,
  schema=schema,
  sep=',',
  nullValue='NA'
)

telco_data.show()

telco_data.printSchema()

telco_data.coalesce(1).write.csv(
  "file:/home/cdsw/raw/telco-data/",
  mode='overwrite',
  header=True
)

spark.sql("show databases").show()

spark.sql("show tables in default").show()


#telco_data\
#  .write.format("parquet")\
#  .mode("overwrite")\
#  .saveAsTable(
#    'default.telco_churn'
#)

spark.sql("select * from default.telco_churn").show()
