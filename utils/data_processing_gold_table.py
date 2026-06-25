import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_labels_gold_table(snapshot_date_str, silver_directory, gold_label_store_directory, spark, dpd, mob):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    filepath = silver_directory + f"silver_loans_{snapshot_date_str.replace('-','_')}.parquet"
    df = spark.read.parquet(filepath)
    print(f"loaded: {filepath} | rows: {df.count()}")

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    filepath_out = gold_label_store_directory + f"gold_label_store_{snapshot_date_str.replace('-','_')}.parquet"
    df.write.mode("overwrite").parquet(filepath_out)
    print(f"saved to: {filepath_out}")
    return df

def process_features_gold_table(snapshot_date_str, silver_directory, gold_feature_store_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    df_attr = spark.read.parquet(silver_directory + f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    df_fin = spark.read.parquet(silver_directory + f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet")
    df_click = spark.read.parquet(silver_directory + f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet")

    agg_cols = [F.mean(f"fe_{i}").alias(f"fe_{i}_mean") for i in range(1, 21)]
    df_click_agg = df_click.groupBy("Customer_ID").agg(*agg_cols)
    df = df_attr.join(df_fin.drop("snapshot_date"), on="Customer_ID", how="left")
    df = df.join(df_click_agg, on="Customer_ID", how="left")

    filepath_out = gold_feature_store_directory + f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    df.write.mode("overwrite").parquet(filepath_out)
    print(f"saved to: {filepath_out}")
    return df
