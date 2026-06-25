import os
from datetime import datetime
from pyspark.sql.functions import col

def process_bronze_table(snapshot_date_str, bronze_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    sources = {
        "attributes": "data/features_attributes.csv",
        "financials": "data/features_financials.csv",
        "clickstream": "data/feature_clickstream.csv",
        "loans": "data/lms_loan_daily.csv",
    }

    for table_name, csv_path in sources.items():

        df = spark.read.csv(csv_path, header=True, inferSchema=True) \
                  .filter(col("snapshot_date") == snapshot_date)

        row_count = df.count()
        print(f"{snapshot_date_str} | {table_name} | row count: {row_count}")

        partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-','_')}.csv"
        filepath = bronze_directory + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print(f"saved to: {filepath}")
