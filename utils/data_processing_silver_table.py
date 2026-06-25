import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    filepath = bronze_directory + f"bronze_loans_{snapshot_date_str.replace('-','_')}.csv"
    df_loans = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"loaded: {filepath} | rows: {df_loans.count()}")

    column_type_map = {
        "loan_id": StringType(), "Customer_ID": StringType(),
        "loan_start_date": DateType(), "tenure": IntegerType(),
        "installment_num": IntegerType(), "loan_amt": FloatType(),
        "due_amt": FloatType(), "paid_amt": FloatType(),
        "overdue_amt": FloatType(), "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df_loans = df_loans.withColumn(column, col(column).cast(new_type))
    df_loans = df_loans.withColumn("mob", col("installment_num").cast(IntegerType()))
    df_loans = df_loans.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df_loans = df_loans.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df_loans = df_loans.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    filepath_out = silver_directory + f"silver_loans_{snapshot_date_str.replace('-','_')}.parquet"
    df_loans.write.mode("overwrite").parquet(filepath_out)
    print(f"saved to: {filepath_out}")

    filepath = bronze_directory + f"bronze_attributes_{snapshot_date_str.replace('-','_')}.csv"
    df_attr = spark.read.csv(filepath, header=True, inferSchema=True)
    df_attr = df_attr.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df_attr = df_attr.withColumn("Age", col("Age").cast(IntegerType()))
    df_attr = df_attr.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df_attr = df_attr.drop("Name", "SSN")
    filepath_out = silver_directory + f"silver_attributes_{snapshot_date_str.replace('-','_')}.parquet"
    df_attr.write.mode("overwrite").parquet(filepath_out)
    print(f"saved to: {filepath_out}")

    filepath = bronze_directory + f"bronze_financials_{snapshot_date_str.replace('-','_')}.csv"
    df_fin = spark.read.csv(filepath, header=True, inferSchema=True)
    df_fin = df_fin.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df_fin = df_fin.withColumn("Annual_Income", col("Annual_Income").cast(FloatType()))
    df_fin = df_fin.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))
    df_fin = df_fin.withColumn("Outstanding_Debt", col("Outstanding_Debt").cast(FloatType()))
    df_fin = df_fin.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    filepath_out = silver_directory + f"silver_financials_{snapshot_date_str.replace('-','_')}.parquet"
    df_fin.write.mode("overwrite").parquet(filepath_out)
    print(f"saved to: {filepath_out}")

    filepath = bronze_directory + f"bronze_clickstream_{snapshot_date_str.replace('-','_')}.csv"
    df_click = spark.read.csv(filepath, header=True, inferSchema=True)
    df_click = df_click.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df_click = df_click.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    filepath_out = silver_directory + f"silver_clickstream_{snapshot_date_str.replace('-','_')}.parquet"
    df_click.write.mode("overwrite").parquet(filepath_out)
    print(f"saved to: {filepath_out}")
