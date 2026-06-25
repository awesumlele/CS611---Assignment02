import os
import sys
from datetime import datetime

import as02_config as config

sys.path.insert(0, config.ROOT)
import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold

def _months(start, end):
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    out, cur = [], datetime(s.year, s.month, 1)
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return out

def _first_of_month(ds):
    d = datetime.strptime(ds[:10], "%Y-%m-%d")
    return datetime(d.year, d.month, 1).strftime("%Y-%m-%d")

def build_datamart(run_date=None):
    import pyspark
    os.chdir(config.ROOT)
    spark = (pyspark.sql.SparkSession.builder
             .appName("CS611_AS02_datamart").master("local[*]")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    if run_date:
        ds = _first_of_month(run_date)
        if not (config.SNAPSHOT_START <= ds <= config.SNAPSHOT_END):
            print(f"[datamart] {ds} outside snapshot range - skipping")
            spark.stop(); return
        dates = [ds]
    else:
        dates = _months(config.SNAPSHOT_START, config.SNAPSHOT_END)
    print(f"[datamart] processing {len(dates)} monthly snapshot(s): {dates}")

    bronze_dir = config.BRONZE + "/"
    silver_dir = config.SILVER + "/"
    label_dir  = config.LABEL_STORE + "/"
    feat_dir   = config.FEATURE_STORE + "/"
    for d in (bronze_dir, silver_dir, label_dir, feat_dir):
        os.makedirs(d, exist_ok=True)

    for ds in dates:
        bronze.process_bronze_table(ds, bronze_dir, spark)
    for ds in dates:
        silver.process_silver_table(ds, bronze_dir, silver_dir, spark)
    for ds in dates:
        gold.process_labels_gold_table(ds, silver_dir, label_dir, spark,
                                       dpd=config.DPD_THRESHOLD, mob=config.MOB_LABEL)
    for ds in dates:
        gold.process_features_gold_table(ds, silver_dir, feat_dir, spark)

    spark.stop()
    print("[datamart] complete")

if __name__ == "__main__":
    build_datamart()
