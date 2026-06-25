import os
import sys

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

SCRIPTS = os.environ.get("PIPELINE_SCRIPTS", "/opt/airflow/scripts")
ROOT = os.environ.get("PIPELINE_ROOT", "/opt/airflow")
for p in (SCRIPTS, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

def _build_datamart(ds=None, **_):
    import build_datamart
    build_datamart.build_datamart(run_date=ds)

def _train(ds=None, **_):
    import train_model
    train_model.run_training(run_date=ds)

def _inference(ds=None, **_):
    import inference
    inference.run_inference(run_date=ds)

def _monitor(ds=None, **_):
    import monitor
    monitor.run_monitoring(run_date=ds)

def _visualise(ds=None, **_):
    import visualise
    visualise.run_visualise(run_date=ds)

with DAG(
    dag_id="loan_default_ml_pipeline",
    description="CS611 AS02: datamart -> governance train -> infer -> monitor -> visualise (monthly, backfillable)",
    default_args={"owner": "Wang Lejun", "retries": 0},
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    end_date=pendulum.datetime(2024, 12, 1, tz="UTC"),
    schedule="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["cs611", "ml-pipeline", "loan-default"],
) as dag:
    build_datamart = PythonOperator(task_id="build_datamart", python_callable=_build_datamart)
    train = PythonOperator(task_id="train_model", python_callable=_train)
    run_inference = PythonOperator(task_id="run_inference", python_callable=_inference)
    run_monitoring = PythonOperator(task_id="run_monitoring", python_callable=_monitor)
    visualise = PythonOperator(task_id="visualise", python_callable=_visualise)

    build_datamart >> train >> run_inference >> run_monitoring >> visualise
