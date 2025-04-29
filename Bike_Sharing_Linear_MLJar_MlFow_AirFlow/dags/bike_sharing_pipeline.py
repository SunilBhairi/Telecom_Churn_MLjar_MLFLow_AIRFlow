from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Script imports
from scripts.preprocessing import preprocess_data
from scripts.model_inference import load_model_and_predict
from scripts.drift_check import monitor_drift
from scripts.retrain_model import retrain_and_log_model

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='bike_sharing_model_pipeline',
    default_args=default_args,
    description='Bike sharing H2O model pipeline with MLflow & drift monitoring',
    start_date=datetime(2025, 4, 14),
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'h2o', 'mlflow', 'drift']
) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    predict_task = PythonOperator(
        task_id='predict_with_registered_model',
        python_callable=load_model_and_predict
    )

    drift_check_task = PythonOperator(
        task_id='check_data_drift',
        python_callable=monitor_drift
    )

    retrain_task = PythonOperator(
        task_id='retrain_model_if_drift',
        python_callable=retrain_and_log_model
    )

    preprocess_task >> predict_task >> drift_check_task >> retrain_task
