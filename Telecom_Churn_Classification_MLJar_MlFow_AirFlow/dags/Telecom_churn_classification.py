from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Script imports
from scripts.telecom_preprocessing import telecom_preprocess_data
from scripts.telecom_model_inference import telecom_load_model_and_predict
from scripts.telecom_drift_check import telecom_monitor_drift
from scripts.telecom_retrain_model import telecom_retrain_and_log_model

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='Telecom_churn_classification',
    default_args=default_args,
    description='Telecom_churn_classificatio model pipeline with MLflow & drift monitoring',
    start_date=datetime(2025, 4, 14),
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'h2o', 'mlflow', 'drift']
) as dag:

    telecom_preprocess_task = PythonOperator(
        task_id='telecom_preprocess_data',
        python_callable=telecom_preprocess_data
    )

    telecom_predict_task = PythonOperator(
        task_id='predict_with_registered_model',
        python_callable=telecom_load_model_and_predict
    )

    telecom_drift_check_task = PythonOperator(
        task_id='check_data_drift',
        python_callable=telecom_monitor_drift
    )

    telecom_retrain_task = PythonOperator(
        task_id='retrain_model_if_drift',
        python_callable=telecom_retrain_and_log_model
    )

    telecom_preprocess_task >> telecom_predict_task >> telecom_drift_check_task >> telecom_retrain_task
