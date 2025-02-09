from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from batch_ingest_CMS import ingest_data
from transform import transform_data
from featureExtraction_CMS import feature_extract
from build_train_model_CMS import build_train
from predict_CMS import predict
from load_db_CMS import load_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 11),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'CMS_Chronic_Condition_Dag',
    default_args=default_args,
    description='Ingest, transform, visualize CMS data',
    schedule_interval='@once',
)

ingest_etl = PythonOperator(
    task_id='ingest_dataset',
    python_callable=ingest_data,
    dag=dag,
)

transform_etl = PythonOperator(
    task_id='transform_dataset',
    python_callable=transform_data,
    dag=dag,
)

feature_Extraction_etl = PythonOperator(
    task_id='feature_Extraction',
    python_callable=feature_extract,
    dag=dag,
)

build_train_etl = PythonOperator(
    task_id='build_train',
    python_callable=build_train,
    dag=dag,
)

predict_etl = PythonOperator(
    task_id='predict',
    python_callable=predict,
    dag=dag,
)

load_etl = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag,
)

ingest_etl >> transform_etl >> feature_Extraction_etl >> build_train_etl >> predict_etl  >> load_etl