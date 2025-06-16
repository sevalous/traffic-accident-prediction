from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import logging
from fyp_package import ingestion, preprocessing, model_dev_and_monitoring
#setup of logger to record notable events
logging.basicConfig(level=logging.WARN) #log only levels including: WARNING, ERROR and CRITICAL
logger = logging.getLogger(__name__) #create the a longger instance with the name of '__main__'
#default arguments for the DAG; required

DEFAULT_ARGS = {
    "owner":"Alex",
    "depends_on_past":True, #every upstream task must be successful for the DAG to continue
    "email":["alexander.roberts2@mail.bcu.ac.uk"],
    "email_on_failure":False,
    "email_on_retry":False,
    "retries":0,
}
DEFAULT_ROAD_DATA_PATH = '/home/alex/FYP_dir/fyp_data/tfl_road_data'
DEFAULT_TRAFFIC_FLOW_DATA_PATH = '/home/alex/FYP_dir/fyp_data/traffic-flow-borough-all-vehicles.csv'
DEFAULT_POPULATION_DATA_PATH = '/home/alex/FYP_dir/fyp_data/ONS mid-year population estimates London boroughs.csv'
DEFAULT_GDP_DATA_PATH = '/home/alex/FYP_dir/fyp_data/gdp at current basic rates.csv'
DEFAULT_CRIME_DATA_PATH = '/home/alex/FYP_dir/fyp_data/MPS Borough Level Crime (Historical).csv'

# connection dictionary to the ClickHouse server instance
ENGINE_CONN = {
    "database":'airflow_storage',
    "host":'localhost',
    "user":'AirflowUser122',
    "password":'',
    "port":8123
}
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379

MLFLOW_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "fyp-model-development"

# creates unique table names in the ClickHouser server instance
# e.g., 'london_road_accidents_12Jun24
ROAD_DATA_TABLE_NAME = "london_road_accidents_"+datetime.today().strftime('%d%b%y')
TRAFFIC_FLOW_TABLE_NAME = "traffic_flow_london_boroughs_"+datetime.today().strftime('%d%b%y')
POPULATION_TABLE_NAME = "population_london_boroughs_"+datetime.today().strftime('%d%b%y')
GDP_TABLE_NAME = "gdp_london_boroughs_"+datetime.today().strftime('%d%b%y')
CRIME_TABLE_NAME = "crime_london_boroughs_"+datetime.today().strftime('%d%b%y')

with DAG(
    "fyp_model_creation", # DAG name
    default_args=DEFAULT_ARGS,
    description="This pipeline is for the development of a machine learning model.",
    start_date=datetime(2022,10,10), # required otherwise the DAG cannot proceed
    schedule_interval=None,
    catchup=False, # prevent the pipeline from performing scheduler catchup
    tags=["fyp", "mlops"],
) as dag:
    start=EmptyOperator(task_id="start")
    ingestion_task=PythonOperator(task_id="ingestion_stage", python_callable=ingestion.ingest_data_main, 
        op_kwargs={
            'london_acc_data_path':DEFAULT_ROAD_DATA_PATH,
            'traffic_flow_data_path':DEFAULT_TRAFFIC_FLOW_DATA_PATH,
            'population_data_path':DEFAULT_POPULATION_DATA_PATH,
            'gdp_data_path':DEFAULT_GDP_DATA_PATH,
            'crime_data_path':DEFAULT_CRIME_DATA_PATH,
            'logger':logger,
            'engine_conn':ENGINE_CONN,
            'london_acc_table_name':ROAD_DATA_TABLE_NAME,
            'traffic_flow_table_name':TRAFFIC_FLOW_TABLE_NAME,
            'population_table_name':POPULATION_TABLE_NAME,
            'gdp_table_name':GDP_TABLE_NAME,
            'crime_table_name':CRIME_TABLE_NAME,
        })
    preprocessing_task=PythonOperator(task_id="preprocessing_stage", python_callable=preprocessing.preprocess_data_main,
        op_kwargs={
            'logger':logger,
            'engine_conn':ENGINE_CONN,
            'london_acc_table_name':ROAD_DATA_TABLE_NAME,
            'population_table_name':POPULATION_TABLE_NAME,
            'gdp_table_name':GDP_TABLE_NAME,
            'traffic_flow_table_name':TRAFFIC_FLOW_TABLE_NAME,
            'crime_table_name':CRIME_TABLE_NAME,
            'redis_host':REDIS_HOST,
            'redis_port':REDIS_PORT
        })
    model_dev_and_monitoring_task=PythonOperator(task_id="model_dev_and_monitoring_stage",
                                                 python_callable=model_dev_and_monitoring.model_dev_and_monitoring__main,
        op_kwargs={
            'logger':logger,
            'redis_host':REDIS_HOST,
            'redis_port':REDIS_PORT,
            'mlflow_uri':MLFLOW_URI,
            'experiment_name':MLFLOW_EXPERIMENT_NAME
        })
    end=EmptyOperator(task_id="end")
    start >> ingestion_task >> preprocessing_task >> model_dev_and_monitoring_task >> end