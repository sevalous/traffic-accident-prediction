from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from fyp_package import deployment
import logging
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
    "retries":3,
}

ENGINE_CONN = {
    "database":'airflow_storage',
    "host":'localhost',
    "user":'AirflowUser122',
    "password":'',
    "port":8123
}

LONDON_BOROUGH_LAT_LONG = {
    "barking and dagenham": {
        "lat": 51.5607,
        "long": 0.1557
    },
    "barnet": {
        "lat": 52.6252,
        "long": -0.1517
    },
    "bexley": {
        "lat": 51.4549,
        "long": 0.1505
    },
    "brent": {
        "lat": 51.5588,
        "long": -0.2817
    },
    "bromley": {
        "lat": 51.4039,
        "long": 0.0198
    },
    "camden": {
        "lat": 51.5290,
        "long": -0.1255
    },
    "croydon": {
        "lat": 51.3714,
        "long": -0.0977
    },
    "ealing": {
        "lat": 51.5130,
        "long": -0.3089
    },
    "enfield": {
        "lat": 51.6538,
        "long": -0.0799
    },
    "greenwich": {
        "lat": 51.4892,
        "long": 0.0648
    },
    "hackney": {
        "lat": 51.5450,
        "long": -0.0553
    },
    "hammersmith and fulham": {
        "lat": 51.4927,
        "long": -0.2339
    },
    "haringey": {
        "lat": 51.6000,
        "long": -0.1119
    },
    "harrow": {
        "lat": 51.5898,
        "long": -0.3346
    },
    "havering": {
        "lat": 51.5812,
        "long": 0.1837
    },
    "hillingdon": {
        "lat": 51.5441,
        "long": -0.4760
    },
    "hounslow": {
        "lat": 51.4746,
        "long": -0.3680
    },
    "islington": {
        "lat": 51.5416,
        "long": -0.1022
    },
    "kensington and chelsea": {
        "lat": 51.5020,
        "long": -0.1947
    },
    "kingston upon thames": {
        "lat": 51.4085,
        "long": -0.3064
    },
    "lambeth": {
        "lat": 51.4607,
        "long": -0.1163
    },
    "lewisham": {
        "lat": 51.4452,
        "long": -0.0209
    },
    "merton": {
        "lat": 51.4014,
        "long": -0.1958
    },
    "newham": {
        "lat": 51.5077,
        "long": 0.0469
    },
    "redbridge": {
        "lat": 51.5590,
        "long": 0.0741
    },
    "richmond upon thames": {
        "lat": 51.4479,
        "long": -0.3260
    },
    "southwark": {
        "lat": 51.5035,
        "long": -0.0804
    },
    "sutton": {
        "lat": 51.3618,
        "long": -0.1945
    },
    "tower hamlets": {
        "lat": 51.5099,
        "long": -0.0059
    },
    "waltham forest": {
        "lat": 51.5908,
        "long": -0.0134
    },
    "wandsworth": {
        "lat": 51.4567,
        "long": -0.1910
    },
    "westminster": {
        "lat": 51.4973,
        "long": -0.1372
    }
}

MLFLOW_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "fyp-model-development"

DEFAULT_PREDICTIONS_TABLE_NAME = "london_borough_road_accident_predictions"

OPENWEATHER_API_KEY = 'e5d35b72f07a5050110e80715a67fbb6'

# these would possibly be set manually by data scientists
# but is designed to work after model creation pipeline but demonstration purposes
ROAD_DATA_TABLE_NAME = "london_road_accidents_"+datetime.today().strftime('%d%b%y')
TRAFFIC_FLOW_TABLE_NAME = "traffic_flow_london_boroughs_"+datetime.today().strftime('%d%b%y')
POPULATION_TABLE_NAME = "population_london_boroughs_"+datetime.today().strftime('%d%b%y')
GDP_TABLE_NAME = "gdp_london_boroughs_"+datetime.today().strftime('%d%b%y')
CRIME_TABLE_NAME = "crime_london_boroughs_"+datetime.today().strftime('%d%b%y')
PREDICTIONS_TABLE_NAME = "borough_risk_level_predictions"

CHOSEN_YEAR_FOR_PREDICTIONS = '2021'

MODEL_URI = 'runs:/39a2441b181240bd83ef9491b9407914/model'

with DAG(
    "fyp_predictions_pipeline",
    default_args=DEFAULT_ARGS,
    description="This pipeline is for creating new predictions for road traffic accidents.",
    start_date=datetime(2020,10,10), #required otherwise the DAG cannot proceed
    schedule_interval=timedelta(hours=2), # run the DAG every 2 hours
    catchup=False,
    tags=["fyp", "mlops", "predictions"],
) as dag:
    start=EmptyOperator(task_id="start")
    deployment_task=PythonOperator(task_id="deployment_stage", python_callable=deployment.create_new_predictions,
        op_kwargs={
            'logger':logger,
            'engine_conn':ENGINE_CONN,
            'year':CHOSEN_YEAR_FOR_PREDICTIONS,
            'predictions_table_name':PREDICTIONS_TABLE_NAME,
            'borough_coords':LONDON_BOROUGH_LAT_LONG,
            'gdp_table_name':GDP_TABLE_NAME,
            'population_table_name':POPULATION_TABLE_NAME,
            'traffic_flow_table_name':TRAFFIC_FLOW_TABLE_NAME,
            'crime_table_name':CRIME_TABLE_NAME,
            'openweather_api_key':OPENWEATHER_API_KEY,
            'model_uri':MODEL_URI,
            'mlflow_uri':MLFLOW_URI
        })
    end=EmptyOperator(task_id="end")
    start >> deployment_task >>end