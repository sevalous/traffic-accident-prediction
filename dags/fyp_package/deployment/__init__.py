# import logging to log any errors or info to Airflow
import logging

import mlflow

from .functions import *

def create_new_predictions(logger:logging.Logger, engine_conn:dict,
                           borough_coords:dict[str,dict[str,float]], predictions_table_name:str,
                           openweather_api_key:dict[str,str], model_uri:str, mlflow_uri:str,
                           gdp_table_name:str, population_table_name:str,
                           traffic_flow_table_name:str, crime_table_name:str, year:str):
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # set the connection to the MLflow server to access its runs and models

        create_clickhouse_predictions_table(engine_conn=engine_conn, table_name=predictions_table_name)

        store_predictions(
            predictions=convert_dtypes(
            combine_input_data_and_predictions(
            convert_predictions(
            make_predictions(
            convert_to_tensor_and_unsqueeze(
            data_scaling(
                label_encoding(
                    time_discretisation(
                        fuse_data(
                            weather_df=get_borough_weather_conditions(borough_coords=borough_coords, openweather_apikey=openweather_api_key),
                            light_conditions=get_light_conditions(
                                timeapi_uri='https://www.timeapi.io/api/Time/current/coordinate?latitude=51.3026&longitude=-0.739',
                                sunrisesunsetapi_uri='https://api.sunrisesunset.io/json?lat=51.3026&lng=-0.739'
                            ),
                            prev_clickhouse_data_df=retrieve_past_data_from_clickhouse(engine_conn=engine_conn, borough_coords=borough_coords,
                                                                gdp_table_name=gdp_table_name, year=year,
                                                                population_table_name=population_table_name,
                                                                traffic_flow_table_name=traffic_flow_table_name,
                                                                crime_table_name=crime_table_name
                                                            ),
                            times_df=get_time_dayofweek_and_dayofweek(
                                timeapi_uri='https://www.timeapi.io/api/Time/current/coordinate?latitude=51.3026&longitude=-0.739')
                        )
                    )
                )
            )
            ),
            mlflow_model_uri=model_uri
            )
            )
            )
            ),
            table_name=predictions_table_name,
            engine_conn=engine_conn
        )
            
    except Exception as e:
        logger.critical("Error occured during deployment stage.\n{}".format(e), exc_info=1)
    else:
        logger.info("Deployment stage successful!")