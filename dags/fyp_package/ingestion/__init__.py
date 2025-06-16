# import logging to log any errors or info to Airflow
import logging

from .functions import *

def ingest_data_main(london_acc_data_path:str,
                     traffic_flow_data_path:str,
                     population_data_path:str,
                     gdp_data_path:str,
                     crime_data_path:str,
                     logger:logging.Logger,
                     engine_conn:dict,
                     london_acc_table_name:str,
                     traffic_flow_table_name:str,
                     population_table_name:str,
                     gdp_table_name:str,
                     crime_table_name:str) -> None:
    """
    Function that manages the program flow of the ingestion stage.
    """
    try:
        create_clickhouse_tables(engine_conn=engine_conn,
                                road_table_name=london_acc_table_name,
                                traffic_flow_table_name=traffic_flow_table_name,
                                population_table_name=population_table_name,
                                gdp_table_name=gdp_table_name,
                                crime_table_name=crime_table_name)

        # ingest Greater London road accident data
        # this data requires extra transformations compared to the other datasets,
            # so some custom functions are needed
        if not table_populated(engine_conn=engine_conn, table_name=london_acc_table_name):
            upload_ingested_data(
                preprocess_road_data(
                    ingest_road_data(directory_path=london_acc_data_path)
                    ),
                engine_conn=engine_conn,
                table_name=london_acc_table_name
            )

        # ingest traffic flow data
        if not table_populated(engine_conn=engine_conn, table_name=traffic_flow_table_name):
            upload_ingested_data(
                ingest_data(traffic_flow_data_path),
                engine_conn=engine_conn,
                table_name=traffic_flow_table_name
            )

        if not table_populated(engine_conn=engine_conn, table_name=population_table_name):
            # ingest population data
            upload_ingested_data(
                ingest_data(population_data_path),
                engine_conn=engine_conn,
                table_name=population_table_name
            )

        if not table_populated(engine_conn=engine_conn, table_name=gdp_table_name):
            # ingest GDP data
            upload_ingested_data(
                ingest_data(gdp_data_path),
                engine_conn=engine_conn,
                table_name=gdp_table_name
            )

        if not table_populated(engine_conn=engine_conn, table_name=crime_table_name):
            # ingest crime data
            upload_ingested_data(
                ingest_data(crime_data_path),
                engine_conn=engine_conn,
                table_name=crime_table_name
            )

    except Exception as e:
        logger.critical("Error occured during ingestion stage.\n{}".format(e), exc_info=1)
    else:
        logger.info("Ingestion stage successful!")