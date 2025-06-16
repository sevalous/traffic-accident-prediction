# import logging to log any errors or info to Airflow
import logging

from .functions import *

def preprocess_data_main(logger:logging.Logger, engine_conn:dict, london_acc_table_name:str, population_table_name:str,
                         gdp_table_name:str, traffic_flow_table_name: str, crime_table_name:str, redis_host:str,
                         redis_port:int) -> None:
    """
    Function that manages the program flow of the preprocessing stage.
    """
    try:
        upload_to_redis(
            torch_tensor_conversion(
                scale_X(
                    train_val_test_X_y_split(
                        drop_collision_date(
                            sort_values(
                                create_risk_level(
                                    label_encoding(
                                        time_discretisation(
                                            remove_duplicates_and_nan(
                                                data_fusion(
                                                    london_acc_df=date_selection_and_seperation(
                                                        create_weekday_col(
                                                            make_time_consistent(
                                                                make_light_conditions_consistent(
                                                                    remove_unknown_weather_conditions(
                                                                        standardise_borough_names(
                                                                            decapitalise_col_values(
                                                                                london_acc_feature_selection(
                                                                                    retrieve_data_from_ClickHouse(
                                                                                        engine_conn=engine_conn,
                                                                                        table_name=london_acc_table_name)
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    ),
                                                    population_df=standardise_borough_names(
                                                        rename_cols(
                                                            df=retrieve_data_from_ClickHouse(
                                                                engine_conn=engine_conn,
                                                                table_name=population_table_name),
                                                            col_rename_dict={'LA name':'Borough'}
                                                        )
                                                    ),
                                                    gdp_df=standardise_borough_names(
                                                        rename_cols(
                                                            df=retrieve_data_from_ClickHouse(
                                                                engine_conn=engine_conn,
                                                                table_name=gdp_table_name),
                                                            col_rename_dict={'LA name':'Borough'}
                                                        )
                                                    ),
                                                    traffic_flow_df=standardise_borough_names(
                                                        rename_cols(
                                                            df=retrieve_data_from_ClickHouse(
                                                                engine_conn=engine_conn,
                                                                table_name=traffic_flow_table_name),
                                                            col_rename_dict={'Local Authority':'Borough'}
                                                        )
                                                    ),
                                                    crime_df=crime_dataset_feature_selection(
                                                        standardise_borough_names(
                                                            rename_cols(
                                                                df=retrieve_data_from_ClickHouse(
                                                                    engine_conn=engine_conn, table_name=crime_table_name),
                                                                col_rename_dict={'BoroughName':'Borough'}
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            redis_host=redis_host,
            redis_port=redis_port
        )
    except Exception as e:
        logger.critical("Error occured during preprocessing stage.\n{}".format(e), exc_info=1)
    else:
        logger.info("Preprocessing stage successful!")