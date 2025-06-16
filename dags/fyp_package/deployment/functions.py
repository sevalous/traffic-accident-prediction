import requests
from mlflow.pytorch import load_model
from pandas import DataFrame, concat, cut
from json import loads
from datetime import datetime
import numpy as np
from numpy import ndarray

from datetime import datetime

from copy import deepcopy

from torch import Tensor

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.orm import Session
from clickhouse_sqlalchemy import Table
from clickhouse_sqlalchemy.engines import MergeTree
import pandahouse as ph

def convert_predictions(data: tuple[DataFrame, Tensor, datetime]) -> tuple[DataFrame, list[int], datetime]:
    """
    Convert predictions made by the LSTM model into relevant risk levels. If the tensor is [w, x, y, z], the risk level
    is equal to the number of these values past the threshold of 1. E.g., so if w > 1, x > 1, y > 1 and z > 1, then
    the risk level would be 5 due to the nature of ordinal classification.
    """
    predictions:Tensor = data[1]
    list_predictions = predictions.tolist()
    for prediction_set_idx in range(len(list_predictions)):
        if ((list_predictions[prediction_set_idx][0] > 1) and (list_predictions[prediction_set_idx][1] > 1)
            and (list_predictions[prediction_set_idx][2] > 1) and (list_predictions[prediction_set_idx][3] > 1)):
            list_predictions[prediction_set_idx] = 5
        elif ((list_predictions[prediction_set_idx][0] > 1) and (list_predictions[prediction_set_idx][1] > 1)
            and (list_predictions[prediction_set_idx][2] > 1)):
            list_predictions[prediction_set_idx] = 4
        elif ((list_predictions[prediction_set_idx][0] > 1) and (list_predictions[prediction_set_idx][1] > 1)):
            list_predictions[prediction_set_idx] = 3
        elif (list_predictions[prediction_set_idx][0] > 1):
            list_predictions[prediction_set_idx] = 2
        elif (list_predictions[prediction_set_idx][0] < 1):
            list_predictions[prediction_set_idx] = 1

    return (data[0], list_predictions, data[2])


def convert_dtypes(df: DataFrame) -> DataFrame:
    """
    Convert all float types in a DataFrame into integers. Also, convert a 'Time' column
    into string type.
    """
    float_cols = df.select_dtypes(float)
    df[float_cols.columns] = df[float_cols.columns].astype(int)
    df['Time'] = df['Time'].astype(str)
    return df

def create_clickhouse_predictions_table(engine_conn:dict, table_name:str) -> None:
    """
    A function that will create a table in a CiickHouse server for storing predictions from generated models.
    """
    engine_path_str = "clickhouse://{user}:{password}@{host}:{port}/{database}".format(user=engine_conn['user'],
                                                                         password=engine_conn['password'],
                                                                         host=engine_conn['host'],
                                                                         port=engine_conn['port'],
                                                                         database=engine_conn['database'])
    
    engine = create_engine(engine_path_str)

    # declaring the table type
    predictions_table = Table(table_name, MetaData(),
        Column(Integer, name='ID', primary_key=True),
        Column(String, name='Borough'),
        Column(Integer, name='GDP Prev Year'),
        Column(Integer, name='Population Prev Year'),
        Column(Integer, name='Traffic Flow Prev Year'),
        Column(Integer, name='Vehicle Crime Prev Month'),
        Column(String, name='Weather Conditions'),
        Column(String, name='Light Conditions'),
        Column(String, name='Time'),
        Column(Integer, name='Year'),
        Column(Integer, name='Month'),
        Column(Integer, name='Day'),
        Column(String, name='DayOfWeek'),
        Column(Integer, name='Predictions'),
        MergeTree(order_by=('ID')),
        schema=engine_conn['database'])
    
    with Session(engine) as session:
        predictions_table.create(session.connection(), checkfirst=True)
        # 'checkfirst' will prevent a new table being created if the table already exists
        session.commit()

def store_predictions(engine_conn:dict, table_name:str, predictions:DataFrame) -> None:
    """
    Take model predictions and store them in a ClickHouse table.

    Parameters
    ----------
        | `engine_conn`: contains the connection info to the ClickHouse server database.
        | `table_name`: the name of the table to write the predictions to.
        | `predictions`: a Pandas DataFrame containing the model input data and the prediction results.
    """
    connection = {
        "database":engine_conn['database'],
        "host":'http://{host}:{port}'.format(host=engine_conn['host'], port=engine_conn['port']),
        "user":engine_conn['user'],
        "password":engine_conn['password']
    }

    ph.to_clickhouse(predictions, table_name, index = False, connection=connection)
    # index of the dataframe is not needed as the prediction time will act as the index

def get_light_conditions(timeapi_uri:str, sunrisesunsetapi_uri:str) -> str:
    """
    Retrieve the current light conditions using the specified APIs.
    """
    # retrieve current time
    current_time_london = loads(
        requests.get(timeapi_uri).content.decode('utf-8')
    )
    current_light_conditions = loads(
        requests.get(sunrisesunsetapi_uri).content.decode('utf-8')
    )

    sunset = datetime.strptime(current_light_conditions['results']['sunset'], "%I:%M:%S %p").time()
    sunrise = datetime.strptime(current_light_conditions['results']['sunrise'], "%I:%M:%S %p").time()
    current_datetime = datetime.strptime(current_time_london['dateTime'].split('.')[0], "%Y-%m-%dT%H:%M:%S" ).time()

    if((current_datetime >=  sunrise) and (current_datetime < sunset)):
        return'light'
    
    return'dark'

def get_borough_weather_conditions(borough_coords:dict[str, dict[str,float]], openweather_apikey:str) -> DataFrame:
    """
    Get the weather conditions of the specified borough.
    """
    df = DataFrame()
    df['Borough'] = np.nan
    df['Weather Conditions'] = np.nan

    for key, value in borough_coords.items():
        current_borough_weather = loads(
            requests.get('https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={long}&appid={api_key}'.format(
                lat=value['lat'],
                long=value['long'],
                api_key=openweather_apikey
            )).content.decode('utf-8'))['weather'][0]['main']

        match current_borough_weather.lower():
            case 'clear':
                current_borough_weather = 'fine'
            case 'rain':
                current_borough_weather = 'raining'
            case 'snow':
                current_borough_weather = 'snowing'
            case 'cloud':
                current_borough_weather = 'fine'
            case 'drizzle':
                current_borough_weather = 'other'
            case _:
                current_borough_weather = 'other'
            
        df = concat([df, DataFrame({
            'Borough': key,
            'Weather Conditions': current_borough_weather
        }, index=[0])], axis=0)

    return df

def retrieve_past_data_from_clickhouse(engine_conn:dict, borough_coords:dict[str, dict[str, float]], gdp_table_name:str,
                                       traffic_flow_table_name:str,
                                       population_table_name:str, crime_table_name:str,
                                       year:str="2021") -> DataFrame:
    """
    Retrieve specific past data from a ClickHouse server instance.
    """
    connection = {
        "database":engine_conn['database'],
        "host":'http://{host}:{port}'.format(host=engine_conn['host'], port=engine_conn['port']),
        "user":engine_conn['user'],
        "password":engine_conn['password']
    }

    gdp:DataFrame = ph.read_clickhouse(
        'SELECT `{}`, `LA name` FROM {}.{}'.format(year, engine_conn['database'], gdp_table_name),
        connection=connection)
    traffic_flow:DataFrame = ph.read_clickhouse(
        'SELECT `{}`, `Local Authority` FROM {}.{}'.format(year, engine_conn['database'], traffic_flow_table_name),
        connection=connection)
    population:DataFrame = ph.read_clickhouse(
        'SELECT `{}`, `LA name` FROM {}.{}'.format(year, engine_conn['database'], population_table_name),
        connection=connection)
    
    crime_col_to_get = year + str(datetime.now().month).zfill(2)

    vehicle_crime:DataFrame = ph.read_clickhouse(
        'SELECT `{}`, `BoroughName`, `MajorText`, `MinorText` FROM {}.{}'.format(crime_col_to_get, engine_conn['database'],
                                                                                crime_table_name),
        connection=connection)

    vehicle_crime = vehicle_crime[vehicle_crime['MinorText'].isin(['THEFT FROM A VEHICLE', 'TRAFFICKING OF DRUGS'])]
    
    gdp['LA name'] = gdp['LA name'].str.replace('-', ' ')
    gdp['LA name'] = gdp['LA name'].str.replace('&', 'and')
    gdp['LA name'] = gdp['LA name'].str.lower()

    population['LA name'] = population['LA name'].str.replace('-', ' ')
    population['LA name'] = population['LA name'].str.replace('&', 'and')
    population['LA name'] = population['LA name'].str.lower()

    traffic_flow['Local Authority'] = traffic_flow['Local Authority'].str.replace('-', ' ')
    traffic_flow['Local Authority'] = traffic_flow['Local Authority'].str.replace('&', 'and')
    traffic_flow['Local Authority'] = traffic_flow['Local Authority'].str.lower()

    vehicle_crime['BoroughName'] = vehicle_crime['BoroughName'].str.replace('-', ' ')
    vehicle_crime['BoroughName'] = vehicle_crime['BoroughName'].str.replace('&', 'and')
    vehicle_crime['BoroughName'] = vehicle_crime['BoroughName'].str.lower()

    df = DataFrame()
    df['Borough'] = np.nan
    df['GDP Prev Year'] = np.nan
    df['Population Prev Year'] = np.nan
    df['Traffic Flow Prev Year'] = np.nan
    df['Vehicle Crime Prev Month'] = np.nan
    #df['Vehicle Crime Prev Month'] = np.nan

    for key, _ in borough_coords.items():
        gdp_prev_year = gdp[gdp['LA name'] == key][year].values[0]
        population_prev_year = population[population['LA name'] == key][year].values[0]
        traffic_flow_prev_year = traffic_flow[traffic_flow['Local Authority'] == key][year].values[0]
        crime_prev_month = vehicle_crime[vehicle_crime['BoroughName'] == key].groupby('BoroughName').agg('sum')[crime_col_to_get].values[0]

        df = concat([df, DataFrame({
            'Borough': key,
            'GDP Prev Year':gdp_prev_year,
            'Population Prev Year':population_prev_year,
            'Traffic Flow Prev Year':traffic_flow_prev_year,
            'Vehicle Crime Prev Month':crime_prev_month
        }, index=[0])], axis=0)

    return df
    #vehicle_crime:Series = ph.read_clickhouse('SELECT {} FROM {}.{}'.format(year, engine_conn['database'], gdp_table_name), index = True, index_col = 'index', connection=connection)

def make_predictions(data: tuple[DataFrame, Tensor], mlflow_model_uri:str) -> tuple[DataFrame, Tensor, datetime]:
    """
    Make predictions using unseen data on a modal stored in MLflow.

    Returns
    -------
        | A tuple containing the original data, the predictions, and the time of prediction.
    """
    loaded_model = load_model(mlflow_model_uri)
    return (data[0], loaded_model(data[1]), datetime.now())

def combine_input_data_and_predictions(data:tuple[DataFrame, list[int], datetime]) -> DataFrame:
    input_data:DataFrame = data[0]
    predictions:ndarray = data[1]
    prediction_time:datetime = data[2]

    prediction_time = prediction_time.replace(microsecond=0)

    # replicate the prediction time to be the same as the length of the number of predictions
    # this is so each row can get the prediction time
    # prediction_time_df = concat([DataFrame({
    #         'Prediction DateTime':prediction_time
    #     }, index=[0])]*len(input_data), ignore_index=True)
    
    # add the predictions to a new column in the dataframe
    input_data['Predictions'] = predictions

    #merged_data = concat([input_data.reset_index(drop=True), prediction_time_df.reset_index(drop=True)], axis=1, join='inner')
    return input_data

def label_encoding(original_data:DataFrame) -> tuple[DataFrame,DataFrame]:
    input_df = deepcopy(original_data) # make a deepcopy so does not reference original data
    le_borough = LabelEncoder()
    le_time = LabelEncoder()
    le_light_conditions = LabelEncoder()
    le_weather = LabelEncoder()
    le_dayofweek = LabelEncoder()

    input_df['Borough'] = le_borough.fit_transform(input_df['Borough'])
    input_df['Time'] = le_time.fit_transform(input_df['Time'])
    input_df['Light Conditions'] = le_light_conditions.fit_transform(input_df['Light Conditions'])
    input_df['Weather Conditions'] = le_weather.fit_transform(input_df['Weather Conditions'])
    input_df['DayOfWeek'] = le_dayofweek.fit_transform(input_df['DayOfWeek'])

    return (original_data, input_df)

def data_scaling(data: tuple[DataFrame,DataFrame]) -> tuple[DataFrame,DataFrame]:
    min_max_scaler = MinMaxScaler(feature_range=(0,1))
    return (data[0], min_max_scaler.fit_transform(data[1]))

def time_discretisation(df: DataFrame) -> DataFrame:
    # as 'Time' contains integer values, the bins need to be real number as well
    # they are float values as the bins work by collecting values between each index
        # e.g., values between 0 and 2 (not including 2) will be put in bin 1
        # which has a label of '00:00-01:59'
    bins = [0, 1.99, 3.99, 5.99, 7.99, 9.99, 11.99, 13.99, 15.99, 17.99, 19.99, 21.99, 23.99]

    # the labels represent the time of day using a 24-hour clock
    labels = ['00:00-01:59', '02:00-03:59', '04:00-05:59', '06:00-07:59', '08:00-09:59', '10:00-11:59', '12:00-13:59',
        '14:00-15:59', '16:00-17:59', '18:00-19:59', '20:00-21:59', '22:00-23:59']
    
    df['Time'] = cut(x=df['Time'], bins=bins, labels=labels, include_lowest=True, ordered=True)

    return df

def get_time_dayofweek_and_dayofweek(timeapi_uri:str) -> DataFrame:
    current_time_london = loads(
        requests.get(timeapi_uri).content.decode('utf-8')
    )

    return DataFrame({
        'Time': current_time_london['hour'],
        'Year': current_time_london['year'],
        'Month': current_time_london['month'],
        'Day': current_time_london['day'],
        'DayOfWeek': current_time_london['dayOfWeek']
    }, index=[0])

def fuse_data(weather_df:DataFrame, prev_clickhouse_data_df:DataFrame, light_conditions:str, times_df:DataFrame) -> DataFrame:
    merged_df = prev_clickhouse_data_df.merge(weather_df, on='Borough')

    # replicate the single light conditions across the data
    light_conditions_df = concat([DataFrame({
            'Light Conditions':light_conditions
        }, index=[0])]*len(merged_df), ignore_index=True)
    times_df = concat([times_df]*len(merged_df), axis=0)

    merged_df = concat([merged_df.reset_index(drop=True), light_conditions_df.reset_index(drop=True)], axis=1, join='inner')

    merged_df = concat([merged_df.reset_index(drop=True), times_df.reset_index(drop=True)], axis=1, join='inner')

    return merged_df

def convert_to_tensor_and_unsqueeze(data: tuple[DataFrame,DataFrame]) -> tuple[DataFrame,Tensor]:
    """
    Convert a Pandas DataFrame into a PyTorch Tensor.

    Parameters
    ----------
        | `data`: a tuple containing two DataFrames, one for the original data at index 0,
        while the other at index 1 is the transformed and to be transformed data into a Tensor.

    Returns
    -------
        | A tuple with the original data, as a DataFrame, at index 0, and a Tensor at index 1.
    """
    return (data[0], Tensor(data[1]))