# data manipulation
from pandas import cut, DateOffset, DataFrame as PandasDataFrame, Series as PandasSeries
from dask.dataframe import DataFrame, to_datetime, from_pandas
from dask_ml.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray

# for static type checking, but this will not be enforced
from typing import Union

# server connection
import pandahouse as ph
from pickle import dumps
from redis import Redis

# for converting data into tensors
from torch import Tensor, tensor

def categorise_risk_level(value:int) -> int:
    if value < 2:
        return 1
    elif (value >= 2) and (value < 4):
        return 2
    elif (value>=4) and (value < 8):
        return 3
    elif (value>=8) and (value < 12):
        return 4
    elif (value>=12):
        return 5

def create_risk_level(df: DataFrame) -> DataFrame:
    """
    Create a 'Risk Level' column in the merged dataframe by transforming the 'Accident Severity' column.
    This function requires all labels of the dataframe to be numerically encoded.

    Returns
    -------
        | A Dask DataFrame with a 'Accident Severity' column transformed into a 'Risk Level' column.
    """
    # aggregation dictionary - tells the 'agg' function how to combine elements
    agg_dict = {
        'DayOfWeek': 'first',
        'Year': 'first',
        'Month': 'first',
        'Day': 'first',
        'Light Conditions': 'first',
        'Weather Details': 'first',
        'GDP Prev Year': 'first',
        'Population Prev Year': 'first',
        'Traffic Flow Prev Year': 'first',
        'Vehicle Crime Offences Prev Month': 'first',
        'Accident Severity': 'sum'
    }

    df : DataFrame = from_pandas(df.compute().groupby(by=['Borough', 'Collision Date', 'Time'],
                                                           as_index=False, dropna=True).agg(agg_dict).reset_index(drop=True), 
                                                           npartitions=16)
    # 'as_index' prevents the designated columns becoming the index
    # 'dropna' is present as the nature of 'groupby' can lead to NaN values in some columns (which getting rid of does not
        # affect the original data)
    # the above is essentially used to combine rows together into one, and aggregate the individual values based on 'agg_dict'
    
    df = df.rename(columns={
        'Accident Severity':'Risk Level'
    })

    # now the values have been aggregated together, it is time to designate the actual risk levels
    # this can be seen in the 'categorise_risk_level' function
    df['Risk Level'] = df['Risk Level'].map_partitions(lambda df: df.apply(categorise_risk_level))

    return df

def create_weekday_col(df: DataFrame) -> DataFrame:
    """
    Take the specific date of a datetime entry (e.g., 16th September 2014) and add the corresponding
    day of the week to a new column, 'DayOfWeek'.
    """
    df['DayOfWeek'] = df['Collision Date'].dt.weekday
    return df

def crime_dataset_feature_selection(df: DataFrame) -> DataFrame:
    """
    Select features of the crime-related dataset that are linked with vehicle offences (e.g., vehicle theft, drug trafficking, etc.).

    Returns
    -------
        | A Dask DataFrame with certain vehicle-related features selected.
    """
    return df[df['MajorText'] == 'VEHICLE OFFENCES']

def data_fusion(london_acc_df: DataFrame, crime_df: DataFrame, population_df: DataFrame,
                gdp_df: DataFrame, traffic_flow_df: DataFrame) -> DataFrame:
    """
    Merge the five designated datasets together into one Dask DataFrame.
    """
    # get only the numerical columns of the other dataframes
    gdp_numeric_cols = [col for col in gdp_df.columns if col.isdigit()]
    pop_numeric_cols = [col for col in population_df.columns if col.isdigit()]
    traffic_flow_numerical_cols = [col for col in traffic_flow_df.columns if col.isdigit()]
    crime_num_cols = [col for col in crime_df.columns if col.isdigit()]

    # 'melt' is a special function used to essentially convert a wide-dataframe into a long-dataframe (reduce columns and increase rows)
    # this is used due to some of the dataframes having the year and/or month set as the column name
    gdp_long : PandasDataFrame = gdp_df.compute().melt(
        id_vars=['Borough'],
        value_vars=gdp_numeric_cols, var_name='Year',
        value_name='GDP Prev Year')
    population_long : PandasDataFrame = population_df.compute().melt(
        id_vars=['Borough'],
        value_vars=pop_numeric_cols, var_name='Year',
        value_name='Population Prev Year')
    traffic_long : PandasDataFrame = traffic_flow_df.compute().melt(
        id_vars=['Borough'],
        value_vars=traffic_flow_numerical_cols, var_name='Year',
        value_name='Traffic Flow Prev Year')
    crime_long : PandasDataFrame = crime_df.compute().melt(
        id_vars=['Borough'],
        value_vars=crime_num_cols, var_name='Year_Month',
        value_name='Vehicle Crime Offences Prev Month')

    # seperate datetime entries in the crime dataframe, e.g., '201004' becomes year 2010, month 04 (April)
    crime_long['Year'] = crime_long['Year_Month'].str[:4]
    crime_long['Month'] = crime_long['Year_Month'].str[4:]

    crime_long['Year'] = crime_long['Year'].astype(int)
    crime_long['Month'] = crime_long['Month'].astype(int)

    # convert the dates in the crime dataframe into a datetime format
    # this is so each month can be pushed forward by 1 month properly
        # this is to match up entries such that current data becomes past data
    crime_long['Year_Month'] = to_datetime(crime_long['Year'].astype(str) + '-' + crime_long['Month'].astype(str),
                                           format='%Y-%m')
    crime_long['Year_Month'] = crime_long['Year_Month'] + DateOffset(months=1)

    # aggregate entries together that have the same borough and time to get the total vehicle offences for that time
    crime_long = crime_long.groupby(['Borough', 'Year',
                                     'Month'], as_index=False)['Vehicle Crime Offences Prev Month'].sum()

    # Convert the years from a string type to an integer type, ready for combining them
    gdp_long['Year'] = gdp_long['Year'].astype(int)
    population_long['Year'] = population_long['Year'].astype(int)
    traffic_long['Year'] = traffic_long['Year'].astype(int)

    # make certain data go from future data to past data for entries the year after
        # a similar thing was done for the crime dataset, except the month was pushed
        # forward by 1, as it is on a month-by-month basis
    gdp_long['Year'] = gdp_long['Year']+1
    population_long['Year'] = population_long['Year']+1
    traffic_long['Year'] = traffic_long['Year']+1

    # Merge the DataFrames
    merged_df : DataFrame = london_acc_df.merge(gdp_long, on=['Borough', 'Year'], how='left')
    merged_df = merged_df.merge(population_long, on=['Borough', 'Year'], how='left')
    merged_df = merged_df.merge(traffic_long, on=['Borough', 'Year'], how='left')
    
    return merged_df.merge(crime_long, on=['Borough', 'Year', 'Month'], how='left')

def date_selection_and_seperation(df: DataFrame) -> DataFrame:
    """
    For the London road accidents dataset, remove all entries before 2011 and after 2021.
    This will also break down any remaining datetimes into the respective year, month and day of the month.
    """
    df[(df['Collision Date'] > datetime(2011, 1, 1)) & (df['Collision Date'] < datetime(2022, 1, 1))]
    # date determined through EDA analysis
    df['Year'] = df['Collision Date'].dt.year
    df['Month'] = df['Collision Date'].dt.month
    df['Day'] = df['Collision Date'].dt.day
    return df

def decapitalise_col_values(df: DataFrame) -> DataFrame:
    """
    Decapitalise values of certain columns of the Greater London accident data, so values are consistent.
    """

    df['Borough'] = df['Borough'].str.lower()
    df['Weather Details'] = df['Weather Details'].str.lower()
    df['Accident Severity'] = df['Accident Severity'].str.lower()
    df['Light Conditions'] = df['Light Conditions'].str.lower()
    return df

def drop_collision_date(df: PandasDataFrame) -> PandasDataFrame:
    """
    Drop the collision date column from the merged dataframe.
    """
    # 'Collision Date' no longer needed
    return df.drop(columns='Collision Date')

def london_acc_feature_selection(df: DataFrame) -> DataFrame:
    """
    Select specific features of the London road accidents dataset.
    """
    # selected features that relate to a Borough level
    return df[['Borough', 'Collision Date', 'Time', 'Accident Severity', 'Light Conditions', 'Weather Details']]

def label_encoding(df: DataFrame) -> DataFrame:
    """
    Encode categorical features of the merged dataframe.
    """
    le_borough = LabelEncoder()
    le_time = LabelEncoder()
    le_light_conditions = LabelEncoder()
    le_weather = LabelEncoder()

    df['Borough'] = le_borough.fit_transform(df['Borough'])
    df['Time'] = le_time.fit_transform(df['Time'])
    df['Light Conditions'] = le_light_conditions.fit_transform(df['Light Conditions'])
    df['Weather Details'] = le_weather.fit_transform(df['Weather Details'])

    df['Accident Severity'] = df['Accident Severity'].replace('slight', 1)
    df['Accident Severity'] = df['Accident Severity'].replace('serious', 4)
    df['Accident Severity'] = df['Accident Severity'].replace('fatal', 12)

    return df

def make_light_conditions_consistent(df: DataFrame) -> DataFrame:
    """
    A function to make the values of the light conditions column of the London road accidents dataset consistent.
    """
    df['Light Conditions'] = df['Light Conditions'].str.replace(r"dark.+", "dark", regex=True)
    return df

def make_time_consistent(df: DataFrame) -> DataFrame:
    """
    A function to make the values of the time of day column of the London road accidents dataset consistent.
    Additionally, this column type will be converted to datetime.
    """
    df['Time'] = df['Time'].str.replace("'", "")
    df['Time'] = df['Time'].str.replace(":", "")
    df['Time'] = df['Time'].str.replace("\\", "")
    df['Time'] = df['Time'].str.replace(r"00$", "", regex=True)
    df['Time'] = to_datetime(df['Time'], format='%H%M')
    df['Time'] = df['Time'].dt.hour
    df['Time'] = df['Time'].astype(int)
    return df

def remove_duplicates_and_nan(df: DataFrame) -> DataFrame:
    """
    Remove duplicates and NaN values of a dataset using Complete Case Analysis.
    """
    return df.drop_duplicates().dropna(how='any')

def remove_unknown_weather_conditions(df: DataFrame) -> DataFrame:
    """
    Remove entries which have 'unknown' weather conditions.
    """
    return df[df['Weather Details'] != "unknown"]

def retrieve_data_from_ClickHouse(engine_conn: dict, table_name: str) -> DataFrame:
    """
    Retrieves data from designated tables from a ClickHouse server.

    Parameters
    ----------
        | ``engine_conn``: A dictionary containing the connection information to the ClickHouse server.
        | ``table_name``: The name of the table matching that in the ClickHouse server; for the data to be retrieved from.

    Returns
    -------
        Data in the form of a Dask DataFrame.
    """
    connection = {
        "database":engine_conn['database'],
        "host":'http://{host}:{port}'.format(host=engine_conn['host'], port=engine_conn['port']),
        "user":engine_conn['user'],
        "password":engine_conn['password']
    }

    return from_pandas(ph.read_clickhouse('SELECT * FROM {}.{}'.format(
        engine_conn['database'], table_name), index = True, index_col = 'index', connection=connection), npartitions=16)

def rename_cols(df: DataFrame, col_rename_dict: dict[str, str]) -> DataFrame:
    """
    Rename columns within a Dask DataFrame.
    """
    return df.rename(columns=col_rename_dict)

def scale_X(split_data: dict[str, Union[PandasDataFrame, PandasSeries]]) -> dict[str, Union[ndarray, PandasSeries]]:
    """
    Scale the indepedent variables of the training and testing sets between the values of 0 and 1.

    Parameters
    --------
        ``split_data``: the training and testing sets (e.g., X_train, X_test, y_train, y_test).
    """
    minmaxscaler = MinMaxScaler(feature_range=(0,1))

    split_data['X_train'] = minmaxscaler.fit_transform(split_data['X_train'])
    split_data['X_val'] = minmaxscaler.transform(split_data['X_val'])
    split_data['X_test'] = minmaxscaler.transform(split_data['X_test'])

    return split_data

def standardise_borough_names(df: DataFrame) -> DataFrame:
    """
    Make the names of boroughs consistent by removing and editing punctuation and capitalisation.
    """
    df['Borough'] = df['Borough'].str.replace('-', ' ')
    df['Borough'] = df['Borough'].str.replace('&', 'and')
    df['Borough'] = df['Borough'].str.lower()
    return df

def sort_values(df: DataFrame) -> PandasDataFrame:
    """
    Change the order of rows by organising the date and time (of day) of each entry.
    Then, the index will be reset so as to accomodate these changes. This makes the
    data become chronological.

    Returns
    -------
        | A Pandas DataFrame of the sorted data. It is a Pandas DataFrame so the index order
        is maintained, unlike in Dask due to partitioning.
    """
    return df.compute().sort_values(by=['Collision Date', 'Time']).reset_index(drop=True)
    # must be converted into a Pandas DataFrame so the index is maintained when split into independent variables and target variable

def time_discretisation(df: DataFrame) -> DataFrame:
    """
    Convert a positive integer sequence of values which represent each hour of the day
    into a select number of bins with associated labels.
    """
    # as 'Time' contains integer values, the bins need to be real number as well
    # they are float values as the bins work by collecting values between each index
        # e.g., values between 0 and 2 (not including 2) will be put in bin 1
        # which has a label of '00:00-01:59'
    bins = [0, 1.99, 3.99, 5.99, 7.99, 9.99, 11.99, 13.99, 15.99, 17.99, 19.99, 21.99, 23.99]

    # the labels represent the time of day using a 24-hour clock
    labels = ['00:00-01:59', '02:00-03:59', '04:00-05:59', '06:00-07:59', '08:00-09:59', '10:00-11:59', '12:00-13:59',
        '14:00-15:59', '16:00-17:59', '18:00-19:59', '20:00-21:59', '22:00-23:59']
    
    df['Time'] = df['Time'].map_partitions(cut, bins=bins, labels=labels, include_lowest=True, ordered=True)
    # 'map_partitions' is a Dask function used to apply a function to each partition in the dataframe
        # in this case, it will be the Pandas 'cut' method - which is used to bin values into discrete values
    # 'ordered=True' will match each index in the bins and labels together
    # 'include_lowest=True' will include the first index, so instead of values being binned like
        # 0 < x < 2, it becomes 0 <= x < 2
    return df

def train_val_test_X_y_split(df: PandasDataFrame) -> dict[str, Union[PandasDataFrame, PandasSeries]]:
    """
    Split the dataframe into one dataframe for independent variables and one dataframe
    for the target variable. Then, split these dataframes into the training and testing sets.
    """
    train_size = int(0.7 * len(df))
    test_size = int(0.2 * len(df))
    val_size = len(df)-test_size
    # 70% training, 10% validation, 20% testing
    # in chronological order, so 70% will be oldest entries, etc.

    X_train = df.iloc[0:train_size, :].drop(columns='Risk Level')
    X_val = df.iloc[train_size:val_size, :].drop(columns='Risk Level')
    X_test = df.iloc[val_size:len(df), :].drop(columns='Risk Level')
    y_train = df.iloc[0:train_size, :]['Risk Level']
    y_val = df.iloc[train_size:val_size, :]['Risk Level']
    y_test = df.iloc[val_size:len(df), :]['Risk Level']
    # separation of independent variables and target variable
    # the rows are sorted by the 'Collision Date' column, then the index is reset

    return {'X_train':X_train, 'X_val':X_val, 'X_test':X_test, 'y_train':y_train, 'y_val':y_val, 'y_test':y_test}

def torch_tensor_conversion(split_data: dict[str, Union[ndarray, PandasSeries]]) -> dict[str, Tensor]:
    """
    Convert training and testing sets into PyTorch Tensors, so they can be processed by PyTorch.

    Returns
    -------
        | A dictionary of the training and testing sets, each converted into a PyTorch tensor.
    """
    X_train = split_data['X_train']
    X_val = split_data['X_val']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_val = split_data['y_val']
    y_test = split_data['y_test']
    # 'X_train', 'X_val' and 'X_test' are already numpy arrays

    return {
        'X_train': tensor(X_train),
        'X_val': tensor(X_val),
        'X_test': tensor(X_test),
        'y_train': tensor(y_train.to_numpy()),
        'y_val': tensor(y_val.to_numpy()),
        'y_test': tensor(y_test.to_numpy())
    }

def upload_to_redis(data: dict[str, Tensor], redis_host: str, redis_port: int) -> None:
    """
    Upload training and testing sets into a Redis datastore, so it can be passed to the next pipeline task.
    """
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    # using 'with' as a context manager, so connection is closed automatically after the code has been run
    with Redis(host=redis_host, port=redis_port) as redis_conn:
        redis_conn.set(name="X_train_preprocessed", value=dumps(X_train))
        redis_conn.set(name="X_val_preprocessed", value=dumps(X_val))
        redis_conn.set(name="X_test_preprocessed", value=dumps(X_test))
        redis_conn.set(name="y_train_preprocessed", value=dumps(y_train))
        redis_conn.set(name="y_val_preprocessed", value=dumps(y_val))
        redis_conn.set(name="y_test_preprocessed", value=dumps(y_test))

    # Pickle 'dumps' converts the data into a bytes Python object, so it can be stored in Redis more effectively