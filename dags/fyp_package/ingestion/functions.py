# data manipulation
from dask.dataframe import DataFrame, concat, to_datetime, from_pandas
from pandas import read_csv

# server connection and table construction
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.orm import Session
from clickhouse_sqlalchemy import Table
from clickhouse_sqlalchemy.engines import MergeTree
import pandahouse as ph

# for listing needed files
from os import listdir

def table_populated(engine_conn:dict, table_name:str):
    """
    Check to see if a table in the ClickHouse server is already populated.
    This is to help prevent data duplication.
    """
    connection = {
        "database":engine_conn['database'],
        "host":'http://{host}:{port}'.format(host=engine_conn['host'], port=engine_conn['port']),
        "user":engine_conn['user'],
        "password":engine_conn['password']
    }

    if(ph.execute("SELECT EXISTS(SELECT 1 FROM airflow_storage.{});".format(table_name),
                  connection=connection).decode('utf-8').find('1') != -1):
        return True
    # the above SQL attempts to retrieve a single record from the designated table in the ClickHouse server
    # the result from the executed SQL is interpreted as byte-code, which is converted
        # into a Python string using 'decode'
    # 'find' makes it so if a specific string value is not present, it will return -1
        # often known as False
    # if the table is populated, the resulting byte-code will contain '1' (true), otherwise
        # it will contain '0' (false)
    
    return False


def create_clickhouse_tables(engine_conn:dict,
                             road_table_name:str,
                             traffic_flow_table_name:str,
                             population_table_name:str,
                             gdp_table_name:str,
                             crime_table_name:str) -> None:
    """
    This will attempt to create the required tables in the ClickHouse server instance using the designated schemas.

    Parameters
    ----------
        | ``engine_conn``: A dictionary containing the connection details to the ClickHouse server instance.
        | ``road_table_name``: The desired table name for creating an appropriate schema for the Greater London road data.
        | ``traffic_flow_table_name``: ...
        | ``population_table_name``: ...
        | ``gdp_table_name``: ...
        | ``crime_table_name``: ...
    """
    engine_path_str = "clickhouse://{user}:{password}@{host}:{port}/{database}".format(user=engine_conn['user'],
                                                                         password=engine_conn['password'],
                                                                         host=engine_conn['host'],
                                                                         port=engine_conn['port'],
                                                                         database=engine_conn['database'])
    engine = create_engine(engine_path_str)

    road_table = Table(road_table_name, MetaData(),
                  Column(Integer, name = 'index', primary_key=True),
                  Column(String, name = 'Accident Ref'),
                  Column(String, name = 'Borough'),
                  Column(Integer, name = 'Borough Number'),
                  Column(Float, name = 'Easting'),
                  Column(Float, name = 'Northing'),
                  Column(String, name='Accident Severity'),
                  Column(Integer, name = 'Casualty Count'),
                  Column(Integer, name = 'Vehicle Count'),
                  Column(DateTime, name = 'Collision Date'),
                  Column(String, name = 'Day'),
                  Column(String, name = 'Time'),
                  Column(String, name = 'First Road Class'),
                  Column(String, name = "First Road Number"),
                  Column(String, name = "Road Type"),
                  Column(String, name = "Speed Limit"),
                  Column(String, name = "Junction Detail"),
                  Column(String, name = "Junction Control"),
                  Column(String, name = 'Second Road Class'),
                  Column(String, name = "Second Road Number"),
                  Column(String, name = "Pedestrian Crossing Facilities"),
                  Column(String, name = "Light Conditions"),
                  Column(String, name = "Weather Details"),
                  Column(String, name = "Road Surface Condition"),
                  Column(String, name = "Special Conditions at Site"),
                  Column(String, name = "Carriageway Hazards"),
                  Column(String, name = "Place Collision Reported"),
                  Column(String, name = "Collision Location Details"),
                  Column(Float, name = "Attendant Count"),
                  Column(String, name = "Highway Authority"),
                  MergeTree(order_by=('index')),
                  schema=engine_conn['database'])
    
    crime_table = Table(crime_table_name, MetaData(),
                    Column(Integer, name = 'index', primary_key=True),
                    Column(String, name = 'MajorText'),
                    Column(String, name = 'MinorText'),
                    Column(String, name = 'BoroughName'),
                    Column(Integer, name = '201004'),
                    Column(Integer, name = '201005'),
                    Column(Integer, name = '201006'),
                    Column(Integer, name = '201007'),
                    Column(Integer, name = '201008'),
                    Column(Integer, name = '201009'),
                    Column(Integer, name = '201010'),
                    Column(Integer, name = '201011'),
                    Column(Integer, name = '201012'),
                    Column(Integer, name = '201101'),
                    Column(Integer, name = '201102'),
                    Column(Integer, name = '201103'),
                    Column(Integer, name = '201104'),
                    Column(Integer, name = '201105'),
                    Column(Integer, name = '201106'),
                    Column(Integer, name = '201107'),
                    Column(Integer, name = '201108'),
                    Column(Integer, name = '201109'),
                    Column(Integer, name = '201110'),
                    Column(Integer, name = '201111'),
                    Column(Integer, name = '201112'),
                    Column(Integer, name = '201201'),
                    Column(Integer, name = '201202'),
                    Column(Integer, name = '201203'),
                    Column(Integer, name = '201204'),
                    Column(Integer, name = '201205'),
                    Column(Integer, name = '201206'),
                    Column(Integer, name = '201207'),
                    Column(Integer, name = '201208'),
                    Column(Integer, name = '201209'),
                    Column(Integer, name = '201210'),
                    Column(Integer, name = '201211'),
                    Column(Integer, name = '201212'),
                    Column(Integer, name = '201301'),
                    Column(Integer, name = '201302'),
                    Column(Integer, name = '201303'),
                    Column(Integer, name = '201304'),
                    Column(Integer, name = '201305'),
                    Column(Integer, name = '201306'),
                    Column(Integer, name = '201307'),
                    Column(Integer, name = '201308'),
                    Column(Integer, name = '201309'),
                    Column(Integer, name = '201310'),
                    Column(Integer, name = '201311'),
                    Column(Integer, name = '201312'),
                    Column(Integer, name = '201401'),
                    Column(Integer, name = '201402'),
                    Column(Integer, name = '201403'),
                    Column(Integer, name = '201404'),
                    Column(Integer, name = '201405'),
                    Column(Integer, name = '201406'),
                    Column(Integer, name = '201407'),
                    Column(Integer, name = '201408'),
                    Column(Integer, name = '201409'),
                    Column(Integer, name = '201410'),
                    Column(Integer, name = '201411'),
                    Column(Integer, name = '201412'),
                    Column(Integer, name = '201501'),
                    Column(Integer, name = '201502'),
                    Column(Integer, name = '201503'),
                    Column(Integer, name = '201504'),
                    Column(Integer, name = '201505'),
                    Column(Integer, name = '201506'),
                    Column(Integer, name = '201507'),
                    Column(Integer, name = '201508'),
                    Column(Integer, name = '201509'),
                    Column(Integer, name = '201510'),
                    Column(Integer, name = '201511'),
                    Column(Integer, name = '201512'),
                    Column(Integer, name = '201601'),
                    Column(Integer, name = '201602'),
                    Column(Integer, name = '201603'),
                    Column(Integer, name = '201604'),
                    Column(Integer, name = '201605'),
                    Column(Integer, name = '201606'),
                    Column(Integer, name = '201607'),
                    Column(Integer, name = '201608'),
                    Column(Integer, name = '201609'),
                    Column(Integer, name = '201610'),
                    Column(Integer, name = '201611'),
                    Column(Integer, name = '201612'),
                    Column(Integer, name = '201701'),
                    Column(Integer, name = '201702'),
                    Column(Integer, name = '201703'),
                    Column(Integer, name = '201704'),
                    Column(Integer, name = '201705'),
                    Column(Integer, name = '201706'),
                    Column(Integer, name = '201707'),
                    Column(Integer, name = '201708'),
                    Column(Integer, name = '201709'),
                    Column(Integer, name = '201710'),
                    Column(Integer, name = '201711'),
                    Column(Integer, name = '201712'),
                    Column(Integer, name = '201801'),
                    Column(Integer, name = '201802'),
                    Column(Integer, name = '201803'),
                    Column(Integer, name = '201804'),
                    Column(Integer, name = '201805'),
                    Column(Integer, name = '201806'),
                    Column(Integer, name = '201807'),
                    Column(Integer, name = '201808'),
                    Column(Integer, name = '201809'),
                    Column(Integer, name = '201810'),
                    Column(Integer, name = '201811'),
                    Column(Integer, name = '201812'),
                    Column(Integer, name = '201901'),
                    Column(Integer, name = '201902'),
                    Column(Integer, name = '201903'),
                    Column(Integer, name = '201904'),
                    Column(Integer, name = '201905'),
                    Column(Integer, name = '201906'),
                    Column(Integer, name = '201907'),
                    Column(Integer, name = '201908'),
                    Column(Integer, name = '201909'),
                    Column(Integer, name = '201910'),
                    Column(Integer, name = '201911'),
                    Column(Integer, name = '201912'),
                    Column(Integer, name = '202001'),
                    Column(Integer, name = '202002'),
                    Column(Integer, name = '202003'),
                    Column(Integer, name = '202004'),
                    Column(Integer, name = '202005'),
                    Column(Integer, name = '202006'),
                    Column(Integer, name = '202007'),
                    Column(Integer, name = '202008'),
                    Column(Integer, name = '202009'),
                    Column(Integer, name = '202010'),
                    Column(Integer, name = '202011'),
                    Column(Integer, name = '202012'),
                    Column(Integer, name = '202101'),
                    Column(Integer, name = '202102'),
                    Column(Integer, name = '202103'),
                    Column(Integer, name = '202104'),
                    Column(Integer, name = '202105'),
                    Column(Integer, name = '202106'),
                    Column(Integer, name = '202107'),
                    Column(Integer, name = '202108'),
                    Column(Integer, name = '202109'),
                    Column(Integer, name = '202110'),
                    Column(Integer, name = '202111'),
                    Column(Integer, name = '202112'),
                    Column(Integer, name = '202201'),
                    Column(Integer, name = '202202'),
                    Column(Integer, name = '202203'),
                    Column(Integer, name = '202204'),
                    Column(Integer, name = '202205'),
                    Column(Integer, name = '202206'),
                    MergeTree(order_by=('index')),
                    schema=engine_conn['database'])

    gdp_table = Table(gdp_table_name, MetaData(),
               Column(Integer, name = 'index', primary_key=True),
               Column(String, name = 'ITL1 Region'),
               Column(String, name = 'LA code'),
               Column(String, name = 'LA name'),
               Column(Integer, name = '1998'),
               Column(Integer, name = '1999'),
               Column(Integer, name = '2000'),
               Column(Integer, name = '2001'),
               Column(Integer, name = '2002'),
               Column(Integer, name = '2003'),
               Column(Integer, name = '2004'),
               Column(Integer, name = '2005'),
               Column(Integer, name = '2006'),
               Column(Integer, name = '2007'),
               Column(Integer, name = '2008'),
               Column(Integer, name = '2009'),
               Column(Integer, name = '2010'),
               Column(Integer, name = '2011'),
               Column(Integer, name = '2012'),
               Column(Integer, name = '2013'),
               Column(Integer, name = '2014'),
               Column(Integer, name = '2015'),
               Column(Integer, name = '2016'),
               Column(Integer, name = '2017'),
               Column(Integer, name = '2018'),
               Column(Integer, name = '2019'),
               Column(Integer, name = '2020'),
               Column(Integer, name = '2021'),
               Column(Integer, name = '2022'),
               MergeTree(order_by=('index')),
               schema=engine_conn['database'])

    population_table = Table(population_table_name, MetaData(),
               Column(Integer, name = 'index', primary_key=True),
               Column(String, name = 'ITL1 Region'),
               Column(String, name = 'LA code'),
               Column(String, name = 'LA name'),
               Column(Integer, name = '1998'),
               Column(Integer, name = '1999'),
               Column(Integer, name = '2000'),
               Column(Integer, name = '2001'),
               Column(Integer, name = '2002'),
               Column(Integer, name = '2003'),
               Column(Integer, name = '2004'),
               Column(Integer, name = '2005'),
               Column(Integer, name = '2006'),
               Column(Integer, name = '2007'),
               Column(Integer, name = '2008'),
               Column(Integer, name = '2009'),
               Column(Integer, name = '2010'),
               Column(Integer, name = '2011'),
               Column(Integer, name = '2012'),
               Column(Integer, name = '2013'),
               Column(Integer, name = '2014'),
               Column(Integer, name = '2015'),
               Column(Integer, name = '2016'),
               Column(Integer, name = '2017'),
               Column(Integer, name = '2018'),
               Column(Integer, name = '2019'),
               Column(Integer, name = '2020'),
               Column(Integer, name = '2021'),
               Column(Integer, name = '2022'),
               MergeTree(order_by=('index')),
               schema=engine_conn['database'])

    traffic_flow_table = Table(traffic_flow_table_name, MetaData(),
                           Column(Integer, name = 'index', primary_key=True),
                           Column(String, name = 'LA Code'),
                           Column(String, name = 'Local Authority'),
                           Column(Integer, name = '1993'),
                           Column(Integer, name = '1994'),
                           Column(Integer, name = '1995'),
                           Column(Integer, name = '1996'),
                           Column(Integer, name = '1997'),
                           Column(Integer, name = '1998'),
                           Column(Integer, name = '1999'),
                           Column(Integer, name = '2000'),
                           Column(Integer, name = '2001'),
                           Column(Integer, name = '2002'),
                           Column(Integer, name = '2003'),
                           Column(Integer, name = '2004'),
                           Column(Integer, name = '2005'),
                           Column(Integer, name = '2006'),
                           Column(Integer, name = '2007'),
                           Column(Integer, name = '2008'),
                           Column(Integer, name = '2009'),
                           Column(Integer, name = '2010'),
                           Column(Integer, name = '2011'),
                           Column(Integer, name = '2012'),
                           Column(Integer, name = '2013'),
                           Column(Integer, name = '2014'),
                           Column(Integer, name = '2015'),
                           Column(Integer, name = '2016'),
                           Column(Integer, name = '2017'),
                           Column(Integer, name = '2018'),
                           Column(Integer, name = '2019'),
                           Column(Integer, name = '2020'),
                           Column(Integer, name = '2021'),
                           Column(Integer, name = '2022'),
                           MergeTree(order_by=('index')),
                           schema=engine_conn['database'])

    # 'with' is being used as a context manager, so once the transaction is complete, the connection is closed automatically
    with Session(engine) as session:
        road_table.create(session.connection(), checkfirst = True)
        traffic_flow_table.create(session.connection(), checkfirst = True)
        population_table.create(session.connection(), checkfirst = True)
        gdp_table.create(session.connection(), checkfirst = True)
        crime_table.create(session.connection(), checkfirst = True)
        # 'checkfirst' will make the program check to see if the table exists already
        # if it does exist, it will not attempt to create a new table. Otherwise, it will
        session.commit() # completes the transaction

def ingest_road_data(directory_path : str) -> DataFrame:
    """
    A custom function that attempts to ingest specific Greater London road data from files in
    a given directory, and then returns the combined data. Column names are adjusted to ensure
    the final result is consistent and uniform.

    Returns
    -------
        The combined and slightly transformed data in the form of a Dask DataFrame.
    """    
    # a custom dictionary containing column replacement names
    # this will be used to make the final combined dataframe columns uniform and consistent
    replace_col_names_dict = {
        'AREFNO':'Accident Ref',
        'Accident Ref.':'Accident Ref',
        'Boro':'Borough Number',
        'Road No. 1':'First Road Number',
        'Road No. 2':'Second Road Number',
        'Road No 2':'Second Road Number',
        'Road Class 1':'First Road Class',
        'Road Class 2':'Second Road Class',
        'No. of Casualties in Acc.':'Casualty Count',
        '_Casualty Count':'Casualty Count',
        'No. of Vehicles in Acc.':'Vehicle Count',
        '_Vehicle Count':'Vehicle Count',
        'Accident Date':'Collision Date',
        '_Collision Date':'Collision Date',
        'Weather':'Weather Details',
        'Light Conditions (Banded)':'Light Conditions',
        'Road Surface':'Road Surface Condition',
        'Ped. Crossing Decoded':'Pedestrian Crossing Facilities',
        'C/W Hazard':'Carriageway Hazards',
        'Special Conditions':'Special Conditions at Site',
        'Day Name':'Day',
        'Borough Name':'Borough',
        '_Collision Severity':'Accident Severity',
        'APOLICER_DECODED':'Place Collision Reported',
        'Collision Location':'Collision Location Details',
        '_Attendant Count':'Attendant Count',
        'Location':'Collision Location Details',
        'Highway':'Highway Authority'
    }

    dd_frames = []
    # array list to temporarily store the TfL files while they are being ingested
    # these file are then concatenated together into one large DataFrame

    for file in listdir(directory_path):
        ddf : DataFrame = from_pandas(read_csv(directory_path+'/'+file, encoding = 'cp1252'), npartitions=16)
        #cp1252 is the standard encoding (codec) for Western Europe
        ddf = ddf.rename(columns=replace_col_names_dict) 
        # rename columns of DataFrames so data is combined properly
        dd_frames.append(ddf)
    
    return concat(dd_frames, interleave_partitions=True, axis = 0)
    # 'interleave_partitions=True' concatenates DataFrame ignoring its order. The order doesn't matter in this case, as it will be reorded later anyway
    #'axis = 0' ensures the dataframes will be concatenated row-wise

def ingest_data(directory_path : str) -> DataFrame:
    """
    A function that attempts to retreve CSV data from a specified file, then return this data in a Dask DataFrame.

    Returns
    -------
        A Dask DataFrame with the CSV data.
    """
    ddf : DataFrame = from_pandas(read_csv(directory_path), npartitions=16)
    return ddf

def preprocess_road_data(ddf : DataFrame) -> DataFrame:
    """
    Performs basic and required transformations to the road data so that it can be stored properly.
    """

    # remove basic outliers
    # every entry should have a Borough name with an associated accident severity for future plans
    ddf = ddf[ddf['Borough'].notnull() & ddf['Accident Severity'].notnull()]

    # a pre-defined list of column names to be replaced
    columns_data_to_replace = ['Speed Limit', 'Accident Severity', 'Junction Detail',
                      'Junction Control', 'Pedestrian Crossing Facilities', 'First Road Class', 'Second Road Class', 'Road Type',
                      'Light Conditions', 'Weather Details', 'Road Surface Condition', 'Special Conditions at Site', 'Carriageway Hazards',
                     'Highway Authority', 'Collision Location Details', 'Place Collision Reported']

    # the below code removed numbers such as '3' from '3 30 MPH', making the result '30 MPH'
    for col in columns_data_to_replace:
        ddf[col] = ddf[col].str.replace(r'^\d\s*|^-\d\s*', '', regex=True)
        # use regular expression (regex) to select prior number to actual data and remove them
        # '^' means at the beginning of the string, '\d' means if a number is present
        # '\s' means any whitespace character, '*' means however many - so however many whitespaces
        # '|' is the equivalent of OR - so if the string matches either expression, the former for positive numbers, the latter for negative numbers

    # some of the dates are in slightly different formats
    ddf['Collision Date'] = ddf['Collision Date'].str.replace('/', '-')
    ddf['Collision Date'] = ddf['Collision Date'].str.replace(" 00:00", '')

    # the below is needed to ensure the dataframe types are consistent with its intended table schema
    ddf['Collision Date'] = to_datetime(ddf['Collision Date'], format = "mixed", dayfirst=True)
    ddf['Attendant Count'] = ddf['Attendant Count'].astype(float) # must be converted to float due to containing 'NaN' values
        # 'NaN' values are classed as floating point values
    ddf['Casualty Count'] = ddf['Casualty Count'].astype(int)
    ddf['Vehicle Count'] = ddf['Vehicle Count'].astype(int)
    ddf['Borough Number'] = ddf['Borough Number'].astype(int)

    return ddf

def upload_ingested_data(ddf : DataFrame, engine_conn : dict, table_name : str) -> None:
    """
    Upload a Dask DataFrame to a ClickHouse server.
    """
    # it is necessary to alter the engine connection dictionary slightly to suit the pandahouse package
    connection = {
        "database":engine_conn['database'],
        "host":'http://{host}:{port}'.format(host=engine_conn['host'], port=engine_conn['port']),
        "user":engine_conn['user'],
        "password":engine_conn['password']
    }
    ph.to_clickhouse(ddf.compute(), table_name, index = True, connection=connection)
