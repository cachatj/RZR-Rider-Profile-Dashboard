# %% [markdown]
# Please review the python script attached. it was developed in JupyterNotebook, and functions very close to expected.
# I would like to ensure Idempotency of these processing components, namely I would like to convert this into a BigQuery UDF / Cloud Function that can take in raw_data (timestamp_unix, device_id, latitude, longitude) and execute the functions.
# 

# %%
# imports
import os
import glob

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
import movingpandas as mpd

from datetime import date
from datetime import datetime

from tzfpy import get_tz
from zoneinfo import ZoneInfo

from gpxcsv import gpxtolist
import h3
import xml.etree.ElementTree as ET

import folium
from IPython.display import display, HTML
import holoviews as hv
from holoviews import dim, opts
import hvplot.pandas
import plotly.express as px
from keplergl import KeplerGl

import skmob
from skmob.preprocessing import detection, clustering, compression, filtering
from skmob.measures.individual import home_location
from skmob.tessellation import tilers

from itables import init_notebook_mode, show
from dataprep.eda import create_report
from dataprep.eda import plot
from dataprep.eda import plot_diff

from google.cloud import bigquery
from google.api_core import exceptions

import warnings
warnings.filterwarnings('ignore')

# notebook extensions
init_notebook_mode(all_interactive=False)
hv.extension()

# %%
# gcp-project STAGE 1 

# generate datetime_local, filter & compress GPS data, detect stay_locations

from stage1_mobility.compression import argcompress_trajectory
from stage1_mobility.detection import (
    detect_stay_locations,
    extract_stay_locations,
)
from stage1_mobility.filtering import argfilter_trajectory
from stage1_mobility.sorting import (
    argsort_trajectory_by_time_ascending,
)

def array_to_str(array: np.array) -> str:
    """
    Converts a numpy array to a string.

    To save on storage, superfluous whitespaces and decimal points are removed.

    :param array: an input array
    :return: string representation of stay location array
    """
    return (
        str(array.tolist())
        .replace(" ", "")  # get rid of whitespaces to reduce length
        .replace(".0,", ",")  # convert from float to int
        .replace(".0]", "]")  # convert from float to int
        .replace("nan", "NaN")  # convert into a json-readable format
        .replace("'", '"')  # convert single-quotation marks to double quotation marks
    )
    
# ... convert timestamp_utc to local timezone
def convert_timestamps_to_datetimes_local(lat_lon_ts: np.array) -> list[str]:
    """
    Converts timestamps to local time zone based on lat/lon location.

    :param lat_lon_ts: an (n, 3) numpy array with (lat, lon) location, its timestamp:
    - 0th column is latitude,
    - 1st column is longitude,
    - 2nd column is observation unix timestamp,
    where n is the number of observations.

    :return: list of timestamps in local timezone
    """
    return [convert_timestamp_to_datetime_local(*lat_lon_ts_row) for lat_lon_ts_row in lat_lon_ts]


# ... convert timestamps to local based on lat/lon location
def convert_timestamp_to_datetime_local(lat: float, lon: float, ts: float) -> str:
    """
    Converts timestamps to local based on lat/lon location.

    :param lat: latitude
    :param lon: longitude
    :param ts: observation unix timestamp

    :return: timestamp in local timezone; 0 if timestamp couldn't be converted between timezones
    """
    date_utc = datetime.fromtimestamp(ts).astimezone(ZoneInfo("UTC"))

    new_tz = ZoneInfo(get_tz(lon, lat))
    return date_utc.astimezone(new_tz).strftime("%Y-%m-%d %H:%M:%S")

# %%
# load files

def rename_latitude_longitude(df):
    longitude_cols = ['longitude', 'lon', 'long', 'lng']
    latitude_cols = ['latitude', 'lat', 'Lat']

    for col in latitude_cols:
        if col in df.columns:
            df.rename(columns={col: 'latitude'}, inplace=True)

    for col in longitude_cols:
        if col in df.columns:
            df.rename(columns={col: 'longitude'}, inplace=True)

    # Print message only if no renaming occurred
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        print("Warning: Could not find longitude/latitude columns in the DataFrame.")

    return df

#-------------------------------------------LOAD GPX, CSV, BIGQUERY or GCS DATA-------------------------------------------------------
def gpx_rzr_load(gpx_file):
    gpx_rzr_df = pd.DataFrame(gpxtolist(gpx_file))
    gpx_rzr_df = rename_latitude_longitude(gpx_rzr_df) 
    gpx_rzr_df.rename(columns={'rcid': 'device_id', 'type': 'trajectory_id_part1', 'name': 'trajectory_id_part2'}, inplace=True)
    gpx_rzr_df['source'] = os.path.basename(gpx_file)
    gpx_rzr_df['timestamp_utc'] = pd.to_datetime(gpx_rzr_df['time'])
    gpx_rzr_df['timestamp_utc'] = gpx_rzr_df['timestamp_utc'].dt.tz_convert('UTC')
    gpx_rzr_df['timestamp_unix'] = gpx_rzr_df['timestamp_utc'].astype('int64') // 10**9
    drop_cols = ["appSku", "id", "appVersion", "time", "color", "Color", "totalDistanceInMeters", "totalDurationInSeconds"]

    # optional handle Polaris-specific columns
    polaris_cols = ["averageSpeed", "maxSpeed", "stoppedTimeInSeconds"]
    drop_cols.extend(col for col in polaris_cols if col in gpx_rzr_df.columns)
    gpx_rzr_df.drop(drop_cols, axis=1, inplace=True)
    gpx_rzr_df['latitude'] = round(gpx_rzr_df['latitude'], 5)
    gpx_rzr_df['longitude'] = round(gpx_rzr_df['longitude'], 5)
    gpx_rzr_df = gpx_rzr_df.sort_values('timestamp_unix', ascending=True)
    gpx_rzr_df['device_id'] = DEVICE_ID
    return gpx_rzr_df[['device_id', 'latitude', 'longitude', 'timestamp_utc', 'timestamp_unix' , 'source']]


def gpx_ios_load(gpx_file):
    xml_data = open(gpx_file, 'r').read()
    root = ET.fromstring(xml_data)

    namespace = '{http://www.topografix.com/GPX/1/1}'
    trkpt_elements = root.findall(f'.//{namespace}trkpt')

    data = []
    for trkpt in trkpt_elements:
        latitude = float(trkpt.attrib['lat'])
        longitude = float(trkpt.attrib['lon'])
        timestamp_utc = pd.to_datetime(trkpt.find(f'{namespace}time').text)
        timestamp_unix = int(timestamp_utc.timestamp())

        data.append({
            'latitude': latitude,
            'longitude': longitude,
            'timestamp_utc': timestamp_utc,
            'timestamp_unix': timestamp_unix
        })

    gpx_ios_df = pd.DataFrame(data)
    gpx_ios_df['source'] = os.path.basename(gpx_file)
    gpx_ios_df['device_id'] = DEVICE_ID
    gpx_ios_df['latitude'] = round(gpx_ios_df['latitude'], 5)
    gpx_ios_df['longitude'] = round(gpx_ios_df['longitude'], 5)
    gpx_ios_df = gpx_ios_df.sort_values('timestamp_unix', ascending=True)
    return gpx_ios_df[['device_id', 'latitude', 'longitude', 'timestamp_utc', 'timestamp_unix' , 'source']]

def csv_load(csv_file):
    csv_df = pd.read_csv(csv_file)
    csv_df = rename_latitude_longitude(csv_df)
    csv_df['timestamp_utc'] = pd.to_datetime(csv_df['timestamp_utc'])
    csv_df['timestamp_unix'] = pd.to_datetime(csv_df['timestamp_utc']).astype('int64') // 10**9
    csv_df['source'] = os.path.basename(csv_file)
    csv_df['latitude'] = round(csv_df['latitude'], 5)
    csv_df['longitude'] = round(csv_df['longitude'], 5)
    csv_df = csv_df.sort_values('timestamp_unix', ascending=True)
    if 'device_id' not in csv_df.columns:
        csv_df['device_id'] = DEVICE_ID
    return csv_df[['device_id', 'latitude', 'longitude', 'timestamp_utc', 'timestamp_unix' , 'source']]

def bigquery_load(query):
    # ...  (logic to load data from BigQuery)
    # ...
    # return bigquery_df[['device_id', 'latitude', 'longitude', 'timestamp_utc', 'timestamp_unix' , 'source']]
    pass

def gcs_bucket_load(bucket_name, file_path):
    # ...  (logic to load data from GCS bucket)
    # ...
    # return gcs_bucket_df[['device_id', 'latitude', 'longitude', 'timestamp_utc', 'timestamp_unix' , 'source']]
    pass

def ingest_raw_data(filepath, file_type):
    if file_type == 'gpx_rzr':
        df = gpx_rzr_load(filepath)
    elif file_type == 'gpx_ios':
        df = gpx_ios_load(filepath)
    elif file_type == 'csv':
        df = csv_load(filepath)
    # elif file_type == 'bigquery':
    #     df = bigquery_load(query)  # Replace with actual BigQuery logic
    # elif file_type == 'gcs':
    #     df = gcs_bucket_load(bucket_name, file_path)  # Replace with actual GCS logic
    
    else:
        raise ValueError("Unsupported file type:", file_type)

    if df is not None and not df.empty:
        location_data = df.copy()
    else:
        location_data = pd.read_csv(filepath) #fallback to CSV

    return location_data

# %%
def stage1_preprocess_all_in_one(raw_df):
     # to_numpy array
    lat_lon_df = raw_df[['latitude', 'longitude', 'timestamp_unix']]
    num_attrs_df = raw_df[[]]
    str_attrs_df = raw_df[['device_id', 'timestamp_utc', 'source']]

    lat_lon_ts = lat_lon_df.to_numpy()
    num_attrs = num_attrs_df.to_numpy()
    str_attrs = str_attrs_df.to_numpy()

    # Sort trajectory
    indices_sorted = argsort_trajectory_by_time_ascending(lat_lon_ts)
    lat_lon_ts_index_sorted = np.hstack((lat_lon_ts[indices_sorted], np.arange(0, lat_lon_ts.shape[0]).reshape(-1, 1)))
    num_attrs_sorted = num_attrs[indices_sorted, :]
    str_attrs_sorted = str_attrs[indices_sorted, :]

    # Filtering > 400 km/h
    indices_filtered = argfilter_trajectory(lat_lon_ts_index_sorted, speed_limit_in_kms=400 / 3600).astype(int)
    lat_lon_ts_filtered = lat_lon_ts_index_sorted[indices_filtered, :3]
    num_attrs_filtered = num_attrs_sorted[indices_filtered, :]
    str_attrs_filtered = str_attrs_sorted[indices_filtered, :]

    # Compression (0.01 to 0.05 km range) for location_data
    location_data_indices_compressed = argcompress_trajectory(lat_lon_ts_filtered, compression_range_in_km=0.01).astype(int)[0]

    location_data_lat_lon_ts_compressed = lat_lon_ts_filtered[location_data_indices_compressed, :]
    location_data_num_attrs_compressed = num_attrs_filtered[location_data_indices_compressed, :]
    location_data_str_attrs_compressed = str_attrs_filtered[location_data_indices_compressed, :]

    # convert compressed numpy arrays to pandas dataframe
    datetime_local = convert_timestamps_to_datetimes_local(location_data_lat_lon_ts_compressed)

    compressed_lat_lon_df = pd.DataFrame(location_data_lat_lon_ts_compressed)
    datetime_local_df = pd.DataFrame(datetime_local)
    compressed_num_data_df = pd.DataFrame(location_data_num_attrs_compressed)
    compressed_str_data_df = pd.DataFrame(location_data_str_attrs_compressed)

    compressed_lat_lon_df.columns = ['latitude', 'longitude', 'timestamp_unix']
    datetime_local_df.columns = ['datetime_local']
    compressed_num_data_df.columns = []
    compressed_str_data_df.columns = ['device_id', 'timestamp_utc', 'source']
    compressed_location_data = pd.concat([compressed_lat_lon_df, datetime_local_df, compressed_num_data_df, compressed_str_data_df], axis=1)

    compressed_location_data['timestamp_utc'] = compressed_location_data['timestamp_unix'].apply(lambda x: datetime.fromtimestamp(x).astimezone(ZoneInfo("UTC")))
    compressed_location_data['datetime_local'] = pd.to_datetime(compressed_location_data['datetime_local'])
    compressed_location_data['trajectory_id'] = compressed_location_data.apply(lambda x: f"{x['datetime_local'].strftime('%Y%m%d')}#{x['device_id']}", axis=1)
    compressed_location_data['trajectory_id'] = compressed_location_data.apply(lambda x: f"#{x['device_id']}#{x['datetime_local'].strftime('%Y%m%d')}", axis=1)
    compressed_location_data['timezone'] = compressed_location_data.apply(lambda row: get_tz(row['longitude'], row['latitude']), axis=1)
    compressed_location_data['last_modified_on'] = TODAY

    compressed_location_data = compressed_location_data[['device_id', 'latitude', 'longitude', 'datetime_local', 'timestamp_utc', 'timestamp_unix', 'timezone', 'trajectory_id', 'source', 'last_modified_on']]
    compressed_location_data = compressed_location_data.sort_values('datetime_local', ascending=True)
    
    # stay_locations - compression & extraction
    stay_location_indices_compressed = argcompress_trajectory(lat_lon_ts_filtered, compression_range_in_km=0.001).astype(int)[0]
    stay_location_lat_lon_ts_compressed = lat_lon_ts_filtered[stay_location_indices_compressed, :]
    stay_location_num_attrs_compressed = num_attrs_filtered[stay_location_indices_compressed, :]
    stay_location_str_attrs_compressed = str_attrs_filtered[stay_location_indices_compressed, :]

    # detect & extract stay_locations_data
    start_stop_indices = detect_stay_locations(
        stay_location_lat_lon_ts_compressed, stay_range_in_km=0.05, stay_duration_in_s=1200
    ).astype(int)[0]

    stay_locations = extract_stay_locations(
        stay_location_lat_lon_ts_compressed, stay_location_num_attrs_compressed, stay_location_str_attrs_compressed, start_stop_indices
    )

    stay_locations_data = pd.DataFrame(stay_locations)
    stay_locations_data.columns = ['stay_start_unix', 'stay_latitude_ctr', 'stay_longitude_ctr', 'stay_end_unix', 'stay_num_points', 'stay_max_diameter', 'speed', 'altitude', 'horizontal_accuracy', 'vertical_accuracy', 'ipv4', 'ipv6', 'bssids', 'ssids']
    stay_locations_data.drop(['horizontal_accuracy', 'vertical_accuracy', 'ipv4', 'ipv6', 'bssids', 'ssids'], axis=1, inplace=True)
    stay_locations_data['device_id'] = DEVICE_ID
    stay_locations_data['last_modified_on'] = TODAY
    
    stay_locations_data = stay_locations_data[['device_id', 'stay_latitude_ctr', 'stay_longitude_ctr', 'stay_start_unix', 'stay_end_unix',
                                               'stay_num_points', 'stay_max_diameter','last_modified_on']]
    stay_locations_data = stay_locations_data.sort_values('stay_start_unix', ascending=True)
    
    return compressed_location_data, stay_locations_data

# %%
# feature engineering

def get_period_of_day(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 6:
        return 'early_am'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 13:
        return 'lunch'
    elif 13 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'
    
def determine_move_activity(row, stay_locations):
    device_id = row['device_id']
    timestamp_unix = row['timestamp_unix']
    
    device_stay_locations = stay_locations[stay_locations['device_id'] == device_id]
    
    # Check if timestamp_unix is between stay_start_unix and stay_end_unix
    is_stay = ((device_stay_locations['stay_start_unix'] <= timestamp_unix) & 
               (device_stay_locations['stay_end_unix'] >= timestamp_unix)).any()
    
    return 'stay' if is_stay else 'trip'

def determine_stay_activity(row, stay_locations):
    device_id = row['device_id']
    timestamp_unix = row['timestamp_unix']
    
    device_stay_locations = stay_locations[stay_locations['device_id'] == device_id]
    
    for _, stay_location in device_stay_locations.iterrows():
        stay_start_unix = stay_location['stay_start_unix']
        stay_end_unix = stay_location['stay_end_unix']
        stay_duration = stay_end_unix - stay_start_unix
        
        if stay_start_unix <= timestamp_unix <= stay_end_unix:
            if timestamp_unix <= stay_start_unix + 0.2 * stay_duration:
                return 'arriving'
            elif timestamp_unix >= stay_end_unix - 0.2 * stay_duration:
                return 'departing'
            else:
                return 'stopped'
    
    return None

def process_spatiotemporal_features(location_data, stay_locations_data):
    # Convert timestamp_utc to datetime_local for location_data
    lat_lon_df = location_data[['latitude', 'longitude', 'timestamp_unix']]
    lat_lon_ts = lat_lon_df.to_numpy()
    location_data['datetime_local'] = convert_timestamps_to_datetimes_local(lat_lon_ts)
    location_data['datetime_local'] = pd.to_datetime(location_data['datetime_local'])
    
    # Convert timestamp_utc to datetime_local for stay_locations_data
    stay_lat_lon_df = stay_locations_data[['stay_latitude_ctr', 'stay_longitude_ctr', 'stay_start_unix']]
    stay_lat_lon_ts = stay_lat_lon_df.to_numpy()
    stay_locations_data['datetime_local'] = convert_timestamps_to_datetimes_local(stay_lat_lon_ts)
    stay_locations_data['datetime_local'] = pd.to_datetime(stay_locations_data['datetime_local'])

    
    #process temporal features
    location_data['min_of_day'] = location_data['datetime_local'].dt.hour * 60 + location_data['datetime_local'].dt.minute
    location_data['hour_of_day'] = location_data['datetime_local'].dt.hour
    location_data['period_of_day'] = location_data['datetime_local'].apply(get_period_of_day)
    location_data['date'] = location_data['datetime_local'].dt.date
    location_data['time_local'] = location_data['datetime_local'].dt.strftime('%I:%M %p')
    location_data['day_of_month'] = location_data['datetime_local'].dt.day
    location_data['day_of_year'] = location_data['datetime_local'].dt.day_of_year
    location_data['day_of_week'] = location_data['datetime_local'].dt.day_of_week
    location_data['day_of_week_name'] = location_data['datetime_local'].dt.day_name()
    location_data['is_workday'] = location_data['datetime_local'].dt.day_of_week.between(0, 4)
    location_data['is_weekend'] = location_data['datetime_local'].dt.weekday >= 5
    location_data['is_business_hours'] = location_data['datetime_local'].dt.hour.between(9, 16)
    location_data['month_name'] = location_data['datetime_local'].dt.month_name()
    location_data['month'] = location_data['datetime_local'].dt.month
    location_data['quarter'] = location_data['datetime_local'].dt.quarter
    location_data['datetime_index'] = location_data['datetime_local'].dt.tz_localize(None, ambiguous="infer", nonexistent='raise')
    
    #spatial pre-processing features
    location_data['h3_lvl10_index'] = location_data.apply(lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], 10), axis=1)
    location_data['h3_lvl4_index'] = location_data.apply(lambda row: h3.geo_to_h3(row['latitude'], row['longitude'], 4), axis=1)
    location_data['altitude1_minOverlap'] = (location_data['day_of_month'] * 240) + (location_data['min_of_day'])
    location_data['altitude2_hourOverlap'] = (location_data['day_of_month'] * 240) + (location_data['hour_of_day'] * 10)
    
    min_datetime = location_data['datetime_local'].min()
    max_datetime = location_data['datetime_local'].max()
    num_intervals = location_data.shape[0]
    interval_size = (max_datetime - min_datetime) / (num_intervals - 1)
    location_data['altitude3_min2max'] = ((location_data['datetime_local'] - min_datetime) / interval_size).astype(int) * 100
    max_value = location_data['altitude3_min2max'].max()
    location_data['altitude3_min2max'] = location_data['altitude3_min2max'] * 10000 / max_value
    
    #move_activity & stay_activity features
    location_data['move_activity'] = location_data.apply(lambda row: determine_move_activity(row, stay_locations_data), axis=1)
    location_data['stay_activity'] = location_data.apply(lambda row: determine_stay_activity(row, stay_locations_data), axis=1)
    location_data['last_modified_on'] = TODAY
    
    #location_data.drop([], axis=1, inplace=True)
    
    #formatting output dataframe columns
    first_columns = ['device_id', 'latitude', 'longitude', 'datetime_local']
    last_columns = ['move_activity','stay_activity','altitude1_minOverlap','altitude2_hourOverlap','altitude3_min2max','h3_lvl10_index','h3_lvl4_index','timestamp_unix','timestamp_utc','timezone','trajectory_id','source','last_modified_on']
    other_columns = [col for col in location_data.columns if col not in (first_columns + last_columns)]
    ordered_output = first_columns + other_columns + last_columns
    
    location_features_output = location_data[ordered_output]
    location_features_output.sort_values(by='datetime_local', inplace=True, ascending=True)
    
    location_features_output.to_csv(ANALYTIC_LOCATION_FEATURES, index=False)
    #location_features_output.to_parquet()
    
    return location_features_output

#-------------------------------------PREPROCESSING STAY_LOCATION FEATURES----------------------------------------------------------------------------------
def process_stay_location_features(stay_locations_data, DEVICE_ID):
    # Check if stay_locations_data has any data
    if len(stay_locations_data) <= 0:
        return display(f"No stay locations detected in device_id: {DEVICE_ID}'s data")
    
    #convert timestamp_utc to datetime_local 
    if 'stay_start_datetime_local' not in stay_locations_data.columns and 'stay_end_datetime_local' not in stay_locations_data.columns:
        lat_lon_start_df = stay_locations_data[['stay_latitude_ctr', 'stay_longitude_ctr', 'stay_start_unix']]
        lat_lon_start_ts = lat_lon_start_df.to_numpy()
        lat_lon_end_df = stay_locations_data[['stay_latitude_ctr', 'stay_longitude_ctr','stay_end_unix']]
        lat_lon_end_ts = lat_lon_end_df.to_numpy()
        stay_locations_data['stay_start_datetime_local'] = convert_timestamps_to_datetimes_local(lat_lon_start_ts)
        stay_locations_data['stay_start_datetime_local'] = pd.to_datetime(stay_locations_data['stay_start_datetime_local'])
        stay_locations_data['stay_end_datetime_local'] = convert_timestamps_to_datetimes_local(lat_lon_end_ts)
        stay_locations_data['stay_end_datetime_local'] = pd.to_datetime(stay_locations_data['stay_end_datetime_local'])
    else:
        pass
    
    stay_locations_data['stay_duration_min'] = np.floor((stay_locations_data['stay_end_unix'] - stay_locations_data['stay_start_unix'])/60)
    stay_locations_data['stay_of_day'] = stay_locations_data['stay_start_unix'].rank(method='dense').astype(int)
    stay_locations_data['stay_id'] = stay_locations_data.apply(lambda x: f"#{x['device_id']}#{x['stay_start_datetime_local'].strftime('%Y%m%d')}#{x['stay_of_day']:04d}", axis=1)
    
    stay_locations_data['last_modified_on'] = TODAY
    
    #formatting the output dataframe
    stay_locations_features = stay_locations_data[['device_id','stay_latitude_ctr','stay_longitude_ctr','stay_start_unix','stay_start_datetime_local','stay_end_unix','stay_end_datetime_local',
                                           'stay_duration_min','stay_num_points','stay_max_diameter','stay_of_day','stay_id','last_modified_on']]
    
    return stay_locations_features

# %%
#stack all daily files & append into CSV & BQ tables
def stack_all_COMPRESSED_location_data(DATA_DIR, file_pattern, output_csv, output_parquet):
    files = glob.glob(os.path.join(DATA_DIR, file_pattern))

    merged_compressed_location_data = []
    processed_compressed_location_data = []  # List to store processed file names
    
    for file in files:
        if file.endswith('.csv') and file not in processed_compressed_location_data:
            df = pd.read_csv(file)
            merged_compressed_location_data.append(df)
            processed_compressed_location_data.append(file)  # Add the processed file name to the list
        
    all_compressed_data_for_export = pd.concat(merged_compressed_location_data, ignore_index=True)
    all_compressed_data_for_export = all_compressed_data_for_export.sort_values('timestamp_unix', ascending=True)

    # Check if the file exists and delete it
    if os.path.exists(output_csv):
        os.remove(output_csv)

    if os.path.exists(output_parquet):
        os.remove(output_parquet)
        
    all_compressed_data_for_export.to_csv(output_csv, index=False)
    all_compressed_data_for_export.to_parquet(output_parquet, index=False)

    # export to BigQuery - COMPRESSED_location_features
    client = bigquery.Client()
    dataset_id = "compressed_location_data"
    table_id = "JC_2024_rzr_compressed_location_data"
    table_ref = client.dataset(dataset_id).table(table_id)

    # Check if the table exists
    try:
        client.get_table(table_ref)
        table_exists = True
    except exceptions.NotFound:
        table_exists = False
        
    # Create the table if it doesn't exist
    if not table_exists:
        location_data_schema = [
            bigquery.SchemaField("device_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("latitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("longitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_utc", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("timezone", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("trajectory_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("last_modified_on", "TIMESTAMP", mode="NULLABLE")
        ]
        table = bigquery.Table(table_ref, schema=location_data_schema)
        table = client.create_table(table)
        print(f"Table {table.project}.{table.dataset_id}.{table.table_id} created.")

    if table_exists:
        # Delete rows with matching 'last_modified_on' values
        last_modified_on_values = pd.to_datetime(all_compressed_data_for_export['last_modified_on']).dt.date.unique().tolist()
        if last_modified_on_values:
            delete_query = f"""
                DELETE FROM `{dataset_id}.{table_id}`
                WHERE DATE(last_modified_on) IN UNNEST(@last_modified_on_values)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("last_modified_on_values", "DATE", last_modified_on_values)
                ]
            )
            delete_job = client.query(delete_query, job_config=job_config)
            delete_job.result()
            print(f"Deleted rows with 'last_modified_on' values: {last_modified_on_values}")
        else:
            print("No rows to delete based on 'last_modified_on' values.")

    # Set the job configuration to overwrite the table if it exists
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = False
    job_config.schema = [
            bigquery.SchemaField("device_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("latitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("longitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_utc", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("timezone", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("trajectory_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("last_modified_on", "TIMESTAMP", mode="NULLABLE")
        ]
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    job_config.skip_leading_rows = 1  # Skip the first row (header row)

    # Load the merged DataFrame into BigQuery
    with open(output_csv, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()

    print("Compressed Location Data loaded into BigQuery table: {}".format(table_ref.path))
    pass

def stack_all_ANALYTICAL_location_features(DATA_DIR, file_pattern, output_csv, output_parquet):
    files = glob.glob(os.path.join(DATA_DIR, file_pattern))

    merged_analytical_location_features = []
    analytical_features_processed = []

    for file in files:
        if file.endswith('.csv') and file not in analytical_features_processed:
            df = pd.read_csv(file)
            merged_analytical_location_features.append(df)
            analytical_features_processed.append(file)
        
    all_features_for_export = pd.concat(merged_analytical_location_features, ignore_index=True)
    all_features_for_export = all_features_for_export.sort_values('timestamp_unix', ascending=True)

    # Check if the file exists and delete it
    if os.path.exists(output_csv):
        os.remove(output_csv)

    if os.path.exists(output_parquet):
        os.remove(output_parquet)
        
    all_features_for_export.to_csv(output_csv, index=False)
    all_features_for_export.to_parquet(output_parquet, index=False)

    # export to BigQuery - ANALYTIC_location_features
    client = bigquery.Client()
    dataset_id = "analytic_location_features"
    table_id = "JC_2024_rzr_analytic_location_features"
    table_ref = client.dataset(dataset_id).table(table_id)

    # Check if the table exists
    try:
        client.get_table(table_ref)
        table_exists = True
    except exceptions.NotFound:
        table_exists = False
        
    # Create the table if it doesn't exist
    if not table_exists:
        location_features_schema = [
            bigquery.SchemaField("device_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("latitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("longitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("min_of_day", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("hour_of_day", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("period_of_day", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("date", "DATE", mode="NULLABLE"),
            bigquery.SchemaField("time_local", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("day_of_month", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_year", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_week", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_week_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("is_workday", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_weekend", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_business_hours", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("month_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("month", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("quarter", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("datetime_index", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("move_activity", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("stay_activity", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("altitude1_minOverlap", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("altitude2_hourOverlap", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("altitude3_min2max", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("h3_lvl10_index", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("h3_lvl4_index", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_utc", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("timezone", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("trajectory_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("last_modified_on", "TIMESTAMP", mode="NULLABLE")
        ]
        table = bigquery.Table(table_ref, schema=location_features_schema)
        table = client.create_table(table)
        print(f"Table {table.project}.{table.dataset_id}.{table.table_id} created.")

    if table_exists:
        # Delete rows with matching 'last_modified_on' values
        last_modified_on_values = pd.to_datetime(all_features_for_export['last_modified_on']).dt.date.unique().tolist()
        if last_modified_on_values:
            delete_query = f"""
                DELETE FROM `{dataset_id}.{table_id}`
                WHERE DATE(last_modified_on) IN UNNEST(@last_modified_on_values)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("last_modified_on_values", "DATE", last_modified_on_values)
                ]
            )
            delete_job = client.query(delete_query, job_config=job_config)
            delete_job.result()
            print(f"Deleted rows with 'last_modified_on' values: {last_modified_on_values}")
        else:
            print("No rows to delete based on 'last_modified_on' values.")

    # Set the job configuration to overwrite the table if it exists
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = False
    job_config.schema = [
            bigquery.SchemaField("device_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("latitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("longitude", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("min_of_day", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("hour_of_day", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("period_of_day", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("date", "DATE", mode="NULLABLE"),
            bigquery.SchemaField("time_local", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("day_of_month", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_year", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_week", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_week_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("is_workday", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_weekend", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_business_hours", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("month_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("month", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("quarter", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("datetime_index", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("move_activity", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("stay_activity", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("altitude1_minOverlap", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("altitude2_hourOverlap", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("altitude3_min2max", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("h3_lvl10_index", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("h3_lvl4_index", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("timestamp_utc", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("timezone", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("trajectory_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("last_modified_on", "TIMESTAMP", mode="NULLABLE")
        ]
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    job_config.skip_leading_rows = 1  # Skip the first row (header row)

    # Load the merged DataFrame into BigQuery
    with open(output_csv, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()

    print("Analytic Location Features loaded into BigQuery table: {}".format(table_ref.path))
    pass 

def stack_all_STAY_location_features(DATA_DIR, file_pattern, output_csv, output_parquet):
    files = glob.glob(os.path.join(DATA_DIR, file_pattern))

    merged_stay_location_features = []
    stay_processed_files = []

    for file in files:
        if file.endswith('.csv') and file not in stay_processed_files:
            df = pd.read_csv(file, dtype={
                'stay_duration_min': str,
                'stay_num_points': str,
                'stay_of_day': str
            })
            merged_stay_location_features.append(df)
            stay_processed_files.append(file)
        
    stay_data_for_export = pd.concat(merged_stay_location_features, ignore_index=True)
    stay_data_for_export = stay_data_for_export.sort_values('stay_start_unix', ascending=True)

    # Convert specific columns to integers
    stay_data_for_export['stay_duration_min'] = pd.to_numeric(stay_data_for_export['stay_duration_min'], errors='coerce')
    stay_data_for_export['stay_num_points'] = pd.to_numeric(stay_data_for_export['stay_num_points'], errors='coerce')
    stay_data_for_export['stay_of_day'] = pd.to_numeric(stay_data_for_export['stay_of_day'], errors='coerce')
    
    # Check if the file exists and delete it
    if os.path.exists(output_csv):
        os.remove(output_csv)

    if os.path.exists(output_parquet):
        os.remove(output_parquet)
        
    stay_data_for_export.to_csv(output_csv, index=False)
    stay_data_for_export.to_parquet(output_parquet, index=False)

    # export to BigQuery - STAY_location_features
    client = bigquery.Client()
    dataset_id = "stay_locations_features"
    table_id = "JC_2024_rzr_stay_locations_features"
    table_ref = client.dataset(dataset_id).table(table_id)

    # Check if the table exists
    try:
        client.get_table(table_ref)
        table_exists = True
    except exceptions.NotFound:
        table_exists = False
        
    # Create the table if it doesn't exist
    if not table_exists:
        stay_locations_schema = [
            bigquery.SchemaField("device_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("stay_latitude_ctr", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_longitude_ctr", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_start_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_start_datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("stay_end_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_end_datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("stay_duration_min", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stay_num_points", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stay_max_diameter", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_of_day", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stay_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("last_modified_on", "TIMESTAMP", mode="NULLABLE")
        ]
        table = bigquery.Table(table_ref, schema=stay_locations_schema)
        table = client.create_table(table)
        print(f"Table {table.project}.{table.dataset_id}.{table.table_id} created.")

    if table_exists:
        # Delete rows with matching 'last_modified_on' values
        last_modified_on_values = pd.to_datetime(stay_data_for_export['last_modified_on']).dt.date.unique().tolist()
        if last_modified_on_values:
            delete_query = f"""
                DELETE FROM `{dataset_id}.{table_id}`
                WHERE DATE(last_modified_on) IN UNNEST(@last_modified_on_values)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("last_modified_on_values", "DATE", last_modified_on_values)
                ]
            )
            delete_job = client.query(delete_query, job_config=job_config)
            delete_job.result()
            print(f"Deleted rows with 'last_modified_on' values: {last_modified_on_values}")
        else:
            print("No rows to delete based on 'last_modified_on' values.")

    # Set the job configuration to overwrite the table if it exists
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = False
    job_config.schema = [
            bigquery.SchemaField("device_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("stay_latitude_ctr", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_longitude_ctr", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_start_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_start_datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("stay_end_unix", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_end_datetime_local", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("stay_duration_min", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stay_num_points", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stay_max_diameter", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("stay_of_day", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("stay_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("last_modified_on", "TIMESTAMP", mode="NULLABLE")
        ]
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
    job_config.skip_leading_rows = 1  # Skip the first row (header row)

    # Load the merged DataFrame into BigQuery
    with open(output_csv, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()

    print("Data loaded into BigQuery table: {}".format(table_ref.path))
    pass

# %% [markdown]
# ---
# 

# %% [markdown]
# ##### [0] Set Source of RAW_location_data objects
# 

# %%
# Location of RAW_location_data objects 
DATA_DIR = "/Users/jonathancachat/..../trajectory_data_processing_cleaned2analytic/data/gpx_rzr/"

# Set the combined filename prefix
COMBINED_FILENAME = "JC_2024_rzr"

TODAY = datetime.now()

# %% [markdown]
# ##### [1] RAW_location_data --> COMPRESSED_location_data & STAY_location_data (Cleaned Phase)
# 

# %%
# RAW_location_data --> COMPRESSED_location_data & STAY_location_data (Cleaned Phase)

# Get a list of all GPX files in the directory
gpx_files = glob.glob(os.path.join(DATA_DIR, "*.gpx"))

# Process each GPX file for stage1_preprocess_all_in_one
for gpx_file in gpx_files:
    # Extract the device ID from the GPX file name
    DEVICE_ID = os.path.splitext(os.path.basename(gpx_file))[0]
    device_id = DEVICE_ID
    
    COMPRESSED_LOCATION_DATA = DATA_DIR+DEVICE_ID+'-COMPRESSED_location_data.csv'
    STAY_LOCATIONS_DATA = DATA_DIR+DEVICE_ID+'-STAY_locations_data.csv'
    
    # Perform the data processing steps for stage1_preprocess_all_in_one
    raw_df = ingest_raw_data(gpx_file, 'gpx_rzr')
    
    compressed_location_data, stay_locations_data = stage1_preprocess_all_in_one(raw_df)
    compressed_location_data.to_csv(os.path.join(DATA_DIR, f"{device_id}-COMPRESSED_location_data.csv"), index=False)
    #stay_locations_data.to_csv(os.path.join(DATA_DIR, f"{device_id}-STAY_locations_data.csv"), index=False)

# %% [markdown]
# ##### [2] COMPRESSED_location_data & STAY_location_data --> ANALYTICAL_location_features & STAY_location_features
# 

# %%
# IN = compressed_location_data & stay_locations_data for features engineering

# Get a list of all GPX files in the directory
gpx_files = glob.glob(os.path.join(DATA_DIR, "*.gpx"))

# Process each GPX file for process_spatiotemporal_features and process_stay_location_features
for gpx_file in gpx_files:
    # Extract the device ID from the GPX file name
    DEVICE_ID = os.path.splitext(os.path.basename(gpx_file))[0]
    device_id = DEVICE_ID
    
    COMPRESSED_LOCATION_DATA = DATA_DIR+DEVICE_ID+'-COMPRESSED_location_data.csv'
    STAY_LOCATIONS_DATA = DATA_DIR+DEVICE_ID+'-STAY_locations_data.csv'
    ANALYTIC_LOCATION_FEATURES = DATA_DIR+DEVICE_ID+'-ANALYTIC_location_features.csv'
    STAY_LOCATIONS_FEATURES = DATA_DIR+DEVICE_ID+'-STAY_locations_features.csv'
    
    # Load the compressed_location_data and stay_locations_data from the output files
    compressed_location_data = pd.read_csv(COMPRESSED_LOCATION_DATA)
    #stay_locations_data = pd.read_csv(STAY_LOCATIONS_DATA)
    
    # Perform process_spatiotemporal_features
    analytic_location_features = process_spatiotemporal_features(compressed_location_data, stay_locations_data)
    analytic_location_features.to_csv(os.path.join(DATA_DIR, f"{device_id}-ANALYTIC_location_features.csv"), index=False)
    
    # Perform process_stay_location_features
    stay_location_features = process_stay_location_features(stay_locations_data, DEVICE_ID)
    stay_location_features.to_csv(os.path.join(DATA_DIR, f"{device_id}-STAY_locations_features.csv"), index=False)

# %%
# Set the output file paths
ALL_COMPRESSED_LOCATION_DATA_CSV = os.path.join(DATA_DIR, f"{COMBINED_FILENAME}-COMPRESSED_location_data.csv")
ALL_STAY_LOCATIONS_FEATURES_CSV = os.path.join(DATA_DIR, f"{COMBINED_FILENAME}-STAY_locations_features.csv")
ALL_ANALYTIC_LOCATION_FEATURES_CSV = os.path.join(DATA_DIR, f"{COMBINED_FILENAME}-ANALYTIC_location_features.csv")

ALL_COMPRESSED_LOCATION_DATA_PARQUET = os.path.join(DATA_DIR, f"{COMBINED_FILENAME}-COMPRESSED_location_data.parquet")
ALL_STAY_LOCATIONS_FEATURES_PARQUET = os.path.join(DATA_DIR, f"{COMBINED_FILENAME}-STAY_locations_features.parquet")
ALL_ANALYTIC_LOCATION_FEATURES_PARQUET = os.path.join(DATA_DIR, f"{COMBINED_FILENAME}-ANALYTIC_location_features.parquet")

# Stack all the processed data files
stack_all_COMPRESSED_location_data(DATA_DIR, "*-COMPRESSED_location_data.csv", ALL_COMPRESSED_LOCATION_DATA_CSV, ALL_COMPRESSED_LOCATION_DATA_PARQUET)
stack_all_ANALYTICAL_location_features(DATA_DIR, "*-ANALYTIC_location_features.csv", ALL_ANALYTIC_LOCATION_FEATURES_CSV, ALL_ANALYTIC_LOCATION_FEATURES_PARQUET)
#stack_all_STAY_location_features(DATA_DIR, "*-STAY_locations_data.csv", ALL_STAY_LOCATIONS_FEATURES_CSV, ALL_STAY_LOCATIONS_FEATURES_PARQUET)


