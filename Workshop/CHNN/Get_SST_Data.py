# %%
import geemap
import ee
import os
import rioxarray
from xarray import DataArray
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import pandas as pd
import to_learning
import os
os.chdir('..')

# %%
ee.Initialize()

# %%
def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]]#.dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['longitude', 'latitude', 'datetime',  *list_of_bands]]

    return df

# %%
# Convert from kelvin to celsius
def kelvin_to_celsius(sst):
    return (sst - 273.15) * 0.01

# %%
sst = (ee.ImageCollection('NOAA/CDR/SST_PATHFINDER/V53')
            .select('sea_surface_temperature'))
# %%
stations = pd.read_excel('Coast_orientation/stations.xlsx', sheet_name='Selected Stations')

for index, station in stations.iterrows():
    station_name = station['Station']
    # df, ds, dir = to_learning.load_file(station_name, input_dir = 'Input_nc')
    
    # start = np.min(np.array(ds.time))
    # start = pd.to_datetime(str(start)).strftime('%Y-%m-%d')
    # end = np.max(np.array(ds.time))
    # end = pd.to_datetime(str(end)).strftime('%Y-%m-%d')
    
    start = '2019-10-01'
    end = '2019-12-31'
    
    latitude, longitude = station['Lat'], station['Lon']
    
    station_sst = (sst.filter(ee.Filter.date(start, end))
           .getRegion(ee.Geometry.Point(longitude, latitude), 30)
           .getInfo())
    
    station_sst_df = ee_array_to_df(station_sst, ['sea_surface_temperature'])
    station_sst_df['SST_C'] =  kelvin_to_celsius(station_sst_df['sea_surface_temperature'])
    station_sst_df['Station'] = station_name
    station_sst_df['Coast'] = station['Coast']
    station_sst_df.drop('sea_surface_temperature', inplace=True, axis=1)
    
    if index == 0:
        sst_df = pd.DataFrame(station_sst_df)
    else:
        sst_df = pd.concat([sst_df, station_sst_df]).reset_index(drop=True)

# %%
# TESTING FOR CUXHAVEN
stations = pd.read_excel('Coast_orientation/stations.xlsx')

station = stations[stations['Station'] == 'cuxhaven-cuxhaven-germany-bsh'].reset_index(drop=True).loc[0,:]

station_name = station['Station']
df, ds, dir = to_learning.load_file(station_name, input_dir = 'Input_nc')

start = np.min(np.array(ds.time))
start = pd.to_datetime(str(start)).strftime('%Y-%m-%d')
end = np.max(np.array(ds.time))
end = pd.to_datetime(str(end)).strftime('%Y-%m-%d')

latitude, longitude = station['Lat'], station['Lon']
# %%
station_sst = (sst.filter(ee.Filter.date(start, end))
    .getRegion(ee.Geometry.Point(longitude, latitude), 30)
    .getInfo())

# %%
station_sst_df = ee_array_to_df(station_sst, ['sea_surface_temperature'])
station_sst_df['SST_C'] =  kelvin_to_celsius(station_sst_df['sea_surface_temperature'])
station_sst_df['Station'] = station_name
# station_sst_df['Coast'] = station['Coast']
station_sst_df.drop('sea_surface_temperature', inplace=True, axis=1)

# %%
# Add the SST_C column of data to the DataArray called ds as a component

# %%
