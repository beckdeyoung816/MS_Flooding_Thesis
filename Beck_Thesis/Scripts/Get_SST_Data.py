'''
Script to get Sea surface temperature for desired stations. Functions had to be broken up to be run on different computers because of GEE authentification
Basic work flow is : 1. get_date_ranges.py on Snellius 2. Transfer results to local machine 3. Get_SST_Data.py on local machine 4. Transfer SST data to Snellius 5. merge_sst_data.py on Snellius
'''
import ee
import xarray as xr
import numpy as np

import pandas as pd
import to_learning
import os

os.chdir('..')

ee.Initialize()

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Select desired columns
    df = df[['longitude', 'latitude', 'time', *list_of_bands]]#.dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['longitude', 'latitude', 'datetime',  *list_of_bands]]

    return df

def kelvin_to_celsius(sst):
    # Convert from kelvin to celsius
    return (sst - 273.15) * 0.01


def add_sst_to_ds(station_name, longitude, latitude):
    """Add Sea Surface Temperate time series to a given station's netcdf file. The data is bi-daily and is upsampled to hourly with linear interpolation between values. Only the exact location is used, no spatial box. The merged ds is written as a new .nc file
    """
    
    # Load in original data
    df, ds, dir = to_learning.load_file(station_name, input_dir = '../Input_nc')

    # Get the start and end dates of the dataset to get those same dates for the SST data
    start = np.min(np.array(ds.time))
    start = pd.to_datetime(str(start)).strftime('%Y-%m-%d')
     
    end = np.max(np.array(ds.time)) + pd.DateOffset(1) # Add one day after end because filtering is exclusive
    end = pd.to_datetime(str(end)).strftime('%Y-%m-%d')
    
    # Filter the SST data to the same dates and location as the original data
    station_sst = (sst.filter(ee.Filter.date(start, end))
                    .getRegion(ee.Geometry.Point(longitude, latitude), 30)
                    .getInfo())
    
    # Convert to a pandas dataframe and convert to celsius
    sst_df = ee_array_to_df(station_sst, ['sea_surface_temperature'])
    sst_df['sst'] =  kelvin_to_celsius(sst_df['sea_surface_temperature'])
    sst_df = sst_df[['datetime', 'sst']]
        
    sst_df['time'] = pd.to_datetime(sst_df['datetime'])
    sst_ds = (sst_df.set_index('time')
            .resample('H').mean().interpolate() # Resample hourly and interpolate between values
            .to_xarray())
    
    print('Merging')
    
    sst_ds = xr.merge([ds, sst_ds]) # Merge datasets
    
    print('Writing')
    sst_ds.to_netcdf('Input_nc_sst/' + station_name + '.nc') # Write to a new file


def get_sst_data(station):

    station_sst = (sst.filter(ee.Filter.date(station['start_date'], station['end_date']))
                   .getRegion(ee.Geometry.Point(station['Lon'], station['Lat']), 30)
                   .getInfo())
    
    # Convert to a pandas dataframe and convert to celsius
    sst_df = ee_array_to_df(station_sst, ['sea_surface_temperature'])
    
    if sst_df['sea_surface_temperature'].count() == 0:
        print(f'No data for {station["Station"]}')
        return
    
    sst_df['sst'] =  kelvin_to_celsius(sst_df['sea_surface_temperature'])
    sst_df = sst_df[['datetime', 'sst']]
        
    sst_df['time'] = pd.to_datetime(sst_df['datetime'])
    sst_ds = (sst_df.set_index('time')
            .resample('H').mean().interpolate() # Resample hourly and interpolate between values
            .to_xarray())
    
    print('Writing')
    sst_ds.to_netcdf('Input_sst_data/' + station['Station'] + '_sst.nc') # Write to a new file


# Load in the SST data
sst = (ee.ImageCollection('NOAA/CDR/SST_PATHFINDER/V53')
            .select('sea_surface_temperature'))

# stations = pd.read_csv('Coast_orientation/Selected_Stations.csv')

# for index, station in stations.iterrows():
#     print(f'Getting SST for Station: {station["Station"]}\n')
    
#     add_sst_to_ds(station['Station'], station['Lon'], station['Lat'])
    
stations2 = pd.read_csv('Coast_orientation/Selected_Stations_dates.csv').dropna().reset_index(drop=True)
stations2['start_date'] = pd.to_datetime(stations2['start_date'])
stations2['end_date'] = pd.to_datetime(stations2['end_date'])

for index, station in stations2.iterrows():
    print(f'\nGetting SST for Station: {station["Station"]}')
    get_sst_data(station)


# REMOVE STATIONS WITHOUT DATA
# stations = pd.read_csv('Coast_orientation/Selected_Stations.csv')
# stations = stations[stations['Station'].isin(stations2['Station'])].reset_index(drop=True)
# stations.to_csv('Coast_orientation/Selected_Stations_w_Data.csv', index=False)
