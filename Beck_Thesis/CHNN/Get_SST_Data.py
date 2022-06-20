# %%
import geemap
import ee
import os
import rioxarray
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import pandas as pd
import to_learning
import os
os.chdir('..')

# %%
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
    df, ds, dir = to_learning.load_file(station_name, input_dir = 'Input_nc')

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

# %%
sst = (ee.ImageCollection('NOAA/CDR/SST_PATHFINDER/V53')
            .select('sea_surface_temperature'))
# %%
stations = pd.read_excel('Coast_orientation/stations.xlsx', sheet_name='Selected Stations')
# %%
for index, station in stations.iterrows():
    print(f'Station: {station["Station"]}')
    
    add_sst_to_ds(station['Station'], station['Lon'], station['Lat'])

# %%
# Testing for Cuxhaven
station = 'cuxhaven-cuxhaven-germany-bsh'
stations = pd.read_excel('Coast_orientation/stations.xlsx')

station = stations[stations['Station'] == station].reset_index(drop=True).loc[0,:]
# %%
add_sst_to_ds(station['Station'], station['Lon'], station['Lat'])

# %%
des_stations = ['calais-calais-france-refmar',         
                'denhelder-hel-nl-rws',
                'aberdeen-p038-uk-bodc',
                'cuxhaven-cuxhaven-germany-bsh',
                'esbjerg-130121-denmark-dmi',
                'brest-brest-france-refmar',
                'delfzijl-del-nl-rws',
                'hoekvanholla-hvh-nl-rws']

des_stations2 = stations[stations['Station'].isin(des_stations)]

# %%
