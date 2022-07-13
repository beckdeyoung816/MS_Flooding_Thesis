# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:02:42 2020

@author: acn980
"""
import glob
import xarray as xr
import pandas as pd
import os

fn1 = 'Input_nc_all_detrend'
all_files = glob.glob(os.path.join(fn1, '*.nc'))

for file in all_files:
    print(file)
    station = file.split('/')[-1].split('.nc')[0]
    print(station)
    ds = xr.open_dataset(file)
    df1 = ds['residual'].to_dataframe()
    df2 = ds['msl'].isel(latitude=0, longitude=0).to_dataframe()
    
    df_final = pd.concat([df1, df2], axis = 1)
    df_final.to_csv(os.path.join(fn1, station+'.csv'), index_label = 'time')

#####    
fn1 = 'Input_nc_linear'
all_files = glob.glob(os.path.join(fn1, '*.nc'))

for file in all_files:
    print(file)
    station = file.split('/')[-1].split('.nc')[0]
    print(station)
    ds = xr.open_dataset(file)
    df1 = ds['residual'].to_dataframe()
    df2 = ds['msl'].isel(latitude=0, longitude=0).to_dataframe()
    
    df_final = pd.concat([df1, df2], axis = 1)
    df_final.to_csv(os.path.join(fn1, station+'.csv'), index_label = 'time')
