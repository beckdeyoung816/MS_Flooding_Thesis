# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:53:34 2020

@author: acn980
"""
import os
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np

#%%
fn_summ = r'E:\surfdrive\Documents\VU\GESLA\public_11092018_summary.csv'
summ = pd.read_csv(fn_summ)

lat_europe_min = -13
lat_europe_max = 35

lon_europe_min = 35
lon_europe_max = 60

sel_stations = summ.where((summ.lat<lat_europe_max) and (summ.lat<lat_europe_max)
                          and (summ.lon<lon_europe_max) and (summ.lon<lon_europe_max))