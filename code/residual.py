# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:32:50 2020

@author: acn980
"""

import os
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error,median_absolute_error
from sklearn.metrics import r2_score,  mean_absolute_error, max_error
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
#%%
def std_normal_trunc(series, thr = 15):
    z_max = (series - series.mean())/series.std()
    series_f = series.where(z_max.abs()< thr)
    return series_f

#%%
fn_data = 'C:/Users/acn980/Desktop'
# list_xr = ['cuxhaven-cuxhaven-germany-bsh.nc', 'hoekvanholla-hvh-nl-rws.nc', 'puerto_armuelles_b-304b-panama-uhslc.nc',]
#list_xr = ['puerto_armuelles_b-304b-panama-uhslc.nc']

z_thr = 15

for xr_file in list_xr:
print(xr_file)
xr_data = xr.open_dataset(os.path.join(fn_data, xr_file))
predictand = xr_data['residual'].to_series()
predictand_trunc = std_normal_trunc(predictand, thr = 15)
predictand_rolling_12 = predictand_trunc.rolling(12,center=True).mean()
predictand_rolling_24 = predictand_trunc.rolling(24,center=True).mean()

# plt.figure()
# plt.plot(predictand.index, predictand, '.-k')
# plt.plot(predictand_trunc.index, predictand_trunc, '.-r')
# plt.show()

# plt.figure()
# plt.plot(predictand_rolling_12.index, predictand_rolling_12, '.-k')
# plt.show()

# plt.figure()
# plt.plot(predictand_rolling_24.index, predictand_rolling_24, '.-k')
# plt.show()




