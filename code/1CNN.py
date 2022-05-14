# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:15:49 2020

@author: acn980
"""

import os
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import xarray as xr
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate



# LOAD DATA
fn_data = 'E:/github/Coastal-hydrographs/MachineLearning/DATA'
file = 'merged_hoekvanholla-hvh-nl-rws.csv'
xr_file = 'hoekvanholla-hvh-nl-rws.nc'

xr_data = xr.open_dataset(os.path.join(fn_data, xr_file))

predictand = xr_data['residual'].to_series()
predictand[predictand<-2] = np.nan
predictand_rolling = predictand.rolling(12,center=True).mean()

bool_series = pd.isnull(predictand_rolling)
to_drop = predictand_rolling[bool_series].index

predictors = ['msl', 'grad', 'uquad', 'vquad', 'rho', 'phi']
predictors_data = xr_data[predictors]
predictors_data['residual_rolling'] = predictand_rolling
predictors_data = predictors_data.where(predictors_data['residual_rolling'].notnull())
predictors_data = predictors_data.dropna(dim = 'time')

Y = predictand_rolling.dropna().values
sc_Y = StandardScaler().fit(np.reshape(Y, (len(Y),1)))
Y_scaled = sc_Y.transform(np.reshape(Y, (len(Y),1)))
Y_scaled = np.reshape(Y_scaled, (len(Y)))

msl = predictors_data['msl'].values

dt = msl.shape[0]
lat_ = msl.shape[1]
lon_ = msl.shape[2]

msl = np.reshape(msl, (msl.shape[0], msl.shape[1]*msl.shape[2]))
sc_msl = PowerTransformer(method='yeo-johnson')
sc_msl = StandardScaler().fit(np.reshape(msl, (msl.shape[0]*msl.shape[1],1)))
msl_scaled = sc_msl.transform(np.reshape(msl, (msl.shape[0]*msl.shape[1],1)))
msl_scaled = np.reshape(msl_scaled, (dt, lat_*lon_))


grad = predictors_data['grad'].values
grad = np.reshape(grad, (grad.shape[0], grad.shape[1]*grad.shape[2]))

sc_grad = StandardScaler().fit(np.reshape(grad, (grad.shape[0]*grad.shape[1],1)))
grad_scaled = sc_grad.transform(np.reshape(grad, (grad.shape[0]*grad.shape[1],1)))
grad_scaled = np.reshape(grad_scaled, (dt, lat_*lon_))


#sc_grad = PowerTransformer(method='yeo-johnson')
#grad_scaled = sc_grad(grad.ravel())

uquad = predictors_data['uquad'].values
uquad = np.reshape(uquad, (uquad.shape[0], uquad.shape[1]*uquad.shape[2]))
sc_uquad = StandardScaler().fit(np.reshape(uquad, (uquad.shape[0]*uquad.shape[1],1)))
uquad_scaled = sc_uquad.transform(np.reshape(uquad, (uquad.shape[0]*uquad.shape[1],1)))
uquad_scaled = np.reshape(uquad_scaled, (dt, lat_*lon_))

#sc_uquad = PowerTransformer(method='yeo-johnson')
#uquad_scaled = sc_uquad(uquad.ravel())

vquad = predictors_data['vquad'].values
vquad = np.reshape(vquad, (vquad.shape[0], vquad.shape[1]*vquad.shape[2]))
sc_vquad = StandardScaler().fit(np.reshape(vquad, (vquad.shape[0]*vquad.shape[1],1)))
vquad_scaled = sc_vquad.transform(np.reshape(vquad, (vquad.shape[0]*vquad.shape[1],1)))
vquad_scaled = np.reshape(vquad_scaled, (dt, lat_*lon_))

#sc_vquad = PowerTransformer(method='yeo-johnson')
#vquad_scaled = sc_vquad(vquad.ravel())

rho = predictors_data['rho'].values
rho = np.reshape(rho, (rho.shape[0], rho.shape[1]*rho.shape[2]))
sc_rho = StandardScaler().fit(np.reshape(rho, (rho.shape[0]*rho.shape[1],1)))
rho_scaled = sc_rho.transform(np.reshape(rho, (rho.shape[0]*rho.shape[1],1)))
rho_scaled = np.reshape(rho_scaled, (dt, lat_*lon_))

#sc_rho = PowerTransformer(method='yeo-johnson')
#rho_scaled = sc_rho(rho.ravel())

phi = predictors_data['phi'].values
phi = np.reshape(phi, (phi.shape[0], phi.shape[1]*phi.shape[2]))
sc_phi = StandardScaler().fit(np.reshape(phi, (phi.shape[0]*phi.shape[1],1)))
phi_scaled = sc_phi.transform(np.reshape(phi, (phi.shape[0]*phi.shape[1],1)))
phi_scaled = np.reshape(phi_scaled, (dt, lat_*lon_))

#%% Building the CNN 1D

#[samples, timesteps, features]

#IN PARALLEL
n_steps = msl_scaled.shape[1] #25 samples per timesteps
n_features = 1 # Because using in parallel otherwise len(predictors) #Number of features or variable

#We reshape the input data
msl_scaled = msl_scaled.reshape(msl.shape[0], msl.shape[1], 1)
grad_scaled = grad_scaled.reshape(grad.shape[0], grad.shape[1], 1)
uquad_scaled = uquad_scaled.reshape(uquad.shape[0], uquad.shape[1], 1)
vquad_scaled = vquad_scaled.reshape(vquad.shape[0], vquad.shape[1], 1)
rho_scaled = rho_scaled.reshape(rho.shape[0], rho.shape[1], 1)
phi_scaled = phi_scaled.reshape(phi.shape[0], phi.shape[1], 1)

visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)

visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
cnn2 = MaxPooling1D(pool_size=2)(cnn2)
cnn2 = Flatten()(cnn2)

visible3 = Input(shape=(n_steps, n_features))
cnn3 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible3)
cnn3 = MaxPooling1D(pool_size=2)(cnn3)
cnn3 = Flatten()(cnn3)

visible4 = Input(shape=(n_steps, n_features))
cnn4 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible4)
cnn4 = MaxPooling1D(pool_size=2)(cnn4)
cnn4 = Flatten()(cnn4)

visible5 = Input(shape=(n_steps, n_features))
cnn5 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible5)
cnn5 = MaxPooling1D(pool_size=2)(cnn5)
cnn5 = Flatten()(cnn5)

visible6 = Input(shape=(n_steps, n_features))
cnn6 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible6)
cnn6 = MaxPooling1D(pool_size=2)(cnn6)
cnn6 = Flatten()(cnn6)

# merge input models
merge = concatenate([cnn1, cnn2, cnn3, cnn4, cnn5, cnn6])
dense = Dense(50, activation='relu')(merge)
output = Dense(1)(dense)

model = Model(inputs=[visible1, visible2, visible3, visible4, visible5, visible6], outputs=output)
model.compile(optimizer='adam', loss='mse')
#%% FITTING MODEL

model.fit([msl_scaled, grad_scaled, uquad_scaled, vquad_scaled, rho_scaled, phi_scaled], Y_scaled , epochs=10, verbose=2)






#%% PREDICTION
yhat = model.predict([msl_scaled, grad_scaled, uquad_scaled, vquad_scaled, rho_scaled, phi_scaled], verbose=0)

plt.figure()
plt.plot(Y_scaled)
plt.plot(yhat)
plt.show()

#%%
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1),
#                 activation='relu',
#                 input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#

