# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:15:49 2020

@author: acn980
"""

import os
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error,median_absolute_error
from sklearn.metrics import r2_score,  mean_absolute_error, max_error
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

#Make function
def split_train_test(nc, var, ratio = 0.70):
    
    # nc = predictors_data, 'rho', ratio = ratio
    
    
    if isinstance(nc, xr.Dataset):
        sel = nc[var].values
        dt = sel.shape[0]    
        i_train = np.int64(np.around(ratio*dt, decimals = 0))    
        train = sel[0:i_train,:,:]
        test = sel[i_train:,:,:]
        
    if isinstance(nc, np.ndarray):
        sel = nc.copy()
        dt = sel.shape[0]    
        i_train = np.int64(np.around(ratio*dt, decimals = 0))    
        train = sel[0:i_train]
        test = sel[i_train:]
        
    return train, test

def reshape_ML(sel):
    if sel.ndim == 1: 
        new_shape = np.reshape(sel, (len(sel),1))

    if sel.ndim == 3:
        new_shape = np.reshape(sel, (sel.shape[0], sel.shape[1],sel.shape[2],1))
    
    return new_shape
            
def scale_predictor_nc(sel, var, method = 'normalize'):
    if sel.ndim == 1:        
        if method == 'normalize':
            sc = StandardScaler().fit(np.reshape(sel, (len(sel),1)))
        elif method == 'yeo-johnson':
            sc = PowerTransformer(method='yeo-johnson').fit(np.reshape(sel, (len(sel),1)))

        sel_scaled = sc.transform(np.reshape(sel, (len(sel),1)))
        sel_scaled = np.reshape(sel_scaled, (len(sel_scaled)))
    
    if sel.ndim == 3:
        dt = sel.shape[0]
        lat_ = sel.shape[1]
        lon_ = sel.shape[2]
        
        sel = np.reshape(sel, (dt, lat_*lon_))
        if method == 'normalize':
            sc = StandardScaler().fit(np.reshape(sel, (sel.shape[0]*sel.shape[1],1)))
        elif method == 'yeo-johnson':
            sc = PowerTransformer(method='yeo-johnson').fit(np.reshape(sel, (sel.shape[0]*sel.shape[1],1)))
        
        sel_scaled = sc.transform(np.reshape(sel, (sel.shape[0]*sel.shape[1],1)))
        sel_scaled = np.reshape(sel_scaled, (dt, lat_,lon_,1))
    
    return sc, sel_scaled

def transform_w_sc(data, sc):
    if data.ndim == 1:        
        data_scaled = sc.transform(np.reshape(data, (len(data),1)))
    if data.ndim == 3:
        dt = data.shape[0]
        lat_ = data.shape[1]
        lon_ = data.shape[2]
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
        data_scaled = sc.transform(np.reshape(data, (data.shape[0]*data.shape[1],1)))
        data_scaled = np.reshape(data_scaled, (dt, lat_,lon_,1))
    
    return data_scaled

def inv_transform_sc(data, sc):
    data = data.reshape((len(data), 1))
    return sc.inverse_transform(data)

def swish(x):
    return (K.sigmoid(x) * x)

def NSE(y_pred, y_true):
    df_nse = pd.DataFrame(zip(y_pred, y_true), columns=['model', 'observed'])
    df_nse['dividend'] = np.power((df_nse['model'] - df_nse['observed']), 2)
    df_nse['divisor'] = np.power((df_nse['observed'] - df_nse['observed'].mean()), 2)
    NSE_val = 1 - (df_nse['dividend'].sum() / df_nse['divisor'].sum())    
    return NSE_val

def std_normal_trunc(series, thr = 15):
    z_max = (series - series.mean())/series.std()
    series_f = series.where(z_max.abs()< thr)
    return series_f
#%%
# LOAD DATA
# fn_data = 'E:/github/Coastal-hydrographs/MachineLearning/DATA'
fn_data = 'C:/Users/acn980/Desktop'

running_avg = 0 #12 #24
#xr_file = 'cuxhaven-cuxhaven-germany-bsh.nc'
#xr_file = 'puerto_armuelles_b-304b-panama-uhslc.nc'
xr_file = 'hoekvanholla-hvh-nl-rws.nc'

xr_data = xr.open_dataset(os.path.join(fn_data, xr_file))
predictand = xr_data['residual'].to_series()
predictand = std_normal_trunc(predictand, thr = 15)

if running_avg == 0:
    case = '0'#'0_H'
elif running_avg == 12:
    case = '12'#'12_H'
    predictand = predictand.rolling(12,center=True).mean()
elif running_avg == 24:
    case = '24'#'24_H'
    predictand = predictand.rolling(24,center=True).mean()

predictand_rolling = predictand.copy()

bool_series = pd.isnull(predictand_rolling)
to_drop = predictand_rolling[bool_series].index
predictors = ['msl', 'grad', 'uquad', 'vquad', 'rho', 'phi']
predictors_data = xr_data[predictors]
predictors_data['residual_rolling'] = predictand_rolling
predictors_data = predictors_data.where(predictors_data['residual_rolling'].notnull()) #Removing Nan
predictors_data = predictors_data.dropna(dim = 'time')
predictors_data = predictors_data.resample(time='24H').max('time')
predictors_data = predictors_data.dropna(dim = 'time')

Y = predictors_data['residual_rolling'].to_series()
Y = Y.values

#Split in train and test
ratio = 0.70
train_Y, test_Y = split_train_test(Y, '', ratio = ratio)
train_rho, test_rho = split_train_test(predictors_data, 'rho', ratio = ratio) #Wind value
train_phi, test_phi = split_train_test(predictors_data, 'phi', ratio = ratio) #Angle speed
train_msl, test_msl = split_train_test(predictors_data, 'msl', ratio = ratio)
train_grad, test_grad = split_train_test(predictors_data, 'grad', ratio = ratio)

#Scaling Predictors based on training only
sc_Y, Y_scaled = scale_predictor_nc(train_Y, '', method = 'normalize')
sc_rho, rho_scaled = scale_predictor_nc(train_rho, 'rho', method = 'normalize')
sc_phi, phi_scaled = scale_predictor_nc(train_phi, 'phi', method = 'normalize')
sc_msl, msl_scaled = scale_predictor_nc(train_msl, 'msl', method = 'normalize')
sc_grad, grad_scaled = scale_predictor_nc(train_grad, 'grad', method = 'normalize')

#Converting test data based on transformation
test_rho_scaled = transform_w_sc(test_rho, sc_rho)
test_phi_scaled = transform_w_sc(test_phi, sc_phi)
test_msl_scaled = transform_w_sc(test_msl, sc_msl)
test_grad_scaled = transform_w_sc(test_grad, sc_grad)
test_Y_scaled = transform_w_sc(test_Y, sc_Y)

comb_msl_grad = np.concatenate((msl_scaled,grad_scaled),axis=3)
test_comb_msl_grad = np.concatenate((test_msl_scaled,test_grad_scaled),axis=3)

#%% Building the CNN 2D
#the CNN Model for feature extraction and the LSTM Model for interpreting the features across time steps.

# From : https://keras.io/examples/vision/mnist_convnet/

batch_size = 80
epochs = 90

dt = train_rho.shape[0]
lat_ = train_rho.shape[1]
lon_ = train_rho.shape[2]

# filters = 32
# kernel_size = 3 #Reading raster in X*X
# pool_size = 2

input_shape = (lat_,lon_,1)

fn_model = r'E:\github\Coastal-hydrographs\MachineLearning\RNN\MODELS_RES'
name_model = xr_file.strip('.nc')+"_"+case

# get_custom_objects().update({'swish': layers.Activation(swish)})
rho_cnn_input = keras.Input(shape=input_shape)
rho_cnn1= layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation= 'relu')(rho_cnn_input) #layers.Activation(swish)
rho_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(rho_cnn1)
rho_cnn1= layers.Conv2D(64*2, kernel_size=(3, 3), padding='same', activation= 'relu')(rho_cnn1)
rho_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(rho_cnn1)
rho_cnn1= layers.Flatten()(rho_cnn1)
rho_cnn1= layers.Dropout(0.2)(rho_cnn1)

phi_cnn_input = keras.Input(shape=input_shape)
phi_cnn1= layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(phi_cnn_input)
phi_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(phi_cnn1)
phi_cnn1= layers.Conv2D(64*2, kernel_size=(3, 3), padding='same', activation='relu')(phi_cnn1)
phi_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(phi_cnn1)
phi_cnn1= layers.Flatten()(phi_cnn1)
phi_cnn1= layers.Dropout(0.2)(phi_cnn1)

comb_msl_grad_cnn_input = keras.Input(shape=(lat_,lon_,2))
comb_msl_grad_cnn= layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(comb_msl_grad_cnn_input)
comb_msl_grad_cnn= layers.MaxPooling2D(pool_size=(2, 2))(comb_msl_grad_cnn)
comb_msl_grad_cnn= layers.Conv2D(64*2, kernel_size=(3, 3), padding='same', activation='relu')(comb_msl_grad_cnn)
comb_msl_grad_cnn= layers.MaxPooling2D(pool_size=(2, 2))(comb_msl_grad_cnn)
comb_msl_grad_cnn= layers.Flatten()(comb_msl_grad_cnn)
comb_msl_grad_cnn= layers.Dropout(0.2)(comb_msl_grad_cnn)

msl_cnn_input = keras.Input(shape=input_shape)
msl_cnn1= layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(msl_cnn_input)
msl_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(msl_cnn1)
msl_cnn1= layers.Conv2D(64*2, kernel_size=(3, 3), padding='same', activation='relu')(msl_cnn1)
msl_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(msl_cnn1)
msl_cnn1= layers.Flatten()(msl_cnn1)
msl_cnn1= layers.Dropout(0.2)(msl_cnn1)

grad_cnn_input = keras.Input(shape=input_shape)
grad_cnn1= layers.Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(grad_cnn_input)
grad_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(grad_cnn1)
grad_cnn1= layers.Conv2D(64*2, kernel_size=(3, 3), padding='same', activation='relu')(grad_cnn1)
grad_cnn1= layers.MaxPooling2D(pool_size=(2, 2))(grad_cnn1)
grad_cnn1= layers.Flatten()(grad_cnn1)
grad_cnn1= layers.Dropout(0.2)(grad_cnn1)

merge = layers.concatenate([rho_cnn1, msl_cnn1, grad_cnn1]) #[rho_cnn1, phi_cnn1, msl_cnn1, grad_cnn1] #[rho_cnn1, msl_cnn1, grad_cnn1]
x = layers.Dense(50, kernel_regularizer=regularizers.l2(0.01), activation='relu')(merge) #kernel_regularizer=regularizers.l2(0.01),
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs=[rho_cnn_input, msl_cnn_input, grad_cnn_input], outputs=outputs, name=name_model) #[rho_cnn_input,phi_cnn_input, msl_cnn_input, grad_cnn_input] #[rho_cnn_input, msl_cnn_input, grad_cnn_input]
model.compile(optimizer='adam', loss=losses.Huber(delta=0.5)) #losses.LogCosh()
model.summary()

keras.utils.plot_model(model, os.path.join(fn_model, name_model+'_BS_'+str(batch_size)+'_E_'+str(epochs)+'.png'), show_shapes=True)
#%%
history = model.fit([rho_scaled, msl_scaled, grad_scaled],  #[rho_scaled, phi_scaled, msl_scaled, grad_scaled] #[rho_scaled, msl_scaled, grad_scaled]
                    Y_scaled, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    verbose = 2, # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                    validation_split=0.3,
                    workers=3, 
                    shuffle = True,
                    use_multiprocessing=True)

model.save(os.path.join(fn_model, name_model+'_BS_'+str(batch_size)+'_E_'+str(epochs)), overwrite=True, include_optimizer=True)

#Checking if saving worked
#del model
## Recreate the exact same model purely from the file:
#model = keras.models.load_model(os.path.join(fn_model, '2D_CNN_CuxHaven_MaxDay_wo_Phi_BS_80_E_90'))

#%%
# plot history
f = plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

f.savefig(os.path.join(fn_model, name_model+'_BS_'+str(batch_size)+'_E_'+str(epochs)+'_perf.png'))

#%%
yhat = model.predict([test_rho_scaled, test_msl_scaled, test_grad_scaled], verbose=0) #[test_rho_scaled, test_phi_scaled, test_msl_scaled, test_grad_scaled] #[test_rho_scaled, test_msl_scaled, test_grad_scaled]

# f, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True)
# axs = axs.reshape(-1)
# axs[0].plot(test_Y_scaled)
# #axs[0].plot(Y_scaled)
# axs[1].plot(yhat)
# plt.show()

# CONVERTING BACK TO UNITS
Y_predict = inv_transform_sc(yhat, sc_Y)

# f, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True)
# axs = axs.reshape(-1)
# axs[0].plot(test_Y)
# axs[1].plot(Y_predict)
# plt.show()

plt.figure()
plt.plot(test_Y, '-r')
plt.plot(Y_predict, '-b')
plt.show()

#%% Calculating some results
result = pd.DataFrame(index = ['RMSE', 'R2', 'MedAE', 'Max', 'NSE'] , columns = [name_model])

result.loc['RMSE',name_model] = mean_squared_error(test_Y, Y_predict, squared=False)
result.loc['R2',name_model] = r2_score(test_Y, Y_predict)
result.loc['Max',name_model] = max_error(test_Y, Y_predict)
result.loc['MedAE',name_model] = median_absolute_error(test_Y, Y_predict)
result.loc['NSE',name_model] = NSE(Y_predict, test_Y)

fn_out = os.path.join(fn_model, name_model+'_BS_'+str(batch_size)+'_E_'+str(epochs)+'_metrics.csv')
result.to_csv(fn_out, index = True, index_label = 'Metrics')

print(result)
