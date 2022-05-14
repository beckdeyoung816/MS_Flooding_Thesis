# -*- coding: utf-8 -*-
"""
Timothy Tiggeloven
"""

from math import sqrt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.backend import sigmoid
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import keras
from keras import layers
from keras import regularizers
import sys
import time
import xarray as xr

from ML_performance_plot import plot_performance

def plot_input(df):
	values = df.values
	# specify columns to plot
	groups = range(len(df.columns))
	i = 1
	# plot each column
	plt.figure()
	for group in groups:
		plt.subplot(len(groups), 1, i)
		plt.plot(values[:, group])
		plt.title(df.columns[group], y=0.5, loc='right')
		i += 1
	plt.savefig('data.png')
	plt.close()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	# convert series to supervised learning
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def reframe_scale(df, prior=False):
	if not prior:
		cols = df.columns.tolist()
		df = df[cols[1:] + cols[:1]]
	values = df.values

	# ensure all data is float
	values = values.astype('float32')

	# normalize features on training data only
	n_train_hours = int(values.shape[0] * tt_value)
	train = values[:n_train_hours, :]
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	scaled = scaler.transform(values)

	# frame as supervised learning
	if prior:
		reframed = series_to_supervised(scaled, 1, 1)
		# drop columns we don't want to predict
		reframed.drop(reframed.columns[5:], axis=1, inplace=True)
	else:
		columns = [f'var{i + 1}(t)' for i in range(len(df.columns) - 1)]
		columns.append('values(t)')
		reframed = pd.DataFrame(scaled, columns=columns)
	return reframed, scaler, scaled

def resample_rolling(df, lat_list, lon_list):
	# df = df.rolling('12H').mean()
	# df['residual'] = df['residual'].rolling('12H').mean().values
	if resample == 'hourly':
		step = 24
		df = df.rolling('12H').mean()
		# df['residual'] = df['residual'].rolling('12H').mean().values
	elif resample == 'daily' and resample_method == 'res_max':
		step = 1
		index_var_name = f'msl_{lat_list[2]}_{lon_list[2]}'
		index = df[index_var_name].loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]].index
		df_res = df['residual'].resample('24H').max().to_frame()
		for var in variables:
			df_var = df.loc[:, df.columns.str.startswith(var)]
			if var != 'msl' and var != 'grad':
				df_var  = df_var.loc[index, :]
			# df_var  = df_var.loc[index, :]
			df_var = df_var.resample('24H').max()
			df_res = df_res.merge(df_var, how='left', right_index=True, left_index=True)
		df = df_res.copy()
	elif resample == 'daily' and resample_method == 'max':
		step = 1
		df = df.resample('24H').max()
	return df, step

def spatial_to_column(station):
	if station == 'Cuxhaven':
		filename, name = os.path.join(workspace, 'cuxhaven-cuxhaven-germany-bsh.nc'), 'ch'
	elif station == 'Hoek van Holland':
		filename, name = os.path.join('MachineLearning', 'DATA', 'hoekvanholla-hvh-nl-rws.nc'), 'hvh'
	elif station == 'Puerto Armuelles':
		filename, name = os.path.join(workspace, 'puerto_armuelles_b-304b-panama-uhslc.nc'), 'pa'
	else:
		sys.exit('Station name not found!')

	# read in variables
	ds = xr.open_dataset(filename)
	df = ds[['gesla_swl', 'tide_wtrend', 'residual']].to_dataframe()
	# df = df.rolling('12H').mean()

	# extract all gridded data to columns
	for lat in ds.latitude.values:
		for lon in ds.longitude.values:
			dfi = ds.sel(latitude=lat, longitude=lon).to_dataframe()
			dfi = dfi[variables].copy()
			dfi.columns = [col + f'_{lat}_{lon}' for col in dfi.columns]
			df = df.merge(dfi, how='left', right_index=True, left_index=True)
	df.drop(df.columns[:2], axis=1, inplace=True)
	return df, name, ds.latitude.values, ds.longitude.values

def column_to_spatial(reframed, columns, lat_list, lon_list):
	# rename columns
	cols = columns[1:]
	cols = np.append(cols, columns[0])
	reframed.columns = cols
	
	# create gridded data per variables
	reframed_list = []
	reframed_list.append(reframed[f'residual'].values)
	if ML == 'CNN':
		reframed_empty = np.zeros((len(reframed), 5, 5, 1))  # time_train, lat, lon, 1
	elif ML == 'ConvLSTM':
		reframed_empty = np.zeros((len(reframed), 1, 5, 5, 1))  # time_train, 1, lat, lon, 1
	for var in variables:
		# find columns of variable
		df_var = reframed[list(reframed.filter(regex=var))]
		reframed_var = reframed_empty.copy()

		# store in empty grid
		for i, lat in enumerate(lat_list):
			for j, lon in enumerate(lon_list):
				if ML == 'CNN':
					reframed_var[:, i, j, 0] = df_var[f'{var}_{lat}_{lon}'].values
				elif ML == 'ConvLSTM':
					reframed_var[:, 0, i, j, 0] = df_var[f'{var}_{lat}_{lon}'].values
		reframed_list.append(reframed_var)
	return reframed_list

def split_tt(tt_value, reframed):
	if ML == 'LSTM':
		values = reframed.values
		n_train = int(values.shape[0] * tt_value)
		train = values[:n_train, :]
		test = values[n_train:, :]

		# split into input and outputs
		train_X, train_y = train[:, :-1], train[:, -1]
		test_X, test_y = test[:, :-1], test[:, -1]

		# reshape input to be 3D [samples, timesteps, features]
		train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
		test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
		# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

	elif ML == 'CNN' or ML == 'ConvLSTM':
		n_train = int(reframed[0].shape[0] * tt_value)

		# split into input and outputs
		train_y, test_y = reframed[0][:n_train], reframed[0][n_train:]
		train_X = [var[:n_train, :, :, :] for var in reframed[1:]]
		test_X = [var[n_train:, :, :, :] for var in reframed[1:]]

	return train_X, train_y, test_X, test_y, n_train

def prepare_station(station):
	start = time.time()
	print(f'\nstart preparing data: {station}')

	# read in variables
	print('flattening data')
	df, name, lat_list, lon_list = spatial_to_column(station)

	# resample or rolling mean
	print('resampling data')
	df, step = resample_rolling(df, lat_list, lon_list)
	df = df[df['residual'].notna()].copy()

	# reframe and scale data
	print('scaling data')
	reframed, scaler, scaled = reframe_scale(df)
	reframed_df = reframed.copy()

	# df to 2d
	if ML == 'CNN' or ML == 'ConvLSTM':
		reframed = column_to_spatial(reframed, df.columns, lat_list, lon_list)

	# split into train and test sets
	print('splitting data')
	train_X, train_y, test_X, test_y, n_train = split_tt(tt_value, reframed)

	print(f'done preparing data: {time.time()-start} sec\n')
	return train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df

def design_network(n_layers, neurons, train_X, ML='LSTM', loss='mae', optimizer='adam',
				   activation='relu'):
	print(f'Building model: {ML}\n')
	# design network
	if ML == 'LSTM':
		model = LSTM_model(n_layers, neurons, activation, train_X)
	elif ML == 'CNN':
		model = CNN_model(n_layers, neurons, activation, train_X)
	elif ML == 'ConvLSTM':
		model = ConvLSTM_model(n_layers, neurons, activation, train_X)
	
	model.compile(loss=loss, optimizer=optimizer)
	# model.summary()

	keras.utils.plot_model(model, os.path.join(workspace, 'Figures', f'{name_model}.png'), show_shapes=True)

	return model

def LSTM_model(n_layers, neurons, activation, train_X):
	# design LSTM
	model = Sequential()
	if n_layers == 1:
		if activation == 'Leaky ReLu':
			model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
			model.add(layers.LeakyReLU(alpha=0.1))
			if dropout:
				model.add(layers.Dropout(drop_value))
		else:
			model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2]), activation=activation))
			if dropout:
				model.add(layers.Dropout(drop_value))
	elif n_layers == 2:
		model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
		if dropout:
			model.add(layers.Dropout(drop_value))
		model.add(LSTM(neurons))
		if dropout:
			model.add(layers.Dropout(drop_value))
	elif n_layers == 3:
		model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
		if dropout:
			model.add(layers.Dropout(drop_value))
		model.add(LSTM(int(neurons / 2), return_sequences=True))
		if dropout:
			model.add(layers.Dropout(drop_value))
		model.add(LSTM(int(neurons / 4)))
		if dropout:
			model.add(layers.Dropout(drop_value))
	model.add(Dense(1))

	return model

def ConvLSTM_model(n_layers, neurons, activation, train_X):
	input_shape = (1, train_X[0].shape[2], train_X[0].shape[3], 1)

	merge_list, input_list = [], []
	for i, var in enumerate(variables):
		cnn_input = keras.Input(shape=input_shape)
		cnn_lay = layers.ConvLSTM2D(filters=neurons, kernel_size=(3, 3), padding="same",
		                            return_sequences=True, activation=activation,
									recurrent_activation=activation)(cnn_input) # , return_sequences=True
		# cnn_lay = layers.BatchNormalization()(cnn_lay)
		cnn_lay = layers.ConvLSTM2D(filters=neurons, kernel_size=(3, 3), padding="same",
		                            return_sequences=True, activation=activation,
									recurrent_activation=activation)(cnn_input)
		# cnn_lay = layers.BatchNormalization()(cnn_lay)
		cnn_lay = layers.Flatten()(cnn_lay)
		merge_list.append(cnn_lay)
		input_list.append(cnn_input)

	merge = layers.concatenate(merge_list)
	outputs = layers.Dense(1)(merge)
	model = keras.Model(inputs=input_list, outputs=outputs, name=name_model)

	return model

def CNN_model(n_layers, neurons, activation, train_X):
	input_shape = (train_X[0].shape[1], train_X[0].shape[2], 1)

	merge_list, input_list = [], []
	for i, var in enumerate(variables):
		cnn_input = keras.Input(shape=input_shape)
		cnn1= layers.Conv2D(neurons, kernel_size=(3, 3), padding='same', activation=activation)(cnn_input)
		cnn1= layers.MaxPooling2D(pool_size=(2, 2))(cnn1)
		cnn1= layers.Conv2D(neurons*2, kernel_size=(3, 3), padding='same', activation=activation)(cnn1)
		cnn1= layers.MaxPooling2D(pool_size=(2, 2))(cnn1)
		cnn1= layers.Flatten()(cnn1)
		if dropout:
			cnn1= layers.Dropout(drop_value)(cnn1)
		merge_list.append(cnn1)
		input_list.append(cnn_input)

	merge = layers.concatenate(merge_list)
	x = layers.Dense(50, kernel_regularizer=regularizers.l2(0.01), activation=activation)(merge)
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(1)(x)
	model = keras.Model(inputs=input_list, outputs=outputs, name=name_model)

	return model

def train_data(model, epochs, batch, train_X, train_y, test_X, test_y):
	# fit network
	if ML == 'LSTM' or ML == 'ConvLSTM':
		history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, use_multiprocessing=True,
							validation_data=(test_X, test_y), verbose=2, shuffle=False, workers=3)
	elif ML == 'CNN':
		history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, shuffle=True,
							validation_split=0.3, workers=3, verbose=2, use_multiprocessing=True)
	
	model.save(os.path.join(workspace, path_model, name_model), overwrite=True, include_optimizer=True)

	# loss values
	train_loss = history.history['loss']
	test_loss = history.history['val_loss']

	return model, train_loss, test_loss

def forescast(model, test_X, reframed_df):
	# make a prediction
	yhat = model.predict(test_X)

	# invert scaling for actual
	inv_y = scaler.inverse_transform(reframed_df.values)[:,-1]

	# invert scaling for actual
	reframed_df['values(t)'] = yhat
	inv_yhat = scaler.inverse_transform(reframed_df.values)[:,-1]

	# # invert scaling for forecast
	# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
	# inv_yhat = np.concatenate((test_X[:, :], yhat), axis=1)
	# inv_yhat = scaler.inverse_transform(inv_yhat)
	# inv_yhat = inv_yhat[:,-1]
	# a = scaler.inverse_transform(reframed_df.values)

	# # invert scaling for actual
	# inv_y = test_y.reshape((len(test_y), 1))
	# inv_y = np.concatenate((test_X[:, :], inv_y), axis=1)
	# inv_y = scaler.inverse_transform(inv_y)
	# inv_y = inv_y[:,-1]

	return inv_yhat, inv_y

def plot_handler(inv_yhat, inv_y, df, n_train, train_loss, test_loss, station, neurons, epochs,
                 batch, resample, tt_value, variables, n_layers, ML, workspace, test_on='self'):
	# plot results
	df_test = df.iloc[n_train:].copy()
	df_test['Observed'] = inv_y
	df_test['Modelled'] = inv_yhat
	df_test = df_test[['Observed', 'Modelled']].copy()
	plot_performance(df_test, train_loss, test_loss, station, neurons, epochs, batch, resample,
					tt_value, len(variables), workspace=os.path.join(workspace, 'Figures'),
					layers=n_layers, ML=ML, test_on=test_on)

def swish_func(x, beta = 1):
    return (x * sigmoid(beta * x))

class swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

get_custom_objects().update({'swish': swish(swish_func)})

# parameters and variables
station = 'Cuxhaven'  # 'Cuxhaven' 'Hoek van Holland', Puerto Armuelles
resample = 'daily' # 'hourly' 'daily'
resample_method = 'res_max'  # 'max' 'res_max'
variables = ['msl', 'grad', 'u10', 'v10', 'rho']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'
tt_value = 0.67 # train-test value
epochs = 50
batch = 100
neurons = 50
n_layers = 2  # now only works for LSTM
activation = 'relu'  # 'relu', 'swish', 'Leaky ReLu', 'sigmoid', 'tanh'
loss = 'mean_squared_error'  # 'mae', 'mean_squared_logarithmic_error', 'mean_squared_error'
optimizer = 'adam'  # SGD(lr=0.01, momentum=0.9), 'adam'
dropout = True
drop_value = 0.2
ML = 'ConvLSTM'  # 'LSTM', 'CNN', 'ConvLSTM'
path_model = 'Models'
name_model = f'{ML}_surge_ERA5'
workspace = 'C:\\Users\\ttn430\\Documents\\Coastal'
print(f'Machine learning to predict surge')

if activation == 'swish':
	get_custom_objects().update({'swish': swish(swish_func)})

# prepare ML station data
train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df = prepare_station(station)

# design network
model = design_network(n_layers, neurons, train_X, ML=ML, loss=loss, optimizer=optimizer,
                       activation=activation)

# fit network
model, train_loss, test_loss = train_data(model, epochs, batch, train_X, train_y, test_X, test_y)

# make a prediction
inv_yhat, inv_y = forescast(model, test_X, reframed_df.iloc[n_train:])

# plot results
plot_handler(inv_yhat, inv_y, df, n_train, train_loss, test_loss, station, neurons, epochs, batch,
             resample, tt_value, variables, n_layers, ML, workspace)

# test on other station
station = 'Hoek van Holland'
# train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df = prepare_station(station)
# inv_yhat, inv_y = forescast(model, test_X, reframed_df.iloc[n_train:])
# plot_handler(inv_yhat, inv_y, df, n_train, train_loss, test_loss, station, neurons, epochs, batch,
#              resample, tt_value, variables, n_layers, ML, workspace, test_on='Cuxhaven')
