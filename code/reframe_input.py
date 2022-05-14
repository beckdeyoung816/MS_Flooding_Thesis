# -*- coding: utf-8 -*-
"""
Preparing input for machine learning

Timothy Tiggeloven and Anaïs Couasnon
"""

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import time
import xarray as xr

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

def reframe_scale(df, tt_value, prior=False):
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

def resample_rolling(df, lat_list, lon_list, variables, resample, resample_method):
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

def spatial_to_column(station, variables, input_dir):
	if station == 'Cuxhaven':
		filename = os.path.join(input_dir, 'cuxhaven-cuxhaven-germany-bsh.nc')
	elif station == 'Hoek van Holland':
		filename = os.path.join(input_dir, 'hoekvanholla-hvh-nl-rws.nc')
	elif station == 'Puerto Armuelles':
		filename = os.path.join(input_dir, 'puerto_armuelles_b-304b-panama-uhslc.nc')
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
	return df, ds.latitude.values, ds.longitude.values

def column_to_spatial(reframed, columns, lat_list, lon_list, variables, ML):
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

def split_tt(reframed, ML, tt_value):
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

def prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method):
	start = time.time()
	print(f'\nstart preparing data: {station}')

	# read in variables
	print('flattening data')
	df, lat_list, lon_list = spatial_to_column(station, variables, input_dir)

	# resample or rolling mean
	print('resampling data')
	df, step = resample_rolling(df, lat_list, lon_list, variables, resample, resample_method)
	df = df[df['residual'].notna()].copy()

	# reframe and scale data
	print('scaling data')
	reframed, scaler, scaled = reframe_scale(df, tt_value)
	reframed_df = reframed.copy()

	# df to 2d
	if ML == 'CNN' or ML == 'ConvLSTM':
		reframed = column_to_spatial(reframed, df.columns, lat_list, lon_list, variables, ML)

	# split into train and test sets
	print('splitting data')
	train_X, train_y, test_X, test_y, n_train = split_tt(reframed, ML, tt_value)

	print(f'done preparing data: {time.time()-start} sec\n')
	return train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df
