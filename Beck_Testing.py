# %%

import pandas as pd
import numpy as np
import os
import sherpa
import time
import xarray as xr

from CHNN import to_learning
from CHNN import ANN
import CHNN
# from CHNN import model_run
from CHNN import performance
from Workshop.CHNN import LSTM



# %%

# parameters and variables
station = 'Cuxhaven'  # 'Cuxhaven' 'Hoek van Holland', Puerto Armuelles
resample = 'hourly' # 'hourly' 'daily'
resample_method = 'rolling_mean'  # 'max' 'res_max' 'rolling_mean' ## res_max for daily and rolling_mean for hourly
variables = ['msl', 'grad', 'u10', 'v10', 'rho']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'
tt_value = 0.67  # train-test value
scaler = 'std_normal'  # std_normal, MinMax
n_ncells = 2
epochs = 50
batch = 100
batch_normalization = False
neurons = 48
filters = 8
n_layers = 1  # now only works for uniform layers with same settings
activation = 'relu'  # 'relu', 'swish', 'Leaky ReLu', 'sigmoid', 'tanh'
loss = 'mae'  # 'mae', 'mean_squared_logarithmic_error', 'mean_squared_error'
optimizer = 'adam'  # SGD(lr=0.01, momentum=0.9), 'adam'
dropout = True
drop_value = 0.2
l1, l2 = 0, 0.01
ML = 'LSTM'  # 'LSTM', 'CNN', 'ConvLSTM', 'ANN', 'ALL'
model_dir = os.path.join(os.getcwd(), 'Models')
name_model = '{}_surge_ERA5'.format(ML)
input_dir = 'Input_nc'
output_dir = ''
figures_dir = 'Figures'

# %%

stations = pd.read_excel('Coast_orientation/stations.xlsx', sheet_name='Sheet2')

# %%

if station == 'Cuxhaven':
    station_name = 'cuxhaven-cuxhaven-germany-bsh'
elif station == 'Hoek van Holland':
    station_name = 'hoekvanholla-hvh-nl-rws'
elif station == 'Puerto Armuelles':
    station_name = 'puerto_armuelles_b-304b-panama-uhslc'
else:
    station_name = station
    # sys.exit('Station name not found!')
filename = os.path.join(input_dir, station_name + '.nc')

df_dir = pd.read_excel(os.path.join('Coast_orientation', 'stations.xlsx'))



# %%
start = time.time()    

# read in variables
df, ds, direction = to_learning.load_file(station, input_dir)
# print('Station is loaded')




# %%
sample='cluster'
scaler_type=scaler
cluster_years=5
extreme_thr=0.02
make_univariate=False
year = 'last'
scaler_op=True
n_ncells=2
mask_val=-999
logger=False

# %%
sample = 'cluster'
if sample == 'cluster':
    print(f'Selecting {cluster_years} years of data')
    selected_dates, df = to_learning.draw_sample(df, cluster_years * 365 * 24, 24, threshold=7)
elif sample == 'extreme':
    print(f'Selecting top {extreme_thr} of data')
    selected_dates, df = to_learning.select_extremes(df, extreme_thr)
else: 
    selected_dates = []

# %%
df, lat_list, lon_list = to_learning.spatial_to_column(df, ds, variables, selected_dates, n_ncells)
# print('df to spatial done')    

# resample or rolling mean
df, step = to_learning.resample_rolling(df, lat_list, lon_list, variables, resample, resample_method, make_univariate)
# df = df[df['residual'].notna()].copy()
# print('Resampling done')  

timesteps = int(365 * step)

# reframe and scale data
reframed, scaler, scaled, dates, i_dates = to_learning.reframe_scale(df, timesteps, scaler_type=scaler_type, year = year, scaler_op=scaler_op)
# reframed_df = reframed.copy()

if logger:
    logger.info(f'done preparing data for {station}: {time.time()-start} sec')
else:
    print('done preparing data: {} sec\n'.format(time.time()-start))
# %%

###################### CLUSTER RUN ##################################
# test Europe
station_list = ['aberdeen-p038-uk-bodc', 'brest-brest-france-refmar',
                'concarneau-concarneau-france-refmar', 'bilbao-bilbao-spain-pde',
                'esbjerg-130121-denmark-dmi', 'ringhals-016-sweden-smhi']

# prepare ML station data
tt_value = 1
df_cluster, values_cluster = {}, []
train_X_cluster, train_y_cluster, test_X_cluster, test_y_cluster, n_train_cluster = {}, {}, {}, {}, {}
for station in station_list:
    train_X, train_y, test_X, test_y, n_train, df, scaler, reframed = to_learning.prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method, sample='cluster', scaler_type=scaler)
    train_X_cluster[station], train_y_cluster[station], test_X_cluster[station], test_y_cluster[station], n_train_cluster[station] = train_X, train_y, test_X, test_y, n_train

# design network
print('design network for cluster')
model = ANN.design_network(n_layers, neurons, filters, train_X_cluster[station_list[0]], dropout, drop_value, variables, batch_normalization,
                           name_model, ML=ML, loss=loss, optimizer=optimizer, activation=activation)

# fit network
print('Train cluster')
for group_epoch in range(5):
    print('\n - Group epoch: {}'.format(group_epoch))
    for station in station_list:
        print('\n    - ', station)
        train_X, train_y, test_X, test_y = train_X_cluster[station], train_y_cluster[station], test_X_cluster[station], test_y_cluster[station]
        model, train_loss, test_loss = ANN.train_model(model, 20, batch, train_X, train_y, test_X,
                                                       test_y, ML, name_model, model_dir)

# blind test
tt_value = 0.67
print('\nBlind tests:')
print('- Cuxhaven')
station = 'Cuxhaven'
train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df, reframed = to_learning.prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method, scaler_type=scaler) #, scaler=scaler)
# train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df, reframed = to_learning.prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method, scaler=scaler)
inv_yhat, inv_y = ANN.predict(model, test_X, reframed_df.iloc[n_train:], scaler)
performance.handler(inv_yhat, inv_y, df, n_train, train_loss, test_loss, station, neurons, epochs,
                    batch, resample, tt_value, variables, n_layers, ML, figures_dir, test_on='Europe')









