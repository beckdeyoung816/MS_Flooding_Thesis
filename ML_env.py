# -*- coding: utf-8 -*-
"""
Using machine learning to predict coastal hydrographs

Timothy Tiggeloven and Anaïs Couasnon
"""
import os
import sys

from CHNN import ANN
from CHNN import to_learning
from CHNN import performance
from CHNN import model_run

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
output_dir = 'ML_model'
figures_dir = 'Figures'

import glob
import os
import pandas as pd
df_prescreening = pd.read_csv('prescreening_station_t0_batch10.csv')
station_list = df_prescreening['station'][df_prescreening['available'] == True].values
ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']
for station in station_list:
    for ML in ML_list:
        for file in glob.iglob(os.path.join('Zenodo', 'Models', station, ML, '*_ensemble_*')):
            src = os.path.join('Models', 'Ensemble_run', station, ML)
            dst = os.path.join('Zenodo', 'Models', station, ML)
            print(file)
            os.rename(file, file + '.keras')
sys.exit(0)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']
out_dir = os.path.join('Results', 'Input_complexity')
df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
station_list = df_prescreening['station'][df_prescreening['available'] == True].values
station_name = ['Anchorage', 'Boston', 'Callao', 'Cuxhaven', 'Dakar', 'Darwin', 'Dunkerque',
                'Honolulu', 'Humboldt', 'Ko Taphao', 'Lord Howe', 'Puerto Armuelles',
                'San Francisco', 'Wakkanai', 'Zanzibar']
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
for ML in ML_list:
    df_learn = pd.DataFrame()
    for station, name in zip(station_list, station_name):
        df = pd.read_excel(os.path.join(out_dir, 'df', f'{station}_input_complexity_{ML}.xlsx'), index_col='Unnamed: 0')
        norm = NormalizeData(df.loc['msl_rho_uv_grad'].values)
        df_learn[name] = (norm - 1) * -1
        df_learn.plot()
        filename = os.path.join(out_dir, f'Input_complexity_learning_{ML}.png')
        if os.path.isfile(filename):
            os.remove(filename)
        plt.savefig(filename, dpi=300)
        plt.close()
sys.exit(0)



# model_run.clim_mean()
# sys.exit(0)


# model_run.post_process_global()
# sys.exit(0)

# import glob
# threshold = 0
# batch = 10
# input_dir = 'Input_nc_all_detrend_all'
# station_list = sorted(glob.glob(f'{input_dir}/*'))
# df_screening, df = to_learning.prescreening(station_list, input_dir, batch, threshold=threshold)
# df_screening.to_csv(os.path.join(f'prescreening_station_t{threshold}_batch{batch}.csv'))
# sys.exit(0)

# stations_file = 'prescreening_station_parametrization.csv'
# for option in ['best', 'ranked', 'mode']:
#     performance.hyper_opt_results(stations_file, option, ['CNN', 'LSTM', 'ANN', 'ConvLSTM']) # , 'LSTM', 'ANN'
# sys.exit(0)

arg_start = int(sys.argv[1])
arg_end = arg_start + 1
model_run.ensemble_handler(arg_start, arg_end) # , hyper_opt=True
sys.exit(0)



# ML_list = ['ANN', 'LSTM', 'CNN', 'ConvLSTM']
# for ML in ML_list:
#     performance.plot_crps_2panel(ML=ML)

# performance.plot_crps_dist('abashiri-347a-japan-uhslc')
# performance.plot_world_best()
# performance.plot_world_best(skill='Persistence')
# sys.exit(0)

# performance.plot_world_best()
# sys.exit(0)

# performance.plot_world_best(weights=True)
# sys.exit(0)

for metric in ['CRPS', 'Reliability']:  # 'R2', 'RMSE', 'MAE', 'NNSE',
    performance.plot_world_metric(metric)
sys.exit(0)



####################### CLUSTER RUN ##################################
# # test Europe
# station_list = ['aberdeen-p038-uk-bodc', 'brest-brest-france-refmar',
#                 'concarneau-concarneau-france-refmar', 'bilbao-bilbao-spain-pde',
#                 'esbjerg-130121-denmark-dmi', 'ringhals-016-sweden-smhi']

# # prepare ML station data
# tt_value = 1
# df_cluster, values_cluster = {}, []
# train_X_cluster, train_y_cluster, test_X_cluster, test_y_cluster, n_train_cluster = {}, {}, {}, {}, {}
# for station in station_list:
#     train_X, train_y, test_X, test_y, n_train, df, scaler, reframed = to_learning.prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method, sample='cluster', scaler_type=scaler)
#     train_X_cluster[station], train_y_cluster[station], test_X_cluster[station], test_y_cluster[station], n_train_cluster[station] = train_X, train_y, test_X, test_y, n_train

# # design network
# print('design network for cluster')
# model = ANN.design_network(n_layers, neurons, filters, train_X_cluster[station_list[0]], dropout, drop_value, variables, batch_normalization,
#                            name_model, ML=ML, loss=loss, optimizer=optimizer, activation=activation)

# # fit network
# print('Train cluster')
# for group_epoch in range(5):
#     print('\n - Group epoch: {}'.format(group_epoch))
#     for station in station_list:
#         print('\n    - ', station)
#         train_X, train_y, test_X, test_y = train_X_cluster[station], train_y_cluster[station], test_X_cluster[station], test_y_cluster[station]
#         model, train_loss, test_loss = ANN.train_model(model, 20, batch, train_X, train_y, test_X,
#                                                        test_y, ML, name_model, model_dir)

# # blind test
# tt_value = 0.67
# print('\nBlind tests:')
# print('- Cuxhaven')
# station = 'Cuxhaven'
# train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df, reframed = to_learning.prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method, scaler_type=scaler) #, scaler=scaler)
# # train_X, train_y, test_X, test_y, n_train, df, scaler, reframed_df, reframed = to_learning.prepare_station(station, variables, ML, tt_value, input_dir, resample, resample_method, scaler=scaler)
# inv_yhat, inv_y = ANN.predict(model, test_X, reframed_df.iloc[n_train:], scaler)
# performance.handler(inv_yhat, inv_y, df, n_train, train_loss, test_loss, station, neurons, epochs,
#                     batch, resample, tt_value, variables, n_layers, ML, figures_dir, test_on='Europe')


