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
ML = 'CNN'  # 'LSTM', 'CNN', 'ConvLSTM', 'ANN', 'ALL'
model_dir = os.path.join(os.getcwd(), 'Models')
name_model = '{}_surge_ERA5'.format(ML)
input_dir = 'Input_nc_all_detrend_all' # _all_detrend_all
output_dir = 'ML_model'
figures_dir = 'Figures'
mask_val =-999

import pandas as pd
import numpy as np
import datetime
import properscoring as ps
import seaborn as sns
import matplotlib.pyplot as plt
import pretty_errors
from ranky import rankz
import cartopy
import cartopy.crs as ccrs
import matplotlib.colors as colors


# df = pd.read_csv(os.path.join('Results', 'Global_performance_metrics_CM.csv'))
# df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
# bounds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

# ML_list = ['ANN', 'LSTM', 'CNN', 'ConvLSTM']
# cmap = 'RdPu'
# abcd_list = ['a', 'b', 'c', 'd']
# subplot_list = [221, 222, 223, 224]
# xtick_list = [False, False, True, True]
# ytick_list = [True, False, True, False]

# metric_list = ['CRPS', 'Uncertainty', 'Reliability', 'Resolution']  # 'corrcoef', 'R2', 'NSE', 'NNSE'

# fig = plt.figure(figsize=[19.3, 9.5])
# isel = 0
# for metric, abcd in zip(metric_list, abcd_list):
#     ax = fig.add_subplot(subplot_list[isel], projection=ccrs.PlateCarree())
#     ax.add_feature(cartopy.feature.LAND, edgecolor='black')
#     rpt = df[f'Best_{metric}'].values * 100
#     sys.exit(0)


# performance.plot_world('Best_NN')  # 'Best_CRPSS', 'Resolution', 'Uncertainty'  # Fig 2-3
# performance.plot_input_complexity(kind='CRPSS', panel=6)  # Fig 5
# performance.plot_input_complexity(kind='CRPSS', panel=9)  # Fig S8
# performance.table_input_complexity(time=True)
# performance.table_hyper_complexity()
# performance.average_model_time()
# performance.plot_world_4panel()  # Fig S3
# performance.plot_world_4panel(var='CRPSS')  # Fig S2
# performance.plot_world_4panel(var='Decomposition')  # Fig S4
# performance.plot_world_metric('Reliability')
performance.complexity_model_time()
sys.exit(0)

ML_list = ['LSTM']
for option in ['best', 'mode', 'ranked']:
    performance.hyper_opt_results(option, ML_list)
sys.exit(0)

ML_list = ['LSTM', 'CNN']
for ML in ML_list:
    df = pd.read_excel(f'test_variables_{ML}.xlsx', index_col='Unnamed: 0')
    df = df.drop(['Lat', 'Lon'], axis=1)

    c = ['1st','2nd','3rd']
    df = (df.apply(lambda x: pd.Series(x.nsmallest(3).index, index=c), axis=1).reset_index())
    df.to_excel(f'test_variables_{ML}_top3.xlsx')

sys.exit(0)

# performance.plot_world_metric('Reliability')
# performance.plot_world_best('Resolution')  # 'Best_CRPS' 'Uncertainty' 'Best_reference_CRPS' 'Resolution'
# performance.plot_world_metric('CM')
# performance.plot_crps_2panel(metric='Best')
# performance.plot_crps_2panel(metric='Absolute')
# sys.exit(0)
performance.table_var_test()

# performance.plot_world_best(weights=True)
sys.exit(0)

for metric in ['CRPS', 'Reliability']:  # 'R2', 'RMSE', 'MAE', 'NNSE',
    performance.plot_world_metric(metric)
sys.exit(0)

def ens_metrics(df_test, model='ensemble'):
    if model == 'ensemble':
        filter_col = [col for col in df_test if col.startswith('Modelled')]
        obs = df_test['Observed'].values
        mods = df_test[filter_col].values
        mod = df_test[filter_col].mean(axis =1).values
    elif model == 'persistence':
        obs = df_test['Observed'].values
        mod = df_test['Modelled'].values
    
    rmse, NSE, r2, mae = performance.performance_metrics(obs, mod)
    
    return rmse, NSE, r2, mae 

def crps_metrics(df_test, model='ensemble'):
    # CRPS = Continuous Ranked Probability Score
    # CRPS is a generalization of mean absolute error
    # Using the properscoring package as in Bruneau et al. (2020)
    if model == 'ensemble':
        filter_col = [col for col in df_test if col.startswith('Modelled')]
        obs = df_test['Observed'].values
        mods = df_test[filter_col].values
    elif model == 'persistence':
        obs = df_test['Observed'].values
        mods = df_test['Modelled'].values
    
    model_score = ps.crps_ensemble(obs, mods).mean() #returns a value per timestep. Here we take the mean of those values to get one value per location
    return model_score

fn_ens = 'Models/Ensemble_run'
df_prescreening = pd.read_csv('prescreening_station_t0_batch10.csv')
station_list = df_prescreening['station'][df_prescreening['available'] == True].values

df_prescreening = pd.read_csv('prescreening_station_t0_batch10_LastYearCheck.csv')
station_list = df_prescreening['station'][df_prescreening['last_year'] == True].values

ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']

date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
metric = pd.DataFrame(index=station_list, data=None)
df_loc = pd.read_excel(os.path.join('Coast_orientation', 'stations.xlsx'))
df_loc = df_loc.set_index('Station')
metric = metric.merge(df_loc[['Lat', 'Lon']], how='left', left_index=True, right_index=True)

for station in station_list:
    print(station)
    # station = 'cuxhaven-cuxhaven-germany-bsh'
    ML_list_plus = ML_list.copy()
    ML_list_plus.append('Persistence')
    for ML in ML_list_plus:
        if ML == 'Persistence':
            fn = os.path.join(fn_ens, station, 'ANN', station + '_' + 'ANN' + '_prediction.csv')
            df_result = pd.read_csv(fn, index_col='time', date_parser=date_parser)
            model = 'persistence'

            # calculate persistence
            df_copy = df_result.copy()
            df_result = df_copy.iloc[1:].copy()
            df_result['Modelled'] = df_copy['Observed'].iloc[:-1].values
        else:
            fn = os.path.join(fn_ens, station, ML, station + '_' + ML + '_prediction.csv')
            df_result = pd.read_csv(fn, index_col='time', date_parser=date_parser)
            model = 'ensemble'       
        metric.loc[station, f'{ML}_CRPS'] = crps_metrics(df_result.dropna(axis=0, how='any'), model=model)
        rmse, NSE, r2, mae = ens_metrics(df_result.dropna(axis=0, how='any'), model=model)
        metric.loc[station, f'{ML}_RMSE'] = rmse
        metric.loc[station, f'{ML}_NSE'] = NSE
        metric.loc[station, f'{ML}_R2'] = r2
        metric.loc[station, f'{ML}_MAE'] = mae

    # metric.loc[station] = pd.to_numeric(metric.loc[station], errors='coerce')
    for sel_metric in ['CRPS', 'RMSE', 'NSE', 'R2', 'MAE']:
        sel_columns = [f'{ML}_{sel_metric}' for ML in ML_list]
        if (sel_metric == 'CRPS') or (sel_metric == 'MAE') or (sel_metric == 'RMSE'):
            metric.loc[station, f'Best_{sel_metric}'] = metric.loc[station][sel_columns].astype(float).idxmin()
            metric.loc[station, f'Best_{sel_metric}_val'] = metric.loc[station][sel_columns].astype(float).min()
        else:
            metric.loc[station, f'Best_{sel_metric}'] = metric.loc[station][sel_columns].astype(float).idxmax()
            metric.loc[station, f'Best_{sel_metric}_val'] = metric.loc[station][sel_columns].astype(float).max()
    
metric.to_csv(os.path.join('Results', 'Global_performance_metrics.csv'))
metric.to_excel(os.path.join('Results', 'Global_performance_metrics.xlsx'))
sys.exit(0)

station = 'aburatsu,japan-aburatsu,japan-glossdm-bodc'
station = 'cuxhaven-cuxhaven-germany-bsh'
out_dir = os.path.join('Models', 'Ensemble_run', station, 'CNN')
# df_result = pd.read_csv(os.path.join(out_dir, f'{station}_{ML}_prediction.csv' ), parse_dates=True, index_col='time')
# df_train = pd.read_csv(os.path.join(out_dir, f'{station}_{ML}_training.csv' ))
# df_test = pd.read_csv(os.path.join(out_dir, f'{station}_{ML}_testing.csv' ))

# performance.plot_ensemble_performance(df_result, df_train, df_test, station, neurons, epochs, batch, resample,
#                          tt_value, 5, out_dir, layers=1, ML=ML, test_on='ensemble')
# sys.exit(0)

df_prescreening = pd.read_csv('prescreening_station_t0_batch10.csv')
df_prescreening['last_year'] = False
station_list = df_prescreening['station'][df_prescreening['available'] == True].values

for station in station_list:
    df, ds, direction = to_learning.load_file(station, input_dir)

    col = 'residual'
    timesteps = 365 * 24
    threshold = 7 * 24
    year = 'last'

    # count timesteps in 5 years and insert ID's
    df = df.copy()
    df.insert(0, 'ID', range(0, len(df)))

    # count consecutive NaN
    na_count = df[col].isnull().astype(int).groupby(df[col].notnull().astype(int).cumsum()).cumsum()

    # count consecutive timesteps without consecutive NaN of more than 'threshold' days
    na_count[na_count > threshold] = np.nan
    value_count = na_count.notnull().astype(int).groupby(na_count.isnull().astype(int).cumsum()).cumsum()

    # binary na identification
    na_binary = na_count.copy()
    na_binary[na_binary > 0] = 1

    # create pool of dates and randomly select dates interval
    pool = value_count[value_count > timesteps].index.values
    if len(pool) < 1:
        sys.exit(f'No consecutive {timesteps} timesteps found without specified NaN interval')

    if year == 'random':
        # check if at least has 75% values
        check, count = False, 0
        while check == False:
            if count > 20:
                sys.exit(f'No consecutive {timesteps} timesteps found with less than 25% NaN')
            end_date = pool[np.random.randint(0, len(pool))]
            end_ID = df.ID.loc[end_date]
            begin_ID = end_ID - timesteps
            if np.all(na_count[begin_ID:end_ID].notnull()) and na_binary[begin_ID:end_ID].sum() < timesteps * 0.25:
                check = True
            count += 1
    elif year == 'last':
        for end_date in reversed(pool):
            end_ID = df.ID.loc[end_date]
            begin_ID = end_ID - timesteps
            if np.all(na_count[begin_ID:end_ID].notnull()) and na_binary[begin_ID:end_ID].sum() < timesteps * 0.25:
                break

    if end_ID == len(df) - 1:
        df_prescreening['last_year'][df_prescreening['station'] == station] = True
        print(f'True:  {station}')
    else:
        print(f'Fasle: {station}')

df_prescreening.to_csv(os.path.join(f'prescreening_station_t0_batch10_LastYearCheck.csv'))
df_prescreening = pd.read_csv('prescreening_station_t0_batch10_LastYearCheck.csv')

# dates = df.index.values[begin_ID:end_ID]
# df = df.iloc[begin_ID:end_ID].copy()
# df.drop(['ID'], axis=1, inplace=True)


kind = 'Time'
ML_list = ['ANN', 'CNN', 'LSTM', 'ConvLSTM']
out_dir = os.path.join('Results', 'Input_complexity')
df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition.csv'))
df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
station_list = df_prescreening['station'][df_prescreening['available'] == True].values
station_excl = ['cuxhaven-825a-germany-uhslc', 'puerto_armuelles_b-304b-panama-uhslc', 'ko_taphao_noi-148a-thailand-uhslc',
                'darwin-168a-australia-uhslc', 'humboldt_bay,_ca-576a-usa-uhslc', 'lord_howe_b-399b-australia-uhslc']
grid_columns = [f'ncells_{i}' for i in np.arange(13)]
var_name_list = ['msl', 'msl_rho', 'msl_rho_uv', 'msl_rho_uv_grad', 'msl_rho_uv_grad_uv2']
df_var_all = pd.DataFrame(index=var_name_list, columns=ML_list)
df_grid_all = pd.DataFrame(index=grid_columns, columns=ML_list)

for ML in ML_list:
    df_var = pd.DataFrame(index=station_list, columns=var_name_list)
    df_grid = pd.DataFrame(index=station_list, columns=grid_columns)
    for station in station_list:
        if (ML == 'ANN' or ML == 'ConvLSTM') and station not in station_excl:
            continue
        if kind == 'CRPSS':
            df = pd.read_excel(os.path.join(out_dir, 'df', f'{station}_input_complexity_{ML}.xlsx'), index_col='Unnamed: 0')
            crps_base = df_dec[df_dec.station == station][f'{ML}_CRPS_value'].values[0]
            df = (df - crps_base) / crps_base * -1 * 100
        else:
            df = pd.read_excel(os.path.join(out_dir, 'df_time', f'{station}_input_complexity_{ML}_Time.xlsx'), index_col='Unnamed: 0')
            df = df / df.loc['msl_rho_uv_grad']['ncells_2']
        df_grid.loc[station] = df.transpose().diff().mean(axis=1)
        df_var.loc[station] = df.diff().mean(axis=1)
    df_grid_all[ML] = df_grid.mean()
    df_var_all[ML] = df_var.mean()
sys.exit(0)