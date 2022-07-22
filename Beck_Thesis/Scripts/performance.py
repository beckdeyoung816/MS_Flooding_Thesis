# -*- coding: utf-8 -*-
#matplotlib.use('Agg')
import cartopy
import cartopy.crs as ccrs
import datetime
import glob
from shapely.geometry import Point
import geopandas as gpd
from math import sqrt
import numpy as np
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,  mean_absolute_error
from sklearn.metrics import precision_score, recall_score, fbeta_score
import properscoring as ps
# from ranky import rankz
import statsmodels.api as sm
import scipy.fftpack
import datetime
import shutil
from distutils.dir_util import copy_tree
import sys

months = mdates.MonthLocator()  # every month
weeks = mdates.WeekdayLocator()  # every month
days = mdates.DayLocator()

def get_coastline_results(stations):
    results = {}
    results['RMSE'] = np.array([station.rmse for station in stations])
    results['Rel_RMSE'] = np.array([station.rel_rmse for station in stations])
    results['NSE'] = np.array([station.NSE for station in stations])
    results['R2'] = np.array([station.r2 for station in stations])
    results['MAE'] = np.array([station.mae for station in stations])
    # results['corrcoef'] = np.array([station.corrcoef for station in stations])
    results['RMSE\nExtremes'] = np.array([station.rmse_ext for station in stations])
    results['Rel RMSE\nExtremes'] = np.array([station.rel_rmse_ext for station in stations])
    results['Precision'] = np.array([station.precision for station in stations])
    results['Recall'] = np.array([station.recall for station in stations])
    results['F_beta'] = np.array([station.fbeta_ext for station in stations])
    
    results_df = pd.DataFrame(index = ['RMSE', 'Rel_RMSE', 'NSE', 'R2', 'MAE', 'RMSE\nExtremes', 'Rel RMSE\nExtremes', 'Precision', 'Recall', 'F_beta'], 
                           columns = ['Min', 'Max', 'Mean', 'Median'])
    
    for metric in results.keys():
        results_df.loc[metric, 'Min'] = np.min(results[metric])
        results_df.loc[metric, 'Max'] = np.max(results[metric])
        results_df.loc[metric, 'Mean'] = np.mean(results[metric])
        results_df.loc[metric, 'Median'] = np.median(results[metric])
    
    return results_df

#**************************************** COASTAL FUNCTIONS ****************************************
def get_ens_class_metrics(station, df_test):
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    obs = df_test['Observed'].values
    ensemble_preds = df_test[filter_col].values
    preds = df_test[filter_col].mean(axis =1).values # Average predictions across all models
    
    rmse, NSE, r2, mae = performance_metrics(obs, preds)
    corrcoef = np.corrcoef(obs, preds)**2
    
    # Calculating RMSE
    rel_rmse = rmse/np.mean(obs)
    
    # Calculating RMSE for Extremes    
    extremes = (pd.DataFrame(obs)
                .nlargest(round(.10*len(obs)), 0) # Largest 10%
                .sort_index())
    min_ext = extremes.iloc[:,0].min() # Minimum of Largest 10% to use as threshold lower bound
    extremes_indices = extremes.index.values
                        
    obs_ext = obs[extremes_indices]
    pred_ext = preds[extremes_indices]

    rmse_ext = np.sqrt(mean_squared_error(obs_ext, pred_ext))
    rel_rmse_ext = rmse_ext / obs.mean()
    
    # Precision and recall and f1 score for extremes
    
    ext_df = pd.DataFrame([obs, preds], index = ['Obs', 'Pred']).T
    ext_df['Extreme_obs'] = ext_df['Obs'] >= min_ext
    ext_df['Extreme_pred'] = ext_df['Pred'] >= min_ext
    
    precision_ext = precision_score(ext_df['Extreme_obs'], ext_df['Extreme_pred'])
    recall_ext = recall_score(ext_df['Extreme_obs'], ext_df['Extreme_pred'])
    fbeta_ext = fbeta_score(ext_df['Extreme_obs'], ext_df['Extreme_pred'], beta=2)
    
    # Store results for the station
    station.rmse = rmse
    station.rel_rmse = rel_rmse
    station.NSE = NSE
    station.r2 = r2
    station.mae = mae
    station.corrcoef = corrcoef
    station.rmse_ext = rmse_ext
    station.rel_rmse_ext = rel_rmse_ext
    station.precision=precision_ext
    station.recall=recall_ext
    station.fbeta_ext=fbeta_ext
    
    return np.around(rmse,4), np.around(rel_rmse,4), np.around(NSE,3), np.around(r2,2), np.around(mae,3), \
        np.around(corrcoef,4), np.around(rmse_ext,4), np.around(rel_rmse_ext,4), np.around(precision_ext,3), np.around(recall_ext,3), np.around(fbeta_ext, 3)
        
def plot_ens_metrics(station, df_test, ax):
    rmse, rel_rmse, NSE, r2, mae, corrcoef, rmse_ext, rel_rmse_ext, precision_ext, recall_ext, fbeta_ext = get_ens_class_metrics(station, df_test)
    
    # ax.set_title('Ensemble Metrics')
    # hide axes
    ax.grid(False)
    ax.axis('off')

    row_labels=['Value']
    col_labels=['RMSE', 'Rel_RMSE', 'NSE', 'R2', 'MAE', 'RMSE\nExtremes', 'Rel RMSE \n Extremes', 'Precision', 'Recall', 'F_beta'] 
    table_vals=[[[rmse], [rel_rmse], [NSE], [r2], [mae], [rmse_ext], [rel_rmse_ext], [precision_ext], [recall_ext], [fbeta_ext]]]
    table_vals=[[rmse, rel_rmse, NSE, r2, mae, rmse_ext, rel_rmse_ext, precision_ext, recall_ext, fbeta_ext]]
    
    row_labels=['Value']
    col_labels=['RMSE', 'Rel_RMSE', 'RMSE\nExtremes', 'Rel RMSE \n Extremes', 'Precision', 'Recall', 'F_beta'] 
    table_vals=[[rmse, rel_rmse,rmse_ext, rel_rmse_ext, precision_ext, recall_ext, fbeta_ext]]

    colcolors = plt.cm.Blues([0.1] *  len(col_labels))
    rowcolors = plt.cm.Oranges([0.1] *  len(row_labels))
    table = ax.table(cellText = table_vals, rowLabels =row_labels, colLabels = col_labels, loc='center',
                     colColours=colcolors, rowColours=rowcolors, colWidths = [0.1]*len(col_labels), cellLoc='center')
    
    cellDict = table.get_celld()
    for i in range(0,len(col_labels)):
        cellDict[(0,i)].set_height(.15)
    table.set_fontsize(12)
    table.scale(2, 2) 



def store_result(inv_yhat, inv_y):
    # plot results
    df_test = pd.DataFrame()
    df_test['Observed'] = inv_y
    df_test['Modelled'] = inv_yhat
    df_test = df_test[['Observed', 'Modelled']].copy()
    return df_test

def ensemble_handler(station, result_dict, station_name, neurons, epochs, batch, resample, tt_value, var_num,
                     out_dir, layers=1, ML='LSTM', test_on='ensemble', plot=True, save=False, loss='mae', i = 0):
    
    train = station.train_test == 'Train'
    
    df_result = result_dict['data'][i].copy()
    df_result.rename(columns = {'Modelled': "Modelled_0"}, inplace = True)
    if train:
        df_train_loss = pd.DataFrame(result_dict['train_loss'][i].copy(), columns = [0])
        df_test_loss = pd.DataFrame(result_dict['test_loss'][i].copy(), columns = [0])
    else:
        df_train_loss, df_test_loss = None, None
    for key in np.arange(1, len(result_dict['data']), 1):
        # print(key)
        df_result = pd.concat([df_result, result_dict['data'][key].rename(columns={'Modelled':f"Modelled_{key}"}).loc[:,f"Modelled_{key}"]], axis = 1)
        if train:    
            df_train_loss = pd.concat([df_train_loss, pd.DataFrame(result_dict['train_loss'][key], columns = [key])], axis = 1, ignore_index=True)      
            df_test_loss = pd.concat([df_test_loss, pd.DataFrame(result_dict['test_loss'][key], columns = [key])], axis = 1, ignore_index=True)    



    df_modelled = df_result.drop('Observed', axis = 1)
    df_result['max'] = df_modelled.max(axis = 1)
    df_result['min'] = df_modelled.min(axis = 1)
    df_result['median'] = df_modelled.median(axis = 1) 
    
    if train:
        cols = [f'Modelled_{col_name}' for col_name in df_train_loss.columns] 
        df_train_loss.columns = cols
        df_test_loss.columns = cols
        
        df_train_loss['max'] = df_train_loss.max(axis = 1)
        df_train_loss['min'] = df_train_loss.min(axis = 1)
        
        df_test_loss['max'] = df_test_loss.max(axis = 1)
        df_test_loss['min'] = df_test_loss.min(axis = 1)
    
    if save == True:
        os.makedirs(os.path.join(out_dir, 'Data'), exist_ok=True)
        
        fn = f'{station_name}_{ML}_{loss}_prediction.csv'       
        df_result.to_csv(os.path.join(out_dir, 'Data', fn))
        if train:
            fn = f'{station_name}_{ML}_{loss}_training.csv'
            df_train_loss.to_csv(os.path.join(out_dir, 'Data',fn))
            
            fn = f'{station_name}_{ML}_{loss}_testing.csv'
            df_test_loss.to_csv(os.path.join(out_dir, 'Data', fn))
    
    if plot == True:
        plot_ensemble_performance(station, df_result, df_train_loss, df_test_loss, station_name, neurons, epochs, batch, resample,
                         tt_value, var_num, out_dir, layers=layers, ML=ML, test_on=test_on, loss=loss)
    
    return df_result, df_train_loss, df_test_loss

def plot_ensemble_performance(station, df, train_loss, test_loss, station_name, neurons, epochs, batch, resample,
                     tt_value, var_num, out_dir, layers=1, ML='LSTM', test_on='ensemble', logger=False, loss='mae'):
    if resample == 'hourly':
        step = 24
    elif resample == 'daily':
        step = 1

    if logger:
        fig = plt.figure(figsize=[14.5, 9.5])
    else:
        fig = plt.figure(figsize=[14.5, 9.5])
    
    
    fig = plt.figure(figsize=[15, 15])

    gs = GridSpec(8, 6)

    plot_ensemble_testing_ts(df, fig.add_subplot(gs[:2, 0:4]))
    plot_ensemble_testing_max_ts(df, fig.add_subplot(gs[2:4, 0:4]), resample)
    plot_meta(fig.add_subplot(gs[:2, 4:]), station_name, neurons, epochs, batch, resample, tt_value, var_num)    
    plot_ensemble_metrics(df.dropna(axis=0, how='any'), fig.add_subplot(gs[2:4, 4:]))
    plot_ensemble_scatter(df[['Observed', 'median']], fig.add_subplot(gs[4:6, :2]))
    plot_ensemble_qq(df.dropna(axis=0, how='any'), fig.add_subplot(gs[4:6, 2:4]))
    if station.train_test == 'Train':
        plot_ensemble_loss(train_loss, test_loss, fig.add_subplot(gs[4:6, 4:]))
    plot_ens_metrics(station, df.dropna(axis=0, how='any'), fig.add_subplot(gs[6:, 1:5]))

    fig.suptitle(station_name, fontsize=32)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # [left, bottom, right, top]
    
    
    fn = f'{station_name}_{ML}_{loss}.png'
    os.makedirs(os.path.join(os.path.split(out_dir)[0], ML, 'Figures'), exist_ok=True)
    plt.savefig(os.path.join(os.path.split(out_dir)[0], ML, 'Figures', fn), dpi=100)
    plt.close()

def plot_ensemble_metrics(df_test, ax):
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    rmse_all = pd.DataFrame(index = filter_col, columns = [0])
    NSE_all = pd.DataFrame(index = filter_col, columns = [0])
    R2_all = pd.DataFrame(index = filter_col, columns = [0])
    mae_all = pd.DataFrame(index = filter_col, columns = [0])
    for i in filter_col:
        inv_y = df_test['Observed'].values
        inv_yhat = df_test[i].values
        rmse_all.loc[i,0], NSE_all.loc[i,0], R2_all.loc[i,0], mae_all.loc[i,0] = performance_metrics(inv_y, inv_yhat) #Add print statement?

    rmse_all.loc['median',0] = np.around(float(rmse_all.median(axis = 0)),4)
    rmse_all.loc['max',0] = np.around(float(rmse_all.max(axis = 0)),4)
    rmse_all.loc['min',0] = np.around(float(rmse_all.min(axis = 0)),4)  
    NSE_all.loc['max',0] = np.around(float(NSE_all.max(axis = 0)),4)  
    NSE_all.loc['min',0] = np.around(float(NSE_all.min(axis = 0)),4)         
    R2_all.loc['max',0] = np.around(float(R2_all.max(axis = 0)),4)  
    R2_all.loc['min',0] = np.around(float(R2_all.min(axis = 0)),4)      
    mae_all.loc['max',0] = np.around(float(mae_all.max(axis = 0)),4)  
    mae_all.loc['min',0] = np.around(float(mae_all.min(axis = 0)),4)  
    NSE_all.loc['median',0] = np.around(float(NSE_all.median(axis = 0)),4)
    R2_all.loc['median',0] = np.around(float(R2_all.median(axis = 0)),4)
    mae_all.loc['median',0] = np.around(float(mae_all.median(axis = 0)),4)      
    # hide axes
    ax.grid(False)
    ax.axis('off')

    col_labels=['median', 'max', 'min']
    row_labels=['RMSE', 'MAE', 'R2']
    table_vals=[[rmse_all.loc['median',0], rmse_all.loc['max',0], rmse_all.loc['min',0]],
                [mae_all.loc['median',0], mae_all.loc['max',0], mae_all.loc['min',0]],
                [R2_all.loc['median',0], R2_all.loc['max',0], R2_all.loc['min',0]]]

    colcolors = plt.cm.Blues([0.1, 0.1, 0.1])
    rowcolors = plt.cm.Oranges([0.1, 0.1, 0.1])
    table = ax.table(cellText=table_vals, colWidths = [0.1]*4, rowLabels=row_labels, loc='center',
                     colLabels=col_labels, colColours=colcolors, rowColours=rowcolors)
    table.set_fontsize(14)
    table.scale(2, 2) 

def plot_ensemble_testing_ts(df_result, ax, station='', crps_val = None):    
    ax.fill_between(df_result.index, df_result['max'], df_result['min'], color='r', alpha = 0.5)
    ax.plot(df_result['median'].index, df_result['median'], '-r', linewidth=0.5)
    ax.plot(df_result['Observed'].index, df_result['Observed'], '-k', linewidth=0.3)
    #ax.set_ylabel('Surge height (m)')
    locator = mdates.AutoDateLocator(tz=None, minticks=10, maxticks=16, interval_multiples=False)
    formatter = mdates.ConciseDateFormatter(locator, formats=['%b', '%b', '%b', '%b', '%b', '%b'], zero_formats=['%b', '%b', '%b', '%b', '%b', '%b'], show_offset=False)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)    
    ax.annotate(station, xy=(0.01,0.01),xycoords='axes fraction', fontsize = 7, fontweight='bold')
    if crps_val is not None:
        ax.annotate("CRPS = {}".format(str(crps_val)), xy=(0.80,0.90),xycoords='axes fraction', fontsize = 5, fontweight='bold')        
    ax.tick_params(axis='x', which='major', labelsize=6)
    # ax.xaxis.set_minor_locator(weeks)

def plot_ensemble_testing_max_ts(df_result, ax, resample):    
    max_time = df_result['Observed'].idxmax()  # .notnull()
    
    ax.fill_between(df_result.index, df_result['max'], df_result['min'], color='r', alpha = 0.5)
    ax.plot(df_result['median'].index, df_result['median'], '-r', linewidth=0.8)
    ax.plot(df_result['Observed'].index, df_result['Observed'], '-k', linewidth=0.5)

    ax.set_ylabel('Surge height (m)')
    ax.set_xlabel(None) 
    
    if resample == 'daily':
        start_date = max_time - pd.offsets.Day(14)
        end_date = max_time + pd.offsets.Day(15)
    else:
        start_date = max_time - pd.offsets.Day(4)
        end_date = max_time + pd.offsets.Day(4)
        ax.xaxis.set_major_locator(days)

    ax.set_xlim([start_date, end_date])
    # ax.xaxis.set_minor_locator(days)

#plot_scatter(df_result[['Observed', 'median']], ax)
def plot_ensemble_qq(df_test, ax):
    q_obs = df_test['Observed'].quantile(np.arange(0.01, 1.00, 0.01)).values
    q_mod = pd.DataFrame()
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    for i in filter_col:
        q_mod_i = df_test[i].quantile(np.arange(0.01, 1.00, 0.01)).values
        q_mod = pd.concat([q_mod, pd.DataFrame(q_mod_i, columns = [i])], axis = 1)
    q_mod['min'] = q_mod.min(axis = 1)
    q_mod['max'] = q_mod.max(axis = 1)
    
    ax.plot(q_obs, q_mod['min'], '-', c='k', linewidth=0.5)
    ax.plot(q_obs, q_mod['max'], '-', c='k', linewidth=0.5)
    ax.fill_between(q_obs, q_mod['min'], q_mod['max'], color='k', alpha = 0.5)
    ax.plot([q_obs.min(), q_obs.max()], [q_obs.min(), q_obs.max()], color='k', linewidth=0.4)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_xlabel('Observed')
    ax.set_ylabel('Modelled')

def plot_ensemble_loss(train_loss, test_loss, ax):
    # plot history
    for col in train_loss.columns:
        ax.plot(train_loss.loc[:,col], color='orange', linewidth=0.5)
        ax.plot(test_loss.loc[:,col], color='blue', linewidth=0.5)
    ax.set_ylabel('Loss value')
    ax.set_xlabel('epoch number')
    #ax.legend()
    
def plot_ensemble_scatter(df_test, ax):
    median_ts = df_test['median'].copy()
    obs_ts = df_test['Observed'].copy()   
#    df_test.plot.scatter(x='Observed', y='median', ax=ax, c='k', alpha = 0.6, edgecolors = 'none', s=1)
#    x = np.linspace(*ax.get_xlim())
    ax.plot(obs_ts, median_ts, markersize=1, marker = '.', color ='k')
    ax.plot(obs_ts, obs_ts, 'k', linewidth=0.45)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_xlabel('Observed')
    ax.set_ylabel('Median')
    

def handler(inv_yhat, inv_y, df, n_train, train_loss, test_loss, station, neurons, epochs,
            batch, resample, tt_value, variables, n_layers, ML, out_dir, test_on='self'):
    # plot results
    df_test = df.iloc[n_train:].copy()
    df_test['Observed'] = inv_y
    df_test['Modelled'] = inv_yhat
    df_test = df_test[['Observed', 'Modelled']].copy()
    plot_performance(df_test, train_loss, test_loss, station, neurons, epochs, batch, resample,
                    tt_value, len(variables), out_dir, layers=n_layers, ML=ML, test_on=test_on)
    return df_test

def plot_performance(df, train_loss, test_loss, station, neurons, epochs, batch, resample,
                     tt_value, var_num, out_dir, layers=1, ML='LSTM', test_on='self'):
    """ This function makes a plot of the results of the run of the Neural network. It includes
        plots for all predicted data, one year data, scatter plot, Q-Q plot, loss value plot,
        performance metrics and metadata of each run.

    Parameters
    ----------
    df : [pd.DataFrame]
        [should include datetime as index and columns decribing 'Observed' and 'Modelled' data]
    train_loss : [array]
        [array of loss value of training data]
    test_loss : [array]
        [array of loss value of test data]
    station : [string]
        [name of station]
    neurons : [int]
        [amount of neurons used in model run]
    epochs : [int]
        [amount of epochs used in model run]
    batch : [int]
        [size used in model run]
    resample : [string]
        ['daily' or 'hourly']
    tt_value : [float]
        [percentage of test/train data: 0.67]
    var_num : [int]
        [number of variables used]
    workspace : str, optional
        [assign workspace to save image], by default 'cur'
    layers : int, optional
        [description], by default 1
    ml : str, optional
        [type of machine learning], by default LSTM
    """  
    if resample == 'hourly':
        step = 24
    elif resample == 'daily':
        step = 1

    fig = plt.figure(figsize=[14.5, 9.5]) 
    gs = GridSpec(3, 3)

    plot_series(df, 'all', step, fig.add_subplot(gs[0, 0:2]))
    plot_meta(fig.add_subplot(gs[0, 2]), station, neurons, epochs, batch, resample, tt_value, var_num)
    plot_series(df, 'year', step, fig.add_subplot(gs[1, 0:2]))
    plot_metrics(df, step, fig.add_subplot(gs[1, 2]))
    plot_scatter(df, fig.add_subplot(gs[2, 0]))
    plot_qq(df, fig.add_subplot(gs[2, 1]))
    plot_loss(train_loss, test_loss, fig.add_subplot(gs[2, 2]))

    fig.suptitle(station, fontsize=32)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # [left, bottom, right, top]
    fn = '{}_{}_n{}_e{}_b{}_var{}_{}_l{}_t_{}.png'.format(station, ML, neurons, epochs, batch, var_num, resample, layers, test_on)
    plt.savefig(os.path.join(out_dir, fn), dpi=100)
    plt.close()

def plot_meta(ax, station, neurons, epochs, batch, resample, tt_value, var_num):
    # hide axes
    ax.grid(False) 
    ax.axis('off')

    col_labels=['Metadata']
    row_labels=['Neurons', 'Batch', 'Epochs', 'tt_value', 'resample', 'variables']
    table_vals=[[neurons], [batch], [epochs], [tt_value], [resample], [var_num]]
    
    # the rectangle is where I want to place the table
    colors = plt.cm.Oranges([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table = ax.table(cellText=table_vals, colWidths = [0.1]*3, rowLabels=row_labels,
                     loc='center', rowColours=colors)
    table.set_fontsize(14)
    table.scale(2.5, 2.5) 

def plot_metrics(df_test, step, ax):
    inv_y = df_test['Observed'].values
    inv_yhat = df_test['Modelled'].values
    rmse, NSE, R2, mae = performance_metrics(inv_y, inv_yhat)
    rmse_y, NSE_y, R2_y, mae_y = performance_metrics(inv_y[-365*step:], inv_yhat[-365*step:])

    rmse, NSE, R2, mae = round(rmse,4), round(NSE,3), round(R2,3), round(mae,3) 
    rmse_y, NSE_y, R2_y, mae_y = round(rmse_y,4), round(NSE_y,3), round(R2_y,3), round(mae_y,3)
    # hide axes
    ax.grid(False)
    ax.axis('off')

    col_labels=['All', 'year']
    row_labels=['RMSE', 'MAE', 'R2']
    table_vals=[[rmse, rmse_y], [mae, mae_y], [R2, R2_y]]

    colcolors = plt.cm.Blues([0.1, 0.1])
    rowcolors = plt.cm.Oranges([0.1, 0.1, 0.1])
    table = ax.table(cellText=table_vals, colWidths = [0.1]*3, rowLabels=row_labels, loc='center',
                     colLabels=col_labels, colColours=colcolors, rowColours=rowcolors)
    table.set_fontsize(14)
    table.scale(2, 2) 
    
def plot_series(df_test, s_range, step, ax):
    if s_range == 'all':
        df_test[['Observed', 'Modelled']].plot(ax=ax)
    elif s_range == 'year':
        df_test[['Observed', 'Modelled']].iloc[-365*step:].plot(ax=ax)
    ax.set_ylabel('Surge height')
    ax.set_xlabel(None)

def plot_loss(train_loss, test_loss, ax):
    # plot history
    ax.plot(train_loss, label='train')
    ax.plot(test_loss, label='test')
    ax.set_ylabel('Loss value')
    ax.set_xlabel('epoch number')
    ax.legend()

def plot_scatter(df_test, ax):
    df_test.plot.scatter(x='Observed', y='Modelled', ax=ax, c='k', alpha = 0.6, edgecolors = 'none')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, 'r')

def plot_qq(df_test, ax):
    q_obs = df_test['Observed'].quantile(np.arange(0.01, 1.00, 0.01)).values
    q_mod = df_test['Modelled'].quantile(np.arange(0.01, 1.00, 0.01)).values
    ax.scatter(q_obs, q_mod, c='k')
    ax.plot([q_obs.min(), q_obs.max()], [q_obs.min(), q_obs.max()], color='red')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Modelled')

def performance_metrics(inv_y, inv_yhat):
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #print('Test RMSE: ', rmse)

    # calculate NSE
    df_nse = pd.DataFrame({'model': inv_yhat, 'observed': inv_y})
    df_nse['dividend'] = np.power((df_nse['model'] - df_nse['observed']), 2)
    df_nse['divisor'] = np.power((df_nse['observed'] - df_nse['observed'].mean()), 2)
    NSE = 1 - (df_nse['dividend'].sum() / df_nse['divisor'].sum())
    #print('Test NSE: ', NSE)

    # calculate R2
    r2 = r2_score(inv_y, inv_yhat)
    #print('Test R2: ', r2)

    # calculate max error
    mae = mean_absolute_error(inv_y, inv_yhat)
    #print('Test mean absolute error: ', mae)


    return rmse, NSE, r2, mae

def IDF_cont_station(data, sel_duration=[], fig_plot = False):
    '''
    ts should be a pandas series
    sel_duration mentions the duration in hours for which to calculate the exc. prob. 
    '''     
    ts = pd.DataFrame(data)
    if len(sel_duration)==0:
        sel_duration = np.arange(1,round(data.shape[0]/100),1)
        
    for i in sel_duration:
        print(i)
        ts.loc[:,str(int(i))] = ts.loc[:,ts.columns[0]].rolling(window = i).mean()

    #For each duration, we rank the results
    ts.drop(ts.columns[0], axis = 1, inplace = True)
    ts.reset_index(inplace = True, drop = True)
    rank_ts = ts.rank(axis=0, method='average', numeric_only=True, ascending=False)
    excprob_ts = rank_ts/(ts.shape[0]+1)

    if fig_plot == True:
        plt.figure()
        plt.plot(excprob_ts.iloc[:,0],ts.iloc[:,0], '.b', label = str(ts.columns[0]))
        plt.plot(excprob_ts.iloc[:,sel_duration[round(len(sel_duration)/2)]],ts.iloc[:,sel_duration[round(len(sel_duration)/2)]], '.r', label = str(ts.columns[sel_duration[round(len(sel_duration)/2)]]))
        plt.plot(excprob_ts.iloc[:,sel_duration[-1]-1],ts.iloc[:,sel_duration[-1]-1], '.g', label = str(ts.columns[sel_duration[-1]-1]))
        plt.xscale('log')
        plt.ylabel('Surge (m)')
        plt.xlabel('Exc. prob.')
        plt.legend()
        plt.show()    
    return ts, excprob_ts

def IDF_continuous_comparison_station(df_test, sel_duration=[], fig_plot = False):
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    obs = df_test['Observed']  
    mod = df_test[filter_col].mean(axis =1) #Ensemble mean
    
    if len(sel_duration)==0:
        sel_duration = np.arange(1,round(df_test.shape[0]/100),1)
    
    ts_obs, excprob_obs = IDF_cont_station(obs, sel_duration=sel_duration, fig_plot = False)
    ts_mod, excprob_mod = IDF_cont_station(mod, sel_duration=sel_duration, fig_plot = False)

    if fig_plot == True:
        plt.figure()
        plt.plot(excprob_obs.iloc[:,0].sort_values(ascending=True),ts_obs.iloc[:,0].sort_values(ascending=False), '.-b', label = str(ts_obs.columns[0]))
        plt.plot(excprob_obs.iloc[:,sel_duration[round(len(sel_duration)/2)]].sort_values(ascending=True),ts_obs.iloc[:,sel_duration[round(len(sel_duration)/2)]].sort_values(ascending=False), '.-r', label = str(ts_obs.columns[sel_duration[round(len(sel_duration)/2)]]))
        plt.plot(excprob_obs.iloc[:,sel_duration[-1]-1].sort_values(ascending=True),ts_obs.iloc[:,sel_duration[-1]-1].sort_values(ascending=False), '.-g', label = str(ts_obs.columns[sel_duration[-1]-1]))
        
        plt.plot(excprob_mod.iloc[:,0].sort_values(ascending=True),ts_mod.iloc[:,0].sort_values(ascending=False), '.--b')
        plt.plot(excprob_mod.iloc[:,sel_duration[round(len(sel_duration)/2)]].sort_values(ascending=True),ts_mod.iloc[:,sel_duration[round(len(sel_duration)/2)]].sort_values(ascending=False), '.--r')
        plt.plot(excprob_mod.iloc[:,sel_duration[-1]-1].sort_values(ascending=True),ts_mod.iloc[:,sel_duration[-1]-1].sort_values(ascending=False), '.--g')
        
        plt.legend(title='Duration (hours)')
        plt.xscale('log')
        plt.ylabel('Surge (m)')
        plt.xlabel('Exc. prob.')
        plt.grid()
        plt.show()           
    
    return ts_obs, excprob_obs, ts_mod, excprob_mod

def IDF_range_station(df_test, sel_duration=[], fig_plot = False):
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    obs = df_test['Observed']
    
    if len(sel_duration)==0:
        sel_duration = np.arange(1,round(df_test.shape[0]/100),1)
    
    ts_obs, excprob_obs = IDF_cont_station(obs, sel_duration=sel_duration, fig_plot = False)
    
    res={}
    for dur in sel_duration:
        res[dur] = {}
        print(dur)
        res_values = pd.DataFrame(columns = filter_col)
        for mod in df_test[filter_col]:
            print(mod)
            sel = df_test[mod]
            ts_mod, excprob_mod = IDF_cont_station(sel, sel_duration=[dur], fig_plot = False)
            excprob_mod.rename(columns={excprob_mod.columns[0]:'exc_prob'}, inplace = True)
            both = pd.concat([ts_mod, excprob_mod], axis = 1)
            both.sort_values('exc_prob', axis = 0, inplace = True)
            res_values[mod] = both.iloc[:,0].values  
        minval = res_values.min(axis=1).rename('minval')
        maxval = res_values.max(axis=1).rename('maxval')
        vals = pd.concat([pd.DataFrame(both['exc_prob']).reset_index(drop=True), minval,maxval], axis = 1)
        res[dur] = vals

    if fig_plot == True:
        fig, ax = plt.subplots()
        for dur in sel_duration:
            ax.plot(excprob_obs.iloc[:,0].sort_values(ascending=True),ts_obs.iloc[:,0].sort_values(ascending=False), '.-k', label = str(ts_obs.columns[0]))
            #ax.plot(res[dur].exc_prob, res[dur].minval, '-r')
            ax.fill_between(res[dur].exc_prob, res[dur].minval, res[dur].maxval, color='r', alpha = 0.5)
        plt.xscale('log')
        plt.ylabel('Surge (m)')
        plt.xlabel('Exc. prob.')
            
def energy_spectrum_plot(data, nfft, Fs = 1/(60*60), plot_fig = False):
    """data is the time series
    nfft is the length of one block
    nfft = len(data) > raw estimate
        
    Fs is the sampling frequency in (/s)"""
    # data = df_result.iloc[:,0]
    # Fs = 1/(60*60) #in Hz
    # nfft = len(data) #raw estimate
    # nfft = nfft/2

    data = np.array(data)
    n = len(data)    
    nfft = int(nfft - (nfft%2))
    nBlocks = int(n/nfft) 
    data_new = data[0:nBlocks*nfft] 
    dataBlock = np.reshape(data_new,(nBlocks,nfft)) 
    df = Fs/nfft
    f = np.arange(0,Fs/2+df,df)
    fId = np.arange(0,len(f))
    
    fft_data = scipy.fftpack.fft(dataBlock,n = nfft,axis = 1)
    fft_data = fft_data[:,fId]
    A = 2.0/nfft*np.real(fft_data)
    B = 2.0/nfft*np.imag(fft_data)
    E = (A**2 + B**2)/2
    E = np.mean(E, axis = 0)/df
    
    if plot_fig == True:
        plt.figure()
        plt.plot((1/f)/3600,E,'-k')
        plt.xlim([1,7])
        plt.xticks(ticks=[2,3,4,5,6,12,24,48,7*24], labels=['2','3', '4','5','6','12','24','48','7\n days'])
        plt.yscale('log')
        plt.xscale('log')       
    
    return E,f #, ConfLow, ConfUpper > not done here

def plot_ensemble_spectrum(df, ax, n=2, station=''):
    filter_col = [col for col in df if col.startswith('Modelled')]
    
    E_res = pd.DataFrame(data = None, columns=filter_col)
    f_res = pd.DataFrame(data = None, columns=filter_col)
    for model_col in filter_col:
        data = df[model_col].values
        E,f = energy_spectrum_plot(data, len(data)/n, Fs = 1/(60*60), plot_fig = False)
        E = np.delete(E,0)
        f = np.delete(f,0)
        E_res[model_col] = E
        f_res[model_col] = f
        #ax.plot((1/f)/3600,E,'-r', linewidth = 0.3)
        del data, E, f 
    E_resmax = E_res.max(axis=1)
    E_resmin = E_res.min(axis=1)
    ax.fill_between((1/f_res.iloc[:,0].values)/3600,E_resmax, E_resmin, color='r', alpha = 0.3)

    data = df[filter_col].median(axis=1).values
    E,f = energy_spectrum_plot(data, len(data)/n, Fs = 1/(60*60), plot_fig = False)
    E = np.delete(E,0)
    f = np.delete(f,0)       
    ax.plot((1/f)/3600,E,'-r', linewidth = 0.7)    
    del data, E, f
    
    data = df['Observed'].values
    E,f = energy_spectrum_plot(data, len(data)/n, Fs = 1/(60*60), plot_fig = False)
    E = np.delete(E,0)
    f = np.delete(f,0)    
    ax.plot((1/f)/3600,E,'-k', linewidth = 0.7)    
    
    ax.set_xlim(2,7*24)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([2,3,4,5,6,12,24,48,7*24])
    ax.set_xticklabels(['2','3', '4','5','6','12','24','48','7\n days'])
    #ax.grid(which ='both', axis = 'both')    
    ax.set_ylabel('Energy spectrum [m$^2$/Hz]')
    ax.annotate(station, xy=(0.03,0.03),xycoords='axes fraction', fontsize = 7, fontweight='bold')  
    return #E_res, f_res
    

def ens_metrics(df_test, model='ensemble'):
    if model == 'ensemble':
        filter_col = [col for col in df_test if col.startswith('Modelled')]
        obs = df_test['Observed'].values
        mods = df_test[filter_col].values
        mod = df_test[filter_col].mean(axis =1).values
    elif model == 'persistence':
        obs = df_test['Observed'].values
        mod = df_test['Modelled'].values
    
    rmse, NSE, r2, mae = performance_metrics(obs, mod)
    corrcoef = np.corrcoef(obs, mod)**2
    
    return rmse, NSE, r2, mae, corrcoef

def crps_metrics(df_test, model='ensemble', mean=True):
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
    
    model_score = ps.crps_ensemble(obs, mods) #returns a value per timestep. Here we take the mean of those values to get one value per location
    if mean:
        model_score = model_score.mean()
    return model_score

def crps_metrics_cont(df_test, model='ensemble'):
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
    
    model_score = ps.crps_ensemble(obs, mods) #returns a value per timestep. Here we take the mean of those values to get one value per location
    return model_score

def ignorance_metric(df_test):
    #NOT RUNNING
    # Also called logartihmic score.
    # See this paper for a discussion of the 3 scores: https://arxiv.org/pdf/1908.08980.pdf
    
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    obs = df_test['Observed'].values
    mods = df_test[filter_col].values
    
    ls = pd.DataFrame(data=None, index= np.arange(0, len(obs)), columns = [0])
    i=0
    for obs_i, mods_i in zip(obs, mods):
        kde = sm.nonparametric.KDEUnivariate(mods_i)
        kde.fit()
        delta = kde.support[1]-kde.support[0]        
        # fig = plt.figure(figsize=(12, 5))
        # ax = fig.add_subplot(111)
        
        # ax.plot(kde.support, kde.density*delta, lw=3, label='KDE from samples', zorder=10)
        
        prob_obs_i=np.interp(obs_i, kde.support, kde.density)
        print(prob_obs_i)
        # if prob_obs_i == 0:
        #     prob_obs_i = 0.001
        ls.iloc[i,0] = -np.log2(prob_obs_i)
        i+=1        
    return ls

def calc_skill_rel(baseline_score, forecast_score):
    # Skill calculates the improvement of the forecast compared with a baseline forecast (ANN) in our case  
    # X% skill means that the forecast is X% better/worse than the baseline model (ANN)
    skill = (forecast_score - baseline_score) / baseline_score    
    return skill

def calc_skill_abs(baseline_score, forecast_score):
    # Skill calculates the improvement of the forecast compared with a baseline forecast (ANN) in absolute terms
    skill = forecast_score - baseline_score 
    return skill

def hyper_opt_results(option, ML_list):
    # load prescreening check
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values

    df_results = pd.DataFrame(columns=['station', 'ML', 'Trial-ID', 'Status', 'Iteration',
                                       'hidden', 'neurons', 'Objective'])  # 'filters', 
    # 'batch', 'dropout', 'l2', 'neurons'
    for station in station_list:
        for ML in ML_list:
            df = pd.read_csv(os.path.join('Hyper_opt', 'complexity', station, ML, 'results.csv'))
            if option == 'mode':
                row = df.sort_values('Objective').head(10).mode().iloc[0].values
            elif option == 'best':
                row = df.loc[df['Objective'].argmin()].values
            elif option == 'ranked':
                sel = df.sort_values('Objective').head(10)
                sel['rank'] = range(10, 0, -1)
                neurons = sel[['neurons', 'rank']].groupby('neurons').sum().idxmax().values
                hidden = sel[['hidden', 'rank']].groupby('hidden').sum().idxmax().values
                # l2 = sel[['l2', 'rank']].groupby('l2').sum().idxmax().values
                # dropout = sel[['dropout', 'rank']].groupby('dropout').sum().idxmax().values
                # batch = sel[['batch', 'rank']].groupby('batch').sum().idxmax().values
                filters = 0
                if ML == 'CNN':
                    filters = sel[['filters', 'rank']].groupby('filters').sum().idxmax().values
                row = np.array([0, 'INTERMEDIATE', 0, hidden, neurons, 0])
                # row = np.array([0, 'INTERMEDIATE', 0, batch, dropout, l2, neurons, 0])
            row = np.insert(row, 0, [station, ML], axis=0)
            df_results.loc[len(df_results)] = row

    for hyper in ['neurons', 'hidden']:  # 'batch', 'dropout', 'l2', 'neurons', 'filters'
        hyper_dict = {}
        hyper_dict['station'] = range(len(station_list))
        for ML in ML_list:
            if ML == 'LSTM' and hyper == 'filters':
                continue
            a = df_results[df_results.ML == ML].sort_values('station')[hyper].values
            if hyper == 'neurons':
                a = [1 if i == 24 else 2 if i == 48 else 3 if i == 96 else 4 for i in a]
            elif hyper == 'l2':
                # a = [1 if i == 0 else 2 if i == 0.001 else 3 if i == 0.01 else 4 for i in a]
                a = [1 if i == 0.001 else 2 if i == 0.01 else 3 for i in a]
            elif hyper == 'dropout':
                a = [1 if i == 0 else 2 if i == 0.1 else 3 if i == 0.2 else 4 for i in a]
            elif hyper == 'batch':
                a = [1 if i == 240 else 2 if i == 1200 else 3 for i in a]
            elif hyper == 'filters':
                a = [1 if i == 8 else 2 if i == 16 else 24 for i in a]
            elif hyper == 'hidden':
                a = [1 if i == 1 else 2 if i == 2 else 3 if i == 3 else 4 if i == 4 else 5 for i in a]
            hyper_dict[ML] = a
        print(hyper_dict)
        df_hyper = pd.DataFrame.from_dict(hyper_dict)
        # df_hyper.set_index('station')
        df_hyper.plot(x='station', y=ML_list)
        plt.title(hyper)
        if hyper == 'neurons':
            plt.yticks([1, 2, 3, 4], ['24', '48', '96', '192'])
        elif hyper == 'l2':
            # plt.yticks([1, 2, 3, 4], ['0', '0.001', '0.01', '0.1'])
            plt.yticks([1, 2, 3], ['0.001', '0.01', '0.1'])
        elif hyper == 'dropout':
            plt.yticks([1, 2, 3, 4], ['0', '0.1', '0.2', '0.5'])
        elif hyper == 'batch':
            plt.yticks([1, 2, 3], ['240', '1200', '2400'])
        elif hyper == 'filters':
            plt.yticks([1, 2, 3], ['8', '16', '24'])
        plt.xticks(range(len(station_list)), station_list, rotation=30)
        # plt.show()
        plt.savefig(os.path.join('Hyper_opt', f'lineplot_ic_{option}_{hyper}.png'), dpi=100)
        plt.close()
        
        df_hist = df_hyper.drop(['station'], axis=1)
        df_hist = df_hist.apply(pd.Series.value_counts)
        df_hist.reset_index().plot(x='index', y=ML_list, kind="bar")

        plt.title(hyper)
        if hyper == 'neurons':
            plt.xticks([0, 1, 2], ['24', '48', '96'])
        elif hyper == 'l2':
            # plt.xticks([0, 1, 2, 3], ['0', '0.001', '0.01', '0.1'])
            plt.xticks([0, 1, 2], ['0.001', '0.01', '0.1'])
        elif hyper == 'dropout':
            plt.xticks([0, 1, 2, 3], ['0', '0.1', '0.2', '0.5'])
        elif hyper == 'batch':
            plt.xticks([0, 1, 2], ['240', '1200', '2400'])
        elif hyper == 'filters':
            plt.xticks([0, 1, 2], ['8', '16', '24'])
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(os.path.join('Hyper_opt', 'complexity', f'histogram_ic_{hyper}_{option}_{ML}.png'), dpi=100)
        plt.close()
        
def plot_stations(list_station, model_type='LSTM', save_fig=False):
    #If statement for lenght not correct

    # os.chdir(r'E:\github\ML_COAST\Cartesius')
    # stations = pd.read_csv('prescreening_station_parametrization.csv', index_col = 0)
    # list_station = list(stations.iloc[[0,3,6,7,11,12],0])
    
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    spot = 0
    
    plt.rcParams.update({'font.size': 7})
    plt.rcParams.update({'xtick.major.size':1.5})
    plt.rcParams.update({'xtick.major.pad':1})
    plt.rcParams.update({'ytick.major.size':1.5})
    plt.rcParams.update({'ytick.major.pad':1})    
    plt.rcParams.update({'axes.labelpad':0})
    # plt.rcParams.update({'axes.axisbelow':False})
    
    fig = plt.figure(figsize=[8, 11])
    gs = GridSpec(6, 3, left=0.05, bottom=0.1, right=0.95, top=0.90, hspace=0.20, width_ratios=[2,1,1], height_ratios=[1,1,1,1,1,1]) #, wspace=0.20, hspace=0.30, width_ratios=None, height_ratios=[0.9,0.9,0.9,0.9,0.9,0.9])
    for station in list_station:
        print(station)            
        df = pd.read_csv(os.path.join('Models','Ensemble_run',station,model_type,station+'_'+model_type+'_prediction.csv'), index_col = 'time', date_parser = date_parser)
        df.dropna(axis=0, how='any', inplace = True)
        
        crps_val = crps_metrics(df, model='ensemble')
        crps_val = np.around(np.float(crps_val),4)       
        
        ax1 = fig.add_subplot(gs[spot, 0])
        plot_ensemble_testing_ts(df, ax1, station=station, crps_val = crps_val)
        ax2 = fig.add_subplot(gs[spot, 1])
        plot_ensemble_scatter(df[['Observed', 'median']], ax2)
        ax3 = fig.add_subplot(gs[spot, 2])
        plot_ensemble_qq(df, ax3)
        spot += 1
        del ax1, ax2, ax3
        
    if save_fig == True:
        filename = os.path.join('Figures', 'stations_set1.png')
        plt.savefig(filename, dpi=300)
        plt.close()
    
def plot_stations_energy_spectrum(list_station, model_type='LSTM', save_fig=False):
    #If statement for lenght not correct

    # os.chdir(r'E:\github\ML_COAST\Cartesius')
    # stations = pd.read_csv('prescreening_station_parametrization.csv', index_col = 0)
    # list_station = list(stations.iloc[[0,3,6,7,11,12],0])
    
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    spot = 0
    
    plt.rcParams.update({'font.size': 5})
    plt.rcParams.update({'xtick.major.size':1.5})
    plt.rcParams.update({'xtick.major.pad':1})
    plt.rcParams.update({'ytick.major.size':1.5})
    plt.rcParams.update({'ytick.major.pad':1})    
    plt.rcParams.update({'axes.labelpad':0})
    
    fig = plt.figure(figsize=[8, 11])
    gs = GridSpec(6, 1, left=0.05, bottom=0.1, right=0.95, top=0.90, hspace=0.20) 
    for station in list_station:
        print(station)            
        df_result = pd.read_csv(os.path.join('Models','Ensemble_run',station,model_type,station+'_'+model_type+'_prediction.csv'), index_col = 'time', date_parser = date_parser)
        df_result.dropna(axis=0, inplace = True)
        ax = fig.add_subplot(gs[spot, 0])
        plot_ensemble_spectrum(df_result, ax, n=4, station=station)
        spot += 1
    if save_fig == True:
        filename = os.path.join('Figures', 'stations_set1_spectrum.png')
        plt.savefig(filename, dpi=300)
        plt.close()    

def plot_world_metric(metric, save=True):
    df = pd.read_csv(os.path.join('Results', 'Global_performance_metrics_CM.csv'))
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
    r = ''
    ML_list = ['ANN', 'CNN', 'LSTM', 'ConvLSTM']
    cmap = 'plasma'
    label = f'{metric}'
    if metric == 'CRPS':
        for ML in ML_list:
            df[f'{ML}_CRPS'] = df[f'{ML}_CRPS'].values * 100
        bounds = np.arange(0, 11, 1)  # np.arange(0.0, 10.5, 0.5)
    elif metric == 'Reliability':
        bounds = np.arange(0, 5.5, 0.5)
    elif metric == 'MAE' or metric == 'RMSE':
        bounds = np.array([0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    elif metric == 'CM' or metric == 'Resolution':
        bounds = np.array([-100, -75, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 75, 100])
        cmap = 'RdYlGn'
        label = 'CRPSS'
    else:
        bounds = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        r = '_r'

    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    fontsize = 15
    ML_list = ['ANN', 'LSTM', 'CNN', 'ConvLSTM']
    abcd_list = ['a', 'b', 'c', 'd']
    subplot_list = [221, 222, 223, 224]
    xtick_list = [False, False, True, True]
    ytick_list = [True, False, True, False]

    # plt.figure(figsize=(12, 6))
    fig = plt.figure(figsize=[19.3, 9.5])
    isel = 0
    for ML, abcd in zip(ML_list, abcd_list):
        ax = fig.add_subplot(subplot_list[isel], projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        if metric == 'Reliability':
            weights = df_dec[f'{ML}_CRPS_Reliability'].values * 100
        elif metric == 'Resolution':
            resolution = df[f'CM_CRPS'].values - df_dec[f'{ML}_CRPS_Potential'].values
            weights = resolution / df_dec[f'{ML}_CRPS_value'].values * 100
        elif metric == 'CM':
            weights = (df[f'{ML}_CRPS'].values - df[f'{metric}_CRPS'].values) / df[f'{metric}_CRPS'].values * -1 * 100
        else:
            weights = df[f'{ML}_{metric}'].values
        ax.scatter(df.Lon.values, df.Lat.values, c=weights, cmap=cmap + r, norm=norm,
                   edgecolors='k', linewidths=0.5, s=15)
        ax.set_ylim([-60, 90])
        # ax.set_xlim([-185, 185])
        ax.set_xticks([-120, -60, 0, 60, 120])
        if not xtick_list[isel]:
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_xticklabels(['', '', '', '', ''])
        else:
            ax.set_xticklabels(ax.get_xticks(), {'size': fontsize})
        ax.set_yticks([-60, -30, 0, 30, 60])
        if not ytick_list[isel]:
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_yticklabels(['', '', '', '', ''])
        else:
            ax.set_yticklabels(ax.get_yticks(), {'size': fontsize})
        ax.set_title(ML, size=fontsize, weight='bold')
        ax.grid(alpha=0.5)  # clip_box=ax.bbox
        ax.text(0.03, 0.065, abcd, transform=ax.transAxes, weight='bold', size=20,
                bbox=dict(facecolor='none', edgecolor='black'))
        isel += 1
    
    cax = fig.add_axes([0.222, 0.06, 0.6, 0.02])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap + r, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=bounds)  # pad=0.2)
    cbar.set_label(label, size=fontsize, weight='bold')
    cbar.ax.tick_params(axis='x', which='minor')
    cbar.ax.set_xticklabels([str(i) for i in bounds], size=fontsize, minor=True)

    plt.subplots_adjust(wspace=0.1, hspace=-0.1)
    plt.tight_layout(rect=[0, 0.08, 1, 1]) # [left, bottom, right, top] rect=[0, 0, .93, 1]

    if save:
        filename = os.path.join('Figures', f'Global_performance_{metric}_4panel.png')

        if os.path.isfile(filename):
            os.remove(filename)
        fig.savefig(filename, dpi=300)
        plt.close()

def calc_resolution():
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition.csv'), index_col='station')
    df = pd.read_csv(os.path.join('Results', 'Global_performance_metrics_CM.csv'), index_col='Unnamed: 0')
    df_dec = df_dec.merge(df['CM_CRPS'], how='left', left_index=True, right_index=True)
    df_dec = df_dec.rename(columns={"CM_CRPS": "Uncertainty"})
    df_dec = df_dec.sort_index()
    ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']
    for ML in ML_list:
        df_dec[f'{ML}_CRPS_Resolution'] = df_dec[f'Uncertainty'].values - df_dec[f'{ML}_CRPS_Potential'].values
    
    df_dec['Best_CRPS'] = np.nan
    df_dec['Best_Reliability'] = np.nan
    df_dec['Best_Potential'] = np.nan
    df_dec['Best_Resolution'] = np.nan
    sel_columns = [f'{ML}_CRPS_value' for ML in ML_list]
    for station in df.index.values:
        NN = df_dec.loc[station][sel_columns].astype(float).idxmin()[:-11]
        df_dec.loc[station, f'Best_CRPS'] = df_dec.loc[station][f'{NN}_CRPS_value']
        df_dec.loc[station, f'Best_Reliability'] = df_dec.loc[station][f'{NN}_CRPS_Reliability']
        df_dec.loc[station, f'Best_Potential'] = df_dec.loc[station][f'{NN}_CRPS_Potential']
        df_dec.loc[station, f'Best_Resolution'] = df_dec.loc[station][f'{NN}_CRPS_Resolution']
    
    df_dec.to_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv')) 

def plot_world(weight, save=True):
    # https://stackoverflow.com/questions/60807254/align-a-cartopy-2d-map-plot-with-a-1d-line-plot
    # Figure best CRPSS Fig 2
    # Figure best NN Fig 3
    df = pd.read_csv(os.path.join('Results', 'Global_performance_metrics_CM.csv'))
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
    ML_list = ['LSTM', 'CNN', 'ConvLSTM', 'ANN']
    if weight == 'Best_NN':
        bounds = np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        c_list = ['b', 'r', 'gold', 'g']
        ticks = [0, 1, 2, 3, 4, 5]
        minorlocator = 1
    elif weight == 'Uncertainty':
        bounds = np.arange(0, 11, 1)
        weights = df[f'CM_CRPS'].values * 100
        cmap = 'plasma'
        label = 'Uncertainty (cm)'
        ticks = [0, 2, 4, 6, 8, 10]
        minorlocator = 1
    elif weight == 'Resolution':
        bounds = np.arange(0, 11, 1)
        cmap = 'plasma'
        label = 'Resolution (cm)'
        weights = df_dec[f'Best_Resolution'].values * 100
        ticks = [0, 2, 4, 6, 8, 10]
        minorlocator = 1
    elif weight == 'Best_CRPSS':
        bounds = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])
        cmap = 'RdYlGn'
        label = 'CRPSS'
        ticks = [-80, -40, 0, 40, 80]
        minorlocator = 10
        weights = (df[f'Best_CRPS_val'].values - df[f'CM_CRPS'].values) / df[f'CM_CRPS'].values * -1 * 100

    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    array = [
        [1, 1, 1, 1, 1, 2],
        [1, 1, 1, 1, 1, 2],
        [1, 1, 1, 1, 1, 2],
        [3, 3, 3, 3, 3, 0]
    ]

    import proplot as plot
    fig, ax = plot.subplots(array, proj={1:'pcarree'}, width=7, space=0.1)  # 'eqearth', 'pcarree'
    ax[0].add_feature(cartopy.feature.LAND, edgecolor='black')
    
    if weight == 'Best_NN':
        df['weights'] = np.nan
        df['colors'] = np.nan
        count = 1
        hs = []
        for ML, c in zip(ML_list, c_list):
            df_scatter = df[df[f'Best_CRPS'] == f'{ML}_CRPS']
            df['weights'][df[f'Best_CRPS'] == f'{ML}_CRPS'] = count
            count += 1
            h = ax[0].scatter(df_scatter.Lon.values, df_scatter.Lat.values, c=c, label=ML, edgecolors='k', linewidths=0.5, s=15)  # , label=df[f'Best_CRPS'].values
            hs.append(h)
        ax[0].legend(hs, loc='top', ncols=4, fancybox=True)# frame=True)
        # ax[0].legend_wrapper(loc='l')
    else:
        df['weights'] = weights
        sca = ax[0].scatter(df.Lon.values, df.Lat.values, c=weights, cmap=cmap, norm=norm, edgecolors='k', linewidths=0.5, s=15)
        ax[0].colorbar(sca, loc='l', extend='both', label=label)


    df_zonal = df[['weights', 'Lat']].sort_values(by=['Lat'])
    ax[1].scatter(df_zonal['weights'], df_zonal.Lat.values, edgecolors='k', s=1, facecolors='white')
    df_meridional = df[['weights', 'Lon']].sort_values(by=['Lon'])
    ax[2].scatter(df_meridional.Lon.values, df_meridional['weights'], edgecolors='k', s=1, facecolors='white')
    plt.gca().invert_xaxis()
 
    if weight is not 'Best_NN':
        df_zonal['weights_ma'] = df_zonal['weights'].rolling(10, center=True).mean()
        ax[1].plot(df_zonal['weights_ma'], df_zonal.Lat.values, c='k', linewidth=2)
        df_meridional['weights_ma'] = df_meridional['weights'].rolling(10, center=True).mean()
        ax[2].plot(df_meridional.Lon.values, df_meridional['weights_ma'], c='k', linewidth=2)
    else:
        ML_list.insert(0, '')
        ML_list.append('')
        ax[1].set(xticks=[1, 2, 3, 4], xticklabels=ML_list)
        ax[1].format(xminorlocator='null', xrotation=90)
        ax[2].set(yticks=[1, 2, 3, 4], yticklabels=ML_list)
        ax[2].format(yminorlocator='null')
        # ax[2].invert_xaxis()

    # ax.format(abc=True, abcloc='l', abcstyle='(a)')
    ax[1].set(xlim=[ticks[0], ticks[-1]], ylabel=None, yticks=[-90, -60, -30, 0, 30, 60, 90], #yminorlocator=10,
              yticklabels=['90°S','60°S','30°S','0°','30°N','60°N','90°N'], ylim=[-90, 90],
              xticks=ticks)
    ax[1].format(yminorlocator=minorlocator)
    ax[1].yaxis.tick_right()

    ax[2].set(ylim=[ticks[0], ticks[-1]], ylabel=None, xticks=[-180, -120, -60, 0, 60, 120, 180], #yminorlocator=10,
              xticklabels=['180°W','120°W','60°W','0°','60°E','120°E','180°E'], xlim=[-180, 180],
              yticks=ticks)
    ax[1].format(xminorlocator=minorlocator)

    if weight == 'Best_NN':
        ax[1].invert_xaxis()

    # plt.subplots_adjust(wspace=0.1, hspace=-0.1)
    # plt.tight_layout(rect=[0, 0.08, 1, 1]) # [left, bottom, right, top] rect=[0, 0, .93, 1]

    if save:
        filename = os.path.join('Figures', f'Global_performance_{weight}.png')

        if os.path.isfile(filename):
            os.remove(filename)
        fig.savefig(filename, dpi=300, facecolor='white')
        plt.close()

def plot_world_4panel(save=True, fontsize=20, var='Decomposition'):
    # 4-panel CRPSS NN Fig S2
    # 4-panel CRPS decomposition best NN Fig S3
    # 4-panel Reliability NN Fig S4
    df = pd.read_csv(os.path.join('Results', 'Global_performance_metrics_CM.csv'))
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
    if var == 'Decomposition':
        bounds = np.arange(0, 10.1, 0.2)
        bounds1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cbar_label = f'CRPS (decomposition) (cm)'
        cmap = 'gnuplot2_r'
        extend = 'both'
        name = f'Global_{var}_Best_NN.png'
    elif var == 'CRPSS':
        bounds = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80])
        bounds1 = bounds
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cbar_label = f'CRPSS'
        cmap = 'RdYlGn'
        extend = 'both'
        name = f'Global_{var}_NN.png'
    elif var == 'Reliability':
        bounds = np.arange(0, 10.1, 0.2)
        bounds1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cbar_label = f'Reliability (cm)'
        cmap = 'gnuplot2_r'
        extend = 'max'
        name = f'Global_{var}_NN.png'

    ML_list = ['ANN', 'LSTM', 'CNN', 'ConvLSTM']
    abcd_list = ['a', 'b', 'c', 'd']
    subplot_list = [221, 222, 223, 224]
    xtick_list = [False, False, True, True]
    ytick_list = [True, False, True, False]

    metric_list = ['CRPS', 'Uncertainty', 'Reliability', 'Resolution']  # 'corrcoef', 'R2', 'NSE', 'NNSE'

    fig = plt.figure(figsize=[19.3, 10.5])
    isel = 0
    for metric, abcd, ML in zip(metric_list, abcd_list, ML_list):
        ax = fig.add_subplot(subplot_list[isel], projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        if var == 'Decomposition':
            if metric == 'Uncertainty':
                rpt = df_dec[f'{metric}'].values * 100
            else:
                rpt = df_dec[f'Best_{metric}'].values * 100
        elif var == 'CRPSS':
            rpt = (df[f'{ML}_CRPS'].values - df[f'CM_CRPS'].values) / df[f'CM_CRPS'].values * -1 * 100
        elif var == 'Reliability':
            rpt = df_dec[f'{ML}_CRPS_{var}'].values * 100
        ax.scatter(df.Lon.values, df.Lat.values, cmap=cmap, norm=norm, edgecolors='k', linewidths=0.5, c=rpt, s=30)

        # ax.set_ylim([-60, 90])
        # ax.set_xlim([-185, 185])
        ax.set_xticks([-120, -60, 0, 60, 120])
        if not xtick_list[isel]:
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_xticklabels(['', '', '', '', ''])
        else:
            ax.set_xticklabels(['120°W','60°W','0°','60°E','120°E'], {'size': fontsize})  # ax.get_xticks()
        ax.set_yticks([-60, -30, 0, 30, 60])
        if not ytick_list[isel]:
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_yticklabels(['', '', '', '', ''])
        else:
            ax.set_yticklabels(['60°S','30°S','0°','30°N','60°N'], {'size': fontsize})  # ax.get_yticks()
        if var == 'Decomposition':
            ax.set_title(metric, size=fontsize, weight='bold')
        else:
            ax.set_title(ML, size=fontsize, weight='bold')
        ax.grid(alpha=0.5)  # clip_box=ax.bbox
        ax.text(0.03, 0.065, abcd, transform=ax.transAxes, weight='bold', size=20,
                bbox=dict(facecolor='none', edgecolor='black'))
        isel += 1

    cax = fig.add_axes([0.221, 0.075, 0.6, 0.02])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=bounds1, extend=extend)  # pad=0.2)
    cbar.set_label(cbar_label, size=fontsize, weight='bold')
    cbar.ax.tick_params(axis='x', which='minor')
    cbar.ax.set_xticklabels([str(i) for i in bounds1], size=fontsize)
    
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # [left, bottom, right, top] rect=[0, 0, .93, 1]

    if save:
        filename = os.path.join('Figures', name)

        if os.path.isfile(filename):
            os.remove(filename)
        fig.savefig(filename, dpi=300)
        plt.close()

def plot_crps_2panel(metric='Best', fontsize=15, save=True):
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
    df = pd.read_csv(os.path.join('Results', 'Global_performance_metrics.csv'))

    abcd_list = ['a', 'b']
    subplot_list = [121, 122]
    if metric == 'Best':
        metric_list = ['Best', 'Resolution']
    elif metric == 'Absolute':
        metric_list = ['CRPS', 'Uncertainty']

    fig = plt.figure(figsize=[19.3, 5.5]) # 5.13
    bounds = np.arange(0, 11, 1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = 'plasma'
    ML_list = ['ANN', 'LSTM', 'CNN', 'ConvLSTM']
    c_list = ['g', 'b', 'r', 'gold']

    for i in range(2):
        ax = fig.add_subplot(subplot_list[i], projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')

        if metric_list[i] == 'Best':
            for ML, c in zip(ML_list, c_list):
                df_scatter = df[df[f'Best_CRPS'] == f'{ML}_CRPS']
                # print(f'{metric} {ML}: {len(df_scatter)}')
                ax.scatter(df_scatter.Lon.values, df_scatter.Lat.values, c=c, label=ML,
                        edgecolors='k', linewidths=0.5, s=15)
        else:
            if metric_list[i] == 'Resolution':
                weights = df_dec[f'Best_Resolution'].values * 100
            elif metric_list[i] == 'Uncertainty':
                weights = df_dec[f'Uncertainty'].values * 100
            elif metric_list[i] == 'CRPS':
                weights = df_dec[f'Best_CRPS'].values * 100
            ax.scatter(df.Lon.values, df.Lat.values, c=weights, cmap=cmap, norm=norm,
                       edgecolors='k', linewidths=0.5, s=15)
        ax.set_ylim([-60, 90])
        # ax.set_xlim([-185, 185])
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_xticklabels(ax.get_xticks(), {'size': fontsize})
        ax.set_yticks([-60, -30, 0, 30, 60])
        if i == 1:
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_yticklabels(['', '', '', '', ''])
        else:
            ax.set_yticklabels(ax.get_yticks(), {'size': fontsize})
        ax.set_title(metric_list[i], size=fontsize, weight='bold')
        ax.grid(alpha=0.5)  # clip_box=ax.bbox
        ax.text(0.03, 0.065, abcd_list[i], transform=ax.transAxes, weight='bold', size=20,
                bbox=dict(facecolor='none', edgecolor='black'))
    

    cax = fig.add_axes([0.222, 0.09, 0.6, 0.04])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")  # pad=0.2)
    cbar.set_label(f'CRPS', size=fontsize, weight='bold')
    cbar.ax.tick_params(axis='x', which='minor')
    cbar.ax.set_xticklabels([str(i) for i in bounds], size=fontsize, minor=True)

    plt.subplots_adjust(wspace=0.1, hspace=-0.1)
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # [left, bottom, right, top] rect=[0, 0, .93, 1]

    if save:
        filename = os.path.join('Figures', f'Global_decomposition_CRPS_{metric}_2panel.png')

        if os.path.isfile(filename):
            os.remove(filename)
        fig.savefig(filename, dpi=300)
        plt.close()

def plot_crps_dist(station, fontsize=15, save=True):
    fn_ens = 'Models/Ensemble_run'
    ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    fig = plt.figure(figsize=[19.3, 9.5])
    # count = 1
    # for ML in ML_list:
    ML = 'LSTM'
    fn = os.path.join(fn_ens, station, ML, station + '_' + ML + '_prediction.csv')
    df = pd.read_csv(fn, index_col='time', date_parser=date_parser)
    df = df.dropna(axis=0, how='any')
    crps = crps_metrics(df, mean=False)

    ax = fig.add_subplot(111)
    ax.scatter(df['Observed'].values, crps, c='grey', edgecolors='k', linewidths=0.5)
    ax.set_title('Threshold decomposition', size=fontsize, weight='bold')

    if save:
        filename = os.path.join('Figures', f'CRPS_threshold.png')
        if os.path.isfile(filename):
            os.remove(filename)
        fig.savefig(filename, dpi=300)
        plt.close()

def rank_hist(df_test):
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    obs = df_test['Observed'].values
    ensemble = np.transpose(df_test[filter_col].values)
    mask = obs != np.nan
    result = rankz(obs, ensemble, mask)
    prob =  result[0] / np.sum(result[0])
    return prob

def rank_plot():
    ML_list = ['ConvLSTM', 'ANN', 'LSTM', 'CNN']
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    station_names = ['Anchorage', 'Boston', 'Callao', 'Cuxhaven', 'Dakar', 'Darwin', 'Dunkerque',
                    'Honolulu', 'Humboldt', 'Ko Taphao', 'Lord Howe', 'Puerto Armuelles',
                    'San Francisco', 'Wakkanai', 'Zanzibar']
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    for ML in ML_list:
        fig, axs = plt.subplots(3, 5, constrained_layout=True)
        for i, ax in enumerate(axs.flat):
            station = station_list[i]
            name = station_names[i]
            # station = 'cuxhaven-825a-germany-uhslc'
            fn_exp = os.path.join('Models', 'hyper_complexity', 'Ensemble_run', station, ML)
            df_pred = pd.read_csv(os.path.join(fn_exp, f'{station}_{ML}_prediction.csv'), index_col='time', date_parser=date_parser)
            # result = rank_hist(df_pred)
            df_test = df_pred.copy()
            # df_test = df_pred[df_pred['Observed'] > df_pred['Observed'].quantile(0.95)]
            prob = rank_hist(df_test)
            ax.bar(range(1, 22), prob)
            ax.set_title(name, fontsize=6)
        filename = os.path.join('Results', f'Rank_histogram_15_{ML}_hyper.png')
        if os.path.isfile(filename):
            os.remove(filename)
        plt.savefig(filename, dpi=300)
        plt.close()

def table_var_test():
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    var_ilist = [[0], [0, 2, 3], [0, 5, 6], [0, 2, 3, 5, 6],
                 [0, 4], [0, 2, 3, 4], [0, 4, 5, 6], [0, 2, 3, 4, 5, 6],
                 [0, 1, 4], [0, 1, 2, 3, 4], [0, 1, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
    var_name_list = ['msl', 'msl_uv', 'msl_uv2', 'msl_uv_uv2',
                     'msl_rho', 'msl_rho_uv', 'msl_rho_uv2', 'msl_rho_uv_uv2',
                     'msl_rho_grad', 'msl_rho_grad_uv', 'msl_rho_grad_uv2', 'msl_rho_grad_uv_uv2']
    variables = ['msl', 'grad', 'u10', 'v10', 'rho', 'uquad', 'vquad']
    ML_list = ['LSTM', 'CNN']

    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    metric = pd.DataFrame(index=station_list, data=None)
    df_loc = pd.read_excel(os.path.join('Stations', 'stations.xlsx'))
    df_loc = df_loc.set_index('Station')
    metric_tmp = metric.merge(df_loc[['Lat', 'Lon']], how='left', left_index=True, right_index=True)
    # station_list = ['puerto_armuelles_b-304b-panama-uhslc', 'cuxhaven-825a-germany-uhslc',
    #                 'lord_howe_b-399b-australia-uhslc', 'humboldt_bay,_ca-576a-usa-uhslc',
    #                 'dakar_e-223e-senegal-uhslc']
    for ML in ML_list:
        metric = metric_tmp.copy()
        for station in station_list:
            for var_isel, var_name in zip(var_ilist, var_name_list):
                var_sel = np.array(variables)[var_isel].tolist()
                fn_ens = os.path.join('Models', 'Var_test', var_name, 'Ensemble_run')
                fn = os.path.join(fn_ens, station, ML, station + '_' + ML + '_prediction.csv')
                df_result = pd.read_csv(fn, index_col='time', date_parser=date_parser)
                model = 'ensemble'    
                metric.loc[station, var_name] = crps_metrics(df_result.dropna(axis=0, how='any'), model=model)
        metric = metric.dropna()
        metric.to_excel(f'test_variables_{ML}.xlsx')

def table_hyper_complexity():
    ML_list = ['ConvLSTM', 'ANN', 'LSTM', 'CNN']
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition_three.csv'))
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    for ML in ML_list:
        df_results = pd.DataFrame()
        for station in station_list:
            df_hyper = pd.read_csv(os.path.join('Hyper_opt', 'complexity', station, ML, 'results.csv'))
            df_hyper = df_hyper.loc[df_hyper['Objective'].argmin()]
            df_hyper['Station'] = station
            fn_exp = os.path.join('Models', 'hyper_complexity', 'Ensemble_run', station, ML)
            df_pred = pd.read_csv(os.path.join(fn_exp, f'{station}_{ML}_prediction.csv'), index_col='time', date_parser=date_parser)
            crps = crps_metrics(df_pred.dropna(axis=0, how='any'))
            uncertainty = df_dec[df_dec['station'] == station][f'Uncertainty'].values[0]
            df_hyper['CRPSS'] = (uncertainty - crps) / uncertainty * 100
            crps_base = df_dec[df_dec.station == station][f'{ML}_CRPS_value'].values[0]
            crpss_base = (uncertainty - crps_base) / uncertainty * 100
            df_hyper['Change_CRPSS'] = df_hyper['CRPSS'] - crpss_base
            prob = rank_hist(df_pred)
            hr_hyper = 1 - (prob[0] + prob[-1])

            fn_exp = os.path.join('Models', 'Ensemble_run', station, ML)
            df_pred = pd.read_csv(os.path.join(fn_exp, f'{station}_{ML}_prediction.csv'), index_col='time', date_parser=date_parser)
            prob = rank_hist(df_pred)
            hr_normal = 1 - (prob[0] + prob[-1])
            df_hyper['HR_change'] = (hr_hyper - hr_normal) * 100
            df_results = df_results.append(df_hyper)
        df_results = df_results.drop(['Iteration', 'Objective', 'Status', 'Trial-ID'], axis=1)
        df_results = df_results.set_index('Station')
        del df_results.index.name
        df_results.to_excel(os.path.join('Results', f'Complexity_hyper_{ML}_table_all.xlsx'))
        select = ['anchorage-9455920-usa-noaa', 'cuxhaven-825a-germany-uhslc','dunkerque-dunkerque-france-refmar',
                  'honolulu_b,hawaii-057b-usa-uhslc', 'puerto_armuelles_b-304b-panama-uhslc', 'san_francisco,ca-551a-usa-uhslc']
        df_6 = df_results.loc[select]
        df_6.to_excel(os.path.join('Results', f'Complexity_hyper_{ML}_table_6.xlsx'))
        df_9 = df_results.drop(select, axis=0)
        df_9.to_excel(os.path.join('Results', f'Complexity_hyper_{ML}_table_9.xlsx'))

def table_input_complexity(time=False):
    n_ncells = np.arange(13)
    ML_list = ['ConvLSTM', 'ANN']  # 'ConvLSTM', 'ANN', 'LSTM', 'CNN'
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    # station_list = ['callao_b-093b-peru-uhslc']
    # station_list = ['san_francisco,ca-551a-usa-uhslc', 'zanzibar-151a-tanzania-uhslc']
    var_name_list = ['msl', 'msl_rho', 'msl_rho_uv', 'msl_rho_uv_grad', 'msl_rho_uv_grad_uv2']
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    if time:
        out_dir = os.path.join('Results', 'Input_complexity', 'df_time')
    else:
        out_dir = os.path.join('Results', 'Input_complexity', 'df')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for station in station_list:
        print(station)
        for ML in ML_list:
            print(ML)
            df_results = pd.DataFrame(index=var_name_list, data=None)
            for cells in n_ncells:
                for var_name in var_name_list:
                    fn_exp = os.path.join('Models', 'Input_complexity', f'ncells_{cells}', var_name, 'Ensemble_run', station, ML)
                    if time:
                        diff = -1
                        count = 1
                        while diff < 0:
                            f1 = os.stat(os.path.join(fn_exp, f'{ML}_ensemble_{count}')).st_mtime
                            f20 = os.stat(os.path.join(fn_exp, f'{ML}_ensemble_20')).st_mtime
                            diff = (f20 - f1) / 60 / (20 - count)
                            count += 1
                        df_results.loc[var_name, f'ncells_{cells}'] = diff
                    else:
                        df_pred = pd.read_csv(os.path.join(fn_exp, f'{station}_{ML}_prediction.csv'), index_col='time', date_parser=date_parser)
                        df_results.loc[var_name, f'ncells_{cells}'] = crps_metrics(df_pred.dropna(axis=0, how='any'))
            if time:
                df_results.to_excel(os.path.join(out_dir, f'{station}_input_complexity_{ML}_Time.xlsx'))
            else:
                df_results.to_excel(os.path.join(out_dir, f'{station}_input_complexity_{ML}.xlsx'))

def complexity_model_time():
    ML_list = ['ConvLSTM', 'ANN', 'LSTM', 'CNN']  # 'ConvLSTM', 'ANN', 'LSTM', 'CNN'
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    var_name_list = ['msl_rho_uv_grad', 'msl_rho_uv_grad_uv2']
    n_ncells = np.array([2, 12])
    date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    for ML in ML_list:
        print(ML)
        df_results = pd.DataFrame(index=var_name_list, data=None)
        for cells, var_name in zip(n_ncells, var_name_list):
            ctime = 0
            for station in station_list:
                fn_exp = os.path.join('Models', 'Input_complexity', f'ncells_{cells}', var_name, 'Ensemble_run', station, ML)
                diff = -1
                count = 1
                while diff < 0:
                    f1 = os.stat(os.path.join(fn_exp, f'{ML}_ensemble_{count}')).st_mtime
                    f20 = os.stat(os.path.join(fn_exp, f'{ML}_ensemble_20')).st_mtime
                    diff = (f20 - f1) / 60 / (20 - count)
                    count += 1
                ctime += diff
                # df_results.loc[var_name, f'ncells_{cells}'] = diff
            print(f'{cells} {var_name}: {ctime / 15}')

def average_model_time():
    df_prescreening = pd.read_csv('prescreening_station_t0_batch10.csv', index_col='station')
    station_list = df_prescreening[df_prescreening['available'] == True].index.values
    df = df_prescreening[df_prescreening['available'] == True]
    ML_list = ['ConvLSTM', 'ANN', 'LSTM', 'CNN']
    for ML in ML_list:
        df[ML] = np.nan
    for station in station_list:
        for ML in ML_list:
            fn_exp = os.path.join('Models', 'Ensemble_run', station, ML)
            diff = -1
            count = 1
            while diff < 0:
                try:
                    f1 = os.stat(os.path.join(fn_exp, f'{ML}_ensemble_{count}')).st_mtime
                except FileNotFoundError as e:
                    count += 1
                    if count > 20:
                        sys.exit(f'Check station: {station} and ML {ML}')
                    continue
                f20 = os.stat(os.path.join(fn_exp, f'{ML}_ensemble_20')).st_mtime
                diff = (f20 - f1) / 60 / (20 - count)
                count += 1
            print(count, station)
            df.loc[station, ML] = diff
    df.to_excel(os.path.join('Results', f'Models_Computation_Time.xlsx'))
    for ML in ML_list:
        print(f'Average time {ML}: {df[ML].mean()}')

def input_complexity_change(kind='CRPSS'):
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
              
def plot_input_complexity(kind='CRPS', panel=6, fontsize=5):
    ML_list = ['LSTM', 'CNN']
    ML_list = ['ANN', 'CNN', 'LSTM', 'ConvLSTM']
    out_dir = os.path.join('Results', 'Input_complexity')
    df_dec = pd.read_csv(os.path.join('Results', 'CRPS_Decomposition.csv'))
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    if panel == 6:
        station_list = ['anchorage-9455920-usa-noaa', 'cuxhaven-825a-germany-uhslc', 'dunkerque-dunkerque-france-refmar',
                        'honolulu_b,hawaii-057b-usa-uhslc', 'puerto_armuelles_b-304b-panama-uhslc', 'san_francisco,ca-551a-usa-uhslc']
        station_name = ['Anchorage', 'Cuxhaven', 'Dunkerque', 'Honolulu', 'Puerto Armuelles', 'San Francisco']
    elif panel == 9:
        station_list = ['boston,ma-741a-usa-uhslc', 'callao_b-093b-peru-uhslc', 'dakar_e-223e-senegal-uhslc',
                        'darwin-168a-australia-uhslc', 'humboldt_bay,_ca-576a-usa-uhslc', 'ko_taphao_noi-148a-thailand-uhslc',
                        'lord_howe_b-399b-australia-uhslc', 'wakkanai-wakkanai-japan-jma', 'zanzibar-151a-tanzania-uhslc']
        station_name = ['Boston', 'Callao', 'Dakar', 'Darwin', 'Humboldt', 'Ko Taphao', 'Lord Howe',
                        'Wakkanai', 'Zanzibar']

    if kind == 'Time':
        df_dir = 'df_time'
        fn_add = '_Time'
    else:
        df_dir = 'df'
        fn_add = ''

    
    if kind == 'CRPSS' or kind == 'Time':
        cmap = sns.cm.rocket_r
        cmap = 'RdYlGn'
    else:
        cmap = sns.cm.rocket

    ML_count = 0
    station_count = 0
    fig, axs = plt.subplots(panel, 4, constrained_layout=True)
    for i, ax in enumerate(axs.flat):
        if ML_count > 3:
            ML_count = 0
            station_count += 1
        ML = ML_list[ML_count] 
        station = station_list[station_count]
        # vmin, vmax = -80, 80
        if ML == 'ANN':
            vmin, vmax = 1000, 0
            for MLi in ML_list:
                # if (MLi == 'ANN' or MLi == 'ConvLSTM') and station not in station_excl:
                #     continue
                df = pd.read_excel(os.path.join(out_dir, df_dir, f'{station}_input_complexity_{MLi}{fn_add}.xlsx'), index_col='Unnamed: 0')
                if kind == 'CRPS':
                    df = df * 100
                elif kind == 'CRPSS':
                    crps_base = df_dec[df_dec.station == station][f'{ML}_CRPS_value'].values[0]
                    df = (df - crps_base) / crps_base * -1 * 100
                elif kind == 'Time':
                    df = df / df.loc['msl_rho_uv_grad']['ncells_2']
                if vmin > df.min().min():
                    vmin = df.min().min()
                if vmax < df.max().max():
                    vmax = df.max().max()
        if station == 'anchorage-9455920-usa-noaa':
            vmax = 15
        norm = colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
        ML_count += 1
        
        if station_count == 0:
            ax.title.set_text(f'{ML}')
        if ML == 'ANN':
            ax.set_ylabel(station_name[station_count], fontsize=5)

        print(station, ML)
        df = pd.read_excel(os.path.join(out_dir, df_dir, f'{station}_input_complexity_{ML}{fn_add}.xlsx'), index_col='Unnamed: 0')
        if kind == 'CRPSS':
            crps_base = df_dec[df_dec.station == station][f'{ML}_CRPS_value'].values[0]
            df = (df - crps_base) / crps_base * -1 * 100
        elif kind == 'CRPS':
            df = df * 100
        df.index = [1,2,3,4,5]
        df.columns = np.arange(13)
        df = df.iloc[::-1]
        if ML == 'ConvLSTM':
            im = sns.heatmap(df, ax=ax, norm=norm, cbar=False, cmap=cmap)#, , cbar=True
                            #  cbar_kws={'label': kind})
            # cbar = ax.collections[0].colorbar

            mtop = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            ax_cbar = axs.flat[4 * (1 + station_count) - 4 : 4 * (1 + station_count)]
            cbar = fig.colorbar(mtop, ax=ax_cbar, orientation='vertical', pad=-0.005)
            if panel == 9:
                locator, formatter = cbar._get_ticker_locator_formatter()
                locator.set_params(nbins=6)
                cbar.update_ticks()
            cbar.ax.tick_params(labelsize=fontsize)
        else:
            im = sns.heatmap(df, ax=ax, norm=norm, cbar=False, cmap=cmap)  # , vmin=vmin, vmax=vmax
        
        if station_count < (panel - 1):
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_xticklabels(['']*13)
        else:
            ax.set_xticks(np.arange(0.5, 13, 1))
            ax.set_xticklabels([str(t) for t in np.arange(13) * 0.25 * 2 + 0.25], rotation=90,
                               fontsize=fontsize)
        
        if ML == 'ANN':
            ax.set_yticks(np.arange(0.5, 5, 1))
            ax.set_yticklabels([7, 5, 4, 2, 1], fontsize=fontsize, rotation=0) # 1, 2, 4, 5, 7
            ax.set_ylabel(station_name[station_count], fontsize=fontsize)
        else:
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_yticklabels(['', '', '', '', ''])
    
    ###
    # plt.subplots_adjust(wspace=0.05)  # , hspace=0.01
    # plt.tight_layout(rect=[0, 0, 0.9, 1]) # [left, bottom, right, top] rect=[0, 0, .93, 1]
    # plt.tight_layout()
    ###
    
    filename = os.path.join(out_dir, f'Input_complexity_{panel}-panel_{kind}.png')
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename, dpi=300)
    plt.close()

def learning_curve_spatial():
    ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']
    out_dir = os.path.join('Results', 'Input_complexity')
    station_list = ['anchorage-9455920-usa-noaa', 'cuxhaven-825a-germany-uhslc', 'dunkerque-dunkerque-france-refmar',
                    'honolulu_b,hawaii-057b-usa-uhslc', 'puerto_armuelles_b-304b-panama-uhslc', 'san_francisco,ca-551a-usa-uhslc']
    station_name = ['Anchorage', 'Cuxhaven', 'Dunkerque', 'Honolulu', 'Puerto Armuelles', 'San Francisco']
    station_list = ['boston,ma-741a-usa-uhslc', 'callao_b-093b-peru-uhslc', 'dakar_e-223e-senegal-uhslc',
                    'darwin-168a-australia-uhslc', 'humboldt_bay,_ca-576a-usa-uhslc', 'ko_taphao_noi-148a-thailand-uhslc',
                    'lord_howe_b-399b-australia-uhslc', 'wakkanai-wakkanai-japan-jma', 'zanzibar-151a-tanzania-uhslc']
    station_name = ['Boston', 'Callao', 'Dakar', 'Darwin', 'Humboldt', 'Ko Taphao', 'Lord Howe',
                    'Wakkanai', 'Zanzibar']
    for ML in ML_list:
        df_hur = pd.DataFrame()
        df_other = pd.DataFrame()
        for station in station_list:
            df = pd.read_excel(os.path.join(out_dir, f'{station}_input_complexity_{ML}.xlsx'), index_col='Unnamed: 0')
            print(df)
            sys.exit(0)

def prepare_sup_dataset():
    df_prescreening = pd.read_csv('prescreening_station_t0_batch10.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    ML_list = ['CNN', 'LSTM', 'ANN', 'ConvLSTM']
    for station in station_list:
        for ML in ML_list:
            src = os.path.join('Models', 'Ensemble_run', station, ML)
            dst = os.path.join('Zenodo', 'Models', station, ML)
            if not os.path.exists(dst):
                os.makedirs(dst)
            copy_tree(src, dst)

def rename_sup_dataset():
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