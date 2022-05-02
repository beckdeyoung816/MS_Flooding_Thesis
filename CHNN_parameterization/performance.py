# -*- coding: utf-8 -*-
"""
Performance of ANN

Timothy Tiggeloven and Anaïs Couasnon
"""

import matplotlib
matplotlib.use('Agg')

from math import sqrt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,  mean_absolute_error
import properscoring as ps
import statsmodels.api as sm




months = mdates.MonthLocator()  # every month
weeks = mdates.WeekdayLocator()  # every month
days = mdates.DayLocator()

def store_result(inv_yhat, inv_y):
    # plot results
    df_test = pd.DataFrame()
    df_test['Observed'] = inv_y
    df_test['Modelled'] = inv_yhat
    df_test = df_test[['Observed', 'Modelled']].copy()
    return df_test

def ensemble_handler(result_dict, station, neurons, epochs, batch, resample, tt_value, var_num, out_dir, layers=1, ML='LSTM', test_on='ensemble', plot = True, save = False):
    df_result = result_dict['data'][0].copy()
    df_result.rename(columns = {'Modelled': "Modelled_0"}, inplace = True)
    
    df_train = pd.DataFrame(result_dict['train_loss'][0].copy(), columns = [0])
    df_test = pd.DataFrame(result_dict['test_loss'][0].copy(), columns = [0])
    
    for key in np.arange(1, len(result_dict['data']),1):
        # print(key)
        df_result = pd.concat([df_result, result_dict['data'][key].rename(columns={'Modelled':f"Modelled_{key}"}).loc[:,f"Modelled_{key}"]], axis = 1)    
        df_train = pd.concat([df_train, pd.DataFrame(result_dict['train_loss'][key], columns = [key])], axis = 1, ignore_index=True)      
        df_test = pd.concat([df_test, pd.DataFrame(result_dict['test_loss'][key], columns = [key])], axis = 1, ignore_index=True)    

    cols = [f'Modelled_{col_name}' for col_name in df_train.columns] 
    df_train.columns = cols
    df_test.columns = cols

    df_modelled = df_result.drop('Observed', axis = 1)
    df_result['max'] = df_modelled.max(axis = 1)
    df_result['min'] = df_modelled.min(axis = 1)
    df_result['median'] = df_modelled.median(axis = 1) 
    
    df_train['max'] = df_train.max(axis = 1)
    df_train['min'] = df_train.min(axis = 1)
    
    df_test['max'] = df_test.max(axis = 1)
    df_test['min'] = df_test.min(axis = 1)
    
    if save == True:
        fn = f'{station}_{ML}_prediction.csv'       
        df_result.to_csv(os.path.join(out_dir, fn))
        
        fn = f'{station}_{ML}_training.csv'
        df_train.to_csv(os.path.join(out_dir, fn))
        
        fn = f'{station}_{ML}_testing.csv'
        df_test.to_csv(os.path.join(out_dir, fn))
    
    if plot == True:
        plot_ensemble_performance(df_result, df_train, df_test, station, neurons, epochs, batch, resample,
                         tt_value, var_num, out_dir, layers=layers, ML=ML, test_on=test_on)
    
    return df_result, df_train, df_test


def plot_ensemble_performance(df, train_loss, test_loss, station, neurons, epochs, batch, resample,
                     tt_value, var_num, out_dir, layers=1, ML='LSTM', test_on='ensemble', logger=False):
    if resample == 'hourly':
        step = 24
    elif resample == 'daily':
        step = 1

    if logger:
        fig = plt.figure(figsize=[14.5, 9.5])
    else:
        fig = plt.figure(figsize=[14.5, 9.5])

    gs = GridSpec(3, 3)

    plot_ensemble_testing_ts(df, fig.add_subplot(gs[0, 0:2]))
    plot_ensemble_testing_max_ts(df, fig.add_subplot(gs[1, 0:2]), resample)
    plot_meta(fig.add_subplot(gs[0, 2]), station, neurons, epochs, batch, resample, tt_value, var_num)    
    plot_ensemble_metrics(df.dropna(axis=0, how='any'), fig.add_subplot(gs[1, 2]))
    plot_ensemble_scatter(df[['Observed', 'median']], fig.add_subplot(gs[2, 0]))
    plot_ensemble_qq(df.dropna(axis=0, how='any'), fig.add_subplot(gs[2, 1]))
    plot_ensemble_loss(train_loss, test_loss, fig.add_subplot(gs[2, 2]))

    fig.suptitle(station, fontsize=32)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # [left, bottom, right, top]
    fn = f'{station}_{ML}.png'
    plt.savefig(os.path.join(os.path.split(out_dir)[0], fn), dpi=100)
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

    rmse_all.loc['median',0] = np.around(np.float(rmse_all.median(axis = 0)),4)
    rmse_all.loc['max',0] = np.around(np.float(rmse_all.max(axis = 0)),4)
    rmse_all.loc['min',0] = np.around(np.float(rmse_all.min(axis = 0)),4)  
    NSE_all.loc['max',0] = np.around(np.float(NSE_all.max(axis = 0)),4)  
    NSE_all.loc['min',0] = np.around(np.float(NSE_all.min(axis = 0)),4)         
    R2_all.loc['max',0] = np.around(np.float(R2_all.max(axis = 0)),4)  
    R2_all.loc['min',0] = np.around(np.float(R2_all.min(axis = 0)),4)      
    mae_all.loc['max',0] = np.around(np.float(mae_all.max(axis = 0)),4)  
    mae_all.loc['min',0] = np.around(np.float(mae_all.min(axis = 0)),4)  
    NSE_all.loc['median',0] = np.around(np.float(NSE_all.median(axis = 0)),4)
    R2_all.loc['median',0] = np.around(np.float(R2_all.median(axis = 0)),4)
    mae_all.loc['median',0] = np.around(np.float(mae_all.median(axis = 0)),4)      
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

def plot_ensemble_testing_ts(df_result, ax):
    ax.fill_between(df_result.index, df_result['max'], df_result['min'], color='r', alpha = 0.5)
    ax.plot(df_result['median'].index, df_result['median'], '-r', linewidth=0.8)
    ax.plot(df_result['Observed'].index, df_result['Observed'], '-k', linewidth=0.5)

    ax.set_ylabel('Surge height (m)')
    ax.set_xlabel(None)  
    ax.xaxis.set_major_locator(months)
#    ax.xaxis.set_minor_locator(weeks)

def plot_ensemble_testing_max_ts(df_result, ax, resample):    
    max_time = df_result['Observed'].idxmax()
    
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
    
    ax.plot(q_obs, q_mod['min'], '-', c='k')
    ax.plot(q_obs, q_mod['max'], '-', c='k')
    ax.fill_between(q_obs, q_mod['min'], q_mod['max'], color='k', alpha = 0.5)
    ax.plot([q_obs.min(), q_obs.max()], [q_obs.min(), q_obs.max()], color='red')
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
    df_test.plot.scatter(x='Observed', y='median', ax=ax, c='k', alpha = 0.6, edgecolors = 'none')
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, 'r')

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


    return round(rmse, 4), round(NSE, 4), round(r2, 4), round(mae, 4)

def crps_metrics(df_test):
    # CRPS = Continuous Ranked Probability Score
    # CRPS is a generalization of mean absolute error
    # Using the properscoring package as in Bruneau et al. (2020)
    
    filter_col = [col for col in df_test if col.startswith('Modelled')]
    obs = df_test['Observed'].values
    mods = df_test[filter_col].values
    
    model_score = ps.crps_ensemble(obs, mods).mean() #returns a value per timestep. Here we take the mean of those values to get one value per location
    return round(model_score, 4)

def brier_metrics():
    #To be coded for the 95th percentile
    return

def ignorance_metric(df_test):
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
    return round(ls, 4)

def calc_skill(baseline_score, forecast_score):
    # Skill calculates the improvement of the forecast compared with a baseline forecast (ANN) in our case  
    # X% skill means that the forecast is X% better/worse than the baseline model (ANN)
    skill = (baseline_score - forecast_score) / baseline_score    
    return round(skill, 3)


# if __name__ == '__main__':
#     # parameters and variables
#     station = 'Cuxhaven'  # 'Cuxhaven' 'Hoek van Holland', Puerto Armuelles
#     resample = 'daily' # 'hourly' 'daily'
#     variables = ['msl', 'uquad', 'vquad', 'grad', 'rho', 'phi']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'
#     tt_value = 0.67 # train-test value
#     epochs = 50
#     batch = 25
#     neurons = 50
#     workspace = 'C:\\Users\\ttn430\\Documents\\Coastal'

#     df_hist = pd.read_excel('hist.xlsx')
#     train = df_hist['train'].values
#     test = df_hist['test'].values
#     df = pd.read_excel('cux.xlsx')
#     df = df.set_index('time')
#     plot_performance(df, train, test, station, neurons, epochs, batch, resample, tt_value,
#                      len(variables), workspace=workspace)
