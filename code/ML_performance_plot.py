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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,  mean_absolute_error, max_error


def plot_performance(df, train_loss, test_loss, station, neurons, epochs, batch, resample,
                     tt_value, var_num, workspace='cur', layers=1, ML='LSTM', test_on='self'):
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
    if workspace == 'cur':
        workspace = os.getcwd()
    
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
    fn = f'{station}_{ML}_n{neurons}_e{epochs}_b{batch}_var{var_num}_{resample}_l{layers}_t_{test_on}.png'
    plt.savefig(os.path.join(workspace, fn), dpi=100)
    plt.close()

def plot_meta(ax, station, neurons, epochs, batch, resample, tt_value, var_num):
	# hide axes
	ax.grid(False) 
	ax.axis('off')

	col_labels=['Metadata']
	row_labels=['Neurons', 'Batch', 'Epochs', 'tt_value', 'resample', 'variables']
	table_vals=[[neurons], [batch], [epochs], [tt_value], [resample], [var_num]]
	
	# the rectangle is where I want to place the table
	colors = plt.cm.Oranges([0.1, 0.1])
	table = ax.table(cellText=table_vals, colWidths = [0.1]*3, rowLabels=row_labels,
	                 loc='center')
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

	colors = plt.cm.Blues([0.1, 0.1])
	table = ax.table(cellText=table_vals, colWidths = [0.1]*3, rowLabels=row_labels,
	                 colLabels=col_labels, loc='center', colColours=colors)
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
	df_test.plot.scatter(x='Observed', y='Modelled', ax=ax)
	x = np.linspace(*ax.get_xlim())
	ax.plot(x, x, 'r')

def plot_qq(df_test, ax):
	q_obs = df_test['Observed'].quantile(np.arange(0.05, 0.96, 0.01)).values
	q_mod = df_test['Modelled'].quantile(np.arange(0.05, 0.96, 0.01)).values
	ax.scatter(q_obs, q_mod)
	ax.plot([q_obs.min(), q_obs.max()], [q_obs.min(), q_obs.max()], color='red')
	ax.set_xlabel('Observed')
	ax.set_ylabel('Modelled')

def performance_metrics(inv_y, inv_yhat):
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'Test RMSE: {rmse}')

    # calculate NSE
    df_nse = pd.DataFrame(zip(inv_yhat, inv_y), columns=['model', 'observed'])
    df_nse['dividend'] = np.power((df_nse['model'] - df_nse['observed']), 2)
    df_nse['divisor'] = np.power((df_nse['observed'] - df_nse['observed'].mean()), 2)
    NSE = 1 - (df_nse['dividend'].sum() / df_nse['divisor'].sum())
    print(f'Test NSE: {NSE}')

    # calculate R2
    r2 = r2_score(inv_y, inv_yhat)
    print(f'Test R2: {r2}')

    # calculate max error
    mae = mean_absolute_error(inv_y, inv_yhat)
    print(f'Test mean absolute error: {mae}')


    return round(rmse, 2), round(NSE, 2), round(r2, 2), round(mae, 3)

if __name__ == '__main__':
    # parameters and variables
    station = 'Cuxhaven'  # 'Cuxhaven' 'Hoek van Holland', Puerto Armuelles
    resample = 'daily' # 'hourly' 'daily'
    variables = ['msl', 'uquad', 'vquad', 'grad', 'rho', 'phi']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'
    tt_value = 0.67 # train-test value
    epochs = 50
    batch = 25
    neurons = 50
    workspace = 'C:\\Users\\ttn430\\Documents\\Coastal'

    df_hist = pd.read_excel('hist.xlsx')
    train = df_hist['train'].values
    test = df_hist['test'].values
    df = pd.read_excel('cux.xlsx')
    df = df.set_index('time')
    plot_performance(df, train, test, station, neurons, epochs, batch, resample, tt_value,
                     len(variables), workspace=workspace)
